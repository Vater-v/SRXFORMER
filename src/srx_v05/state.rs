//! State and Scratchpad Workspace buffers for SRXFORMER v05 (Quantum-Algebraic Core).
//! Provides strictly O(1) context state (EXACTLY 160 bytes: 100% L1D cache resident),
//! and zero-allocation scratchpad buffers for hot-path inference and forward steps.

use crate::classic::cache::MlpWorkspace;
use crate::classic::config::TransformerConfig;

/// Internal recurrent memory state for SRX v05 attention.
/// In contrast to Classical Transformer KV-cache that grows O(N * d),
/// SRX v05 state is strictly O(1) in sequence length N, storing:
/// - Butterfly phase angles Thetas: [n_heads, 4] (32 bytes)
/// - Associative value memory matrix M: [n_heads, 4, 4] (128 bytes)
/// Total core associative state footprint: EXACTLY 160 bytes (100% L1D Cache resident, < 0.5% of 32 KB).
#[derive(Debug, Clone, PartialEq)]
pub struct SrxState {
    /// Number of attention heads (2)
    pub n_heads: usize,
    /// Head dimension d (4)
    pub head_dim: usize,
    /// Butterfly Phase angles Theta: [n_heads, 4] (32 bytes)
    pub thetas: Vec<f32>,
    /// Associative value memory matrix M: [n_heads, head_dim, head_dim] (128 bytes)
    pub m: Vec<f32>,
    /// Current sequence token position
    pub current_pos: usize,
}

impl SrxState {
    /// Allocates initial state for the given configuration.
    /// Memory matrix M is initialized to zero; Thetas initialized to zero.
    pub fn new(config: &TransformerConfig) -> Self {
        let n_heads = config.n_heads;
        let head_dim = config.head_dim();
        assert_eq!(head_dim, 4, "SRX v05 is optimized for head_dim = 4");

        let thetas_len = n_heads * head_dim;
        let m_len = n_heads * head_dim * head_dim;

        Self {
            n_heads,
            head_dim,
            thetas: vec![0.0; thetas_len],
            m: vec![0.0; m_len],
            current_pos: 0,
        }
    }

    /// Resets all angles and memory matrices to initial states:
    /// Thetas -> 0, M -> 0.
    pub fn reset(&mut self) {
        self.thetas.fill(0.0);
        self.m.fill(0.0);
        self.current_pos = 0;
    }

    /// Total memory occupied by the core associative state (Thetas + M) in bytes.
    /// In target config (H=2, d_head=4): (8 + 32) * 4 = EXACTLY 160 bytes!
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        (self.thetas.len() + self.m.len()) * std::mem::size_of::<f32>()
    }

    /// Immutable slice to Theta angles for a given head as a fixed 4-element array.
    #[inline]
    pub fn thetas_head_4(&self, head: usize) -> &[f32; 4] {
        let span = 4;
        let slice = &self.thetas[head * span..(head + 1) * span];
        slice.try_into().expect("head theta slice must be 4 elements")
    }

    /// Mutable slice to Theta angles for a given head as a fixed 4-element array.
    #[inline]
    pub fn thetas_head_4_mut(&mut self, head: usize) -> &mut [f32; 4] {
        let span = 4;
        let slice = &mut self.thetas[head * span..(head + 1) * span];
        slice.try_into().expect("head theta slice must be 4 elements")
    }

    /// Immutable slice to matrix M for a given head [head_dim, head_dim].
    #[inline]
    pub fn m_head(&self, head: usize) -> &[f32] {
        let span = self.head_dim * self.head_dim;
        &self.m[head * span..(head + 1) * span]
    }

    /// Mutable slice to matrix M for a given head [head_dim, head_dim].
    #[inline]
    pub fn m_head_mut(&mut self, head: usize) -> &mut [f32] {
        let span = self.head_dim * self.head_dim;
        &mut self.m[head * span..(head + 1) * span]
    }

    /// Immutable slice to matrix M for a given head as a fixed 16-element array.
    #[inline]
    pub fn m_head_16(&self, head: usize) -> &[f32; 16] {
        let span = 16;
        let slice = &self.m[head * span..(head + 1) * span];
        slice.try_into().expect("head M slice must be 16 elements")
    }

    /// Mutable slice to matrix M for a given head as a fixed 16-element array.
    #[inline]
    pub fn m_head_16_mut(&mut self, head: usize) -> &mut [f32; 16] {
        let span = 16;
        let slice = &mut self.m[head * span..(head + 1) * span];
        slice.try_into().expect("head M slice must be 16 elements")
    }

    /// Simultaneously borrows mutable references to (Thetas, M) for head.
    #[inline]
    pub fn thetas_and_m_mut(&mut self, head: usize) -> (&mut [f32; 4], &mut [f32; 16]) {
        let th_span = 4;
        let m_span = self.head_dim * self.head_dim;
        let t_slice: &mut [f32; 4] = (&mut self.thetas[head * th_span..(head + 1) * th_span])
            .try_into()
            .expect("head theta slice must be 4 elements");
        let m_slice: &mut [f32; 16] = (&mut self.m[head * m_span..(head + 1) * m_span])
            .try_into()
            .expect("head M slice must be 16 elements");
        (t_slice, m_slice)
    }
}

/// Pre-allocated scratchpad buffers for SRX v05 inference and forward unroll.
/// Completely eliminates all dynamic heap allocations on the hot path.
#[derive(Debug, Clone)]
pub struct SrxWorkspace {
    pub max_seq_len: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,

    // Step mode single-token scratchpads
    pub step_x: Vec<f32>,
    pub step_norm_out: Vec<f32>,
    pub step_residual: Vec<f32>,
    pub step_attn_out: Vec<f32>,
    pub step_mlp_hidden: Vec<f32>,
    pub step_mlp_out: Vec<f32>,
    pub step_logits: Vec<f32>,

    // Sequence forward mode buffers
    pub x: Vec<f32>,
    pub norm_out: Vec<f32>,
    pub residual: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub mlp_out: Vec<f32>,
    pub logits: Vec<f32>,

    // Sublayer scratchpads
    pub mlp: MlpWorkspace,
    pub head_buf: Vec<f32>,
}

impl SrxWorkspace {
    /// Allocates workspace buffers configured to model constraints.
    pub fn new(config: &TransformerConfig) -> Self {
        let s = config.max_seq_len;
        let d = config.d_model;
        let d_ff = config.d_ff;
        let v = config.vocab_size;
        let n_heads = config.n_heads;
        let head_dim = config.head_dim();

        Self {
            max_seq_len: s,
            d_model: d,
            d_ff,
            vocab_size: v,
            n_heads,
            head_dim,
            step_x: vec![0.0; d],
            step_norm_out: vec![0.0; d],
            step_residual: vec![0.0; d],
            step_attn_out: vec![0.0; d],
            step_mlp_hidden: vec![0.0; d_ff],
            step_mlp_out: vec![0.0; d],
            step_logits: vec![0.0; v],
            x: vec![0.0; s * d],
            norm_out: vec![0.0; s * d],
            residual: vec![0.0; s * d],
            attn_out: vec![0.0; s * d],
            mlp_out: vec![0.0; s * d],
            logits: vec![0.0; s * v],
            mlp: MlpWorkspace::new(config),
            head_buf: vec![0.0; d * 4],
        }
    }

    /// Resets workspace buffers to zero.
    pub fn reset(&mut self) {
        self.step_x.fill(0.0);
        self.step_norm_out.fill(0.0);
        self.step_residual.fill(0.0);
        self.step_attn_out.fill(0.0);
        self.step_mlp_hidden.fill(0.0);
        self.step_mlp_out.fill(0.0);
        self.step_logits.fill(0.0);
        self.x.fill(0.0);
        self.norm_out.fill(0.0);
        self.residual.fill(0.0);
        self.attn_out.fill(0.0);
        self.mlp_out.fill(0.0);
        self.logits.fill(0.0);
        self.head_buf.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srx_v05_state_size() {
        let cfg = TransformerConfig::lang_chinchilla();
        let state = SrxState::new(&cfg);

        // H = 2, d_head = 4
        // thetas: 2 * 4 = 8 floats = 32 bytes
        // m: 2 * 4 * 4 = 32 floats = 128 bytes
        // Total core associative state = 40 floats = EXACTLY 160 bytes!
        assert_eq!(state.thetas.len(), 8);
        assert_eq!(state.m.len(), 32);
        assert_eq!(state.memory_bytes(), 160);
        assert!(
            state.memory_bytes() <= 160,
            "SRX v05 state must fit in strictly 160 bytes (< 0.5% of 32KB L1D cache)!"
        );
    }
}
