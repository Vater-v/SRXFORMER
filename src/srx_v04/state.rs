//! State and Scratchpad Workspace buffers for SRXFORMER v04.
//! Provides strictly O(1) context state (160 bytes: 100% L1D cache resident),
//! and zero-allocation scratchpad buffers for hot-path inference and forward steps.

use crate::classic::cache::MlpWorkspace;
use crate::classic::config::TransformerConfig;

/// Internal recurrent memory state for SRX v04 attention.
/// In contrast to Classical Transformer KV-cache that grows O(N * d),
/// SRX state is strictly O(1) in sequence length N, storing ONLY Butterfly phase angles Theta and associative matrix M.
#[derive(Debug, Clone, PartialEq)]
pub struct SrxState {
    /// Number of attention heads
    pub n_heads: usize,
    /// Head dimension d (4)
    pub head_dim: usize,
    /// Butterfly Phase angles Theta: [n_heads, 4] (exact 4 angles per head)
    pub thetas: Vec<f32>,
    /// Associative value memory matrix M: [n_heads, head_dim, head_dim]
    pub m: Vec<f32>,
    /// Current sequence token position
    pub current_pos: usize,
}

impl SrxState {
    /// Allocates initial zero state for the given configuration.
    pub fn new(config: &TransformerConfig) -> Self {
        let n_heads = config.n_heads;
        let head_dim = config.head_dim();
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

    /// Resets all angles and memory matrices to zero.
    pub fn reset(&mut self) {
        self.thetas.fill(0.0);
        self.m.fill(0.0);
        self.current_pos = 0;
    }

    /// Total heap memory occupied by this state in bytes.
    /// In default config (H=2, d_head=4): (8 + 32) * 4 = 160 bytes!
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

    /// Simultaneously borrows mutable slices to Theta and M for head `head`.
    #[inline]
    pub fn head_state_mut(&mut self, head: usize) -> (&mut [f32; 4], &mut [f32]) {
        let th_span = 4;
        let m_span = self.head_dim * self.head_dim;
        let thetas_slice = &mut self.thetas[head * th_span..(head + 1) * th_span];
        let thetas_arr: &mut [f32; 4] = thetas_slice.try_into().expect("head thetas slice must be 4");
        let m_slice = &mut self.m[head * m_span..(head + 1) * m_span];
        (thetas_arr, m_slice)
    }
}

/// Pre-allocated scratchpad buffers for SRX v04 inference and forward unroll.
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
    fn test_srx_v04_state_size() {
        let cfg = TransformerConfig::srx_v04_parity();
        let state = SrxState::new(&cfg);

        // H = 2, d_head = 4
        // thetas: 2 * 4 = 8 floats = 32 bytes
        // m: 2 * 4 * 4 = 32 floats = 128 bytes
        // Total = 40 floats = 160 bytes!
        assert_eq!(state.thetas.len(), 8);
        assert_eq!(state.m.len(), 32);
        assert_eq!(state.memory_bytes(), 160);
        assert!(state.memory_bytes() <= 256, "SRX v04 state fits in tiny fraction of L1 Cache!");
    }
}
