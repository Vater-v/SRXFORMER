//! State and Scratchpad Workspace buffers for SRXFORMER v05 (Quantum-Algebraic Core).
//! Provides strictly O(1) context state (288 bytes: 100% L1D cache resident),
//! and zero-allocation scratchpad buffers for hot-path inference and forward steps.

use crate::classic::cache::MlpWorkspace;
use crate::classic::config::TransformerConfig;

/// Internal recurrent memory state for SRX v05 attention.
/// In contrast to Classical Transformer KV-cache that grows O(N * d),
/// SRX v05 state is strictly O(1) in sequence length N, storing:
/// - Butterfly phase angles Thetas: [n_heads, 4] (32 bytes)
/// - Associative value memory matrix M: [n_heads, 4, 4] (128 bytes)
/// - Recursive Least Squares inverse covariance matrix P: [n_heads, 4, 4] (128 bytes)
/// Total core associative state footprint: EXACTLY 288 bytes (100% L1D Cache resident).
#[derive(Debug, Clone, PartialEq)]
pub struct SrxState {
    /// Number of attention heads (2)
    pub n_heads: usize,
    /// Head dimension d (4)
    pub head_dim: usize,
    /// Butterfly Phase angles Theta: [n_heads, 4] (32 bytes)
    pub thetas: Vec<f32>,
    /// Butterfly Phase momentum p_theta: [n_heads, 4]
    pub p_thetas: Vec<f32>,
    /// Associative value memory matrix M: [n_heads, head_dim, head_dim] (128 bytes)
    pub m: Vec<f32>,
    /// RLS inverse covariance matrix P: [n_heads, head_dim, head_dim] (128 bytes)
    pub p: Vec<f32>,
    /// Current sequence token position
    pub current_pos: usize,
}

impl SrxState {
    /// Allocates initial state for the given configuration.
    /// P is initialized to delta^{-1} * I = I_{4x4}.
    pub fn new(config: &TransformerConfig) -> Self {
        let n_heads = config.n_heads;
        let head_dim = config.head_dim();
        assert_eq!(head_dim, 4, "SRX v05 is optimized for head_dim = 4");

        let thetas_len = n_heads * head_dim;
        let m_len = n_heads * head_dim * head_dim;
        let p_len = n_heads * head_dim * head_dim;

        let mut p = vec![0.0; p_len];
        // Initialize P_0 = I_{4x4} for each head
        for h in 0..n_heads {
            let h_off = h * head_dim * head_dim;
            for i in 0..head_dim {
                p[h_off + i * head_dim + i] = 1.0;
            }
        }

        Self {
            n_heads,
            head_dim,
            thetas: vec![0.0; thetas_len],
            p_thetas: vec![0.0; thetas_len],
            m: vec![0.0; m_len],
            p,
            current_pos: 0,
        }
    }

    /// Resets all angles and memory matrices to initial states:
    /// Thetas -> 0, p_thetas -> 0, M -> 0, P -> I_{4x4}.
    pub fn reset(&mut self) {
        self.thetas.fill(0.0);
        self.p_thetas.fill(0.0);
        self.m.fill(0.0);
        self.p.fill(0.0);
        for h in 0..self.n_heads {
            let h_off = h * self.head_dim * self.head_dim;
            for i in 0..self.head_dim {
                self.p[h_off + i * self.head_dim + i] = 1.0;
            }
        }
        self.current_pos = 0;
    }

    /// Total memory occupied by the core associative state (Thetas + M + P) in bytes.
    /// In target config (H=2, d_head=4): (8 + 32 + 32) * 4 = EXACTLY 288 bytes!
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        (self.thetas.len() + self.m.len() + self.p.len()) * std::mem::size_of::<f32>()
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

    /// Mutable slice to Phase Momentum for a given head as a fixed 4-element array.
    #[inline]
    pub fn p_thetas_head_4_mut(&mut self, head: usize) -> &mut [f32; 4] {
        let span = 4;
        let slice = &mut self.p_thetas[head * span..(head + 1) * span];
        slice.try_into().expect("head p_theta slice must be 4 elements")
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

    /// Immutable slice to covariance matrix P for a given head [head_dim, head_dim].
    #[inline]
    pub fn p_head(&self, head: usize) -> &[f32] {
        let span = self.head_dim * self.head_dim;
        &self.p[head * span..(head + 1) * span]
    }

    /// Mutable slice to covariance matrix P for a given head [head_dim, head_dim].
    #[inline]
    pub fn p_head_mut(&mut self, head: usize) -> &mut [f32] {
        let span = self.head_dim * self.head_dim;
        &mut self.p[head * span..(head + 1) * span]
    }

    /// Simultaneously borrows mutable references to (Thetas, p_thetas) for head.
    #[inline]
    pub fn thetas_and_p_thetas_mut(&mut self, head: usize) -> (&mut [f32; 4], &mut [f32; 4]) {
        let span = 4;
        let t_slice: &mut [f32; 4] = (&mut self.thetas[head * span..(head + 1) * span])
            .try_into()
            .expect("head theta slice must be 4 elements");
        let p_slice: &mut [f32; 4] = (&mut self.p_thetas[head * span..(head + 1) * span])
            .try_into()
            .expect("head p_theta slice must be 4 elements");
        (t_slice, p_slice)
    }

    /// Simultaneously borrows mutable references to (M, P) for head.
    #[inline]
    pub fn m_and_p_mut(&mut self, head: usize) -> (&mut [f32], &mut [f32]) {
        let span = self.head_dim * self.head_dim;
        let m_slice = &mut self.m[head * span..(head + 1) * span];
        let p_slice = &mut self.p[head * span..(head + 1) * span];
        (m_slice, p_slice)
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
        let cfg = TransformerConfig::lang_v3();
        let state = SrxState::new(&cfg);

        // H = 2, d_head = 4
        // thetas: 2 * 4 = 8 floats = 32 bytes
        // m: 2 * 4 * 4 = 32 floats = 128 bytes
        // p: 2 * 4 * 4 = 32 floats = 128 bytes
        // Total core associative state = 72 floats = EXACTLY 288 bytes!
        assert_eq!(state.thetas.len(), 8);
        assert_eq!(state.m.len(), 32);
        assert_eq!(state.p.len(), 32);
        assert_eq!(state.memory_bytes(), 288);
        assert!(state.memory_bytes() <= 512, "SRX v05 state fits in tiny fraction of L1 Cache (< 1% of 32KB)!");

        // Verify P_0 initialization: identity matrix
        for h in 0..2 {
            let p_h = state.p_head(h);
            for i in 0..4 {
                for j in 0..4 {
                    if i == j {
                        assert_eq!(p_h[i * 4 + j], 1.0);
                    } else {
                        assert_eq!(p_h[i * 4 + j], 0.0);
                    }
                }
            }
        }
    }
}
