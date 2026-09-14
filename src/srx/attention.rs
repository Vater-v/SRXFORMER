//! SRX Attention: Super-Resolvent Unitary Givens Attention with MUSIC Subspace Resonance.
//! Replaces Softmax Attention with Unitary Phase Modulation and Noise Subspace Projection.

use crate::classic::config::TransformerConfig;
use crate::classic::ops::matvec;
use crate::classic::rng::FastRng;
use super::ops::{apply_givens, l2_normalize};
use super::state::SrxState;

/// Phase modulation rate alpha
pub const SRX_ALPHA: f32 = 0.1;
/// Memory decay factor gamma
pub const SRX_GAMMA: f32 = 0.99;
/// Default inference epsilon for Dirac-like peak
pub const SRX_EPS_DEFAULT: f32 = 1e-4;

/// Super-Resolvent xFormer Attention layer.
/// Contains projection matrices W_q, W_k, W_v, W_o (no biases).
#[derive(Debug, Clone)]
pub struct SrxAttention {
    pub w_q: Vec<f32>, // [d_model, d_model]
    pub w_k: Vec<f32>, // [d_model, d_model]
    pub w_v: Vec<f32>, // [d_model, d_model]
    pub w_o: Vec<f32>, // [d_model, d_model]

    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub alpha: f32,
    pub gamma: f32,
}

impl SrxAttention {
    /// Constructs a new SRX attention sublayer initialized from FastRng.
    pub fn new(config: &TransformerConfig, rng: &mut FastRng) -> Self {
        let d = config.d_model;
        let std = 1.0 / (d as f32).sqrt();

        let w_q = (0..d * d).map(|_| rng.gen_normal(0.0, std)).collect();
        let w_k = (0..d * d).map(|_| rng.gen_normal(0.0, std)).collect();
        let w_v = (0..d * d).map(|_| rng.gen_normal(0.0, std)).collect();
        let w_o = (0..d * d).map(|_| rng.gen_normal(0.0, std)).collect();

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            d_model: d,
            n_heads: config.n_heads,
            head_dim: config.head_dim(),
            alpha: SRX_ALPHA,
            gamma: SRX_GAMMA,
        }
    }

    /// Total number of parameters in this attention layer (4 * d_model^2).
    #[inline]
    pub fn param_count(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len()
    }

    /// Single autoregressive step: computes output token projection and updates state in O(1).
    /// `x`: [d_model] input vector.
    /// `state`: mutable reference to SrxState (updated in-place).
    /// `out`: [d_model] output projection buffer.
    pub fn step(
        &self,
        x: &[f32],
        state: &mut SrxState,
        out: &mut [f32],
        eps: f32,
    ) {
        let d = self.d_model;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let r = head_dim / 2; // Signal rank = d / 2, remainder is noise subspace

        // Temporary scratch for Q, K, V [d_model]
        let mut q = vec![0.0f32; d];
        let mut k = vec![0.0f32; d];
        let mut v = vec![0.0f32; d];
        let mut y_heads = vec![0.0f32; d];

        matvec(&mut q, &self.w_q, x, None, d, d);
        matvec(&mut k, &self.w_k, x, None, d, d);
        matvec(&mut v, &self.w_v, x, None, d, d);

        for h in 0..n_heads {
            let h_off = h * head_dim;
            let q_h = &mut q[h_off..h_off + head_dim];
            let k_h = &mut k[h_off..h_off + head_dim];
            let v_h = &v[h_off..h_off + head_dim];

            // 1. L2-normalize k and q
            let mut q_norm = vec![0.0f32; head_dim];
            let mut k_norm = vec![0.0f32; head_dim];
            l2_normalize(q_h, &mut q_norm, 1e-12);
            l2_normalize(k_h, &mut k_norm, 1e-12);

            // 2. Phase update: delta_theta = alpha * tanh(k[:-1] * v[:-1])
            let (thetas, m) = state.head_state_mut(h);
            for i in 0..head_dim - 1 {
                let delta = self.alpha * (k_norm[i] * v_h[i]).tanh();
                thetas[i] += delta;
            }

            // 3. Rotate k into unitary basis: k_rot = U(Theta) k_norm
            let mut k_rot = k_norm.clone();
            apply_givens(&mut k_rot, thetas, false);

            // 4. Associative memory accumulation: M = gamma * M + k_rot * v^T
            for j in 0..head_dim {
                for col in 0..head_dim {
                    let idx = j * head_dim + col;
                    m[idx] = self.gamma * m[idx] + k_rot[j] * v_h[col];
                }
            }

            // 5. MUSIC Noise Subspace projector
            // q_inv = U^\dagger(Theta) q_norm
            let mut q_inv = q_norm.clone();
            apply_givens(&mut q_inv, thetas, true);

            let mut noise_energy = 0.0f32;
            for j in r..head_dim {
                noise_energy += q_inv[j] * q_inv[j];
            }

            // Dirac-like resonant gain
            let w = 1.0 / (noise_energy + eps);

            // 6. Memory retrieval: y_ret = M^T (U(Theta) q_norm)
            let mut q_rot = q_norm;
            apply_givens(&mut q_rot, thetas, false);

            let out_h = &mut y_heads[h_off..h_off + head_dim];
            for col in 0..head_dim {
                let mut sum_m = 0.0f32;
                for j in 0..head_dim {
                    sum_m += q_rot[j] * m[j * head_dim + col];
                }
                out_h[col] = sum_m * w;
            }
        }

        // 7. Output projection: W_o y
        matvec(out, &self.w_o, &y_heads, None, d, d);
    }

    /// Sequence forward pass: sequentially unrolls causal state for tokens 0 .. seq_len - 1.
    pub fn forward_causal(
        &self,
        x_seq: &[f32],
        seq_len: usize,
        out_seq: &mut [f32],
        eps: f32,
    ) {
        let d = self.d_model;
        let cfg = TransformerConfig {
            vocab_size: 21,
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_layers: 1,
            d_ff: 4,
            max_seq_len: seq_len.max(32),
            eps: 1e-5,
            norm_type: crate::classic::config::NormType::RMSNorm,
            activation: crate::classic::config::ActivationType::Relu,
            pos_encoding: crate::classic::config::PosEncodingType::Sinusoidal,
            tie_word_embeddings: true,
            use_bias: false,
        };
        let mut state = SrxState::new(&cfg);

        for t in 0..seq_len {
            let x_tok = &x_seq[t * d..(t + 1) * d];
            let out_tok = &mut out_seq[t * d..(t + 1) * d];
            self.step(x_tok, &mut state, out_tok, eps);
            state.current_pos = t + 1;
        }
    }
}
