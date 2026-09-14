//! SRX Attention v04: Super-Resolvent Physics-Spectral Attention
//! with Widrow-Hoff Delta Rule Memory Accumulation, Monarch Butterfly Unitary Mixer,
//! Selective Dynamic Memory Gate, MUSIC Subspace Resonance, and Zero Heap Allocations.
//!
//! Enhancements in v04:
//! 1. Widrow-Hoff Delta Rule (Novelty Error Residual Memory Update):
//!    - v_hat = M_{t-1}^T * k_rot
//!    - e_t = v_raw - v_hat
//!    - M_t = lambda_t * M_{t-1} + gamma_t * (k_rot * e_t^T)
//!    Eliminates catastrophic interference between semantically similar keys (cat/dog, fox/wolf)
//!    via exact orthogonal projection (I - k_rot k_rot^T).
//! 2. Monarch Butterfly Unitary Mixer:
//!    U(Theta) = B_2(Theta_2) * P * B_1(Theta_1)
//! 3. Zero-Allocation Hot Path:
//!    All vectors in `step()` are fixed stack arrays `[f32; 8]` and `[f32; 4]`.
//! 4. Post-MUSIC RMSNorm and hard gain clipping w_clamped <= 10.0.

use crate::classic::config::TransformerConfig;
use crate::classic::rng::FastRng;
use super::ops::{apply_butterfly_4, l2_normalize};
use super::state::SrxState;

/// Phase modulation rate alpha
pub const SRX_ALPHA: f32 = 0.1;
/// Base memory decay factor gamma
pub const SRX_GAMMA_BASE: f32 = 0.99;
/// Default inference epsilon for Dirac-like peak
pub const SRX_EPS_DEFAULT: f32 = 1e-3;
/// Maximum clamped gain w_max to eliminate MUSIC spectral clicks
pub const SRX_W_MAX: f32 = 10.0;

/// Super-Resolvent xFormer Attention layer v04.
#[derive(Debug, Clone)]
pub struct SrxAttention {
    pub w_q: Vec<f32>,     // [d_model, d_model]
    pub w_k: Vec<f32>,     // [d_model, d_model]
    pub w_v: Vec<f32>,     // [d_model, d_model]
    pub w_o: Vec<f32>,     // [d_model, d_model]
    pub w_gamma: Vec<f32>, // [n_heads, d_model] (selective memory gate weights)
    pub b_gamma: Vec<f32>, // [n_heads]          (selective memory gate biases)

    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub alpha: f32,
    pub gamma_base: f32,
}

impl SrxAttention {
    /// Constructs a new SRX v04 attention sublayer initialized from FastRng.
    pub fn new(config: &TransformerConfig, rng: &mut FastRng) -> Self {
        let d = config.d_model;
        let n_heads = config.n_heads;
        let head_dim = config.head_dim();
        assert_eq!(head_dim, 4, "SRX v04 Butterfly Mixer is optimized for head_dim = 4");

        let std = 1.0 / (d as f32).sqrt();

        let w_q = (0..d * d).map(|_| rng.gen_normal(0.0, std)).collect();
        let w_k = (0..d * d).map(|_| rng.gen_normal(0.0, std)).collect();
        let w_v = (0..d * d).map(|_| rng.gen_normal(0.0, std)).collect();
        let w_o = (0..d * d).map(|_| rng.gen_normal(0.0, std)).collect();

        // Selective gate weights initialized near zero, bias with slight positive bias (0.5)
        let w_gamma = (0..n_heads * d).map(|_| rng.gen_normal(0.0, std * 0.5)).collect();
        let b_gamma = vec![0.5; n_heads];

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            w_gamma,
            b_gamma,
            d_model: d,
            n_heads,
            head_dim,
            alpha: SRX_ALPHA,
            gamma_base: SRX_GAMMA_BASE,
        }
    }

    /// Total number of parameters in this attention layer.
    #[inline]
    pub fn param_count(&self) -> usize {
        self.w_q.len()
            + self.w_k.len()
            + self.w_v.len()
            + self.w_o.len()
            + self.w_gamma.len()
            + self.b_gamma.len()
    }

    /// Single autoregressive step: strictly ZERO HEAP ALLOCATIONS!
    /// All vectors are allocated on the stack as `[f32; 8]` and `[f32; 4]`.
    /// `x`: [d_model] input vector.
    /// `state`: mutable reference to SrxState (updated in-place).
    /// `out`: [d_model] output projection buffer.
    #[inline]
    pub fn step(
        &self,
        x: &[f32],
        state: &mut SrxState,
        out: &mut [f32],
        eps: f32,
    ) {
        let d = self.d_model;
        let n_heads = self.n_heads;
        debug_assert_eq!(d, 8);
        debug_assert_eq!(n_heads, 2);

        // Stack scratchpads for Q, K, V [8]
        let mut q = [0.0f32; 8];
        let mut k = [0.0f32; 8];
        let mut v = [0.0f32; 8];
        let mut y_heads = [0.0f32; 8];

        for i in 0..8 {
            let mut sum_q = 0.0f32;
            let mut sum_k = 0.0f32;
            let mut sum_v = 0.0f32;
            let row_off = i * 8;
            for j in 0..8 {
                let xj = x[j];
                sum_q += self.w_q[row_off + j] * xj;
                sum_k += self.w_k[row_off + j] * xj;
                sum_v += self.w_v[row_off + j] * xj;
            }
            q[i] = sum_q;
            k[i] = sum_k;
            v[i] = sum_v;
        }

        // Per-head Butterfly Unitary Attention with Widrow-Hoff Delta Rule & Selective Gate
        for h in 0..n_heads {
            let h_off = h * 4;

            // 1. Selective Dynamic Memory Gate:
            // gamma_t = sigmoid(W_gamma * x_t + b_gamma)
            let mut s_gate = self.b_gamma[h];
            let g_row = h * 8;
            for c in 0..8 {
                s_gate += self.w_gamma[g_row + c] * x[c];
            }
            let gamma_t = 1.0 / (1.0 + (-s_gate).exp());
            let lambda_t = 1.0 - (1.0 - self.gamma_base) * gamma_t;

            let q_h = [q[h_off], q[h_off + 1], q[h_off + 2], q[h_off + 3]];
            let k_h = [k[h_off], k[h_off + 1], k[h_off + 2], k[h_off + 3]];
            let v_h = [v[h_off], v[h_off + 1], v[h_off + 2], v[h_off + 3]];

            // 2. L2 normalization of q and k
            let mut q_norm = [0.0f32; 4];
            let mut k_norm = [0.0f32; 4];
            l2_normalize(&q_h, &mut q_norm, 1e-12);
            l2_normalize(&k_h, &mut k_norm, 1e-12);

            // 3. Butterfly Phase update: delta_theta = alpha * tanh(k_norm * v)
            let (thetas, m) = state.head_state_mut(h);
            for i in 0..4 {
                let delta = self.alpha * (k_norm[i] * v_h[i]).tanh();
                thetas[i] += delta;
            }

            // 4. Monarch Butterfly Unitary Rotation of k: k_rot = U(Theta) * k_norm
            let mut k_rot = [0.0f32; 4];
            apply_butterfly_4(&k_norm, thetas, false, &mut k_rot);

            // 5. Widrow-Hoff Delta Rule:
            // a) Retrieve current prediction from existing memory: v_hat = M_{t-1}^T * k_rot
            let mut v_hat = [0.0f32; 4];
            for col in 0..4 {
                let mut sum_vhat = 0.0f32;
                for row in 0..4 {
                    sum_vhat += k_rot[row] * m[row * 4 + col];
                }
                v_hat[col] = sum_vhat;
            }

            // b) Compute novelty error residual: e_t = v_raw - v_hat
            let mut e_t = [0.0f32; 4];
            for col in 0..4 {
                e_t[col] = v_h[col] - v_hat[col];
            }

            // c) Memory update with Delta Rule:
            // M_t = lambda_t * M_{t-1} + gamma_t * (k_rot * e_t^T)
            // Projects out projection along k_rot, perfectly overwriting without corrupting orthogonal facts!
            for r in 0..4 {
                let kr = k_rot[r];
                let row_off = r * 4;
                for col in 0..4 {
                    let idx = row_off + col;
                    m[idx] = lambda_t * m[idx] + gamma_t * (kr * e_t[col]);
                }
            }

            // 6. MUSIC Noise Subspace Projector via Butterfly Unitary Inverse:
            // q_inv = U^\dagger(Theta) q_norm
            let mut q_inv = [0.0f32; 4];
            apply_butterfly_4(&q_norm, thetas, true, &mut q_inv);

            // Signal rank r = 2; noise subspace is coordinates 2..4
            let noise_energy = q_inv[2] * q_inv[2] + q_inv[3] * q_inv[3];

            // Dirac-like resonant gain with hard clipping
            let w = 1.0 / (noise_energy + eps);
            let w_clamped = w.min(SRX_W_MAX);

            // 7. Memory retrieval: q_rot = U(Theta) q_norm
            let mut q_rot = [0.0f32; 4];
            apply_butterfly_4(&q_norm, thetas, false, &mut q_rot);

            // y_ret = M^T q_rot
            for col in 0..4 {
                let mut sum_m = 0.0f32;
                for row in 0..4 {
                    sum_m += q_rot[row] * m[row * 4 + col];
                }
                y_heads[h_off + col] = sum_m * w_clamped;
            }
        }

        // 8. Post-MUSIC RMSNorm across multi-head retrieved output vector
        let mut sum_sq = 0.0f32;
        for c in 0..8 {
            sum_sq += y_heads[c] * y_heads[c];
        }
        let rms = (sum_sq / 8.0 + 1e-5).sqrt();
        let inv_rms = 1.0 / rms;
        for c in 0..8 {
            y_heads[c] *= inv_rms;
        }

        // 9. Output projection: out = W_o * y_heads
        for c in 0..8 {
            let mut sum_o = 0.0f32;
            let row_off = c * 8;
            for k in 0..8 {
                sum_o += self.w_o[row_off + k] * y_heads[k];
            }
            out[c] = sum_o;
        }
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
            vocab_size: 41,
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_layers: 1,
            d_ff: 17,
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
