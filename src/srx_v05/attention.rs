//! SRX Attention v05: Quantum-Algebraic Core
//! with 2nd-Order Associative RLS Memory (Sherman-Morrison), Krylov Recurrent Depth (K=2),
//! Monarch Butterfly Unitary Mixer with Phase Momentum, MUSIC Subspace Resonance,
//! and Strictly Zero Heap Allocations on the Hot Path.
//!
//! Architectural Innovations in v05:
//! 1. Associative RLS-Memory 2nd Order:
//!    - Inverse covariance matrix P_t \in R^{4x4} per head (P_0 = delta^{-1} I, lambda = 0.999).
//!    - Online Sherman-Morrison rank-1 update:
//!      v_p = P_{t-1} * k_rot
//!      denom = lambda + k_rot^T * v_p
//!      k_gain = v_p / denom
//!      e_t = v_raw - M_{t-1}^T * k_rot
//!      M_t = lambda * M_{t-1} + k_gain * e_t^T
//!      P_t = (1 / lambda) * (P_{t-1} - k_gain * (k_rot^T * P_{t-1}))
//!    Zero learned gating parameters needed (100% algebraic and mathematically closed).
//! 2. Krylov Recurrent Depth (K=2):
//!    - q^{(0)} = q_norm
//!    - q^{(1)} = L2_Norm(0.5 * q^{(0)} + 0.5 * (U_t * q^{(0)}))
//!    Refines query vector through unitary resolvent subspace before MUSIC projection and retrieval.
//! 3. Phase Momentum:
//!    - p_{\theta, t} = mu * p_{\theta, t-1} + alpha * (k_norm \odot v_raw[:4]), mu=0.85, alpha=0.1
//!    - \theta_t = \theta_{t-1} + p_{\theta, t}
//! 4. Strictly 288 Bytes Context State (100% L1D cache resident).

use crate::classic::config::TransformerConfig;
use crate::classic::rng::FastRng;
use super::ops::{apply_butterfly_4, l2_normalize};
use super::state::SrxState;

/// Phase modulation rate alpha
pub const SRX_ALPHA: f32 = 0.1;
/// Phase momentum coefficient mu
pub const SRX_MU: f32 = 0.85;
/// RLS forgetting factor lambda
pub const SRX_RLS_LAMBDA: f32 = 0.999;
/// RLS initial regularization delta
pub const SRX_RLS_DELTA: f32 = 1.0;
/// Default inference epsilon for Dirac-like peak
pub const SRX_EPS_DEFAULT: f32 = 1e-3;
/// Maximum clamped gain w_max to eliminate MUSIC spectral clicks
pub const SRX_W_MAX: f32 = 10.0;

/// Super-Resolvent xFormer Attention layer v05 (Quantum-Algebraic Core).
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
    pub mu: f32,
    pub lambda_rls: f32,
}

impl SrxAttention {
    /// Constructs a new SRX v05 attention sublayer initialized from FastRng.
    pub fn new(config: &TransformerConfig, rng: &mut FastRng) -> Self {
        let d = config.d_model;
        let n_heads = config.n_heads;
        let head_dim = config.head_dim();
        assert_eq!(head_dim, 4, "SRX v05 Butterfly Mixer is optimized for head_dim = 4");

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
            n_heads,
            head_dim,
            alpha: SRX_ALPHA,
            mu: SRX_MU,
            lambda_rls: SRX_RLS_LAMBDA,
        }
    }

    /// Total number of trainable parameters in this attention layer (256 parameters).
    #[inline]
    pub fn param_count(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len()
    }

    /// Single autoregressive step: strictly ZERO HEAP ALLOCATIONS!
    /// All vectors and matrices are allocated on the stack as fixed arrays.
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

        // Per-head Quantum-Algebraic Attention with 2nd-Order RLS, Krylov Depth & Butterfly Momentum
        for h in 0..n_heads {
            let h_off = h * 4;
            let q_h = [q[h_off], q[h_off + 1], q[h_off + 2], q[h_off + 3]];
            let k_h = [k[h_off], k[h_off + 1], k[h_off + 2], k[h_off + 3]];
            let v_h = [v[h_off], v[h_off + 1], v[h_off + 2], v[h_off + 3]];

            // 1. L2 normalization of q and k
            let mut q_norm = [0.0f32; 4];
            let mut k_norm = [0.0f32; 4];
            l2_normalize(&q_h, &mut q_norm, 1e-12);
            l2_normalize(&k_h, &mut k_norm, 1e-12);

            // 2. Phase Momentum update:
            // p_{\theta, t} = \mu * p_{\theta, t-1} + \alpha * (k_norm \odot v_raw[:4])
            // \theta_t = \theta_{t-1} + p_{\theta, t}
            let (thetas, p_thetas) = state.thetas_and_p_thetas_mut(h);
            for i in 0..4 {
                p_thetas[i] = self.mu * p_thetas[i] + self.alpha * (k_norm[i] * v_h[i]);
                thetas[i] += p_thetas[i];
            }
            let thetas_curr = *thetas;

            // 3. Monarch Butterfly Unitary Rotation of k: k_rot = U(Theta_t) * k_norm
            let mut k_rot = [0.0f32; 4];
            apply_butterfly_4(&k_norm, &thetas_curr, false, &mut k_rot);

            // 4. 2nd-Order Associative Recursive Least Squares (Sherman-Morrison update)
            let (m, p_mat) = state.m_and_p_mut(h);

            // a) Kalman gain vector:
            // v_p = P_{t-1} * k_rot
            let mut v_p = [0.0f32; 4];
            for r in 0..4 {
                let mut sum = 0.0f32;
                let r_off = r * 4;
                for c in 0..4 {
                    sum += p_mat[r_off + c] * k_rot[c];
                }
                v_p[r] = sum;
            }

            // denom = \lambda + k_rot^T * v_p
            let mut k_dot_vp = 0.0f32;
            for i in 0..4 {
                k_dot_vp += k_rot[i] * v_p[i];
            }
            let denom = self.lambda_rls + k_dot_vp;
            let inv_denom = 1.0 / denom;

            // k_gain = v_p / denom
            let mut k_gain = [0.0f32; 4];
            for i in 0..4 {
                k_gain[i] = v_p[i] * inv_denom;
            }

            // b) Reproduction error:
            // v_hat = M_{t-1}^T * k_rot
            let mut v_hat = [0.0f32; 4];
            for c in 0..4 {
                let mut sum = 0.0f32;
                for r in 0..4 {
                    sum += k_rot[r] * m[r * 4 + c];
                }
                v_hat[c] = sum;
            }

            // e_t = v_raw - v_hat
            let mut e_t = [0.0f32; 4];
            for c in 0..4 {
                e_t[c] = v_h[c] - v_hat[c];
            }

            // c) Memory update:
            // M_t = \lambda * M_{t-1} + k_gain * e_t^T
            for r in 0..4 {
                let kg = k_gain[r];
                let r_off = r * 4;
                for c in 0..4 {
                    let idx = r_off + c;
                    m[idx] = self.lambda_rls * m[idx] + kg * e_t[c];
                }
            }

            // d) Covariance update (Sherman-Morrison formula):
            // k_trans_p = k_rot^T * P_{t-1}
            let mut k_trans_p = [0.0f32; 4];
            for c in 0..4 {
                let mut sum = 0.0f32;
                for r in 0..4 {
                    sum += k_rot[r] * p_mat[r * 4 + c];
                }
                k_trans_p[c] = sum;
            }

            let inv_lambda = 1.0 / self.lambda_rls;
            for r in 0..4 {
                let kg = k_gain[r];
                let r_off = r * 4;
                for c in 0..4 {
                    let idx = r_off + c;
                    p_mat[idx] = inv_lambda * (p_mat[idx] - kg * k_trans_p[c]);
                }
            }

            // 5. Krylov Recurrent Depth (K=2)
            // q^{(0)} = q_norm
            // u_q0 = U_t * q^{(0)}
            let mut u_q0 = [0.0f32; 4];
            apply_butterfly_4(&q_norm, &thetas_curr, false, &mut u_q0);

            // q_combo = 0.5 * q^{(0)} + 0.5 * u_q0
            let mut q_combo = [0.0f32; 4];
            for i in 0..4 {
                q_combo[i] = 0.5 * q_norm[i] + 0.5 * u_q0[i];
            }

            // q^{(1)} = L2_Norm(q_combo)
            let mut q_1 = [0.0f32; 4];
            l2_normalize(&q_combo, &mut q_1, 1e-12);

            // 6. MUSIC Noise Subspace Projector using q^{(1)}:
            // q_inv = U^\dagger(Theta) q^{(1)}
            let mut q_inv = [0.0f32; 4];
            apply_butterfly_4(&q_1, &thetas_curr, true, &mut q_inv);

            // Signal rank r = 2; noise subspace is coordinates 2..4
            let noise_energy = q_inv[2] * q_inv[2] + q_inv[3] * q_inv[3];

            // Dirac-like resonant gain with hard clipping
            let w = 1.0 / (noise_energy + eps);
            let w_clamped = w.min(SRX_W_MAX);

            // 7. Memory retrieval using q^{(1)}:
            // q_rot = U(Theta) q^{(1)}
            let mut q_rot = [0.0f32; 4];
            apply_butterfly_4(&q_1, &thetas_curr, false, &mut q_rot);

            // y_ret = M_t^T q_rot
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
            vocab_size: 53,
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_layers: 1,
            d_ff: 12,
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
