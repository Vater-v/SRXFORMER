//! SRX Attention v05: Quantum-Algebraic Core («Автомат Калашникова»)
//! with Pure Orthogonal Complement Projector Memory, Undistorted MUSIC Subspace Resonance,
//! Monarch Butterfly Unitary Mixer with Dynamic Phase Coupling,
//! and Strictly Zero Heap Allocations on the Hot Path.
//!
//! Architectural Principles in v05:
//! 1. Mathematical Rigor (Orthogonal Projector, Not Reflection):
//!    - Projector onto orthogonal complement of the key:
//!      \Pi_{k^\perp} = I - k_rot * k_rot^T (where ||k_rot||_2 = 1)
//!    - Associative memory update:
//!      e_t = v_raw - M_{t-1}^T * k_rot
//!      M_t = M_{t-1} \Pi_{k_rot^\perp} + k_rot * v_raw^T = M_{t-1} + k_rot * e_t^T
//!    - Exact response identity:
//!      M_t^T * k_rot = (I - k_rot * k_rot^T) M_{t-1}^T * k_rot + v_raw * (k_rot^T * k_rot) = 0 + v_raw (1) = v_raw
//!    - Zero heuristics, zero learned gates, zero manual decay hyperparameters lambda.
//!      If key repeats and v matches: e_t = 0 => M_t = M_{t-1} (strictly zero memory drift!).
//! 2. Undistorted MUSIC Subspace Resonance:
//!    - Direct query inverse rotation: q_inv = U^\dagger(\Theta_t) * q_norm
//!    - Noise subspace energy: E_noise = q_inv[2]^2 + q_inv[3]^2
//!    - Resonant gain: w(q) = min(1 / (E_noise + eps), 15.0)
//!    - Memory retrieval: y_ret = M_t^T * (U(\Theta_t) * q_norm) * w(q)
//!    - When q = k_rot: U^\dagger * U * k_sig = k_sig = [k_0, k_1, 0, 0] => E_noise = 0 => w = 15.0!
//! 3. Strictly 160 Bytes Context State (100% L1D cache resident).

use crate::classic::config::TransformerConfig;
use crate::classic::rng::FastRng;
use super::ops::{apply_butterfly_4, l2_normalize};
use super::state::SrxState;

/// Phase modulation rate alpha
pub const SRX_ALPHA: f32 = 0.1;
/// Default inference epsilon for Dirac-like peak
pub const SRX_EPS_DEFAULT: f32 = 1e-3;
/// Maximum clamped gain w_max to eliminate MUSIC spectral clicks
pub const SRX_W_MAX: f32 = 15.0;

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

        // Per-head Quantum-Algebraic Attention with Pure Orthogonal Projector and Clean MUSIC
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

            // 2. Butterfly Phase update: \theta_t = \theta_{t-1} + \alpha * (k_norm \odot v_raw)
            let (thetas, m) = state.thetas_and_m_mut(h);
            for i in 0..4 {
                thetas[i] += self.alpha * (k_norm[i] * v_h[i]);
            }
            let thetas_curr = *thetas;

            // 3. Monarch Butterfly Unitary Rotation of k: k_rot = U(Theta_t) * k_norm
            let mut k_rot = [0.0f32; 4];
            apply_butterfly_4(&k_norm, &thetas_curr, false, &mut k_rot);

            // 4. Pure Orthogonal Projector Associative Memory Update:
            // v_hat = M_{t-1}^T * k_rot
            let mut v_hat = [0.0f32; 4];
            for col in 0..4 {
                let mut sum = 0.0f32;
                for row in 0..4 {
                    sum += k_rot[row] * m[row * 4 + col];
                }
                v_hat[col] = sum;
            }

            // e_t = v_raw - v_hat
            let mut e_t = [0.0f32; 4];
            for c in 0..4 {
                e_t[c] = v_h[c] - v_hat[c];
            }

            // M_t = M_{t-1} + k_rot * e_t^T (Orthogonal Projector on k^\perp)
            for row in 0..4 {
                let kr = k_rot[row];
                let row_off = row * 4;
                for col in 0..4 {
                    m[row_off + col] += kr * e_t[col];
                }
            }

            // 5. Undistorted MUSIC Noise Subspace Projector directly on q_norm:
            // q_inv = U^\dagger(Theta_t) * q_norm
            let mut q_inv = [0.0f32; 4];
            apply_butterfly_4(&q_norm, &thetas_curr, true, &mut q_inv);

            // Signal rank r = 2; noise subspace is coordinates 2..4
            let noise_energy = q_inv[2] * q_inv[2] + q_inv[3] * q_inv[3];

            // Dirac-like resonant gain with hard clipping at 15.0
            let w = 1.0 / (noise_energy + eps);
            let w_clamped = w.min(SRX_W_MAX);

            // 6. Memory retrieval:
            // q_rot = U(Theta_t) * q_norm
            let mut q_rot = [0.0f32; 4];
            apply_butterfly_4(&q_norm, &thetas_curr, false, &mut q_rot);

            // y_ret = M_t^T * q_rot * w(q)
            for col in 0..4 {
                let mut sum_m = 0.0f32;
                for row in 0..4 {
                    sum_m += q_rot[row] * m[row * 4 + col];
                }
                y_heads[h_off + col] = sum_m * w_clamped;
            }
        }

        // 7. Post-MUSIC RMSNorm across multi-head retrieved output vector
        let mut sum_sq = 0.0f32;
        for c in 0..8 {
            sum_sq += y_heads[c] * y_heads[c];
        }
        let rms = (sum_sq / 8.0 + 1e-5).sqrt();
        let inv_rms = 1.0 / rms;
        for c in 0..8 {
            y_heads[c] *= inv_rms;
        }

        // 8. Output projection: out = W_o * y_heads
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
            vocab_size: 65,
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_layers: 1,
            d_ff: 6,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projector_exact_reproduction() {
        // Validates M_t^T * k_rot == v_raw with tolerance 1e-6
        // for multiple arbitrary unit vectors k_rot and arbitrary vectors v_raw.
        let mut m = [
            0.15f32, -0.42, 0.88, -0.11,
            -0.65, 0.33, 0.12, 0.77,
            0.45, -0.90, 0.23, -0.55,
            0.08, 0.71, -0.34, 0.62,
        ];

        let test_cases = [
            ([0.5f32, -0.5, 0.5, -0.5], [1.2f32, -3.4, 0.7, 2.1]),
            ([0.8f32, 0.0, -0.6, 0.0], [-0.5f32, 4.2, -1.8, 0.9]),
            ([0.1f32, 0.7, -0.2, 0.67823299], [3.0f32, 1.5, -2.2, 0.4]),
        ];

        for (k_unnorm, v_raw) in test_cases {
            let mut k_rot = [0.0f32; 4];
            l2_normalize(&k_unnorm, &mut k_rot, 1e-12);

            // Compute v_hat = M_{t-1}^T * k_rot
            let mut v_hat = [0.0f32; 4];
            for col in 0..4 {
                let mut sum = 0.0f32;
                for row in 0..4 {
                    sum += k_rot[row] * m[row * 4 + col];
                }
                v_hat[col] = sum;
            }

            // e_t = v_raw - v_hat
            let mut e_t = [0.0f32; 4];
            for c in 0..4 {
                e_t[c] = v_raw[c] - v_hat[c];
            }

            // M_t = M_{t-1} + k_rot * e_t^T
            for row in 0..4 {
                let kr = k_rot[row];
                for col in 0..4 {
                    m[row * 4 + col] += kr * e_t[col];
                }
            }

            // Verification: M_t^T * k_rot == v_raw
            let mut v_reproduced = [0.0f32; 4];
            for col in 0..4 {
                let mut sum = 0.0f32;
                for row in 0..4 {
                    sum += k_rot[row] * m[row * 4 + col];
                }
                v_reproduced[col] = sum;
            }

            for c in 0..4 {
                let diff = (v_reproduced[c] - v_raw[c]).abs();
                assert!(
                    diff < 1e-6,
                    "Orthogonal projector reproduction failed at c={}: expected {}, got {}, diff={}",
                    c,
                    v_raw[c],
                    v_reproduced[c],
                    diff
                );
            }

            // Zero memory drift check: repeated key with same v produces e_t = 0 and M unchanged
            let m_before = m;
            let mut v_hat2 = [0.0f32; 4];
            for col in 0..4 {
                let mut sum = 0.0f32;
                for row in 0..4 {
                    sum += k_rot[row] * m[row * 4 + col];
                }
                v_hat2[col] = sum;
            }
            let mut e_t2 = [0.0f32; 4];
            for c in 0..4 {
                e_t2[c] = v_raw[c] - v_hat2[c];
                assert!(e_t2[c].abs() < 1e-6, "Repeated key must have e_t == 0");
            }
            for row in 0..4 {
                for col in 0..4 {
                    m[row * 4 + col] += k_rot[row] * e_t2[col];
                }
            }
            for idx in 0..16 {
                assert!(
                    (m[idx] - m_before[idx]).abs() < 1e-6,
                    "Memory must not drift on repeated key"
                );
            }
        }
    }

    #[test]
    fn test_music_peak_trigger() {
        // When q = k_rot (where k_rot = U(Theta) * k_sig with k_sig = [k0, k1, 0, 0]):
        // noise energy is strictly 0 and gain w reaches exact maximum SRX_W_MAX (15.0).
        let thetas = [0.45f32, -1.2, 0.88, -0.35];

        // Signal subspace has zero in coordinates [2] and [3]
        let k_sig = [0.6f32, 0.8, 0.0, 0.0];

        // k_rot = U(Theta) * k_sig
        let mut k_rot = [0.0f32; 4];
        apply_butterfly_4(&k_sig, &thetas, false, &mut k_rot);

        // In query: q_norm = k_rot
        let q_norm = k_rot;

        // Inverse rotation: q_inv = U^\dagger(Theta) * q_norm
        let mut q_inv = [0.0f32; 4];
        apply_butterfly_4(&q_norm, &thetas, true, &mut q_inv);

        // Noise energy = q_inv[2]^2 + q_inv[3]^2
        let noise_energy = q_inv[2] * q_inv[2] + q_inv[3] * q_inv[3];

        assert!(
            noise_energy < 1e-6,
            "Noise energy must be 0 for signal subspace, got {}",
            noise_energy
        );

        let eps = 1e-3f32;
        let w = (1.0 / (noise_energy + eps)).min(SRX_W_MAX);
        assert_eq!(
            w, SRX_W_MAX,
            "Dirac resonant gain must trigger to maximum 15.0, got {}",
            w
        );
    }
}
