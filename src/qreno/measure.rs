//! # Quantum Renormalization & Operator-Field Tokenizer (Q-RENO): Operator Measurement
//!
//! Projects coarse-grained quasi-particle field vectors $\Phi_m \in \mathbb{R}^{d_f}$ into
//! model embedding space $\mathbb{R}^{d_{\text{model}}}$ via semantic operator matrix $W_O$:
//! - $u_m = W_O \cdot \Phi_m$
//! - $E_m = \text{RMSNorm}(u_m)$
//!
//! Preserves topological continuity: single-byte perturbations (typos/insertions) yield
//! small bounded shifts $\delta \Psi_0$ and maintain high cosine similarity ($\ge 0.85$).

/// Semantic operator measurement projecting quasi-particles into model representations.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorMeasure {
    /// Dimension of model representations ($d_{\text{model}}$).
    pub d_model: usize,
    /// Dimension of quantum field ($d_f$).
    pub field_dim: usize,
    /// Operator weights $W_O \in \mathbb{R}^{d_{\text{model}} \times d_f}$, row-major.
    pub w_o: Vec<f32>,
    /// Epsilon for RMSNorm numerical stability.
    pub rms_eps: f32,
}

impl OperatorMeasure {
    /// Creates a new operator measure with deterministic normalized Xavier-harmonic initialization.
    pub fn new(field_dim: usize, d_model: usize) -> Self {
        assert!(field_dim > 0 && d_model > 0);
        let mut w_o = vec![0.0f32; d_model * field_dim];

        let scale = (2.0 / (field_dim + d_model) as f32).sqrt();
        for i in 0..d_model {
            for j in 0..field_dim {
                let mut rng = ((i * field_dim + j) as u64).wrapping_mul(0x517CC1B727220A95) ^ 0x6A09E667F3BCC908;
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let val = ((rng >> 40) as f32) / 16777216.0 - 0.5;
                w_o[i * field_dim + j] = scale * val * 2.0;
            }
        }

        Self {
            d_model,
            field_dim,
            w_o,
            rms_eps: 1e-5,
        }
    }

    /// Measures the quasi-particle $\Phi \in \mathbb{R}^{d_f}$ to produce embedding $E \in \mathbb{R}^{d_{\text{model}}}$.
    #[inline(always)]
    pub fn measure(&self, phi: &[f32], e_out: &mut [f32]) {
        debug_assert_eq!(phi.len(), self.field_dim);
        debug_assert_eq!(e_out.len(), self.d_model);

        // 1. Linear projection u = W_O * phi
        let mut sum_sq = 0.0f32;
        for i in 0..self.d_model {
            let offset = i * self.field_dim;
            let mut u_i = 0.0f32;
            for j in 0..self.field_dim {
                u_i += self.w_o[offset + j] * phi[j];
            }
            e_out[i] = u_i;
            sum_sq += u_i * u_i;
        }

        // 2. RMSNorm: E = u / sqrt(mean(u^2) + eps)
        let rms = (sum_sq / (self.d_model as f32) + self.rms_eps).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..self.d_model {
            e_out[i] *= inv_rms;
        }
    }

    /// Computes analytical backward gradients through RMSNorm and linear projection $W_O$.
    #[inline(always)]
    pub fn measure_backward(
        &self,
        phi: &[f32],
        e_out: &[f32],
        e_grad: &[f32],
        phi_grad: &mut [f32],
        w_o_grad: &mut [f32],
    ) {
        debug_assert_eq!(phi.len(), self.field_dim);
        debug_assert_eq!(e_out.len(), self.d_model);
        debug_assert_eq!(e_grad.len(), self.d_model);
        debug_assert_eq!(phi_grad.len(), self.field_dim);
        debug_assert_eq!(w_o_grad.len(), self.d_model * self.field_dim);

        // Recompute unnormalized projection u and its RMS
        let mut sum_sq = 0.0f32;
        let mut dot_e_eg = 0.0f32;
        for i in 0..self.d_model {
            let offset = i * self.field_dim;
            let mut u_i = 0.0f32;
            for j in 0..self.field_dim {
                u_i += self.w_o[offset + j] * phi[j];
            }
            sum_sq += u_i * u_i;
            dot_e_eg += e_out[i] * e_grad[i];
        }

        let rms = (sum_sq / (self.d_model as f32) + self.rms_eps).sqrt();
        let inv_rms = 1.0 / rms;
        let d_f32 = self.d_model as f32;

        // Gradient wrt u: dL/du = (1 / rms) * (dL/dE - e * (e . dL/dE) / d)
        for i in 0..self.d_model {
            let grad_u_i = inv_rms * (e_grad[i] - e_out[i] * dot_e_eg / d_f32);
            let offset = i * self.field_dim;

            // Propagate into W_O and phi
            for j in 0..self.field_dim {
                w_o_grad[offset + j] += grad_u_i * phi[j];
                phi_grad[j] += grad_u_i * self.w_o[offset + j];
            }
        }
    }

    /// Total parameter count in this operator.
    pub fn param_count(&self) -> usize {
        self.w_o.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_measure_rmsnorm() {
        let op = OperatorMeasure::new(8, 16);
        let phi = vec![0.5f32; 8];
        let mut e_out = vec![0.0f32; 16];

        op.measure(&phi, &mut e_out);

        // RMS of e_out should be close to 1.0 (modulo small rms_eps)
        let mean_sq: f32 = e_out.iter().map(|&x| x * x).sum::<f32>() / 16.0;
        let rms = mean_sq.sqrt();
        assert!((rms - 1.0).abs() < 1e-3, "RMSNorm must produce unit RMS, got {rms}");
    }
}
