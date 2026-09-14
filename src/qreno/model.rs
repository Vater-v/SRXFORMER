//! # Quantum Renormalization & Operator-Field Tokenizer (Q-RENO): Model & Tokenizer
//!
//! Provides the top-level Q-RENO Tokenizer & Continuous Quasi-Particle Embedding system:
//! - Complete replacement of discrete lookup table $W_E$ and heuristic BPE tokenizers.
//! - Continuous byte-level quantum field theory with tight-binding Hamiltonian.
//! - End-to-end analytical backpropagation and AdamW optimizer.

use crate::qreno::field::QrenoField;
use crate::qreno::hamiltonian::{partition_into_clusters, BondParams, Cluster};
use crate::qreno::measure::OperatorMeasure;
use crate::qreno::solver::{
    coarse_grain_cluster, solve_ground_state, solve_ground_state_vjp, MAX_CLUSTER_LEN,
};

/// Configuration for Q-RENO Tokenizer and Continuous Embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct QrenoConfig {
    /// Dimension of the continuous quantum field $c(b) \in \mathbb{R}^{d_f}$ (default: 8).
    pub field_dim: usize,
    /// Dimension of the output model representation $E \in \mathbb{R}^{d_{\text{model}}}$ (default: 16).
    pub d_model: usize,
    /// Maximum length of a bound cluster in bytes ($1 \le L_m \le 16$, default: 16).
    pub max_cluster_len: usize,
    /// Bond resonance integral threshold for cluster termination (default: 0.5).
    pub bond_threshold: f32,
    /// Epsilon for RMSNorm stability (default: 1e-5).
    pub rms_eps: f32,
}

impl Default for QrenoConfig {
    fn default() -> Self {
        Self {
            field_dim: 8,
            d_model: 16,
            max_cluster_len: 16,
            bond_threshold: 0.5,
            rms_eps: 1e-5,
        }
    }
}

/// Trainable weights of the Q-RENO module.
#[derive(Debug, Clone, PartialEq)]
pub struct QrenoWeights {
    /// Basis wave field parameters ($c, \epsilon, \omega$).
    pub field: QrenoField,
    /// Chemical Hamiltonian bond parameters ($W_{\text{bond}}, b_{\text{bond}}$).
    pub bond: BondParams,
    /// Semantic operator measurement projection ($W_O$).
    pub measure: OperatorMeasure,
}

impl QrenoWeights {
    /// Initializes weights with standard harmonic physical defaults.
    pub fn new(config: &QrenoConfig) -> Self {
        Self {
            field: QrenoField::new(config.field_dim),
            bond: BondParams::new(config.field_dim),
            measure: OperatorMeasure::new(config.field_dim, config.d_model),
        }
    }

    /// Initializes weights with a deterministic random seed.
    pub fn with_seed(config: &QrenoConfig, seed: u64) -> Self {
        Self {
            field: QrenoField::with_seed(config.field_dim, seed),
            bond: BondParams::new(config.field_dim),
            measure: OperatorMeasure::new(config.field_dim, config.d_model),
        }
    }

    /// Returns the total trainable parameter count.
    pub fn param_count(&self) -> usize {
        self.field.param_count() + self.bond.param_count() + self.measure.param_count()
    }
}

/// Gradients for all trainable parameters in `QrenoWeights`.
#[derive(Debug, Clone)]
pub struct QrenoGrad {
    /// Gradients wrt continuous byte charges: `[256, field_dim]`.
    pub charges_grad: Vec<f32>,
    /// Gradients wrt chemical potentials $\epsilon(b)$: `[256]`.
    pub epsilon_grad: [f32; 256],
    /// Gradients wrt phase frequencies $\omega(b)$: `[256]`.
    pub omega_grad: [f32; 256],
    /// Gradients wrt bond weights $W_{\text{bond}}$: `[field_dim]`.
    pub w_bond_grad: Vec<f32>,
    /// Gradient wrt bond bias $b_{\text{bond}}$: scalar.
    pub b_bond_grad: f32,
    /// Gradients wrt operator projection $W_O$: `[d_model, field_dim]`.
    pub w_o_grad: Vec<f32>,
}

impl QrenoGrad {
    /// Creates a zeroed gradient structure matching config.
    pub fn new(config: &QrenoConfig) -> Self {
        Self {
            charges_grad: vec![0.0f32; 256 * config.field_dim],
            epsilon_grad: [0.0f32; 256],
            omega_grad: [0.0f32; 256],
            w_bond_grad: vec![0.0f32; config.field_dim],
            b_bond_grad: 0.0f32,
            w_o_grad: vec![0.0f32; config.d_model * config.field_dim],
        }
    }

    /// Resets all accumulated gradients to zero.
    pub fn zero(&mut self) {
        self.charges_grad.fill(0.0);
        self.epsilon_grad.fill(0.0);
        self.omega_grad.fill(0.0);
        self.w_bond_grad.fill(0.0);
        self.b_bond_grad = 0.0;
        self.w_o_grad.fill(0.0);
    }

    /// Scales accumulated gradients by factor and clips extremes to [-5.0, 5.0].
    pub fn scale(&mut self, factor: f32) {
        for x in &mut self.charges_grad { *x = (*x * factor).clamp(-5.0, 5.0); }
        for x in &mut self.epsilon_grad { *x = (*x * factor).clamp(-5.0, 5.0); }
        for x in &mut self.omega_grad { *x = (*x * factor).clamp(-5.0, 5.0); }
        for x in &mut self.w_bond_grad { *x = (*x * factor).clamp(-5.0, 5.0); }
        self.b_bond_grad = (self.b_bond_grad * factor).clamp(-5.0, 5.0);
        for x in &mut self.w_o_grad { *x = (*x * factor).clamp(-5.0, 5.0); }
    }
}

/// AdamW optimizer state for Q-RENO parameters.
#[derive(Debug, Clone)]
pub struct QrenoAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: usize,
    // First moments
    m_charges: Vec<f32>,
    m_epsilon: [f32; 256],
    m_omega: [f32; 256],
    m_w_bond: Vec<f32>,
    m_b_bond: f32,
    m_w_o: Vec<f32>,
    // Second moments
    v_charges: Vec<f32>,
    v_epsilon: [f32; 256],
    v_omega: [f32; 256],
    v_w_bond: Vec<f32>,
    v_b_bond: f32,
    v_w_o: Vec<f32>,
}

impl QrenoAdamW {
    pub fn new(config: &QrenoConfig, lr: f32, weight_decay: f32) -> Self {
        let fd = config.field_dim;
        let dm = config.d_model;
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step: 0,
            m_charges: vec![0.0; 256 * fd],
            m_epsilon: [0.0; 256],
            m_omega: [0.0; 256],
            m_w_bond: vec![0.0; fd],
            m_b_bond: 0.0,
            m_w_o: vec![0.0; dm * fd],
            v_charges: vec![0.0; 256 * fd],
            v_epsilon: [0.0; 256],
            v_omega: [0.0; 256],
            v_w_bond: vec![0.0; fd],
            v_b_bond: 0.0,
            v_w_o: vec![0.0; dm * fd],
        }
    }

    /// Performs one AdamW optimization update step on `QrenoWeights`.
    pub fn step(&mut self, weights: &mut QrenoWeights, grad: &QrenoGrad) {
        self.step += 1;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;
        let lr = self.lr;

        let bias_correction1 = 1.0 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step as i32);
        let step_size = lr * (bias_correction2.sqrt() / bias_correction1);

        // Update charges
        for i in 0..weights.field.charges.len() {
            let g = grad.charges_grad[i];
            let p = &mut weights.field.charges[i];
            *p -= lr * wd * *p;
            self.m_charges[i] = beta1 * self.m_charges[i] + (1.0 - beta1) * g;
            self.v_charges[i] = beta2 * self.v_charges[i] + (1.0 - beta2) * g * g;
            let denom = self.v_charges[i].sqrt() + eps;
            *p -= step_size * (self.m_charges[i] / denom);
        }

        // Update epsilon
        for i in 0..256 {
            let g = grad.epsilon_grad[i];
            let p = &mut weights.field.epsilon[i];
            *p -= lr * wd * *p;
            self.m_epsilon[i] = beta1 * self.m_epsilon[i] + (1.0 - beta1) * g;
            self.v_epsilon[i] = beta2 * self.v_epsilon[i] + (1.0 - beta2) * g * g;
            let denom = self.v_epsilon[i].sqrt() + eps;
            *p -= step_size * (self.m_epsilon[i] / denom);
        }

        // Update omega
        for i in 0..256 {
            let g = grad.omega_grad[i];
            let p = &mut weights.field.omega[i];
            *p -= lr * wd * *p;
            self.m_omega[i] = beta1 * self.m_omega[i] + (1.0 - beta1) * g;
            self.v_omega[i] = beta2 * self.v_omega[i] + (1.0 - beta2) * g * g;
            let denom = self.v_omega[i].sqrt() + eps;
            *p -= step_size * (self.m_omega[i] / denom);
        }

        // Update w_bond
        for i in 0..weights.bond.w_bond.len() {
            let g = grad.w_bond_grad[i];
            let p = &mut weights.bond.w_bond[i];
            *p -= lr * wd * *p;
            self.m_w_bond[i] = beta1 * self.m_w_bond[i] + (1.0 - beta1) * g;
            self.v_w_bond[i] = beta2 * self.v_w_bond[i] + (1.0 - beta2) * g * g;
            let denom = self.v_w_bond[i].sqrt() + eps;
            *p -= step_size * (self.m_w_bond[i] / denom);
        }

        // Update b_bond
        let g = grad.b_bond_grad;
        let p = &mut weights.bond.b_bond;
        *p -= lr * wd * *p;
        self.m_b_bond = beta1 * self.m_b_bond + (1.0 - beta1) * g;
        self.v_b_bond = beta2 * self.v_b_bond + (1.0 - beta2) * g * g;
        let denom = self.v_b_bond.sqrt() + eps;
        *p -= step_size * (self.m_b_bond / denom);

        // Update w_o
        for i in 0..weights.measure.w_o.len() {
            let g = grad.w_o_grad[i];
            let p = &mut weights.measure.w_o[i];
            *p -= lr * wd * *p;
            self.m_w_o[i] = beta1 * self.m_w_o[i] + (1.0 - beta1) * g;
            self.v_w_o[i] = beta2 * self.v_w_o[i] + (1.0 - beta2) * g * g;
            let denom = self.v_w_o[i].sqrt() + eps;
            *p -= step_size * (self.m_w_o[i] / denom);
        }
    }
}

/// The top-level Quantum Renormalization & Operator-Field Tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub struct QrenoTokenizer {
    /// Hyperparameter configuration.
    pub config: QrenoConfig,
    /// Trainable physical and semantic weights.
    pub weights: QrenoWeights,
}

impl QrenoTokenizer {
    /// Creates a new Q-RENO Tokenizer with default harmonic weights.
    pub fn new(config: QrenoConfig) -> Self {
        let weights = QrenoWeights::new(&config);
        Self { config, weights }
    }

    /// Creates a tokenizer with a random seed.
    pub fn with_seed(config: QrenoConfig, seed: u64) -> Self {
        let weights = QrenoWeights::with_seed(&config, seed);
        Self { config, weights }
    }

    /// Partitions an arbitrary UTF-8 string into bound clusters.
    pub fn tokenize_to_clusters(&self, text: &str) -> Vec<Cluster> {
        partition_into_clusters(
            text.as_bytes(),
            &self.weights.field,
            &self.weights.bond,
            self.config.max_cluster_len,
        )
    }

    /// Encodes text into a sequence of continuous quasi-particle token embeddings $E_m \in \mathbb{R}^{d_{\text{model}}}$.
    pub fn encode(&self, text: &str) -> Vec<Vec<f32>> {
        let bytes = text.as_bytes();
        let clusters = self.tokenize_to_clusters(text);
        let mut embeddings = Vec::with_capacity(clusters.len());

        for c in clusters {
            let cluster_bytes = &bytes[c.start..c.start + c.len];
            let mut e = vec![0.0f32; self.config.d_model];
            self.embed_cluster_bytes(cluster_bytes, &mut e);
            embeddings.push(e);
        }

        embeddings
    }

    /// Embeds a slice of bytes corresponding to a single cluster with zero heap allocations.
    #[inline(always)]
    pub fn embed_cluster_bytes(&self, cluster_bytes: &[u8], e_out: &mut [f32]) {
        let n = cluster_bytes.len().min(MAX_CLUSTER_LEN);
        let mut diag = [0.0f32; MAX_CLUSTER_LEN];
        let mut subdiag = [0.0f32; MAX_CLUSTER_LEN];

        for i in 0..n {
            diag[i] = self.weights.field.epsilon(cluster_bytes[i]);
        }

        for i in 0..n.saturating_sub(1) {
            let c_curr = self.weights.field.charge(cluster_bytes[i]);
            let c_next = self.weights.field.charge(cluster_bytes[i + 1]);
            let t = self.weights.bond.compute_bond(c_curr, c_next);
            subdiag[i] = -t;
        }

        let gs = solve_ground_state(&diag[..n], &subdiag[..n.saturating_sub(1)]);

        let mut phi = vec![0.0f32; self.config.field_dim];
        coarse_grain_cluster(
            &cluster_bytes[..n],
            gs.psi_slice(),
            &self.weights.field,
            &mut phi,
        );

        self.weights.measure.measure(&phi, e_out);
    }

    /// Embeds an entire word or subphrase into a unified normalized embedding vector $E \in \mathbb{R}^{d_{\text{model}}}$.
    ///
    /// If word spans multiple clusters, aggregates quasi-particles via length-weighted pooling.
    pub fn embed_word(&self, word: &str) -> Vec<f32> {
        let bytes = word.as_bytes();
        if bytes.is_empty() {
            return vec![0.0f32; self.config.d_model];
        }

        let clusters = self.tokenize_to_clusters(word);
        if clusters.len() == 1 {
            let mut e = vec![0.0f32; self.config.d_model];
            let c = &clusters[0];
            self.embed_cluster_bytes(&bytes[c.start..c.start + c.len], &mut e);
            return e;
        }

        // Multiple clusters: length-weighted continuous aggregation + RMSNorm
        let mut aggregated = vec![0.0f32; self.config.d_model];
        let mut e_temp = vec![0.0f32; self.config.d_model];

        let total_bytes: f32 = clusters.iter().map(|c| c.len as f32).sum();
        for c in &clusters {
            let cluster_bytes = &bytes[c.start..c.start + c.len];
            self.embed_cluster_bytes(cluster_bytes, &mut e_temp);
            let weight = (c.len as f32) / total_bytes;
            for k in 0..self.config.d_model {
                aggregated[k] += weight * e_temp[k];
            }
        }

        // Normalize aggregated embedding
        let mut sum_sq = 0.0f32;
        for &val in &aggregated {
            sum_sq += val * val;
        }
        let rms = (sum_sq / (self.config.d_model as f32) + self.config.rms_eps).sqrt();
        let inv_rms = 1.0 / rms;
        for k in 0..self.config.d_model {
            aggregated[k] *= inv_rms;
        }

        aggregated
    }

    /// Embeds full text via mean-pooling across all generated quasi-particles.
    pub fn embed_text(&self, text: &str) -> Vec<f32> {
        self.embed_word(text)
    }

    /// Computes cosine similarity between two vector representations: $\cos(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut dot = 0.0f32;
        let mut norm_a_sq = 0.0f32;
        let mut norm_b_sq = 0.0f32;

        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a_sq += a[i] * a[i];
            norm_b_sq += b[i] * b[i];
        }

        let denom = (norm_a_sq.sqrt() * norm_b_sq.sqrt()).max(1e-12);
        dot / denom
    }

    /// Executes full end-to-end forward and analytical backward pass for a single cluster.
    pub fn forward_backward_cluster(
        &self,
        cluster_bytes: &[u8],
        e_grad: &[f32],
        grad: &mut QrenoGrad,
    ) -> Vec<f32> {
        let n = cluster_bytes.len().min(MAX_CLUSTER_LEN);
        let mut diag = [0.0f32; MAX_CLUSTER_LEN];
        let mut subdiag = [0.0f32; MAX_CLUSTER_LEN];
        let mut bonds = [0.0f32; MAX_CLUSTER_LEN];

        for i in 0..n {
            diag[i] = self.weights.field.epsilon(cluster_bytes[i]);
        }

        for i in 0..n.saturating_sub(1) {
            let c_curr = self.weights.field.charge(cluster_bytes[i]);
            let c_next = self.weights.field.charge(cluster_bytes[i + 1]);
            let t = self.weights.bond.compute_bond(c_curr, c_next);
            bonds[i] = t;
            subdiag[i] = -t;
        }

        // 1. Solve eigensystem
        let gs = solve_ground_state(&diag[..n], &subdiag[..n.saturating_sub(1)]);

        // 2. Wilson RG coarse-graining
        let mut phi = vec![0.0f32; self.config.field_dim];
        coarse_grain_cluster(
            &cluster_bytes[..n],
            gs.psi_slice(),
            &self.weights.field,
            &mut phi,
        );

        // 3. Operator measurement
        let mut e_out = vec![0.0f32; self.config.d_model];
        self.weights.measure.measure(&phi, &mut e_out);

        // 4. Operator backward
        let mut phi_grad = vec![0.0f32; self.config.field_dim];
        self.weights.measure.measure_backward(
            &phi,
            &e_out,
            e_grad,
            &mut phi_grad,
            &mut grad.w_o_grad,
        );

        // 5. Wilson RG backward
        let mut psi_grad = vec![0.0f32; n];
        for j in 0..n {
            let b_j = cluster_bytes[j];
            let c_j = self.weights.field.charge(b_j);
            let offset = (b_j as usize) * self.config.field_dim;

            let mut dot_phi_c = 0.0f32;
            for k in 0..self.config.field_dim {
                dot_phi_c += phi_grad[k] * c_j[k];
                grad.charges_grad[offset + k] += gs.psi[j] * phi_grad[k];
            }
            psi_grad[j] = dot_phi_c;
        }

        // 6. Spectral solver VJP backward
        let mut diag_grad = vec![0.0f32; n];
        let mut subdiag_grad = vec![0.0f32; n.saturating_sub(1)];

        solve_ground_state_vjp(
            &diag[..n],
            &subdiag[..n.saturating_sub(1)],
            &gs,
            &psi_grad,
            &mut diag_grad,
            &mut subdiag_grad,
        );

        // Accumulate epsilon gradients
        for j in 0..n {
            let b_j = cluster_bytes[j];
            grad.epsilon_grad[b_j as usize] += diag_grad[j];
        }

        // 7. Bond backward: subdiag[i] = -t_i, so dL/dt_i = - dL/d(subdiag[i])
        for i in 0..n.saturating_sub(1) {
            let t_i = bonds[i];
            let t_grad = -subdiag_grad[i];
            let z_grad = t_grad * t_i * (1.0 - t_i);

            grad.b_bond_grad += z_grad;

            let b_i = cluster_bytes[i];
            let b_next = cluster_bytes[i + 1];
            let c_curr = self.weights.field.charge(b_i);
            let c_next = self.weights.field.charge(b_next);

            let offset_i = (b_i as usize) * self.config.field_dim;
            let offset_next = (b_next as usize) * self.config.field_dim;

            for k in 0..self.config.field_dim {
                let w_k = self.weights.bond.w_bond[k];
                grad.w_bond_grad[k] += z_grad * c_curr[k] * c_next[k];
                grad.charges_grad[offset_i + k] += z_grad * w_k * c_next[k];
                grad.charges_grad[offset_next + k] += z_grad * w_k * c_curr[k];
            }
        }

        e_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encode_and_param_count() {
        let config = QrenoConfig::default();
        let tok = QrenoTokenizer::new(config.clone());

        let expected_params = (256 * 8 + 256 + 256) + (8 + 1) + (16 * 8);
        assert_eq!(tok.weights.param_count(), expected_params);

        let embs = tok.encode("hello world");
        assert_eq!(embs.len(), 3);
        assert_eq!(embs[0].len(), config.d_model);
    }
}
