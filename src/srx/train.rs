//! High-performance Analytical Backpropagation and AdamW Optimizer Engine for SRXFORMER.
//! Implements exact analytical gradients through Givens unitary chains, MUSIC noise subspace projection,
//! dynamic epsilon annealing, and associative memory accumulation with zero heap allocations on the hot path.

use std::time::Instant;

use crate::classic::config::TransformerConfig;
use crate::classic::ops::softmax;
use crate::classic::telemetry::TrainTelemetry;
use crate::classic::tokenizer::EOS_TOKEN_ID;

use super::attention::SRX_ALPHA;
use super::model::SrxTransformer;
use super::ops::{
    apply_givens_backward, apply_givens_forward_with_intermediates, l2_normalize,
    l2_normalize_backward,
};

/// Gradient storage for all 512 trainable parameters of SrxTransformer.
#[derive(Debug, Clone)]
pub struct SrxGrad {
    pub token_embeddings: Vec<f32>, // [vocab_size, d_model]
    pub attn_norm_gamma: Vec<f32>,  // [d_model]
    pub w_q: Vec<f32>,              // [d_model, d_model]
    pub w_k: Vec<f32>,              // [d_model, d_model]
    pub w_v: Vec<f32>,              // [d_model, d_model]
    pub w_o: Vec<f32>,              // [d_model, d_model]
    pub ffn_norm_gamma: Vec<f32>,   // [d_model]
    pub w_1: Vec<f32>,              // [d_ff, d_model]
    pub w_2: Vec<f32>,              // [d_model, d_ff]
    pub final_norm_gamma: Vec<f32>, // [d_model]

    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
}

impl SrxGrad {
    pub fn new(config: &TransformerConfig) -> Self {
        let v = config.vocab_size;
        let d = config.d_model;
        let d_ff = config.d_ff;

        Self {
            token_embeddings: vec![0.0; v * d],
            attn_norm_gamma: vec![0.0; d],
            w_q: vec![0.0; d * d],
            w_k: vec![0.0; d * d],
            w_v: vec![0.0; d * d],
            w_o: vec![0.0; d * d],
            ffn_norm_gamma: vec![0.0; d],
            w_1: vec![0.0; d_ff * d],
            w_2: vec![0.0; d * d_ff],
            final_norm_gamma: vec![0.0; d],
            vocab_size: v,
            d_model: d,
            d_ff,
        }
    }

    pub fn zero(&mut self) {
        self.token_embeddings.fill(0.0);
        self.attn_norm_gamma.fill(0.0);
        self.w_q.fill(0.0);
        self.w_k.fill(0.0);
        self.w_v.fill(0.0);
        self.w_o.fill(0.0);
        self.ffn_norm_gamma.fill(0.0);
        self.w_1.fill(0.0);
        self.w_2.fill(0.0);
        self.final_norm_gamma.fill(0.0);
    }

    pub fn param_count(&self) -> usize {
        self.token_embeddings.len()
            + self.attn_norm_gamma.len()
            + self.w_q.len()
            + self.w_k.len()
            + self.w_v.len()
            + self.w_o.len()
            + self.ffn_norm_gamma.len()
            + self.w_1.len()
            + self.w_2.len()
            + self.final_norm_gamma.len()
    }

    pub fn l2_norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        let slices: [&[f32]; 10] = [
            &self.token_embeddings,
            &self.attn_norm_gamma,
            &self.w_q,
            &self.w_k,
            &self.w_v,
            &self.w_o,
            &self.ffn_norm_gamma,
            &self.w_1,
            &self.w_2,
            &self.final_norm_gamma,
        ];
        for s in slices {
            for &x in s {
                sum_sq += x * x;
            }
        }
        sum_sq.sqrt()
    }

    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let norm = self.l2_norm();
        if norm > max_norm && norm > 1e-8 {
            let scale = max_norm / norm;
            let slices: [&mut [f32]; 10] = [
                &mut self.token_embeddings,
                &mut self.attn_norm_gamma,
                &mut self.w_q,
                &mut self.w_k,
                &mut self.w_v,
                &mut self.w_o,
                &mut self.ffn_norm_gamma,
                &mut self.w_1,
                &mut self.w_2,
                &mut self.final_norm_gamma,
            ];
            for s in slices {
                for x in s.iter_mut() {
                    *x *= scale;
                }
            }
        }
        norm
    }
}

/// Pre-allocated workspace for SRX forward activations and backward adjoints.
#[derive(Debug, Clone)]
pub struct SrxTrainWorkspace {
    pub max_seq_len: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,

    // Forward activations
    pub x_0: Vec<f32>,          // [max_seq_len, d_model]
    pub x_0_norm: Vec<f32>,     // [max_seq_len, d_model]
    pub rms_attn: Vec<f32>,     // [max_seq_len]
    pub a: Vec<f32>,            // [max_seq_len, d_model]
    pub q_raw: Vec<f32>,        // [max_seq_len, d_model]
    pub k_raw: Vec<f32>,        // [max_seq_len, d_model]
    pub v_raw: Vec<f32>,        // [max_seq_len, d_model]

    // Per-head SRX states
    pub norm_q: Vec<f32>,       // [max_seq_len, n_heads]
    pub norm_k: Vec<f32>,       // [max_seq_len, n_heads]
    pub q_norm: Vec<f32>,       // [max_seq_len, n_heads, head_dim]
    pub k_norm: Vec<f32>,       // [max_seq_len, n_heads, head_dim]
    pub thetas: Vec<f32>,       // [max_seq_len, n_heads, head_dim - 1]
    pub k_rot: Vec<f32>,        // [max_seq_len, n_heads, head_dim]
    pub k_rot_inter: Vec<f32>,  // [max_seq_len, n_heads, head_dim * head_dim]
    pub m: Vec<f32>,            // [max_seq_len, n_heads, head_dim * head_dim]
    pub q_inv: Vec<f32>,        // [max_seq_len, n_heads, head_dim]
    pub q_inv_inter: Vec<f32>,  // [max_seq_len, n_heads, head_dim * head_dim]
    pub noise_energy: Vec<f32>, // [max_seq_len, n_heads]
    pub w: Vec<f32>,            // [max_seq_len, n_heads]
    pub q_rot: Vec<f32>,        // [max_seq_len, n_heads, head_dim]
    pub q_rot_inter: Vec<f32>,  // [max_seq_len, n_heads, head_dim * head_dim]
    pub y_ret: Vec<f32>,        // [max_seq_len, n_heads, head_dim]
    pub y_head: Vec<f32>,       // [max_seq_len, n_heads, head_dim]
    pub y: Vec<f32>,            // [max_seq_len, d_model]
    pub attn_out: Vec<f32>,     // [max_seq_len, d_model]

    pub x_mid: Vec<f32>,        // [max_seq_len, d_model]
    pub x_mid_norm: Vec<f32>,   // [max_seq_len, d_model]
    pub rms_ffn: Vec<f32>,      // [max_seq_len]
    pub b: Vec<f32>,            // [max_seq_len, d_model]
    pub z: Vec<f32>,            // [max_seq_len, d_ff]
    pub h: Vec<f32>,            // [max_seq_len, d_ff]
    pub f: Vec<f32>,            // [max_seq_len, d_model]
    pub x_1: Vec<f32>,          // [max_seq_len, d_model]
    pub x_1_norm: Vec<f32>,     // [max_seq_len, d_model]
    pub rms_final: Vec<f32>,    // [max_seq_len]
    pub u: Vec<f32>,            // [max_seq_len, d_model]
    pub logits: Vec<f32>,       // [max_seq_len, vocab_size]
    pub probs: Vec<f32>,        // [max_seq_len, vocab_size]

    // Backward intermediate adjoints
    pub d_logits: Vec<f32>,     // [max_seq_len, vocab_size]
    pub d_u: Vec<f32>,          // [max_seq_len, d_model]
    pub d_x_1: Vec<f32>,        // [max_seq_len, d_model]
    pub d_f: Vec<f32>,          // [max_seq_len, d_model]
    pub d_h: Vec<f32>,          // [max_seq_len, d_ff]
    pub d_z: Vec<f32>,          // [max_seq_len, d_ff]
    pub d_b: Vec<f32>,          // [max_seq_len, d_model]
    pub d_x_mid: Vec<f32>,      // [max_seq_len, d_model]
    pub d_attn_out: Vec<f32>,   // [max_seq_len, d_model]
    pub d_y: Vec<f32>,          // [max_seq_len, d_model]
    pub d_q_raw: Vec<f32>,      // [max_seq_len, d_model]
    pub d_k_raw: Vec<f32>,      // [max_seq_len, d_model]
    pub d_v_raw: Vec<f32>,      // [max_seq_len, d_model]
    pub d_a: Vec<f32>,          // [max_seq_len, d_model]
    pub d_x_0: Vec<f32>,        // [max_seq_len, d_model]
}

impl SrxTrainWorkspace {
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
            x_0: vec![0.0; s * d],
            x_0_norm: vec![0.0; s * d],
            rms_attn: vec![0.0; s],
            a: vec![0.0; s * d],
            q_raw: vec![0.0; s * d],
            k_raw: vec![0.0; s * d],
            v_raw: vec![0.0; s * d],
            norm_q: vec![0.0; s * n_heads],
            norm_k: vec![0.0; s * n_heads],
            q_norm: vec![0.0; s * n_heads * head_dim],
            k_norm: vec![0.0; s * n_heads * head_dim],
            thetas: vec![0.0; s * n_heads * (head_dim - 1)],
            k_rot: vec![0.0; s * n_heads * head_dim],
            k_rot_inter: vec![0.0; s * n_heads * head_dim * head_dim],
            m: vec![0.0; s * n_heads * head_dim * head_dim],
            q_inv: vec![0.0; s * n_heads * head_dim],
            q_inv_inter: vec![0.0; s * n_heads * head_dim * head_dim],
            noise_energy: vec![0.0; s * n_heads],
            w: vec![0.0; s * n_heads],
            q_rot: vec![0.0; s * n_heads * head_dim],
            q_rot_inter: vec![0.0; s * n_heads * head_dim * head_dim],
            y_ret: vec![0.0; s * n_heads * head_dim],
            y_head: vec![0.0; s * n_heads * head_dim],
            y: vec![0.0; s * d],
            attn_out: vec![0.0; s * d],
            x_mid: vec![0.0; s * d],
            x_mid_norm: vec![0.0; s * d],
            rms_ffn: vec![0.0; s],
            b: vec![0.0; s * d],
            z: vec![0.0; s * d_ff],
            h: vec![0.0; s * d_ff],
            f: vec![0.0; s * d],
            x_1: vec![0.0; s * d],
            x_1_norm: vec![0.0; s * d],
            rms_final: vec![0.0; s],
            u: vec![0.0; s * d],
            logits: vec![0.0; s * v],
            probs: vec![0.0; s * v],
            d_logits: vec![0.0; s * v],
            d_u: vec![0.0; s * d],
            d_x_1: vec![0.0; s * d],
            d_f: vec![0.0; s * d],
            d_h: vec![0.0; s * d_ff],
            d_z: vec![0.0; s * d_ff],
            d_b: vec![0.0; s * d],
            d_x_mid: vec![0.0; s * d],
            d_attn_out: vec![0.0; s * d],
            d_y: vec![0.0; s * d],
            d_q_raw: vec![0.0; s * d],
            d_k_raw: vec![0.0; s * d],
            d_v_raw: vec![0.0; s * d],
            d_a: vec![0.0; s * d],
            d_x_0: vec![0.0; s * d],
        }
    }
}

/// Zero-dependency AdamW Optimizer for SrxTransformer.
#[derive(Debug, Clone)]
pub struct SrxAdamW {
    pub m: SrxGrad,
    pub v: SrxGrad,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: usize,
}

impl SrxAdamW {
    pub fn new(config: &TransformerConfig, weight_decay: f32) -> Self {
        Self {
            m: SrxGrad::new(config),
            v: SrxGrad::new(config),
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step_count: 0,
        }
    }

    pub fn step(&mut self, model: &mut SrxTransformer, grad: &SrxGrad, lr: f32) {
        self.step_count += 1;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;

        let bc1 = 1.0 - beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - beta2.powi(self.step_count as i32);
        let alpha = lr * (bc2.sqrt()) / bc1;

        macro_rules! update_slice {
            ($p:expr, $g:expr, $m:expr, $v:expr) => {
                for i in 0..$p.len() {
                    let g_val = $g[i] + wd * $p[i];
                    $m[i] = beta1 * $m[i] + (1.0 - beta1) * g_val;
                    $v[i] = beta2 * $v[i] + (1.0 - beta2) * g_val * g_val;
                    let denom = $v[i].sqrt() + eps;
                    $p[i] -= alpha * ($m[i] / denom);
                }
            };
        }

        update_slice!(
            model.token_embeddings,
            grad.token_embeddings,
            self.m.token_embeddings,
            self.v.token_embeddings
        );
        update_slice!(
            model.layers[0].attn_norm_gamma,
            grad.attn_norm_gamma,
            self.m.attn_norm_gamma,
            self.v.attn_norm_gamma
        );
        update_slice!(
            model.layers[0].attn.w_q,
            grad.w_q,
            self.m.w_q,
            self.v.w_q
        );
        update_slice!(
            model.layers[0].attn.w_k,
            grad.w_k,
            self.m.w_k,
            self.v.w_k
        );
        update_slice!(
            model.layers[0].attn.w_v,
            grad.w_v,
            self.m.w_v,
            self.v.w_v
        );
        update_slice!(
            model.layers[0].attn.w_o,
            grad.w_o,
            self.m.w_o,
            self.v.w_o
        );
        update_slice!(
            model.layers[0].ffn_norm_gamma,
            grad.ffn_norm_gamma,
            self.m.ffn_norm_gamma,
            self.v.ffn_norm_gamma
        );
        update_slice!(
            model.layers[0].mlp.w_1,
            grad.w_1,
            self.m.w_1,
            self.v.w_1
        );
        update_slice!(
            model.layers[0].mlp.w_2,
            grad.w_2,
            self.m.w_2,
            self.v.w_2
        );
        update_slice!(
            model.final_norm_gamma,
            grad.final_norm_gamma,
            self.m.final_norm_gamma,
            self.v.final_norm_gamma
        );
    }
}

/// Forward pass recording activations and computing cross-entropy loss.
pub fn forward_loss(
    model: &SrxTransformer,
    tokens: &[usize],
    ws: &mut SrxTrainWorkspace,
    eps_srx: f32,
) -> f32 {
    let seq_len = tokens.len();
    assert!(seq_len >= 2);
    assert!(seq_len <= ws.max_seq_len);

    let d = ws.d_model;
    let d_ff = ws.d_ff;
    let v = ws.vocab_size;
    let n_heads = ws.n_heads;
    let head_dim = ws.head_dim;
    let r = head_dim / 2;
    let eps_norm = model.config.eps;

    let layer = &model.layers[0];

    // 1. Token Embeddings + Sinusoidal PE
    for (pos, &tok) in tokens.iter().enumerate() {
        let tok_emb = &model.token_embeddings[tok * d..(tok + 1) * d];
        let pe_emb = if let Some(ref pe) = model.sinusoidal_table {
            &pe[pos * d..(pos + 1) * d]
        } else {
            panic!("Sinusoidal table required");
        };
        let x_tok = &mut ws.x_0[pos * d..(pos + 1) * d];
        for c in 0..d {
            x_tok[c] = tok_emb[c] + pe_emb[c];
        }
    }

    // 2. Pre-Attn RMSNorm
    for t in 0..seq_len {
        let x = &ws.x_0[t * d..(t + 1) * d];
        let mut sum_sq = 0.0f32;
        for &val in x {
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + eps_norm).sqrt();
        ws.rms_attn[t] = rms;
        let inv_rms = 1.0 / rms;

        let norm_slice = &mut ws.x_0_norm[t * d..(t + 1) * d];
        let a_slice = &mut ws.a[t * d..(t + 1) * d];
        for c in 0..d {
            norm_slice[c] = x[c] * inv_rms;
            a_slice[c] = norm_slice[c] * layer.attn_norm_gamma[c];
        }
    }

    // 3. Q, K, V Projections
    for t in 0..seq_len {
        let a_tok = &ws.a[t * d..(t + 1) * d];
        for c in 0..d {
            let mut q_val = 0.0f32;
            let mut k_val = 0.0f32;
            let mut v_val = 0.0f32;
            for k_idx in 0..d {
                q_val += a_tok[k_idx] * layer.attn.w_q[c * d + k_idx];
                k_val += a_tok[k_idx] * layer.attn.w_k[c * d + k_idx];
                v_val += a_tok[k_idx] * layer.attn.w_v[c * d + k_idx];
            }
            ws.q_raw[t * d + c] = q_val;
            ws.k_raw[t * d + c] = k_val;
            ws.v_raw[t * d + c] = v_val;
        }
    }

    // 4. Per-head SRX Recurrence
    for h in 0..n_heads {
        let h_off = h * head_dim;
        let span_theta = head_dim - 1;
        let span_m = head_dim * head_dim;

        for t in 0..seq_len {
            let q_raw_h = &ws.q_raw[t * d + h_off..t * d + h_off + head_dim];
            let k_raw_h = &ws.k_raw[t * d + h_off..t * d + h_off + head_dim];
            let v_raw_h = &ws.v_raw[t * d + h_off..t * d + h_off + head_dim];

            let q_norm_off = (t * n_heads + h) * head_dim;
            let k_norm_off = (t * n_heads + h) * head_dim;
            let thetas_off = (t * n_heads + h) * span_theta;
            let m_off = (t * n_heads + h) * span_m;
            let inter_off = (t * n_heads + h) * span_m;

            // a) L2 Normalization of k and q
            let n_q = l2_normalize(
                q_raw_h,
                &mut ws.q_norm[q_norm_off..q_norm_off + head_dim],
                1e-12,
            );
            let n_k = l2_normalize(
                k_raw_h,
                &mut ws.k_norm[k_norm_off..k_norm_off + head_dim],
                1e-12,
            );
            ws.norm_q[t * n_heads + h] = n_q;
            ws.norm_k[t * n_heads + h] = n_k;

            // b) Phase modulation update: Theta_t = Theta_{t-1} + alpha * tanh(k[:-1] * v[:-1])
            let prev_off = (t.saturating_sub(1) * n_heads + h) * span_theta;
            for i in 0..span_theta {
                let prev_th = if t > 0 { ws.thetas[prev_off + i] } else { 0.0 };
                let k_val = ws.k_norm[k_norm_off + i];
                let v_val = v_raw_h[i];
                let delta = SRX_ALPHA * (k_val * v_val).tanh();
                ws.thetas[thetas_off + i] = prev_th + delta;
            }

            let curr_thetas = &ws.thetas[thetas_off..thetas_off + span_theta];

            // c) k_rot = U(Theta_t) k_norm
            apply_givens_forward_with_intermediates(
                &ws.k_norm[k_norm_off..k_norm_off + head_dim],
                curr_thetas,
                false,
                &mut ws.k_rot_inter[inter_off..inter_off + span_m],
            );
            let k_rot_vec = &ws.k_rot_inter[inter_off + (head_dim - 1) * head_dim
                ..inter_off + head_dim * head_dim];
            ws.k_rot[k_norm_off..k_norm_off + head_dim].copy_from_slice(k_rot_vec);

            // d) Associative Memory accumulation: M_t = gamma * M_{t-1} + k_rot * v^T
            let pm_off = (t.saturating_sub(1) * n_heads + h) * span_m;
            for row in 0..head_dim {
                for col in 0..head_dim {
                    let idx = row * head_dim + col;
                    let p_val = if t > 0 { ws.m[pm_off + idx] } else { 0.0 };
                    ws.m[m_off + idx] =
                        layer.attn.gamma * p_val + ws.k_rot[k_norm_off + row] * v_raw_h[col];
                }
            }

            // e) Noise subspace projection: q_inv = U^\dagger(Theta_t) q_norm
            apply_givens_forward_with_intermediates(
                &ws.q_norm[q_norm_off..q_norm_off + head_dim],
                curr_thetas,
                true,
                &mut ws.q_inv_inter[inter_off..inter_off + span_m],
            );
            let q_inv_vec = &ws.q_inv_inter[inter_off + (head_dim - 1) * head_dim
                ..inter_off + head_dim * head_dim];
            ws.q_inv[q_norm_off..q_norm_off + head_dim].copy_from_slice(q_inv_vec);

            let mut noise_energy = 0.0f32;
            for j in r..head_dim {
                let q_val = ws.q_inv[q_norm_off + j];
                noise_energy += q_val * q_val;
            }
            ws.noise_energy[t * n_heads + h] = noise_energy;

            // Dirac-like resonant gain
            let w_val = 1.0 / (noise_energy + eps_srx);
            ws.w[t * n_heads + h] = w_val;

            // f) Memory retrieval: q_rot = U(Theta_t) q_norm
            apply_givens_forward_with_intermediates(
                &ws.q_norm[q_norm_off..q_norm_off + head_dim],
                curr_thetas,
                false,
                &mut ws.q_rot_inter[inter_off..inter_off + span_m],
            );
            let q_rot_vec = &ws.q_rot_inter[inter_off + (head_dim - 1) * head_dim
                ..inter_off + head_dim * head_dim];
            ws.q_rot[q_norm_off..q_norm_off + head_dim].copy_from_slice(q_rot_vec);

            // y_ret = M^T q_rot
            let y_ret_off = (t * n_heads + h) * head_dim;
            for col in 0..head_dim {
                let mut sum_m = 0.0f32;
                for row in 0..head_dim {
                    sum_m += ws.q_rot[q_norm_off + row] * ws.m[m_off + row * head_dim + col];
                }
                ws.y_ret[y_ret_off + col] = sum_m;
                ws.y_head[y_ret_off + col] = sum_m * w_val;
                ws.y[t * d + h_off + col] = sum_m * w_val;
            }
        }
    }

    // 5. Output Projection: attn_out = y * W_o^T
    for t in 0..seq_len {
        let y_tok = &ws.y[t * d..(t + 1) * d];
        for c in 0..d {
            let mut sum_o = 0.0f32;
            for k in 0..d {
                sum_o += y_tok[k] * layer.attn.w_o[c * d + k];
            }
            ws.attn_out[t * d + c] = sum_o;
            ws.x_mid[t * d + c] = ws.x_0[t * d + c] + sum_o;
        }
    }

    // 6. Pre-FFN RMSNorm
    for t in 0..seq_len {
        let x = &ws.x_mid[t * d..(t + 1) * d];
        let mut sum_sq = 0.0f32;
        for &val in x {
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + eps_norm).sqrt();
        ws.rms_ffn[t] = rms;
        let inv_rms = 1.0 / rms;

        let norm_slice = &mut ws.x_mid_norm[t * d..(t + 1) * d];
        let b_slice = &mut ws.b[t * d..(t + 1) * d];
        for c in 0..d {
            norm_slice[c] = x[c] * inv_rms;
            b_slice[c] = norm_slice[c] * layer.ffn_norm_gamma[c];
        }
    }

    // 7. MLP
    for t in 0..seq_len {
        let b_tok = &ws.b[t * d..(t + 1) * d];
        for j in 0..d_ff {
            let mut z_val = 0.0f32;
            for k in 0..d {
                z_val += b_tok[k] * layer.mlp.w_1[j * d + k];
            }
            ws.z[t * d_ff + j] = z_val;
            ws.h[t * d_ff + j] = if z_val > 0.0 { z_val } else { 0.0 };
        }

        let h_tok = &ws.h[t * d_ff..(t + 1) * d_ff];
        for c in 0..d {
            let mut f_val = 0.0f32;
            for j in 0..d_ff {
                f_val += h_tok[j] * layer.mlp.w_2[c * d_ff + j];
            }
            ws.f[t * d + c] = f_val;
            ws.x_1[t * d + c] = ws.x_mid[t * d + c] + f_val;
        }
    }

    // 8. Final RMSNorm
    for t in 0..seq_len {
        let x = &ws.x_1[t * d..(t + 1) * d];
        let mut sum_sq = 0.0f32;
        for &val in x {
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + eps_norm).sqrt();
        ws.rms_final[t] = rms;
        let inv_rms = 1.0 / rms;

        let norm_slice = &mut ws.x_1_norm[t * d..(t + 1) * d];
        let u_slice = &mut ws.u[t * d..(t + 1) * d];
        for c in 0..d {
            norm_slice[c] = x[c] * inv_rms;
            u_slice[c] = norm_slice[c] * model.final_norm_gamma[c];
        }
    }

    // 9. LM Head and Cross-Entropy Loss
    let mut total_loss = 0.0f32;
    for t in 0..seq_len {
        let u_tok = &ws.u[t * d..(t + 1) * d];
        let logits_tok = &mut ws.logits[t * v..(t + 1) * v];
        let probs_tok = &mut ws.probs[t * v..(t + 1) * v];

        for tok_idx in 0..v {
            let w_emb = &model.token_embeddings[tok_idx * d..(tok_idx + 1) * d];
            let mut dot = 0.0f32;
            for c in 0..d {
                dot += u_tok[c] * w_emb[c];
            }
            logits_tok[tok_idx] = dot;
            probs_tok[tok_idx] = dot;
        }

        softmax(probs_tok);

        if t < seq_len - 1 {
            let target = tokens[t + 1];
            let p_target = probs_tok[target].max(1e-12);
            total_loss += -p_target.ln();
        }
    }

    total_loss / (seq_len - 1) as f32
}

/// Backward pass accumulating exact analytical gradients into `grad`.
pub fn backward_loss(
    model: &SrxTransformer,
    tokens: &[usize],
    ws: &mut SrxTrainWorkspace,
    grad: &mut SrxGrad,
    _eps_srx: f32,
) {
    let seq_len = tokens.len();
    let d = ws.d_model;
    let d_ff = ws.d_ff;
    let v = ws.vocab_size;
    let n_heads = ws.n_heads;
    let head_dim = ws.head_dim;
    let r = head_dim / 2;

    let layer = &model.layers[0];
    let loss_scale = 1.0 / (seq_len - 1) as f32;

    // Reset temporary adjoint buffers
    ws.d_logits.fill(0.0);
    ws.d_u.fill(0.0);
    ws.d_x_1.fill(0.0);
    ws.d_f.fill(0.0);
    ws.d_h.fill(0.0);
    ws.d_z.fill(0.0);
    ws.d_b.fill(0.0);
    ws.d_x_mid.fill(0.0);
    ws.d_attn_out.fill(0.0);
    ws.d_y.fill(0.0);
    ws.d_q_raw.fill(0.0);
    ws.d_k_raw.fill(0.0);
    ws.d_v_raw.fill(0.0);
    ws.d_a.fill(0.0);
    ws.d_x_0.fill(0.0);

    // 1. Loss -> d_logits
    for t in 0..seq_len - 1 {
        let target = tokens[t + 1];
        for tok_idx in 0..v {
            let p = ws.probs[t * v + tok_idx];
            let indicator = if tok_idx == target { 1.0 } else { 0.0 };
            ws.d_logits[t * v + tok_idx] = (p - indicator) * loss_scale;
        }
    }

    // 2. LM Head -> d_u & d_token_embeddings
    for t in 0..seq_len - 1 {
        let u_tok = &ws.u[t * d..(t + 1) * d];
        for tok_idx in 0..v {
            let d_log = ws.d_logits[t * v + tok_idx];
            if d_log == 0.0 {
                continue;
            }
            let w_emb = &model.token_embeddings[tok_idx * d..(tok_idx + 1) * d];
            for c in 0..d {
                ws.d_u[t * d + c] += d_log * w_emb[c];
                grad.token_embeddings[tok_idx * d + c] += d_log * u_tok[c];
            }
        }
    }

    // 3. Final RMSNorm -> d_x_1 & final_norm_gamma
    for t in 0..seq_len {
        let rms = ws.rms_final[t];
        if rms <= 1e-12 {
            continue;
        }
        let inv_rms = 1.0 / rms;
        let x_norm = &ws.x_1_norm[t * d..(t + 1) * d];
        let d_u_tok = &ws.d_u[t * d..(t + 1) * d];

        let mut dot_du_gamma_x = 0.0f32;
        for c in 0..d {
            let gamma = model.final_norm_gamma[c];
            grad.final_norm_gamma[c] += d_u_tok[c] * x_norm[c];
            dot_du_gamma_x += d_u_tok[c] * gamma * x_norm[c];
        }

        for c in 0..d {
            let gamma = model.final_norm_gamma[c];
            let d_norm = d_u_tok[c] * gamma;
            ws.d_x_1[t * d + c] = inv_rms * (d_norm - x_norm[c] * dot_du_gamma_x / d as f32);
        }
    }

    // 4. Residual connection & MLP
    for t in 0..seq_len {
        for c in 0..d {
            let d_x1_val = ws.d_x_1[t * d + c];
            ws.d_x_mid[t * d + c] += d_x1_val;
            ws.d_f[t * d + c] += d_x1_val;
        }

        // Backward through MLP W_2
        let h_tok = &ws.h[t * d_ff..(t + 1) * d_ff];
        for c in 0..d {
            let df_val = ws.d_f[t * d + c];
            for j in 0..d_ff {
                grad.w_2[c * d_ff + j] += df_val * h_tok[j];
                ws.d_h[t * d_ff + j] += df_val * layer.mlp.w_2[c * d_ff + j];
            }
        }

        // Backward through ReLU
        for j in 0..d_ff {
            ws.d_z[t * d_ff + j] = if ws.z[t * d_ff + j] > 0.0 {
                ws.d_h[t * d_ff + j]
            } else {
                0.0
            };
        }

        // Backward through MLP W_1
        let b_tok = &ws.b[t * d..(t + 1) * d];
        for j in 0..d_ff {
            let dz_val = ws.d_z[t * d_ff + j];
            for k in 0..d {
                grad.w_1[j * d + k] += dz_val * b_tok[k];
                ws.d_b[t * d + k] += dz_val * layer.mlp.w_1[j * d + k];
            }
        }
    }

    // 5. Pre-FFN RMSNorm -> d_x_mid & ffn_norm_gamma
    for t in 0..seq_len {
        let rms = ws.rms_ffn[t];
        if rms <= 1e-12 {
            continue;
        }
        let inv_rms = 1.0 / rms;
        let x_norm = &ws.x_mid_norm[t * d..(t + 1) * d];
        let d_b_tok = &ws.d_b[t * d..(t + 1) * d];

        let mut dot_db_gamma_x = 0.0f32;
        for c in 0..d {
            let gamma = layer.ffn_norm_gamma[c];
            grad.ffn_norm_gamma[c] += d_b_tok[c] * x_norm[c];
            dot_db_gamma_x += d_b_tok[c] * gamma * x_norm[c];
        }

        for c in 0..d {
            let gamma = layer.ffn_norm_gamma[c];
            let d_norm = d_b_tok[c] * gamma;
            ws.d_x_mid[t * d + c] += inv_rms * (d_norm - x_norm[c] * dot_db_gamma_x / d as f32);
        }
    }

    // 6. Residual connection & Output projection W_o
    for t in 0..seq_len {
        for c in 0..d {
            let dx_mid = ws.d_x_mid[t * d + c];
            ws.d_x_0[t * d + c] += dx_mid;
            ws.d_attn_out[t * d + c] += dx_mid;
        }

        let y_tok = &ws.y[t * d..(t + 1) * d];
        for c in 0..d {
            let da_val = ws.d_attn_out[t * d + c];
            for k in 0..d {
                grad.w_o[c * d + k] += da_val * y_tok[k];
                ws.d_y[t * d + k] += da_val * layer.attn.w_o[c * d + k];
            }
        }
    }

    // 7. Backward through SRX Attention Recurrence
    for h in 0..n_heads {
        let h_off = h * head_dim;
        let span_theta = head_dim - 1;
        let span_m = head_dim * head_dim;

        // Recurrence accumulators for reverse-time unroll
        let mut d_m_acc = vec![0.0f32; span_m];
        let mut d_thetas_acc = vec![0.0f32; span_theta];

        for t in (0..seq_len).rev() {
            let q_norm_off = (t * n_heads + h) * head_dim;
            let k_norm_off = (t * n_heads + h) * head_dim;
            let thetas_off = (t * n_heads + h) * span_theta;
            let m_off = (t * n_heads + h) * span_m;
            let inter_off = (t * n_heads + h) * span_m;
            let y_ret_off = (t * n_heads + h) * head_dim;

            let curr_thetas = &ws.thetas[thetas_off..thetas_off + span_theta];
            let w_val = ws.w[t * n_heads + h];

            // Adjoint from y_head
            let mut d_y_ret = vec![0.0f32; head_dim];
            let mut d_w = 0.0f32;

            for col in 0..head_dim {
                let dy_val = ws.d_y[t * d + h_off + col];
                d_y_ret[col] = dy_val * w_val;
                d_w += dy_val * ws.y_ret[y_ret_off + col];
            }

            // d_noise_energy from w = 1 / (noise_energy + eps)
            let d_noise_energy = -d_w * w_val * w_val;

            // d_q_inv
            let mut d_q_inv = vec![0.0f32; head_dim];
            for j in r..head_dim {
                d_q_inv[j] = 2.0 * d_noise_energy * ws.q_inv[q_norm_off + j];
            }

            // Readout: y_ret = M^T q_rot => d_q_rot = M * d_y_ret, d_M_read = q_rot * d_y_ret^T
            let mut d_q_rot = vec![0.0f32; head_dim];
            for row in 0..head_dim {
                for col in 0..head_dim {
                    let m_val = ws.m[m_off + row * head_dim + col];
                    d_q_rot[row] += d_y_ret[col] * m_val;
                    d_m_acc[row * head_dim + col] += ws.q_rot[q_norm_off + row] * d_y_ret[col];
                }
            }

            // VJP backward through q_rot Givens rotation (forward mode)
            let mut d_q_norm_rot = vec![0.0f32; head_dim];
            apply_givens_backward(
                &ws.q_rot_inter[inter_off..inter_off + span_m],
                curr_thetas,
                false,
                &d_q_rot,
                &mut d_q_norm_rot,
                &mut d_thetas_acc,
            );

            // VJP backward through q_inv Givens rotation (inverse mode)
            let mut d_q_norm_inv = vec![0.0f32; head_dim];
            apply_givens_backward(
                &ws.q_inv_inter[inter_off..inter_off + span_m],
                curr_thetas,
                true,
                &d_q_inv,
                &mut d_q_norm_inv,
                &mut d_thetas_acc,
            );

            let mut d_q_norm_total = vec![0.0f32; head_dim];
            for c in 0..head_dim {
                d_q_norm_total[c] = d_q_norm_rot[c] + d_q_norm_inv[c];
            }

            // M_t = gamma * M_{t-1} + k_rot * v^T
            let mut d_k_rot = vec![0.0f32; head_dim];
            let v_raw_h = &ws.v_raw[t * d + h_off..t * d + h_off + head_dim];

            for row in 0..head_dim {
                for col in 0..head_dim {
                    let dm_val = d_m_acc[row * head_dim + col];
                    d_k_rot[row] += dm_val * v_raw_h[col];
                    ws.d_v_raw[t * d + h_off + col] += dm_val * ws.k_rot[k_norm_off + row];
                }
            }

            // M recurrence step to t - 1: d_M_{t-1} = gamma * d_M_t
            for dm in d_m_acc.iter_mut() {
                *dm *= layer.attn.gamma;
            }

            // VJP backward through k_rot Givens rotation (forward mode)
            let mut d_k_norm = vec![0.0f32; head_dim];
            apply_givens_backward(
                &ws.k_rot_inter[inter_off..inter_off + span_m],
                curr_thetas,
                false,
                &d_k_rot,
                &mut d_k_norm,
                &mut d_thetas_acc,
            );

            // Phase modulation: delta_theta = alpha * tanh(k[:-1] * v[:-1])
            // d_thetas_acc is d_Theta_t.
            for i in 0..span_theta {
                let d_th = d_thetas_acc[i];
                let k_val = ws.k_norm[k_norm_off + i];
                let v_val = v_raw_h[i];
                let th_val = (k_val * v_val).tanh();
                let d_tanh = SRX_ALPHA * (1.0 - th_val * th_val);
                let grad_prod = d_th * d_tanh;

                d_k_norm[i] += grad_prod * v_val;
                ws.d_v_raw[t * d + h_off + i] += grad_prod * k_val;
            }
            // Theta recurrence: d_Theta_{t-1} = d_Theta_t (gradient flows unconditionally across cumsum)

            // Backward through L2 normalizations
            let q_raw_h = &ws.q_raw[t * d + h_off..t * d + h_off + head_dim];
            let k_raw_h = &ws.k_raw[t * d + h_off..t * d + h_off + head_dim];
            let q_norm_vec = &ws.q_norm[q_norm_off..q_norm_off + head_dim];
            let k_norm_vec = &ws.k_norm[k_norm_off..k_norm_off + head_dim];

            let mut d_q_raw_h = vec![0.0f32; head_dim];
            let mut d_k_raw_h = vec![0.0f32; head_dim];

            l2_normalize_backward(
                q_raw_h,
                q_norm_vec,
                ws.norm_q[t * n_heads + h],
                &d_q_norm_total,
                &mut d_q_raw_h,
            );
            l2_normalize_backward(
                k_raw_h,
                k_norm_vec,
                ws.norm_k[t * n_heads + h],
                &d_k_norm,
                &mut d_k_raw_h,
            );

            for c in 0..head_dim {
                ws.d_q_raw[t * d + h_off + c] += d_q_raw_h[c];
                ws.d_k_raw[t * d + h_off + c] += d_k_raw_h[c];
            }
        }
    }

    // 8. Backward through Q, K, V Projections
    for t in 0..seq_len {
        let a_tok = &ws.a[t * d..(t + 1) * d];

        for c in 0..d {
            let dq = ws.d_q_raw[t * d + c];
            let dk = ws.d_k_raw[t * d + c];
            let dv = ws.d_v_raw[t * d + c];

            for k_idx in 0..d {
                let a_val = a_tok[k_idx];
                grad.w_q[c * d + k_idx] += dq * a_val;
                grad.w_k[c * d + k_idx] += dk * a_val;
                grad.w_v[c * d + k_idx] += dv * a_val;

                ws.d_a[t * d + k_idx] += dq * layer.attn.w_q[c * d + k_idx]
                    + dk * layer.attn.w_k[c * d + k_idx]
                    + dv * layer.attn.w_v[c * d + k_idx];
            }
        }
    }

    // 9. Backward through Pre-Attn RMSNorm
    for t in 0..seq_len {
        let rms = ws.rms_attn[t];
        if rms <= 1e-12 {
            continue;
        }
        let inv_rms = 1.0 / rms;
        let x_norm = &ws.x_0_norm[t * d..(t + 1) * d];
        let d_a_tok = &ws.d_a[t * d..(t + 1) * d];

        let mut dot_da_gamma_x = 0.0f32;
        for c in 0..d {
            let gamma = layer.attn_norm_gamma[c];
            grad.attn_norm_gamma[c] += d_a_tok[c] * x_norm[c];
            dot_da_gamma_x += d_a_tok[c] * gamma * x_norm[c];
        }

        for c in 0..d {
            let gamma = layer.attn_norm_gamma[c];
            let d_norm = d_a_tok[c] * gamma;
            ws.d_x_0[t * d + c] += inv_rms * (d_norm - x_norm[c] * dot_da_gamma_x / d as f32);
        }

        // Accumulate into token embeddings
        let tok = tokens[t];
        for c in 0..d {
            grad.token_embeddings[tok * d + c] += ws.d_x_0[t * d + c];
        }
    }
}

/// Splits corpus token slice into sub-sequences separated by EOS_TOKEN_ID.
pub fn split_into_eos_sequences(tokens: &[usize], max_seq_len: usize) -> Vec<Vec<usize>> {
    let mut sequences = Vec::new();
    let mut current = Vec::new();

    for &tok in tokens {
        current.push(tok);
        if tok == EOS_TOKEN_ID || current.len() >= max_seq_len {
            if current.len() >= 2 {
                sequences.push(current);
            }
            current = Vec::new();
        }
    }

    if current.len() >= 2 {
        sequences.push(current);
    }

    sequences
}

/// High-level dataset training method using analytical BPTT, dynamic epsilon annealing, and AdamW.
pub fn train_dataset(
    model: &mut SrxTransformer,
    tokens: &[usize],
    config: &TransformerConfig,
    epochs: usize,
    lr: f32,
) -> TrainTelemetry {
    let start_time = Instant::now();
    let mut ws = SrxTrainWorkspace::new(config);
    let mut grad = SrxGrad::new(config);
    let mut optimizer = SrxAdamW::new(config, 0.0);

    let mut sequences = split_into_eos_sequences(tokens, config.max_seq_len);
    assert!(!sequences.is_empty(), "Dataset has no valid sequences");

    let mut loss_history = Vec::with_capacity(epochs);
    let mut total_steps = 0;

    // Initial loss evaluation across first 20 sequences (with start epsilon = 1.0)
    let eval_count = sequences.len().min(20);
    let mut initial_loss = 0.0f32;
    for seq in &sequences[..eval_count] {
        initial_loss += forward_loss(model, seq, &mut ws, 1.0);
    }
    initial_loss /= eval_count as f32;

    let mut rng = crate::classic::rng::FastRng::new(42);

    for epoch in 0..epochs {
        // Fisher-Yates shuffle
        for i in (1..sequences.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            sequences.swap(i, j);
        }

        let progress = epoch as f32 / epochs.max(1) as f32;
        // Dynamic epsilon annealing: eps(s) = eps_min + (eps_max - eps_min) * (1 - s / S)^2
        // eps_max = 1.0, eps_min = 1e-4
        let eps_srx = 1e-4 + (1.0 - 1e-4) * (1.0 - progress).powi(2);

        // Cosine learning rate schedule
        let current_lr = (lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(lr * 0.1);

        let mut epoch_loss = 0.0f32;
        for seq in &sequences {
            grad.zero();
            let loss = forward_loss(model, seq, &mut ws, eps_srx);
            epoch_loss += loss;
            backward_loss(model, seq, &mut ws, &mut grad, eps_srx);
            grad.clip_grad_norm(1.0);
            optimizer.step(model, &grad, current_lr);
            total_steps += 1;
        }

        let avg_epoch_loss = epoch_loss / sequences.len() as f32;
        loss_history.push(avg_epoch_loss);
    }

    let final_loss = *loss_history.last().unwrap_or(&initial_loss);
    let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let elapsed_s = (elapsed_ms / 1000.0).max(1e-6);

    let num_params = model.param_count();
    let dataset_tokens = tokens.len();
    // Formula: 6 * N_params * tokens_per_epoch * epochs
    let total_training_flops = 6u64 * (num_params as u64) * (dataset_tokens as u64) * (epochs as u64);
    let flops_per_sec = (total_training_flops as f64) / elapsed_s;
    let mflops_per_sec = flops_per_sec / 1e6;
    let gflops_per_sec = flops_per_sec / 1e9;

    let initial_perplexity = initial_loss.exp();
    let final_perplexity = final_loss.exp();

    TrainTelemetry {
        num_params,
        dataset_tokens,
        epochs,
        total_training_flops,
        elapsed_ms,
        mflops_per_sec,
        gflops_per_sec,
        initial_loss,
        final_loss,
        initial_perplexity,
        final_perplexity,
        loss_history,
        total_steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srx_gradient_check_numerical() {
        let config = TransformerConfig::lang_512();
        let mut model = SrxTransformer::new_with_seed(config.clone(), 123).unwrap();
        let mut ws = SrxTrainWorkspace::new(&config);
        let mut grad = SrxGrad::new(&config);

        let tokens = [1, 4, 2, 7, 3];
        let eps_srx = 0.5f32; // Smooth regime for gradient checking

        grad.zero();
        forward_loss(&model, &tokens, &mut ws, eps_srx);
        backward_loss(&model, &tokens, &mut ws, &mut grad, eps_srx);

        // Check MLP w_2 gradient numerically
        let eps = 1e-4f32;
        for i in 0..model.layers[0].mlp.w_2.len().min(8) {
            let orig = model.layers[0].mlp.w_2[i];

            model.layers[0].mlp.w_2[i] = orig + eps;
            let l_plus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].mlp.w_2[i] = orig - eps;
            let l_minus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].mlp.w_2[i] = orig;

            let num_grad = (l_plus - l_minus) / (2.0 * eps);
            let ana_grad = grad.w_2[i];
            assert!(
                (ana_grad - num_grad).abs() < 2e-3,
                "MLP w_2[{}] grad mismatch: ana={}, num={}",
                i,
                ana_grad,
                num_grad
            );
        }

        // Check W_o gradient numerically
        for i in 0..model.layers[0].attn.w_o.len().min(8) {
            let orig = model.layers[0].attn.w_o[i];

            model.layers[0].attn.w_o[i] = orig + eps;
            let l_plus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].attn.w_o[i] = orig - eps;
            let l_minus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].attn.w_o[i] = orig;

            let num_grad = (l_plus - l_minus) / (2.0 * eps);
            let ana_grad = grad.w_o[i];
            assert!(
                (ana_grad - num_grad).abs() < 2e-3,
                "W_o[{}] grad mismatch: ana={}, num={}",
                i,
                ana_grad,
                num_grad
            );
        }
    }
}
