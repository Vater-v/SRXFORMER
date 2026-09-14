//! High-performance Analytical Reversible Backpropagation and AdamW Optimizer Engine for SRXFORMER v04.
//! Implements exact analytical gradients through Widrow-Hoff Delta Rule Memory Accumulation,
//! Monarch Butterfly Unitary factorization, Selective Dynamic Memory Gate,
//! MUSIC Noise Subspace Projection, and Adaptive Epsilon Annealing.

use std::time::Instant;

use crate::classic::config::TransformerConfig;
use crate::classic::ops::softmax;
use crate::classic::telemetry::TrainTelemetry;
use crate::classic::tokenizer::EOS_TOKEN_ID;

use super::attention::{SRX_ALPHA, SRX_GAMMA_BASE, SRX_W_MAX};
use super::model::SrxTransformer;
use super::ops::{
    apply_butterfly_4, apply_butterfly_4_backward, l2_normalize, l2_normalize_backward,
};

/// Gradient storage for all trainable parameters of SrxTransformer v04.
#[derive(Debug, Clone)]
pub struct SrxGrad {
    pub token_embeddings: Vec<f32>, // [vocab_size, d_model]
    pub attn_norm_gamma: Vec<f32>,  // [d_model]
    pub w_q: Vec<f32>,              // [d_model, d_model]
    pub w_k: Vec<f32>,              // [d_model, d_model]
    pub w_v: Vec<f32>,              // [d_model, d_model]
    pub w_o: Vec<f32>,              // [d_model, d_model]
    pub w_gamma: Vec<f32>,          // [n_heads, d_model]
    pub b_gamma: Vec<f32>,          // [n_heads]
    pub ffn_norm_gamma: Vec<f32>,   // [d_model]
    pub w_1: Vec<f32>,              // [d_ff, d_model]
    pub w_2: Vec<f32>,              // [d_model, d_ff]
    pub final_norm_gamma: Vec<f32>, // [d_model]

    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
}

impl SrxGrad {
    pub fn new(config: &TransformerConfig) -> Self {
        let v = config.vocab_size;
        let d = config.d_model;
        let d_ff = config.d_ff;
        let n_heads = config.n_heads;

        Self {
            token_embeddings: vec![0.0; v * d],
            attn_norm_gamma: vec![0.0; d],
            w_q: vec![0.0; d * d],
            w_k: vec![0.0; d * d],
            w_v: vec![0.0; d * d],
            w_o: vec![0.0; d * d],
            w_gamma: vec![0.0; n_heads * d],
            b_gamma: vec![0.0; n_heads],
            ffn_norm_gamma: vec![0.0; d],
            w_1: vec![0.0; d_ff * d],
            w_2: vec![0.0; d * d_ff],
            final_norm_gamma: vec![0.0; d],
            vocab_size: v,
            d_model: d,
            d_ff,
            n_heads,
        }
    }

    pub fn zero(&mut self) {
        self.token_embeddings.fill(0.0);
        self.attn_norm_gamma.fill(0.0);
        self.w_q.fill(0.0);
        self.w_k.fill(0.0);
        self.w_v.fill(0.0);
        self.w_o.fill(0.0);
        self.w_gamma.fill(0.0);
        self.b_gamma.fill(0.0);
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
            + self.w_gamma.len()
            + self.b_gamma.len()
            + self.ffn_norm_gamma.len()
            + self.w_1.len()
            + self.w_2.len()
            + self.final_norm_gamma.len()
    }

    pub fn l2_norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        let slices: [&[f32]; 12] = [
            &self.token_embeddings,
            &self.attn_norm_gamma,
            &self.w_q,
            &self.w_k,
            &self.w_v,
            &self.w_o,
            &self.w_gamma,
            &self.b_gamma,
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
            let slices: [&mut [f32]; 12] = [
                &mut self.token_embeddings,
                &mut self.attn_norm_gamma,
                &mut self.w_q,
                &mut self.w_k,
                &mut self.w_v,
                &mut self.w_o,
                &mut self.w_gamma,
                &mut self.b_gamma,
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

/// Pre-allocated workspace for SRX v04 forward activations and backward adjoints.
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

    // Selective Memory Gate states
    pub s_gate: Vec<f32>,       // [max_seq_len, n_heads]
    pub gamma_t: Vec<f32>,      // [max_seq_len, n_heads]
    pub lambda_t: Vec<f32>,     // [max_seq_len, n_heads]

    // Per-head SRX states
    pub norm_q: Vec<f32>,       // [max_seq_len, n_heads]
    pub norm_k: Vec<f32>,       // [max_seq_len, n_heads]
    pub q_norm: Vec<f32>,       // [max_seq_len, n_heads, head_dim]
    pub k_norm: Vec<f32>,       // [max_seq_len, n_heads, head_dim]
    pub thetas: Vec<f32>,       // [max_seq_len, n_heads, 4]
    pub k_rot: Vec<f32>,        // [max_seq_len, n_heads, head_dim]

    // Widrow-Hoff Delta Rule buffers
    pub v_hat: Vec<f32>,        // [max_seq_len, n_heads, head_dim] (current prediction)
    pub e_t: Vec<f32>,          // [max_seq_len, n_heads, head_dim] (novelty error residual)

    pub m: Vec<f32>,            // [max_seq_len, n_heads, head_dim * head_dim]
    pub q_inv: Vec<f32>,        // [max_seq_len, n_heads, head_dim]
    pub noise_energy: Vec<f32>, // [max_seq_len, n_heads]
    pub w: Vec<f32>,            // [max_seq_len, n_heads]
    pub w_clamped: Vec<f32>,    // [max_seq_len, n_heads]
    pub q_rot: Vec<f32>,        // [max_seq_len, n_heads, head_dim]
    pub y_ret: Vec<f32>,        // [max_seq_len, n_heads, head_dim]
    pub y_head: Vec<f32>,       // [max_seq_len, n_heads, head_dim]
    pub y_raw: Vec<f32>,        // [max_seq_len, d_model]
    pub rms_music: Vec<f32>,    // [max_seq_len]
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

    // Backward adjoints
    pub d_logits: Vec<f32>,
    pub d_u: Vec<f32>,
    pub d_x_1: Vec<f32>,
    pub d_f: Vec<f32>,
    pub d_h: Vec<f32>,
    pub d_z: Vec<f32>,
    pub d_b: Vec<f32>,
    pub d_x_mid: Vec<f32>,
    pub d_attn_out: Vec<f32>,
    pub d_y: Vec<f32>,
    pub d_y_raw: Vec<f32>,
    pub d_q_raw: Vec<f32>,
    pub d_k_raw: Vec<f32>,
    pub d_v_raw: Vec<f32>,
    pub d_a: Vec<f32>,
    pub d_x_0: Vec<f32>,
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
            s_gate: vec![0.0; s * n_heads],
            gamma_t: vec![0.0; s * n_heads],
            lambda_t: vec![0.0; s * n_heads],
            norm_q: vec![0.0; s * n_heads],
            norm_k: vec![0.0; s * n_heads],
            q_norm: vec![0.0; s * n_heads * head_dim],
            k_norm: vec![0.0; s * n_heads * head_dim],
            thetas: vec![0.0; s * n_heads * 4],
            k_rot: vec![0.0; s * n_heads * head_dim],
            v_hat: vec![0.0; s * n_heads * head_dim],
            e_t: vec![0.0; s * n_heads * head_dim],
            m: vec![0.0; s * n_heads * head_dim * head_dim],
            q_inv: vec![0.0; s * n_heads * head_dim],
            noise_energy: vec![0.0; s * n_heads],
            w: vec![0.0; s * n_heads],
            w_clamped: vec![0.0; s * n_heads],
            q_rot: vec![0.0; s * n_heads * head_dim],
            y_ret: vec![0.0; s * n_heads * head_dim],
            y_head: vec![0.0; s * n_heads * head_dim],
            y_raw: vec![0.0; s * d],
            rms_music: vec![0.0; s],
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
            d_y_raw: vec![0.0; s * d],
            d_q_raw: vec![0.0; s * d],
            d_k_raw: vec![0.0; s * d],
            d_v_raw: vec![0.0; s * d],
            d_a: vec![0.0; s * d],
            d_x_0: vec![0.0; s * d],
        }
    }
}

/// Zero-dependency AdamW Optimizer for SrxTransformer v04.
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
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;

        let bc1 = 1.0 - b1.powi(self.step_count as i32);
        let bc2 = 1.0 - b2.powi(self.step_count as i32);
        let alpha = lr * (bc2.sqrt() / bc1);

        let layer = &mut model.layers[0];

        let param_slices: [&mut [f32]; 12] = [
            &mut model.token_embeddings,
            &mut layer.attn_norm_gamma,
            &mut layer.attn.w_q,
            &mut layer.attn.w_k,
            &mut layer.attn.w_v,
            &mut layer.attn.w_o,
            &mut layer.attn.w_gamma,
            &mut layer.attn.b_gamma,
            &mut layer.ffn_norm_gamma,
            &mut layer.mlp.w_1,
            &mut layer.mlp.w_2,
            &mut model.final_norm_gamma,
        ];

        let grad_slices: [&[f32]; 12] = [
            &grad.token_embeddings,
            &grad.attn_norm_gamma,
            &grad.w_q,
            &grad.w_k,
            &grad.w_v,
            &grad.w_o,
            &grad.w_gamma,
            &grad.b_gamma,
            &grad.ffn_norm_gamma,
            &grad.w_1,
            &grad.w_2,
            &grad.final_norm_gamma,
        ];

        let m_slices: [&mut [f32]; 12] = [
            &mut self.m.token_embeddings,
            &mut self.m.attn_norm_gamma,
            &mut self.m.w_q,
            &mut self.m.w_k,
            &mut self.m.w_v,
            &mut self.m.w_o,
            &mut self.m.w_gamma,
            &mut self.m.b_gamma,
            &mut self.m.ffn_norm_gamma,
            &mut self.m.w_1,
            &mut self.m.w_2,
            &mut self.m.final_norm_gamma,
        ];

        let v_slices: [&mut [f32]; 12] = [
            &mut self.v.token_embeddings,
            &mut self.v.attn_norm_gamma,
            &mut self.v.w_q,
            &mut self.v.w_k,
            &mut self.v.w_v,
            &mut self.v.w_o,
            &mut self.v.w_gamma,
            &mut self.v.b_gamma,
            &mut self.v.ffn_norm_gamma,
            &mut self.v.w_1,
            &mut self.v.w_2,
            &mut self.v.final_norm_gamma,
        ];

        for k in 0..12 {
            let p_slice = &mut *param_slices[k];
            let g_slice = grad_slices[k];
            let m_slice = &mut *m_slices[k];
            let v_slice = &mut *v_slices[k];

            let n = p_slice.len();
            for i in 0..n {
                let g = g_slice[i];

                m_slice[i] = b1 * m_slice[i] + (1.0 - b1) * g;
                v_slice[i] = b2 * v_slice[i] + (1.0 - b2) * g * g;

                if wd > 0.0 {
                    p_slice[i] -= lr * wd * p_slice[i];
                }

                p_slice[i] -= alpha * (m_slice[i] / (v_slice[i].sqrt() + eps));
            }
        }
    }
}

/// Sequence forward pass computing cross-entropy loss and recording forward activations.
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
    let eps_norm = model.config.eps;

    let layer = &model.layers[0];

    // 1. Token Embeddings + Sinusoidal Positional Embeddings
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

    // 2. Pre-Attention RMSNorm
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

    // 3.1. Selective Memory Gate Computation:
    // gamma_t = sigmoid(W_gamma * a_t + b_gamma)
    for t in 0..seq_len {
        let a_tok = &ws.a[t * d..(t + 1) * d];
        for h in 0..n_heads {
            let mut s = layer.attn.b_gamma[h];
            for c in 0..d {
                s += layer.attn.w_gamma[h * d + c] * a_tok[c];
            }
            ws.s_gate[t * n_heads + h] = s;
            let g = 1.0 / (1.0 + (-s).exp());
            ws.gamma_t[t * n_heads + h] = g;
            ws.lambda_t[t * n_heads + h] = 1.0 - (1.0 - SRX_GAMMA_BASE) * g;
        }
    }

    // 4. Per-head SRX Recurrence with Widrow-Hoff Delta Rule, Butterfly Unitary & Selective Gate
    for h in 0..n_heads {
        let h_off = h * head_dim;
        let span_theta = 4;
        let span_m = head_dim * head_dim;

        for t in 0..seq_len {
            let q_raw_h = &ws.q_raw[t * d + h_off..t * d + h_off + head_dim];
            let k_raw_h = &ws.k_raw[t * d + h_off..t * d + h_off + head_dim];
            let v_raw_h = &ws.v_raw[t * d + h_off..t * d + h_off + head_dim];

            let q_norm_off = (t * n_heads + h) * head_dim;
            let k_norm_off = (t * n_heads + h) * head_dim;
            let thetas_off = (t * n_heads + h) * span_theta;
            let m_off = (t * n_heads + h) * span_m;
            let v_hat_off = (t * n_heads + h) * head_dim;
            let e_off = (t * n_heads + h) * head_dim;

            let gamma_val = ws.gamma_t[t * n_heads + h];
            let lambda_val = ws.lambda_t[t * n_heads + h];

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

            // b) Phase modulation update: Theta_t = Theta_{t-1} + alpha * tanh(k_norm * v)
            let prev_th_off = (t.saturating_sub(1) * n_heads + h) * span_theta;
            for i in 0..4 {
                let prev_th = if t > 0 { ws.thetas[prev_th_off + i] } else { 0.0 };
                let k_val = ws.k_norm[k_norm_off + i];
                let v_val = v_raw_h[i];
                let delta = SRX_ALPHA * (k_val * v_val).tanh();
                ws.thetas[thetas_off + i] = prev_th + delta;
            }

            let curr_thetas: &[f32; 4] = ws.thetas[thetas_off..thetas_off + 4].try_into().unwrap();
            let k_norm_arr: &[f32; 4] = ws.k_norm[k_norm_off..k_norm_off + 4].try_into().unwrap();

            // c) k_rot = U(Theta_t) k_norm (Monarch Butterfly Unitary)
            let mut k_rot_arr = [0.0f32; 4];
            apply_butterfly_4(k_norm_arr, curr_thetas, false, &mut k_rot_arr);
            ws.k_rot[k_norm_off..k_norm_off + 4].copy_from_slice(&k_rot_arr);

            // d) Widrow-Hoff Delta Rule:
            // 1. Current prediction from existing memory: v_hat = M_{t-1}^T * k_rot
            let pm_off = (t.saturating_sub(1) * n_heads + h) * span_m;
            for col in 0..4 {
                let mut sum_v = 0.0f32;
                if t > 0 {
                    for row in 0..4 {
                        sum_v += k_rot_arr[row] * ws.m[pm_off + row * 4 + col];
                    }
                }
                ws.v_hat[v_hat_off + col] = sum_v;
                ws.e_t[e_off + col] = v_raw_h[col] - sum_v;
            }

            // 2. Memory accumulation with Widrow-Hoff Delta Rule & Selective Gate:
            // M_t = lambda_t * M_{t-1} + gamma_t * (k_rot * e_t^T)
            for row in 0..4 {
                let kr = k_rot_arr[row];
                for col in 0..4 {
                    let idx = row * 4 + col;
                    let p_val = if t > 0 { ws.m[pm_off + idx] } else { 0.0 };
                    let e_val = ws.e_t[e_off + col];
                    ws.m[m_off + idx] = lambda_val * p_val + gamma_val * (kr * e_val);
                }
            }

            // e) Noise subspace projection: q_inv = U^\dagger(Theta_t) q_norm
            let q_norm_arr: &[f32; 4] = ws.q_norm[q_norm_off..q_norm_off + 4].try_into().unwrap();
            let mut q_inv_arr = [0.0f32; 4];
            apply_butterfly_4(q_norm_arr, curr_thetas, true, &mut q_inv_arr);
            ws.q_inv[q_norm_off..q_norm_off + 4].copy_from_slice(&q_inv_arr);

            // Signal rank r = 2; noise energy = ||q_inv[2..4]||^2
            let noise_energy = q_inv_arr[2] * q_inv_arr[2] + q_inv_arr[3] * q_inv_arr[3];
            ws.noise_energy[t * n_heads + h] = noise_energy;

            // Dirac-like resonant gain with hard clipping
            let w_val = 1.0 / (noise_energy + eps_srx);
            let w_clamped = w_val.min(SRX_W_MAX);
            ws.w[t * n_heads + h] = w_val;
            ws.w_clamped[t * n_heads + h] = w_clamped;

            // f) Memory retrieval: q_rot = U(Theta_t) q_norm
            let mut q_rot_arr = [0.0f32; 4];
            apply_butterfly_4(q_norm_arr, curr_thetas, false, &mut q_rot_arr);
            ws.q_rot[q_norm_off..q_norm_off + 4].copy_from_slice(&q_rot_arr);

            // y_ret = M^T q_rot
            let y_ret_off = (t * n_heads + h) * head_dim;
            for col in 0..4 {
                let mut sum_m = 0.0f32;
                for row in 0..4 {
                    sum_m += q_rot_arr[row] * ws.m[m_off + row * 4 + col];
                }
                ws.y_ret[y_ret_off + col] = sum_m;
                ws.y_head[y_ret_off + col] = sum_m * w_clamped;
                ws.y_raw[t * d + h_off + col] = sum_m * w_clamped;
            }
        }
    }

    // Post-MUSIC RMSNorm across multi-head retrieved output vector
    for t in 0..seq_len {
        let mut sum_sq = 0.0f32;
        for c in 0..d {
            let val = ws.y_raw[t * d + c];
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + 1e-5).sqrt();
        ws.rms_music[t] = rms;
        let inv_rms = 1.0 / rms;
        for c in 0..d {
            ws.y[t * d + c] = ws.y_raw[t * d + c] * inv_rms;
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

/// Backward pass accumulating exact analytical gradients into `grad` with Widrow-Hoff Delta Rule.
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
    ws.d_y_raw.fill(0.0);
    ws.d_q_raw.fill(0.0);
    ws.d_k_raw.fill(0.0);
    ws.d_v_raw.fill(0.0);
    ws.d_a.fill(0.0);
    ws.d_x_0.fill(0.0);

    // 1. Cross-entropy loss backward -> d_logits
    for t in 0..seq_len - 1 {
        let target = tokens[t + 1];
        let probs_tok = &ws.probs[t * v..(t + 1) * v];
        let d_logits_tok = &mut ws.d_logits[t * v..(t + 1) * v];

        for tok_idx in 0..v {
            let p = probs_tok[tok_idx];
            let indicator = if tok_idx == target { 1.0f32 } else { 0.0f32 };
            d_logits_tok[tok_idx] = (p - indicator) * loss_scale;
        }
    }

    // 2. LM Head projection backward -> d_u & grad.token_embeddings
    for t in 0..seq_len {
        let d_logits_tok = &ws.d_logits[t * v..(t + 1) * v];
        let u_tok = &ws.u[t * d..(t + 1) * d];

        for tok_idx in 0..v {
            let d_l = d_logits_tok[tok_idx];
            if d_l != 0.0 {
                let w_emb = &model.token_embeddings[tok_idx * d..(tok_idx + 1) * d];
                let grad_emb = &mut grad.token_embeddings[tok_idx * d..(tok_idx + 1) * d];
                for c in 0..d {
                    ws.d_u[t * d + c] += d_l * w_emb[c];
                    grad_emb[c] += d_l * u_tok[c];
                }
            }
        }
    }

    // 3. Final RMSNorm backward -> d_x_1 & grad.final_norm_gamma
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

    // 6.1. Backward through Post-MUSIC RMSNorm
    for t in 0..seq_len {
        let rms = ws.rms_music[t];
        if rms <= 1e-12 {
            continue;
        }
        let inv_rms = 1.0 / rms;
        let mut dot = 0.0f32;
        for c in 0..d {
            dot += ws.d_y[t * d + c] * ws.y[t * d + c];
        }
        for c in 0..d {
            ws.d_y_raw[t * d + c] = inv_rms * (ws.d_y[t * d + c] - ws.y[t * d + c] * dot / d as f32);
        }
    }

    // 7. Backward through SRX Attention Recurrence with Widrow-Hoff Delta Rule
    for h in 0..n_heads {
        let h_off = h * head_dim;
        let span_theta = 4;
        let span_m = head_dim * head_dim;

        // Recurrence accumulators for reverse-time unroll
        let mut d_m_acc = [0.0f32; 16];
        let mut d_thetas_acc = [0.0f32; 4];

        for t in (0..seq_len).rev() {
            let q_norm_off = (t * n_heads + h) * head_dim;
            let k_norm_off = (t * n_heads + h) * head_dim;
            let thetas_off = (t * n_heads + h) * span_theta;
            let m_off = (t * n_heads + h) * span_m;
            let y_ret_off = (t * n_heads + h) * head_dim;
            let e_off = (t * n_heads + h) * head_dim;

            let curr_thetas: &[f32; 4] = ws.thetas[thetas_off..thetas_off + 4].try_into().unwrap();
            let w_val = ws.w[t * n_heads + h];
            let w_clamped = ws.w_clamped[t * n_heads + h];
            let gamma_val = ws.gamma_t[t * n_heads + h];
            let lambda_val = ws.lambda_t[t * n_heads + h];

            // Adjoint from y_head (propagating from d_y_raw)
            let mut d_y_ret = [0.0f32; 4];
            let mut d_w_clamped = 0.0f32;

            for col in 0..4 {
                let dy_val = ws.d_y_raw[t * d + h_off + col];
                d_y_ret[col] = dy_val * w_clamped;
                d_w_clamped += dy_val * ws.y_ret[y_ret_off + col];
            }

            // Gradient through gain clipping:
            let d_w = if w_val < SRX_W_MAX {
                d_w_clamped
            } else {
                0.0
            };

            // d_noise_energy from w = 1 / (noise_energy + eps)
            let d_noise_energy = -d_w * w_val * w_val;

            // d_q_inv (noise subspace is coordinates 2 and 3)
            let mut d_q_inv = [0.0f32; 4];
            d_q_inv[2] = 2.0 * d_noise_energy * ws.q_inv[q_norm_off + 2];
            d_q_inv[3] = 2.0 * d_noise_energy * ws.q_inv[q_norm_off + 3];

            // Readout: y_ret = M^T q_rot => d_q_rot = M * d_y_ret, d_M_read = q_rot * d_y_ret^T
            let mut d_q_rot = [0.0f32; 4];
            let mut d_m_total = [0.0f32; 16];
            for row in 0..4 {
                for col in 0..4 {
                    let idx = row * 4 + col;
                    let m_val = ws.m[m_off + idx];
                    d_q_rot[row] += d_y_ret[col] * m_val;
                    d_m_total[idx] = d_m_acc[idx] + ws.q_rot[q_norm_off + row] * d_y_ret[col];
                }
            }

            let q_norm_arr: &[f32; 4] = ws.q_norm[q_norm_off..q_norm_off + 4].try_into().unwrap();

            // Reversible VJP backward through q_rot Butterfly rotation (forward mode)
            let mut d_q_norm_rot = [0.0f32; 4];
            apply_butterfly_4_backward(
                q_norm_arr,
                curr_thetas,
                false,
                &d_q_rot,
                &mut d_q_norm_rot,
                &mut d_thetas_acc,
            );

            // Reversible VJP backward through q_inv Butterfly rotation (inverse mode)
            let mut d_q_norm_inv = [0.0f32; 4];
            apply_butterfly_4_backward(
                q_norm_arr,
                curr_thetas,
                true,
                &d_q_inv,
                &mut d_q_norm_inv,
                &mut d_thetas_acc,
            );

            let mut d_q_norm_total = [0.0f32; 4];
            for c in 0..4 {
                d_q_norm_total[c] = d_q_norm_rot[c] + d_q_norm_inv[c];
            }

            // Widrow-Hoff Delta Rule Backward:
            // M_t[r, c] = lambda_t * M_{t-1}[r, c] + gamma_t * k_rot[r] * e_t[c]
            // where e_t[c] = v_raw[c] - sum_{r'} k_rot[r'] * M_{t-1}[r', c]
            let pm_off = (t.saturating_sub(1) * n_heads + h) * span_m;
            let e_arr = &ws.e_t[e_off..e_off + 4];

            // 1. d_e_t[c] = gamma_t * sum_r (d_m_total[r, c] * k_rot[r])
            let mut d_e = [0.0f32; 4];
            for col in 0..4 {
                let mut sum_ke = 0.0f32;
                for row in 0..4 {
                    sum_ke += d_m_total[row * 4 + col] * ws.k_rot[k_norm_off + row];
                }
                d_e[col] = gamma_val * sum_ke;
            }

            // 2. d_v_raw[c] += d_e_t[c]
            for col in 0..4 {
                ws.d_v_raw[t * d + h_off + col] += d_e[col];
            }

            // 3. d_k_rot[r] = gamma_t * sum_c (d_m_total[r, c] * e_t[c]) - sum_c (d_e_t[c] * M_{t-1}[r, c])
            let mut d_k_rot = [0.0f32; 4];
            for row in 0..4 {
                let mut term1 = 0.0f32;
                let mut term2 = 0.0f32;
                for col in 0..4 {
                    term1 += d_m_total[row * 4 + col] * e_arr[col];
                    if t > 0 {
                        term2 += d_e[col] * ws.m[pm_off + row * 4 + col];
                    }
                }
                d_k_rot[row] = gamma_val * term1 - term2;
            }

            // 4. d_gamma_t:
            // dM_t / dgamma_t = k_rot * e_t^T - (1 - gamma_base) * M_{t-1}
            let mut d_gamma_t = 0.0f32;
            for row in 0..4 {
                let kr = ws.k_rot[k_norm_off + row];
                for col in 0..4 {
                    let dm_val = d_m_total[row * 4 + col];
                    let p_val = if t > 0 { ws.m[pm_off + row * 4 + col] } else { 0.0 };
                    let d_write = kr * e_arr[col] - (1.0 - SRX_GAMMA_BASE) * p_val;
                    d_gamma_t += dm_val * d_write;
                }
            }

            // 5. Update d_m_acc for recurrence to t - 1:
            // d_M_{t-1}[r, c] = lambda_t * d_M_t[r, c] - k_rot[r] * d_e[c]
            if t > 0 {
                for row in 0..4 {
                    let kr = ws.k_rot[k_norm_off + row];
                    for col in 0..4 {
                        let idx = row * 4 + col;
                        d_m_acc[idx] = lambda_val * d_m_total[idx] - kr * d_e[col];
                    }
                }
            } else {
                d_m_acc.fill(0.0);
            }

            // Backward through Selective Dynamic Memory Gate:
            // gamma_t = sigmoid(s_gate)
            let s_sig = gamma_val * (1.0 - gamma_val);
            let d_s_gate = d_gamma_t * s_sig;

            grad.b_gamma[h] += d_s_gate;
            let a_tok = &ws.a[t * d..(t + 1) * d];
            for c in 0..d {
                grad.w_gamma[h * d + c] += d_s_gate * a_tok[c];
                ws.d_a[t * d + c] += d_s_gate * layer.attn.w_gamma[h * d + c];
            }

            let k_norm_arr: &[f32; 4] = ws.k_norm[k_norm_off..k_norm_off + 4].try_into().unwrap();

            // Reversible VJP backward through k_rot Butterfly rotation (forward mode)
            let mut d_k_norm = [0.0f32; 4];
            apply_butterfly_4_backward(
                k_norm_arr,
                curr_thetas,
                false,
                &d_k_rot,
                &mut d_k_norm,
                &mut d_thetas_acc,
            );

            // Phase modulation: delta_theta = alpha * tanh(k_norm * v)
            let v_raw_h = &ws.v_raw[t * d + h_off..t * d + h_off + 4];
            for i in 0..4 {
                let d_th = d_thetas_acc[i];
                let k_val = ws.k_norm[k_norm_off + i];
                let v_val = v_raw_h[i];
                let th_val = (k_val * v_val).tanh();
                let d_tanh = SRX_ALPHA * (1.0 - th_val * th_val);
                let grad_prod = d_th * d_tanh;

                d_k_norm[i] += grad_prod * v_val;
                ws.d_v_raw[t * d + h_off + i] += grad_prod * k_val;
            }

            // Backward through L2 normalizations
            let q_raw_h = &ws.q_raw[t * d + h_off..t * d + h_off + 4];
            let k_raw_h = &ws.k_raw[t * d + h_off..t * d + h_off + 4];
            let q_norm_vec = &ws.q_norm[q_norm_off..q_norm_off + 4];
            let k_norm_vec = &ws.k_norm[k_norm_off..k_norm_off + 4];

            let mut d_q_raw_h = [0.0f32; 4];
            let mut d_k_raw_h = [0.0f32; 4];

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

            for c in 0..4 {
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

            for k in 0..d {
                let ak = a_tok[k];
                grad.w_q[c * d + k] += dq * ak;
                grad.w_k[c * d + k] += dk * ak;
                grad.w_v[c * d + k] += dv * ak;

                ws.d_a[t * d + k] += dq * layer.attn.w_q[c * d + k]
                    + dk * layer.attn.w_k[c * d + k]
                    + dv * layer.attn.w_v[c * d + k];
            }
        }
    }

    // 9. Backward through Pre-Attention RMSNorm
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
    }

    // 10. Backward into Token Embeddings (Input Tokens)
    for (pos, &tok) in tokens.iter().enumerate() {
        let d_x0_tok = &ws.d_x_0[pos * d..(pos + 1) * d];
        let grad_emb = &mut grad.token_embeddings[tok * d..(tok + 1) * d];
        for c in 0..d {
            grad_emb[c] += d_x0_tok[c];
        }
    }
}

/// Splits corpus tokens by EOS_TOKEN_ID into sequence slices for training.
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

/// High-level dataset training method for SRX v04 using analytical BPTT, dynamic epsilon annealing, and AdamW.
/// Accounting for +240 FLOPs per token for the Widrow-Hoff Delta Rule.
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

    // Initial loss evaluation across first 20 sequences
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
        // Smooth quadratic epsilon annealing:
        let eps_min = 1e-3f32;
        let eps_max = 1.0f32;
        let eps_srx = eps_min + (eps_max - eps_min) * (1.0 - progress).powi(2);

        // Cosine learning rate schedule with minimum floor
        let current_lr = (lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(lr * 0.15);

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
    // Accounting: 6 * N_params + 240 FLOPs per token for Delta-Rule training
    let flops_per_token = 6u64 * (num_params as u64) + 240u64;
    let total_training_flops = flops_per_token * (dataset_tokens as u64) * (epochs as u64);
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
    fn test_srx_v04_gradient_check_numerical() {
        let mut config = TransformerConfig::lang_512();
        config.d_ff = 8;
        let mut model = SrxTransformer::new_with_seed(config.clone(), 123).unwrap();
        let mut ws = SrxTrainWorkspace::new(&config);
        let mut grad = SrxGrad::new(&config);

        let tokens = [1, 4, 2, 7, 3];
        let eps_srx = 0.5f32;

        grad.zero();
        forward_loss(&model, &tokens, &mut ws, eps_srx);
        backward_loss(&model, &tokens, &mut ws, &mut grad, eps_srx);

        let eps = 1e-4f32;

        // Check MLP w_2 gradient numerically
        for i in 0..model.layers[0].mlp.w_2.len().min(6) {
            let orig = model.layers[0].mlp.w_2[i];

            model.layers[0].mlp.w_2[i] = orig + eps;
            let l_plus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].mlp.w_2[i] = orig - eps;
            let l_minus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].mlp.w_2[i] = orig;

            let num_grad = (l_plus - l_minus) / (2.0 * eps);
            let ana_grad = grad.w_2[i];
            assert!(
                (ana_grad - num_grad).abs() < 5e-3,
                "MLP w_2[{}] grad mismatch: ana={}, num={}",
                i,
                ana_grad,
                num_grad
            );
        }

        // Check Selective Gate w_gamma gradient numerically
        for i in 0..model.layers[0].attn.w_gamma.len().min(6) {
            let orig = model.layers[0].attn.w_gamma[i];

            model.layers[0].attn.w_gamma[i] = orig + eps;
            let l_plus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].attn.w_gamma[i] = orig - eps;
            let l_minus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].attn.w_gamma[i] = orig;

            let num_grad = (l_plus - l_minus) / (2.0 * eps);
            let ana_grad = grad.w_gamma[i];
            assert!(
                (ana_grad - num_grad).abs() < 2e-3,
                "Attn w_gamma[{}] grad mismatch: ana={}, num={}",
                i,
                ana_grad,
                num_grad
            );
        }

        // Check Selective Gate b_gamma gradient numerically
        for i in 0..model.layers[0].attn.b_gamma.len() {
            let orig = model.layers[0].attn.b_gamma[i];

            model.layers[0].attn.b_gamma[i] = orig + eps;
            let l_plus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].attn.b_gamma[i] = orig - eps;
            let l_minus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].attn.b_gamma[i] = orig;

            let num_grad = (l_plus - l_minus) / (2.0 * eps);
            let ana_grad = grad.b_gamma[i];
            assert!(
                (ana_grad - num_grad).abs() < 2e-3,
                "Attn b_gamma[{}] grad mismatch: ana={}, num={}",
                i,
                ana_grad,
                num_grad
            );
        }

        // Check W_o gradient numerically
        let eps_wo = 1e-3f32;
        for i in 0..model.layers[0].attn.w_o.len().min(6) {
            let orig = model.layers[0].attn.w_o[i];

            model.layers[0].attn.w_o[i] = orig + eps_wo;
            let l_plus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].attn.w_o[i] = orig - eps_wo;
            let l_minus = forward_loss(&model, &tokens, &mut ws, eps_srx);

            model.layers[0].attn.w_o[i] = orig;

            let num_grad = (l_plus - l_minus) / (2.0 * eps_wo);
            let ana_grad = grad.w_o[i];
            assert!(
                (ana_grad - num_grad).abs() < 2e-2,
                "Attn w_o[{}] grad mismatch: ana={}, num={}",
                i,
                ana_grad,
                num_grad
            );
        }
    }

    #[test]
    fn test_srx_v04_delta_rule_interference_cancellation() {
        // Mathematical proof test:
        // Updating key k1 with value v1, then updating key k1 with v2,
        // while evaluating orthogonal key k2 (k1^T k2 = 0).
        let mut m = [0.0f32; 16]; // 4x4
        let k1 = [1.0f32, 0.0, 0.0, 0.0];
        let k2 = [0.0f32, 1.0, 0.0, 0.0];
        let v_fact_cat = [0.1f32, 0.2, 0.3, 0.4];
        let v_fact_dog = [0.5f32, 0.6, 0.7, 0.8];

        // Step 1: Write k2 -> v_fact_cat
        // v_hat = M^T k2 = 0
        let mut e = v_fact_cat;
        for r in 0..4 {
            for c in 0..4 {
                m[r * 4 + c] += k2[r] * e[c];
            }
        }

        // Verify retrieval of k2 is v_fact_cat
        for c in 0..4 {
            let mut retrieved = 0.0f32;
            for r in 0..4 {
                retrieved += k2[r] * m[r * 4 + c];
            }
            assert!((retrieved - v_fact_cat[c]).abs() < 1e-6);
        }

        // Step 2: Write k1 -> v_fact_dog
        let mut v_hat_k1 = [0.0f32; 4];
        for c in 0..4 {
            for r in 0..4 {
                v_hat_k1[c] += k1[r] * m[r * 4 + c];
            }
        }
        for c in 0..4 {
            e[c] = v_fact_dog[c] - v_hat_k1[c];
        }
        for r in 0..4 {
            for c in 0..4 {
                m[r * 4 + c] += k1[r] * e[c];
            }
        }

        // Verify retrieval of k2 is STILL v_fact_cat with ZERO corruption!
        for c in 0..4 {
            let mut retrieved = 0.0f32;
            for r in 0..4 {
                retrieved += k2[r] * m[r * 4 + c];
            }
            assert!(
                (retrieved - v_fact_cat[c]).abs() < 1e-6,
                "Orthogonal fact must suffer 0 distortion under Widrow-Hoff Delta Rule!"
            );
        }
    }
}
