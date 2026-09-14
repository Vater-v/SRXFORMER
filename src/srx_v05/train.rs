//! High-performance Analytical Reversible Backpropagation and AdamW Optimizer Engine for SRXFORMER v05.
//! Implements exact analytical gradients through 2nd-Order Associative RLS-Memory (Sherman-Morrison update),
//! Krylov Recurrent Depth (K=2) Resolvent Query Refinement, Monarch Butterfly Unitary Factorization,
//! and Phase Momentum Dynamics.

use std::time::Instant;

use crate::classic::config::TransformerConfig;
use crate::classic::ops::softmax;
use crate::classic::telemetry::TrainTelemetry;
use crate::classic::tokenizer::EOS_TOKEN_ID;

use super::attention::{SRX_ALPHA, SRX_MU, SRX_RLS_DELTA, SRX_RLS_LAMBDA, SRX_W_MAX};
use super::model::SrxTransformer;
use super::ops::{
    apply_butterfly_4, apply_butterfly_4_backward, l2_normalize, l2_normalize_backward,
};

/// Gradient storage for all 896 trainable parameters of SrxTransformer v05.
#[derive(Debug, Clone)]
pub struct SrxGrad {
    pub token_embeddings: Vec<f32>, // [vocab_size, d_model] = 53 * 8 = 424
    pub attn_norm_gamma: Vec<f32>,  // [d_model] = 8
    pub w_q: Vec<f32>,              // [d_model, d_model] = 64
    pub w_k: Vec<f32>,              // [d_model, d_model] = 64
    pub w_v: Vec<f32>,              // [d_model, d_model] = 64
    pub w_o: Vec<f32>,              // [d_model, d_model] = 64
    pub ffn_norm_gamma: Vec<f32>,   // [d_model] = 8
    pub w_1: Vec<f32>,              // [d_ff, d_model] = 12 * 8 = 96
    pub w_2: Vec<f32>,              // [d_model, d_ff] = 8 * 12 = 96
    pub final_norm_gamma: Vec<f32>, // [d_model] = 8

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
        self.ffn_norm_gamma.fill(0.0);
        self.w_1.fill(0.0);
        self.w_2.fill(0.0);
        self.final_norm_gamma.fill(0.0);
    }

    #[inline]
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

/// Pre-allocated workspace for SRX v05 forward activations and backward adjoints.
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

    pub norm_q: Vec<f32>,       // [max_seq_len, n_heads]
    pub norm_k: Vec<f32>,       // [max_seq_len, n_heads]
    pub q_norm: Vec<f32>,       // [max_seq_len, n_heads, head_dim]
    pub k_norm: Vec<f32>,       // [max_seq_len, n_heads, head_dim]
    pub p_thetas: Vec<f32>,     // [max_seq_len, n_heads, 4]
    pub thetas: Vec<f32>,       // [max_seq_len, n_heads, 4]
    pub k_rot: Vec<f32>,        // [max_seq_len, n_heads, head_dim]

    // RLS Sherman-Morrison state history
    pub v_p: Vec<f32>,          // [max_seq_len, n_heads, head_dim]
    pub denom: Vec<f32>,        // [max_seq_len, n_heads]
    pub k_gain: Vec<f32>,       // [max_seq_len, n_heads, head_dim]
    pub v_hat: Vec<f32>,        // [max_seq_len, n_heads, head_dim]
    pub e_t: Vec<f32>,          // [max_seq_len, n_heads, head_dim]
    pub m: Vec<f32>,            // [max_seq_len, n_heads, 16]
    pub p_mat: Vec<f32>,        // [max_seq_len, n_heads, 16]

    // Krylov Recurrent Depth buffers (K=2)
    pub u_q0: Vec<f32>,         // [max_seq_len, n_heads, head_dim]
    pub q_combo: Vec<f32>,      // [max_seq_len, n_heads, head_dim]
    pub norm_q_combo: Vec<f32>, // [max_seq_len, n_heads]
    pub q_1: Vec<f32>,          // [max_seq_len, n_heads, head_dim]

    // MUSIC noise projector and retrieval
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
            norm_q: vec![0.0; s * n_heads],
            norm_k: vec![0.0; s * n_heads],
            q_norm: vec![0.0; s * n_heads * head_dim],
            k_norm: vec![0.0; s * n_heads * head_dim],
            p_thetas: vec![0.0; s * n_heads * 4],
            thetas: vec![0.0; s * n_heads * 4],
            k_rot: vec![0.0; s * n_heads * head_dim],
            v_p: vec![0.0; s * n_heads * head_dim],
            denom: vec![0.0; s * n_heads],
            k_gain: vec![0.0; s * n_heads * head_dim],
            v_hat: vec![0.0; s * n_heads * head_dim],
            e_t: vec![0.0; s * n_heads * head_dim],
            m: vec![0.0; s * n_heads * head_dim * head_dim],
            p_mat: vec![0.0; s * n_heads * head_dim * head_dim],
            u_q0: vec![0.0; s * n_heads * head_dim],
            q_combo: vec![0.0; s * n_heads * head_dim],
            norm_q_combo: vec![0.0; s * n_heads],
            q_1: vec![0.0; s * n_heads * head_dim],
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

/// Zero-dependency AdamW Optimizer for SrxTransformer v05.
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

        let param_slices: [&mut [f32]; 10] = [
            &mut model.token_embeddings,
            &mut layer.attn_norm_gamma,
            &mut layer.attn.w_q,
            &mut layer.attn.w_k,
            &mut layer.attn.w_v,
            &mut layer.attn.w_o,
            &mut layer.ffn_norm_gamma,
            &mut layer.mlp.w_1,
            &mut layer.mlp.w_2,
            &mut model.final_norm_gamma,
        ];

        let grad_slices: [&[f32]; 10] = [
            &grad.token_embeddings,
            &grad.attn_norm_gamma,
            &grad.w_q,
            &grad.w_k,
            &grad.w_v,
            &grad.w_o,
            &grad.ffn_norm_gamma,
            &grad.w_1,
            &grad.w_2,
            &grad.final_norm_gamma,
        ];

        let m_slices: [&mut [f32]; 10] = [
            &mut self.m.token_embeddings,
            &mut self.m.attn_norm_gamma,
            &mut self.m.w_q,
            &mut self.m.w_k,
            &mut self.m.w_v,
            &mut self.m.w_o,
            &mut self.m.ffn_norm_gamma,
            &mut self.m.w_1,
            &mut self.m.w_2,
            &mut self.m.final_norm_gamma,
        ];

        let v_slices: [&mut [f32]; 10] = [
            &mut self.v.token_embeddings,
            &mut self.v.attn_norm_gamma,
            &mut self.v.w_q,
            &mut self.v.w_k,
            &mut self.v.w_v,
            &mut self.v.w_o,
            &mut self.v.ffn_norm_gamma,
            &mut self.v.w_1,
            &mut self.v.w_2,
            &mut self.v.final_norm_gamma,
        ];

        for k in 0..10 {
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

    // 1. Embedding lookup
    for t in 0..seq_len {
        let tok = tokens[t];
        let emb_off = tok * d;
        ws.x_0[t * d..(t + 1) * d].copy_from_slice(&model.token_embeddings[emb_off..emb_off + d]);
    }

    // 2. Pre-Attention RMSNorm
    for t in 0..seq_len {
        let x_tok = &ws.x_0[t * d..(t + 1) * d];
        let mut sum_sq = 0.0f32;
        for &val in x_tok {
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + eps_norm).sqrt();
        ws.rms_attn[t] = rms;
        let inv_rms = 1.0 / rms;

        for c in 0..d {
            let x_n = x_tok[c] * inv_rms;
            ws.x_0_norm[t * d + c] = x_n;
            ws.a[t * d + c] = x_n * layer.attn_norm_gamma[c];
        }
    }

    // 3. Q, K, V linear projections
    for t in 0..seq_len {
        let a_tok = &ws.a[t * d..(t + 1) * d];
        for i in 0..d {
            let row_off = i * d;
            let mut sq = 0.0f32;
            let mut sk = 0.0f32;
            let mut sv = 0.0f32;
            for j in 0..d {
                let aj = a_tok[j];
                sq += layer.attn.w_q[row_off + j] * aj;
                sk += layer.attn.w_k[row_off + j] * aj;
                sv += layer.attn.w_v[row_off + j] * aj;
            }
            ws.q_raw[t * d + i] = sq;
            ws.k_raw[t * d + i] = sk;
            ws.v_raw[t * d + i] = sv;
        }
    }

    // 4. Per-head SRX Recurrence with 2nd-Order RLS, Krylov Depth, Butterfly Mixer & Phase Momentum
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
            let p_mat_off = (t * n_heads + h) * span_m;
            let v_p_off = (t * n_heads + h) * head_dim;
            let k_gain_off = (t * n_heads + h) * head_dim;
            let v_hat_off = (t * n_heads + h) * head_dim;
            let e_off = (t * n_heads + h) * head_dim;

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

            // b) Phase Momentum update:
            // p_{\theta, t} = \mu * p_{\theta, t-1} + \alpha * (k_norm \odot v_raw_h)
            // \theta_t = \theta_{t-1} + p_{\theta, t}
            let prev_th_off = (t.saturating_sub(1) * n_heads + h) * span_theta;
            for i in 0..4 {
                let prev_p = if t > 0 { ws.p_thetas[prev_th_off + i] } else { 0.0 };
                let prev_th = if t > 0 { ws.thetas[prev_th_off + i] } else { 0.0 };
                let k_val = ws.k_norm[k_norm_off + i];
                let v_val = v_raw_h[i];
                let p_curr = SRX_MU * prev_p + SRX_ALPHA * (k_val * v_val);
                ws.p_thetas[thetas_off + i] = p_curr;
                ws.thetas[thetas_off + i] = prev_th + p_curr;
            }

            let curr_thetas: &[f32; 4] = ws.thetas[thetas_off..thetas_off + 4].try_into().unwrap();
            let k_norm_arr: &[f32; 4] = ws.k_norm[k_norm_off..k_norm_off + 4].try_into().unwrap();

            // c) k_rot = U(Theta_t) * k_norm (Monarch Butterfly Unitary)
            let mut k_rot_arr = [0.0f32; 4];
            apply_butterfly_4(k_norm_arr, curr_thetas, false, &mut k_rot_arr);
            ws.k_rot[k_norm_off..k_norm_off + 4].copy_from_slice(&k_rot_arr);

            // d) 2nd-Order Associative RLS Memory (Sherman-Morrison update):
            let pp_off = (t.saturating_sub(1) * n_heads + h) * span_m;
            let pm_off = (t.saturating_sub(1) * n_heads + h) * span_m;

            // 1. v_p = P_{t-1} * k_rot
            let mut v_p_arr = [0.0f32; 4];
            for r in 0..4 {
                let mut sum = 0.0f32;
                for c in 0..4 {
                    let p_val = if t > 0 {
                        ws.p_mat[pp_off + r * 4 + c]
                    } else if r == c {
                        1.0 / SRX_RLS_DELTA
                    } else {
                        0.0
                    };
                    sum += p_val * k_rot_arr[c];
                }
                v_p_arr[r] = sum;
            }
            ws.v_p[v_p_off..v_p_off + 4].copy_from_slice(&v_p_arr);

            // 2. denom = \lambda + k_rot^T * v_p
            let mut k_dot_vp = 0.0f32;
            for i in 0..4 {
                k_dot_vp += k_rot_arr[i] * v_p_arr[i];
            }
            let denom_val = SRX_RLS_LAMBDA + k_dot_vp;
            let inv_denom = 1.0 / denom_val;
            ws.denom[t * n_heads + h] = denom_val;

            // 3. k_gain = v_p / denom
            let mut k_gain_arr = [0.0f32; 4];
            for i in 0..4 {
                k_gain_arr[i] = v_p_arr[i] * inv_denom;
            }
            ws.k_gain[k_gain_off..k_gain_off + 4].copy_from_slice(&k_gain_arr);

            // 4. v_hat = M_{t-1}^T * k_rot
            let mut v_hat_arr = [0.0f32; 4];
            for col in 0..4 {
                let mut sum_v = 0.0f32;
                if t > 0 {
                    for row in 0..4 {
                        sum_v += k_rot_arr[row] * ws.m[pm_off + row * 4 + col];
                    }
                }
                v_hat_arr[col] = sum_v;
                ws.e_t[e_off + col] = v_raw_h[col] - sum_v;
            }
            ws.v_hat[v_hat_off..v_hat_off + 4].copy_from_slice(&v_hat_arr);

            // 5. Memory update: M_t = \lambda * M_{t-1} + k_gain * e_t^T
            for row in 0..4 {
                let kg = k_gain_arr[row];
                for col in 0..4 {
                    let idx = row * 4 + col;
                    let p_val = if t > 0 { ws.m[pm_off + idx] } else { 0.0 };
                    let e_val = ws.e_t[e_off + col];
                    ws.m[m_off + idx] = SRX_RLS_LAMBDA * p_val + kg * e_val;
                }
            }

            // 6. Covariance update: P_t = (1 / \lambda) * (P_{t-1} - k_gain * (k_rot^T * P_{t-1}))
            let mut k_trans_p = [0.0f32; 4];
            for col in 0..4 {
                let mut sum = 0.0f32;
                for row in 0..4 {
                    let p_val = if t > 0 {
                        ws.p_mat[pp_off + row * 4 + col]
                    } else if row == col {
                        1.0 / SRX_RLS_DELTA
                    } else {
                        0.0
                    };
                    sum += k_rot_arr[row] * p_val;
                }
                k_trans_p[col] = sum;
            }

            let inv_lambda = 1.0 / SRX_RLS_LAMBDA;
            for row in 0..4 {
                let kg = k_gain_arr[row];
                for col in 0..4 {
                    let idx = row * 4 + col;
                    let p_val = if t > 0 {
                        ws.p_mat[pp_off + idx]
                    } else if row == col {
                        1.0 / SRX_RLS_DELTA
                    } else {
                        0.0
                    };
                    ws.p_mat[p_mat_off + idx] = inv_lambda * (p_val - kg * k_trans_p[col]);
                }
            }

            // e) Krylov Recurrent Depth (K=2):
            // q^{(0)} = q_norm
            // u_q0 = U_t * q^{(0)}
            let q_norm_arr: &[f32; 4] = ws.q_norm[q_norm_off..q_norm_off + 4].try_into().unwrap();
            let mut u_q0_arr = [0.0f32; 4];
            apply_butterfly_4(q_norm_arr, curr_thetas, false, &mut u_q0_arr);
            ws.u_q0[q_norm_off..q_norm_off + 4].copy_from_slice(&u_q0_arr);

            // q_combo = 0.5 * q^{(0)} + 0.5 * u_q0
            let mut q_combo_arr = [0.0f32; 4];
            for i in 0..4 {
                q_combo_arr[i] = 0.5 * q_norm_arr[i] + 0.5 * u_q0_arr[i];
            }
            ws.q_combo[q_norm_off..q_norm_off + 4].copy_from_slice(&q_combo_arr);

            // q^{(1)} = L2_Norm(q_combo)
            let mut q_1_arr = [0.0f32; 4];
            let norm_combo = l2_normalize(&q_combo_arr, &mut q_1_arr, 1e-12);
            ws.norm_q_combo[t * n_heads + h] = norm_combo;
            ws.q_1[q_norm_off..q_norm_off + 4].copy_from_slice(&q_1_arr);

            // f) Noise subspace projection using q^{(1)}:
            // q_inv = U^\dagger(Theta_t) * q^{(1)}
            let mut q_inv_arr = [0.0f32; 4];
            apply_butterfly_4(&q_1_arr, curr_thetas, true, &mut q_inv_arr);
            ws.q_inv[q_norm_off..q_norm_off + 4].copy_from_slice(&q_inv_arr);

            // Signal rank r = 2; noise energy = ||q_inv[2..4]||^2
            let noise_energy = q_inv_arr[2] * q_inv_arr[2] + q_inv_arr[3] * q_inv_arr[3];
            ws.noise_energy[t * n_heads + h] = noise_energy;

            // Dirac-like resonant gain with hard clipping
            let w_val = 1.0 / (noise_energy + eps_srx);
            let w_clamped = w_val.min(SRX_W_MAX);
            ws.w[t * n_heads + h] = w_val;
            ws.w_clamped[t * n_heads + h] = w_clamped;

            // g) Memory retrieval using q^{(1)}:
            // q_rot = U(Theta_t) * q^{(1)}
            let mut q_rot_arr = [0.0f32; 4];
            apply_butterfly_4(&q_1_arr, curr_thetas, false, &mut q_rot_arr);
            ws.q_rot[q_norm_off..q_norm_off + 4].copy_from_slice(&q_rot_arr);

            // y_ret = M_t^T q_rot
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
        for i in 0..d {
            let row_off = i * d;
            let mut sum_o = 0.0f32;
            for j in 0..d {
                sum_o += layer.attn.w_o[row_off + j] * y_tok[j];
            }
            ws.attn_out[t * d + i] = sum_o;
            ws.x_mid[t * d + i] = ws.x_0[t * d + i] + sum_o;
        }
    }

    // 6. Pre-FFN RMSNorm
    for t in 0..seq_len {
        let x_tok = &ws.x_mid[t * d..(t + 1) * d];
        let mut sum_sq = 0.0f32;
        for &val in x_tok {
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + eps_norm).sqrt();
        ws.rms_ffn[t] = rms;
        let inv_rms = 1.0 / rms;

        for c in 0..d {
            let x_n = x_tok[c] * inv_rms;
            ws.x_mid_norm[t * d + c] = x_n;
            ws.b[t * d + c] = x_n * layer.ffn_norm_gamma[c];
        }
    }

    // 7. MLP Forward (W_1 -> ReLU -> W_2)
    for t in 0..seq_len {
        let b_tok = &ws.b[t * d..(t + 1) * d];
        for i in 0..d_ff {
            let row_off = i * d;
            let mut sum_1 = 0.0f32;
            for j in 0..d {
                sum_1 += layer.mlp.w_1[row_off + j] * b_tok[j];
            }
            ws.z[t * d_ff + i] = sum_1;
            ws.h[t * d_ff + i] = if sum_1 > 0.0 { sum_1 } else { 0.0 };
        }

        let h_tok = &ws.h[t * d_ff..(t + 1) * d_ff];
        for i in 0..d {
            let row_off = i * d_ff;
            let mut sum_2 = 0.0f32;
            for j in 0..d_ff {
                sum_2 += layer.mlp.w_2[row_off + j] * h_tok[j];
            }
            ws.f[t * d + i] = sum_2;
            ws.x_1[t * d + i] = ws.x_mid[t * d + i] + sum_2;
        }
    }

    // 8. Final RMSNorm
    for t in 0..seq_len {
        let x_tok = &ws.x_1[t * d..(t + 1) * d];
        let mut sum_sq = 0.0f32;
        for &val in x_tok {
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + eps_norm).sqrt();
        ws.rms_final[t] = rms;
        let inv_rms = 1.0 / rms;

        for c in 0..d {
            let x_n = x_tok[c] * inv_rms;
            ws.x_1_norm[t * d + c] = x_n;
            ws.u[t * d + c] = x_n * model.final_norm_gamma[c];
        }
    }

    // 9. Tied LM Head & Cross-Entropy Loss
    let mut total_loss = 0.0f32;
    for t in 0..seq_len - 1 {
        let u_tok = &ws.u[t * d..(t + 1) * d];
        for i in 0..v {
            let row_off = i * d;
            let mut dot = 0.0f32;
            for j in 0..d {
                dot += model.token_embeddings[row_off + j] * u_tok[j];
            }
            ws.logits[t * v + i] = dot;
        }

        let target_tok = tokens[t + 1];
        let probs_tok = &mut ws.probs[t * v..(t + 1) * v];
        probs_tok.copy_from_slice(&ws.logits[t * v..(t + 1) * v]);
        softmax(probs_tok);

        let p = probs_tok[target_tok].max(1e-12);
        total_loss -= p.ln();
    }

    total_loss / (seq_len - 1) as f32
}

/// Sequence backward pass computing exact analytical gradients via reversible BPTT.
pub fn backward_loss(
    model: &SrxTransformer,
    tokens: &[usize],
    ws: &mut SrxTrainWorkspace,
    grad: &mut SrxGrad,
    _eps_srx: f32,
) {
    let seq_len = tokens.len();
    assert!(seq_len >= 2);
    assert!(seq_len <= ws.max_seq_len);

    let d = ws.d_model;
    let d_ff = ws.d_ff;
    let v = ws.vocab_size;
    let n_heads = ws.n_heads;
    let head_dim = ws.head_dim;
    let inv_steps = 1.0 / (seq_len - 1) as f32;

    let layer = &model.layers[0];

    // Reset adjoint buffers
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

    // 1. Cross-entropy loss derivative w.r.t logits
    for t in 0..seq_len - 1 {
        let target_tok = tokens[t + 1];
        let probs_tok = &ws.probs[t * v..(t + 1) * v];
        let d_logits_tok = &mut ws.d_logits[t * v..(t + 1) * v];

        for i in 0..v {
            let p = probs_tok[i];
            let indicator = if i == target_tok { 1.0 } else { 0.0 };
            d_logits_tok[i] = (p - indicator) * inv_steps;
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

    // 4. Residual connection & MLP backward
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

    // 7. Backward through SRX Attention Recurrence with 2nd-Order RLS, Krylov Depth & Phase Momentum
    for h in 0..n_heads {
        let h_off = h * head_dim;
        let span_theta = 4;
        let span_m = head_dim * head_dim;

        // Recurrence accumulators for reverse-time unroll
        let mut d_m_acc = [0.0f32; 16];
        let mut d_thetas_acc = [0.0f32; 4];
        let mut d_p_thetas_acc = [0.0f32; 4];

        for t in (0..seq_len).rev() {
            let q_norm_off = (t * n_heads + h) * head_dim;
            let k_norm_off = (t * n_heads + h) * head_dim;
            let thetas_off = (t * n_heads + h) * span_theta;
            let m_off = (t * n_heads + h) * span_m;
            let y_ret_off = (t * n_heads + h) * head_dim;
            let e_off = (t * n_heads + h) * head_dim;
            let v_p_off = (t * n_heads + h) * head_dim;

            let curr_thetas: &[f32; 4] = ws.thetas[thetas_off..thetas_off + 4].try_into().unwrap();
            let w_val = ws.w[t * n_heads + h];
            let w_clamped = ws.w_clamped[t * n_heads + h];

            // Adjoint from y_head
            let mut d_y_ret = [0.0f32; 4];
            let mut d_w_clamped = 0.0f32;

            for col in 0..4 {
                let dy_val = ws.d_y_raw[t * d + h_off + col];
                d_y_ret[col] = dy_val * w_clamped;
                d_w_clamped += dy_val * ws.y_ret[y_ret_off + col];
            }

            // Gradient through gain clipping
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

            // Readout: y_ret = M_t^T q_rot => d_q_rot = M_t * d_y_ret, d_M_read = q_rot * d_y_ret^T
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

            let q_1_arr: &[f32; 4] = ws.q_1[q_norm_off..q_norm_off + 4].try_into().unwrap();

            // Backward through q_rot Butterfly rotation (forward mode)
            let mut d_q1_rot = [0.0f32; 4];
            apply_butterfly_4_backward(
                q_1_arr,
                curr_thetas,
                false,
                &d_q_rot,
                &mut d_q1_rot,
                &mut d_thetas_acc,
            );

            // Backward through q_inv Butterfly rotation (inverse mode)
            let mut d_q1_inv = [0.0f32; 4];
            apply_butterfly_4_backward(
                q_1_arr,
                curr_thetas,
                true,
                &d_q_inv,
                &mut d_q1_inv,
                &mut d_thetas_acc,
            );

            let mut d_q1 = [0.0f32; 4];
            for c in 0..4 {
                d_q1[c] = d_q1_rot[c] + d_q1_inv[c];
            }

            // Backward through Krylov L2-norm: q^{(1)} = l2_normalize(q_combo)
            let q_combo_arr: &[f32; 4] = ws.q_combo[q_norm_off..q_norm_off + 4].try_into().unwrap();
            let norm_combo = ws.norm_q_combo[t * n_heads + h];
            let mut d_q_combo = [0.0f32; 4];
            l2_normalize_backward(q_combo_arr, q_1_arr, norm_combo, &d_q1, &mut d_q_combo);

            // Backward through q_combo = 0.5 * q^{(0)} + 0.5 * u_q0
            let mut d_u_q0 = [0.0f32; 4];
            let mut d_q0_direct = [0.0f32; 4];
            for i in 0..4 {
                d_u_q0[i] = 0.5 * d_q_combo[i];
                d_q0_direct[i] = 0.5 * d_q_combo[i];
            }

            // Backward through u_q0 = U(thetas) * q^{(0)}
            let q_norm_arr: &[f32; 4] = ws.q_norm[q_norm_off..q_norm_off + 4].try_into().unwrap();
            let mut d_q0_rot = [0.0f32; 4];
            apply_butterfly_4_backward(
                q_norm_arr,
                curr_thetas,
                false,
                &d_u_q0,
                &mut d_q0_rot,
                &mut d_thetas_acc,
            );

            let mut d_q_norm_total = [0.0f32; 4];
            for i in 0..4 {
                d_q_norm_total[i] = d_q0_direct[i] + d_q0_rot[i];
            }

            // 2nd-Order RLS Backward:
            // M_t[r, c] = \lambda * M_{t-1}[r, c] + k_gain[r] * e_t[c]
            // where e_t[c] = v_raw_h[c] - sum_{r'} k_rot[r'] * M_{t-1}[r', c]
            let pm_off = (t.saturating_sub(1) * n_heads + h) * span_m;
            let pp_off = (t.saturating_sub(1) * n_heads + h) * span_m;
            let e_arr = &ws.e_t[e_off..e_off + 4];
            let k_gain_arr = &ws.k_gain[q_norm_off..q_norm_off + 4];
            let denom_val = ws.denom[t * n_heads + h];
            let v_p_arr = &ws.v_p[v_p_off..v_p_off + 4];

            // 1. d_e_t[c] = sum_r (d_m_total[r, c] * k_gain[r])
            let mut d_e = [0.0f32; 4];
            for col in 0..4 {
                let mut sum_ke = 0.0f32;
                for row in 0..4 {
                    sum_ke += d_m_total[row * 4 + col] * k_gain_arr[row];
                }
                d_e[col] = sum_ke;
            }

            // 2. d_v_raw[c] += d_e_t[c]
            for col in 0..4 {
                ws.d_v_raw[t * d + h_off + col] += d_e[col];
            }

            // 3. d_k_gain[r] = sum_c (d_m_total[r, c] * e_t[c])
            let mut d_k_gain = [0.0f32; 4];
            for row in 0..4 {
                let mut sum_ge = 0.0f32;
                for col in 0..4 {
                    sum_ge += d_m_total[row * 4 + col] * e_arr[col];
                }
                d_k_gain[row] = sum_ge;
            }

            // 4. From k_gain = v_p / denom:
            let inv_denom = 1.0 / denom_val;
            let mut d_v_p = [0.0f32; 4];
            for r in 0..4 {
                d_v_p[r] = d_k_gain[r] * inv_denom;
            }

            let mut dot_gk_vp = 0.0f32;
            for r in 0..4 {
                dot_gk_vp += d_k_gain[r] * v_p_arr[r];
            }
            let d_denom = -dot_gk_vp * (inv_denom * inv_denom);

            // denom = \lambda + k_rot^T * v_p
            for r in 0..4 {
                d_v_p[r] += d_denom * ws.k_rot[k_norm_off + r];
            }

            // 5. Total d_k_rot:
            // From v_hat: - sum_c (d_e_t[c] * M_{t-1}[r, c])
            // From denom: d_denom * v_p[r]
            // From v_p = P_{t-1} * k_rot: sum_c P_{t-1}[c, r] * d_v_p[c]
            let mut d_k_rot = [0.0f32; 4];
            for row in 0..4 {
                let mut term_vhat = 0.0f32;
                for col in 0..4 {
                    if t > 0 {
                        term_vhat += d_e[col] * ws.m[pm_off + row * 4 + col];
                    }
                }

                let mut term_p = 0.0f32;
                for c in 0..4 {
                    let p_val = if t > 0 {
                        ws.p_mat[pp_off + c * 4 + row]
                    } else if c == row {
                        1.0 / SRX_RLS_DELTA
                    } else {
                        0.0
                    };
                    term_p += p_val * d_v_p[c];
                }

                d_k_rot[row] = -term_vhat + d_denom * v_p_arr[row] + term_p;
            }

            // 6. Recurrence accumulator for M_{t-1}:
            // d_M_{t-1}[r, c] = \lambda * d_M_t[r, c] - k_rot[r] * d_e[c]
            if t > 0 {
                for row in 0..4 {
                    let kr = ws.k_rot[k_norm_off + row];
                    for col in 0..4 {
                        let idx = row * 4 + col;
                        d_m_acc[idx] = SRX_RLS_LAMBDA * d_m_total[idx] - kr * d_e[col];
                    }
                }
            } else {
                d_m_acc.fill(0.0);
            }

            // 7. Backward through k_rot = U(\theta_t) * k_norm
            let mut d_k_norm = [0.0f32; 4];
            let k_norm_arr: &[f32; 4] = ws.k_norm[k_norm_off..k_norm_off + 4].try_into().unwrap();
            apply_butterfly_4_backward(
                k_norm_arr,
                curr_thetas,
                false,
                &d_k_rot,
                &mut d_k_norm,
                &mut d_thetas_acc,
            );

            // 8. Backward through Phase Momentum:
            // Forward:
            // p_{\theta, t} = \mu * p_{\theta, t-1} + \alpha * (k_norm \odot v_raw_h)
            // \theta_t = \theta_{t-1} + p_{\theta, t}
            let d_theta_total = d_thetas_acc;
            for i in 0..4 {
                let d_p = d_p_thetas_acc[i] + d_theta_total[i];
                let k_val = ws.k_norm[k_norm_off + i];
                let v_val = ws.v_raw[t * d + h_off + i];
                d_k_norm[i] += SRX_ALPHA * d_p * v_val;
                ws.d_v_raw[t * d + h_off + i] += SRX_ALPHA * d_p * k_val;
                d_p_thetas_acc[i] = if t > 0 { SRX_MU * d_p } else { 0.0 };
                d_thetas_acc[i] = if t > 0 { d_theta_total[i] } else { 0.0 };
            }

            // 9. Backward through L2-normalization of q and k
            let q_raw_h = &ws.q_raw[t * d + h_off..t * d + h_off + head_dim];
            let k_raw_h = &ws.k_raw[t * d + h_off..t * d + h_off + head_dim];
            let n_q = ws.norm_q[t * n_heads + h];
            let n_k = ws.norm_k[t * n_heads + h];

            l2_normalize_backward(
                q_raw_h,
                q_norm_arr,
                n_q,
                &d_q_norm_total,
                &mut ws.d_q_raw[t * d + h_off..t * d + h_off + head_dim],
            );
            l2_normalize_backward(
                k_raw_h,
                k_norm_arr,
                n_k,
                &d_k_norm,
                &mut ws.d_k_raw[t * d + h_off..t * d + h_off + head_dim],
            );
        }
    }

    // 8. Backward through Q, K, V Projections
    for t in 0..seq_len {
        let a_tok = &ws.a[t * d..(t + 1) * d];
        for i in 0..d {
            let row_off = i * d;
            let dq = ws.d_q_raw[t * d + i];
            let dk = ws.d_k_raw[t * d + i];
            let dv = ws.d_v_raw[t * d + i];

            for j in 0..d {
                let aj = a_tok[j];
                grad.w_q[row_off + j] += dq * aj;
                grad.w_k[row_off + j] += dk * aj;
                grad.w_v[row_off + j] += dv * aj;

                let da = dq * layer.attn.w_q[row_off + j]
                    + dk * layer.attn.w_k[row_off + j]
                    + dv * layer.attn.w_v[row_off + j];
                ws.d_a[t * d + j] += da;
            }
        }
    }

    // 9. Pre-Attention RMSNorm backward -> d_x_0 & grad.attn_norm_gamma
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

    // 10. Accumulate d_x_0 into input token embeddings
    for t in 0..seq_len {
        let tok = tokens[t];
        let emb_off = tok * d;
        let grad_emb = &mut grad.token_embeddings[emb_off..emb_off + d];
        for c in 0..d {
            grad_emb[c] += ws.d_x_0[t * d + c];
        }
    }
}

/// Splits token stream into sub-sequences strictly segmented by `<eos>` delimiters.
pub fn split_into_eos_sequences(tokens: &[usize], max_seq_len: usize) -> Vec<Vec<usize>> {
    let mut seqs = Vec::new();
    let mut current = Vec::new();

    for &tok in tokens {
        current.push(tok);
        if tok == EOS_TOKEN_ID || current.len() >= max_seq_len {
            if current.len() >= 2 {
                seqs.push(current);
            }
            current = Vec::new();
        }
    }

    if current.len() >= 2 {
        seqs.push(current);
    }

    seqs
}

/// Trains SrxTransformer v05 deterministically over a tokenized dataset for specified epochs.
pub fn train_dataset(
    model: &mut SrxTransformer,
    tokens: &[usize],
    epochs: usize,
    lr: f32,
) -> TrainTelemetry {
    let mut ws = SrxTrainWorkspace::new(&model.config);
    let mut grad = SrxGrad::new(&model.config);
    let mut optimizer = SrxAdamW::new(&model.config, 0.0);
    let seqs = split_into_eos_sequences(tokens, model.config.max_seq_len);
    let mut loss_history = Vec::with_capacity(epochs);

    let eval_count = seqs.len().min(20);
    let mut initial_loss = 0.0f32;
    for seq in &seqs[..eval_count] {
        initial_loss += forward_loss(model, seq, &mut ws, 1.0);
    }
    initial_loss /= eval_count as f32;

    let start = Instant::now();
    let mut total_steps = 0;

    for epoch in 1..=epochs {
        let progress = epoch as f32 / epochs as f32;
        let eps_min = 1e-3f32;
        let eps_max = 1.0f32;
        let eps_srx = eps_min + (eps_max - eps_min) * (1.0 - progress).powi(2);
        let current_lr = (lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(lr * 0.1);

        let mut epoch_loss = 0.0f32;
        for seq in &seqs {
            grad.zero();
            let loss = forward_loss(model, seq, &mut ws, eps_srx);
            epoch_loss += loss;
            backward_loss(model, seq, &mut ws, &mut grad, eps_srx);
            grad.clip_grad_norm(1.0);
            optimizer.step(model, &grad, current_lr);
            total_steps += 1;
        }

        let avg_loss = epoch_loss / seqs.len() as f32;
        loss_history.push(avg_loss);
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let final_loss = *loss_history.last().unwrap_or(&initial_loss);
    let total_flops = 6240 * tokens.len() as u64 * epochs as u64;

    TrainTelemetry {
        num_params: model.param_count(),
        dataset_tokens: tokens.len(),
        epochs,
        total_training_flops: total_flops,
        elapsed_ms,
        mflops_per_sec: (total_flops as f64 / 1e6) / (elapsed_ms / 1000.0),
        gflops_per_sec: (total_flops as f64 / 1e9) / (elapsed_ms / 1000.0),
        initial_loss,
        final_loss,
        initial_perplexity: initial_loss.exp(),
        final_perplexity: final_loss.exp(),
        total_steps,
        loss_history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srx_v05_gradient_check_numerical() {
        let config = TransformerConfig::lang_v3();
        let mut model = SrxTransformer::new_with_seed(config.clone(), 42).unwrap();
        let mut ws = SrxTrainWorkspace::new(&config);
        let mut grad = SrxGrad::new(&config);

        let tokens = [2, 6, 41, 6, 12, 3, 7, EOS_TOKEN_ID]; // "<user> 2 / 2 = <bot> 1 <eos>"

        // Compute analytical gradients
        grad.zero();
        let loss = forward_loss(&model, &tokens, &mut ws, 1.0);
        backward_loss(&model, &tokens, &mut ws, &mut grad, 1.0);

        assert!(loss > 0.0 && loss.is_finite());
        assert!(grad.l2_norm() > 0.0);

        // Finite difference check on W_q
        let eps = 1e-3f32;
        for i in 0..4 {
            let orig = model.layers[0].attn.w_q[i];

            model.layers[0].attn.w_q[i] = orig + eps;
            let loss_plus = forward_loss(&model, &tokens, &mut ws, 1.0);

            model.layers[0].attn.w_q[i] = orig - eps;
            let loss_minus = forward_loss(&model, &tokens, &mut ws, 1.0);

            model.layers[0].attn.w_q[i] = orig;

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
            let ana_grad = grad.w_q[i];

            assert!(
                (ana_grad - num_grad).abs() < 2e-2,
                "W_q[{}] gradient mismatch: ana={}, num={}",
                i,
                ana_grad,
                num_grad
            );
        }

        // Finite difference check on W_1
        for i in 0..4 {
            let orig = model.layers[0].mlp.w_1[i];

            model.layers[0].mlp.w_1[i] = orig + eps;
            let loss_plus = forward_loss(&model, &tokens, &mut ws, 1.0);

            model.layers[0].mlp.w_1[i] = orig - eps;
            let loss_minus = forward_loss(&model, &tokens, &mut ws, 1.0);

            model.layers[0].mlp.w_1[i] = orig;

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
            let ana_grad = grad.w_1[i];

            assert!(
                (ana_grad - num_grad).abs() < 2e-2,
                "W_1[{}] gradient mismatch: ana={}, num={}",
                i,
                ana_grad,
                num_grad
            );
        }
    }
}
