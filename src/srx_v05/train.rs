//! High-performance Analytical Reversible Backpropagation and AdamW Optimizer Engine for SRXFORMER v05.
//! Implements exact analytical gradients through Pure Orthogonal Complement Projector Memory,
//! Undistorted MUSIC Subspace Pseudo-Spectrum, and Monarch Butterfly Unitary Factorization.

use std::time::Instant;

use crate::classic::config::TransformerConfig;
use crate::classic::ops::softmax;
use crate::classic::telemetry::TrainTelemetry;
use crate::classic::tokenizer::EOS_TOKEN_ID;

use super::attention::{SRX_ALPHA, SRX_W_MAX};
use super::model::SrxTransformer;
use super::ops::{
    apply_butterfly_4, apply_butterfly_4_backward, l2_normalize, l2_normalize_backward,
};

/// Gradient storage for all trainable parameters of SrxTransformer v05 (896 parameters under Chinchilla).
#[derive(Debug, Clone)]
pub struct SrxGrad {
    pub token_embeddings: Vec<f32>, // [vocab_size, d_model] = 65 * 8 = 520
    pub attn_norm_gamma: Vec<f32>,  // [d_model] = 8
    pub w_q: Vec<f32>,              // [d_model, d_model] = 64
    pub w_k: Vec<f32>,              // [d_model, d_model] = 64
    pub w_v: Vec<f32>,              // [d_model, d_model] = 64
    pub w_o: Vec<f32>,              // [d_model, d_model] = 64
    pub ffn_norm_gamma: Vec<f32>,   // [d_model] = 8
    pub w_1: Vec<f32>,              // [d_ff, d_model] = 6 * 8 = 48
    pub w_2: Vec<f32>,              // [d_model, d_ff] = 8 * 6 = 48
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
        for slice in slices {
            for &val in slice {
                sum_sq += val * val;
            }
        }
        sum_sq.sqrt()
    }

    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let norm = self.l2_norm();
        if norm > max_norm && norm > 1e-12 {
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
            for slice in slices {
                for val in slice.iter_mut() {
                    *val *= scale;
                }
            }
        }
        norm
    }
}

/// Pre-allocated workspace for analytical forward-backward unrolling over sequence.
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
    pub thetas: Vec<f32>,       // [max_seq_len, n_heads, 4]
    pub k_rot: Vec<f32>,        // [max_seq_len, n_heads, head_dim]

    // Pure Orthogonal Projector state history
    pub v_hat: Vec<f32>,        // [max_seq_len, n_heads, head_dim]
    pub e_t: Vec<f32>,          // [max_seq_len, n_heads, head_dim]
    pub m: Vec<f32>,            // [max_seq_len, n_heads, 16]

    // Undistorted MUSIC noise projector and retrieval
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

    // 2. Pre-Attention RMSNorm & Scale: a = RMSNorm(x_0) * gamma_attn
    for t in 0..seq_len {
        let mut sum_sq = 0.0f32;
        for c in 0..d {
            let val = ws.x_0[t * d + c];
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + eps_norm).sqrt();
        ws.rms_attn[t] = rms;
        let inv_rms = 1.0 / rms;
        for c in 0..d {
            let norm_val = ws.x_0[t * d + c] * inv_rms;
            ws.x_0_norm[t * d + c] = norm_val;
            ws.a[t * d + c] = norm_val * layer.attn_norm_gamma[c];
        }
    }

    // 3. QKV Projections: q_raw = W_q * a, k_raw = W_k * a, v_raw = W_v * a
    for t in 0..seq_len {
        let a_tok = &ws.a[t * d..(t + 1) * d];
        for i in 0..d {
            let row_off = i * d;
            let mut sum_q = 0.0f32;
            let mut sum_k = 0.0f32;
            let mut sum_v = 0.0f32;
            for j in 0..d {
                let aj = a_tok[j];
                sum_q += layer.attn.w_q[row_off + j] * aj;
                sum_k += layer.attn.w_k[row_off + j] * aj;
                sum_v += layer.attn.w_v[row_off + j] * aj;
            }
            ws.q_raw[t * d + i] = sum_q;
            ws.k_raw[t * d + i] = sum_k;
            ws.v_raw[t * d + i] = sum_v;
        }
    }

    // 4. Per-head SRX Recurrence with Pure Orthogonal Projector and Clean MUSIC
    let span_theta = 4;
    let span_m = head_dim * head_dim;

    for t in 0..seq_len {
        for h in 0..n_heads {
            let h_off = h * head_dim;
            let q_raw_h = &ws.q_raw[t * d + h_off..t * d + h_off + head_dim];
            let k_raw_h = &ws.k_raw[t * d + h_off..t * d + h_off + head_dim];
            let v_raw_h = &ws.v_raw[t * d + h_off..t * d + h_off + head_dim];

            let q_norm_off = (t * n_heads + h) * head_dim;
            let k_norm_off = (t * n_heads + h) * head_dim;
            let thetas_off = (t * n_heads + h) * span_theta;
            let m_off = (t * n_heads + h) * span_m;
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

            // b) Dynamic Phase Coupling:
            // \theta_t = \theta_{t-1} + \alpha * (k_norm \odot v_raw_h)
            let prev_th_off = (t.saturating_sub(1) * n_heads + h) * span_theta;
            for i in 0..4 {
                let prev_th = if t > 0 { ws.thetas[prev_th_off + i] } else { 0.0 };
                let k_val = ws.k_norm[k_norm_off + i];
                let v_val = v_raw_h[i];
                ws.thetas[thetas_off + i] = prev_th + SRX_ALPHA * (k_val * v_val);
            }

            let curr_thetas: &[f32; 4] = ws.thetas[thetas_off..thetas_off + 4].try_into().unwrap();
            let k_norm_arr: &[f32; 4] = ws.k_norm[k_norm_off..k_norm_off + 4].try_into().unwrap();

            // c) k_rot = U(Theta_t) * k_norm (Monarch Butterfly Unitary)
            let mut k_rot_arr = [0.0f32; 4];
            apply_butterfly_4(k_norm_arr, curr_thetas, false, &mut k_rot_arr);
            ws.k_rot[k_norm_off..k_norm_off + 4].copy_from_slice(&k_rot_arr);

            // d) Pure Orthogonal Projector Associative Memory Update:
            // v_hat = M_{t-1}^T * k_rot
            let pm_off = (t.saturating_sub(1) * n_heads + h) * span_m;
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

            // M_t = M_{t-1} + k_rot * e_t^T
            for row in 0..4 {
                let kr = k_rot_arr[row];
                for col in 0..4 {
                    let idx = row * 4 + col;
                    let p_val = if t > 0 { ws.m[pm_off + idx] } else { 0.0 };
                    let e_val = ws.e_t[e_off + col];
                    ws.m[m_off + idx] = p_val + kr * e_val;
                }
            }

            // e) Undistorted MUSIC Noise Subspace Projection:
            // q_inv = U^\dagger(Theta_t) * q_norm
            let q_norm_arr: &[f32; 4] = ws.q_norm[q_norm_off..q_norm_off + 4].try_into().unwrap();
            let mut q_inv_arr = [0.0f32; 4];
            apply_butterfly_4(q_norm_arr, curr_thetas, true, &mut q_inv_arr);
            ws.q_inv[q_norm_off..q_norm_off + 4].copy_from_slice(&q_inv_arr);

            // Signal rank r = 2; noise energy = ||q_inv[2..4]||^2
            let noise_energy = q_inv_arr[2] * q_inv_arr[2] + q_inv_arr[3] * q_inv_arr[3];
            ws.noise_energy[t * n_heads + h] = noise_energy;

            // Dirac-like resonant gain with hard clipping at SRX_W_MAX (15.0)
            let w_val = 1.0 / (noise_energy + eps_srx);
            let w_clamped = w_val.min(SRX_W_MAX);
            ws.w[t * n_heads + h] = w_val;
            ws.w_clamped[t * n_heads + h] = w_clamped;

            // f) Memory retrieval:
            // q_rot = U(Theta_t) * q_norm
            let mut q_rot_arr = [0.0f32; 4];
            apply_butterfly_4(q_norm_arr, curr_thetas, false, &mut q_rot_arr);
            ws.q_rot[q_norm_off..q_norm_off + 4].copy_from_slice(&q_rot_arr);

            // y_ret = M_t^T * q_rot
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
        }
    }

    // 6. First Residual: x_mid = x_0 + attn_out
    for i in 0..seq_len * d {
        ws.x_mid[i] = ws.x_0[i] + ws.attn_out[i];
    }

    // 7. Pre-FFN RMSNorm & Scale: b = RMSNorm(x_mid) * gamma_ffn
    for t in 0..seq_len {
        let mut sum_sq = 0.0f32;
        for c in 0..d {
            let val = ws.x_mid[t * d + c];
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + eps_norm).sqrt();
        ws.rms_ffn[t] = rms;
        let inv_rms = 1.0 / rms;
        for c in 0..d {
            let norm_val = ws.x_mid[t * d + c] * inv_rms;
            ws.x_mid_norm[t * d + c] = norm_val;
            ws.b[t * d + c] = norm_val * layer.ffn_norm_gamma[c];
        }
    }

    // 8. FFN: z = W_1 * b, h = ReLU(z), f = W_2 * h
    for t in 0..seq_len {
        let b_tok = &ws.b[t * d..(t + 1) * d];
        for i in 0..d_ff {
            let row_off = i * d;
            let mut sum_z = 0.0f32;
            for j in 0..d {
                sum_z += layer.mlp.w_1[row_off + j] * b_tok[j];
            }
            ws.z[t * d_ff + i] = sum_z;
            ws.h[t * d_ff + i] = sum_z.max(0.0);
        }

        let h_tok = &ws.h[t * d_ff..(t + 1) * d_ff];
        for i in 0..d {
            let row_off = i * d_ff;
            let mut sum_f = 0.0f32;
            for j in 0..d_ff {
                sum_f += layer.mlp.w_2[row_off + j] * h_tok[j];
            }
            ws.f[t * d + i] = sum_f;
        }
    }

    // 9. Second Residual: x_1 = x_mid + f
    for i in 0..seq_len * d {
        ws.x_1[i] = ws.x_mid[i] + ws.f[i];
    }

    // 10. Final RMSNorm: u = RMSNorm(x_1) * gamma_final
    for t in 0..seq_len {
        let mut sum_sq = 0.0f32;
        for c in 0..d {
            let val = ws.x_1[t * d + c];
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + eps_norm).sqrt();
        ws.rms_final[t] = rms;
        let inv_rms = 1.0 / rms;
        for c in 0..d {
            let norm_val = ws.x_1[t * d + c] * inv_rms;
            ws.x_1_norm[t * d + c] = norm_val;
            ws.u[t * d + c] = norm_val * model.final_norm_gamma[c];
        }
    }

    // 11. Tied LM Head & Softmax Cross-Entropy Loss
    let mut total_loss = 0.0f32;
    let n_targets = seq_len - 1;

    for t in 0..seq_len {
        let u_tok = &ws.u[t * d..(t + 1) * d];
        for tok in 0..v {
            let emb_off = tok * d;
            let mut logit = 0.0f32;
            for c in 0..d {
                logit += u_tok[c] * model.token_embeddings[emb_off + c];
            }
            ws.logits[t * v + tok] = logit;
        }

        ws.probs[t * v..(t + 1) * v].copy_from_slice(&ws.logits[t * v..(t + 1) * v]);
        softmax(&mut ws.probs[t * v..(t + 1) * v]);

        if t < n_targets {
            let target_tok = tokens[t + 1];
            let p = ws.probs[t * v + target_tok].max(1e-12);
            total_loss -= p.ln();
        }
    }

    total_loss / n_targets as f32
}

/// Exact Analytical Reversible BPTT backward pass for SRXFORMER v05.
pub fn backward_loss(
    model: &SrxTransformer,
    tokens: &[usize],
    ws: &mut SrxTrainWorkspace,
    grad: &mut SrxGrad,
    _eps_srx: f32,
) {
    let seq_len = tokens.len();
    assert!(seq_len >= 2);
    let n_targets = seq_len - 1;
    let scale = 1.0 / n_targets as f32;

    let d = ws.d_model;
    let d_ff = ws.d_ff;
    let v = ws.vocab_size;
    let n_heads = ws.n_heads;
    let head_dim = ws.head_dim;
    let _eps_norm = model.config.eps;

    let layer = &model.layers[0];

    // Reset workspace backward buffers
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

    // 1. Cross-entropy gradient on logits
    for t in 0..n_targets {
        let target_tok = tokens[t + 1];
        let logits_off = t * v;
        for tok in 0..v {
            let p = ws.probs[logits_off + tok];
            let grad_val = if tok == target_tok {
                (p - 1.0) * scale
            } else {
                p * scale
            };
            ws.d_logits[logits_off + tok] = grad_val;
        }
    }

    // 2. Backward through Tied LM Head: logits[t, tok] = u[t] dot E[tok]
    for t in 0..n_targets {
        let u_tok = &ws.u[t * d..(t + 1) * d];
        for tok in 0..v {
            let dl = ws.d_logits[t * v + tok];
            if dl == 0.0 {
                continue;
            }
            let emb_off = tok * d;
            for c in 0..d {
                grad.token_embeddings[emb_off + c] += dl * u_tok[c];
                ws.d_u[t * d + c] += dl * model.token_embeddings[emb_off + c];
            }
        }
    }

    // 3. Backward through Final RMSNorm: u = RMSNorm(x_1) * gamma_final
    for t in 0..seq_len {
        let rms = ws.rms_final[t];
        if rms <= 1e-12 {
            continue;
        }
        let inv_rms = 1.0 / rms;
        let mut dot = 0.0f32;
        for c in 0..d {
            let du = ws.d_u[t * d + c];
            let x1_norm = ws.x_1_norm[t * d + c];
            grad.final_norm_gamma[c] += du * x1_norm;

            let d_norm = du * model.final_norm_gamma[c];
            dot += d_norm * ws.x_1[t * d + c];
        }

        let mean_dot = dot / (d as f32 * rms * rms);
        for c in 0..d {
            let d_norm = ws.d_u[t * d + c] * model.final_norm_gamma[c];
            let dx1 = inv_rms * (d_norm - ws.x_1[t * d + c] * mean_dot);
            ws.d_x_1[t * d + c] = dx1;
            ws.d_x_mid[t * d + c] += dx1;
            ws.d_f[t * d + c] = dx1;
        }
    }

    // 4. Backward through FFN: f = W_2 * h, h = ReLU(z), z = W_1 * b
    for t in 0..seq_len {
        let df_tok = &ws.d_f[t * d..(t + 1) * d];
        let h_tok = &ws.h[t * d_ff..(t + 1) * d_ff];

        // Gradient for W_2 and adjoint for h
        for i in 0..d {
            let df_val = df_tok[i];
            let row_off = i * d_ff;
            for j in 0..d_ff {
                grad.w_2[row_off + j] += df_val * h_tok[j];
                ws.d_h[t * d_ff + j] += df_val * layer.mlp.w_2[row_off + j];
            }
        }

        // Backward through ReLU: z -> h
        for j in 0..d_ff {
            let z_val = ws.z[t * d_ff + j];
            let dh_val = ws.d_h[t * d_ff + j];
            ws.d_z[t * d_ff + j] = if z_val > 0.0 { dh_val } else { 0.0 };
        }

        // Gradient for W_1 and adjoint for b
        let dz_tok = &ws.d_z[t * d_ff..(t + 1) * d_ff];
        let b_tok = &ws.b[t * d..(t + 1) * d];
        for i in 0..d_ff {
            let dz_val = dz_tok[i];
            let row_off = i * d;
            for j in 0..d {
                grad.w_1[row_off + j] += dz_val * b_tok[j];
                ws.d_b[t * d + j] += dz_val * layer.mlp.w_1[row_off + j];
            }
        }

        // Backward through Pre-FFN RMSNorm: b = RMSNorm(x_mid) * gamma_ffn
        let rms = ws.rms_ffn[t];
        if rms <= 1e-12 {
            continue;
        }
        let inv_rms = 1.0 / rms;
        let mut dot = 0.0f32;
        for c in 0..d {
            let db = ws.d_b[t * d + c];
            let xmid_norm = ws.x_mid_norm[t * d + c];
            grad.ffn_norm_gamma[c] += db * xmid_norm;

            let d_norm = db * layer.ffn_norm_gamma[c];
            dot += d_norm * ws.x_mid[t * d + c];
        }

        let mean_dot = dot / (d as f32 * rms * rms);
        for c in 0..d {
            let d_norm = ws.d_b[t * d + c] * layer.ffn_norm_gamma[c];
            let dxmid = inv_rms * (d_norm - ws.x_mid[t * d + c] * mean_dot);
            ws.d_x_mid[t * d + c] += dxmid;
        }
    }

    // 5. Backward through First Residual: x_mid = x_0 + attn_out
    for i in 0..seq_len * d {
        let dxm = ws.d_x_mid[i];
        ws.d_x_0[i] += dxm;
        ws.d_attn_out[i] += dxm;
    }

    // 6. Backward through Output Projection: attn_out = y * W_o^T
    for t in 0..seq_len {
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

    // 7. Backward through SRX Attention Recurrence with Pure Orthogonal Projector and Clean MUSIC
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

            // 1. Adjoint from y_head
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

            // 2. Readout: y_ret = M_t^T * q_rot => d_q_rot = M_t * d_y_ret, d_M_read = q_rot * d_y_ret^T
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

            // 3. Backward through query Butterfly rotations:
            let q_norm_arr: &[f32; 4] = ws.q_norm[q_norm_off..q_norm_off + 4].try_into().unwrap();

            // Backward through q_rot = U(thetas) * q_norm (forward mode)
            let mut d_q_norm_rot = [0.0f32; 4];
            apply_butterfly_4_backward(
                q_norm_arr,
                curr_thetas,
                false,
                &d_q_rot,
                &mut d_q_norm_rot,
                &mut d_thetas_acc,
            );

            // Backward through q_inv = U^\dagger(thetas) * q_norm (inverse mode)
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
            for i in 0..4 {
                d_q_norm_total[i] = d_q_norm_rot[i] + d_q_norm_inv[i];
            }

            // Backward through L2-normalization of q
            let q_raw_h = &ws.q_raw[t * d + h_off..t * d + h_off + head_dim];
            let n_q = ws.norm_q[t * n_heads + h];
            l2_normalize_backward(
                q_raw_h,
                q_norm_arr,
                n_q,
                &d_q_norm_total,
                &mut ws.d_q_raw[t * d + h_off..t * d + h_off + head_dim],
            );

            // 4. Exact Orthogonal Projector Backward:
            // Forward:
            // v_hat[c] = sum_r k_rot[r] * M_{t-1}[r, c]
            // e_t[c] = v_raw[c] - v_hat[c]
            // M_t[r, c] = M_{t-1}[r, c] + k_rot[r] * e_t[c]
            let k_rot_arr: &[f32; 4] = ws.k_rot[k_norm_off..k_norm_off + 4].try_into().unwrap();
            let e_arr = &ws.e_t[e_off..e_off + 4];
            let pm_off = (t.saturating_sub(1) * n_heads + h) * span_m;

            // a) de_t[c] = sum_r k_rot[r] * dM_t[r, c]
            let mut d_e = [0.0f32; 4];
            for col in 0..4 {
                let mut sum = 0.0f32;
                for row in 0..4 {
                    sum += k_rot_arr[row] * d_m_total[row * 4 + col];
                }
                d_e[col] = sum;
            }

            // b) dv_raw[c] += de_t[c]
            for col in 0..4 {
                ws.d_v_raw[t * d + h_off + col] += d_e[col];
            }

            // c) dk_rot[r] += sum_c dM_t[r, c] * e_t[c] - sum_c de_t[c] * M_{t-1}[r, c]
            let mut d_k_rot = [0.0f32; 4];
            for row in 0..4 {
                let mut sum_me = 0.0f32;
                let mut sum_de_m = 0.0f32;
                for col in 0..4 {
                    sum_me += d_m_total[row * 4 + col] * e_arr[col];
                    let prev_m = if t > 0 { ws.m[pm_off + row * 4 + col] } else { 0.0 };
                    sum_de_m += d_e[col] * prev_m;
                }
                d_k_rot[row] = sum_me - sum_de_m;
            }

            // d) dM_{t-1}[r, c] = dM_t[r, c] - k_rot[r] * de_t[c]
            if t > 0 {
                for row in 0..4 {
                    let kr = k_rot_arr[row];
                    for col in 0..4 {
                        let idx = row * 4 + col;
                        d_m_acc[idx] = d_m_total[idx] - kr * d_e[col];
                    }
                }
            } else {
                d_m_acc.fill(0.0);
            }

            // 5. Backward through k_rot = U(thetas) * k_norm
            let mut d_k_norm_rot = [0.0f32; 4];
            let k_norm_arr: &[f32; 4] = ws.k_norm[k_norm_off..k_norm_off + 4].try_into().unwrap();
            apply_butterfly_4_backward(
                k_norm_arr,
                curr_thetas,
                false,
                &d_k_rot,
                &mut d_k_norm_rot,
                &mut d_thetas_acc,
            );

            // 6. Backward through Phase Dynamics:
            // Forward: thetas_t = thetas_{t-1} + alpha * (k_norm * v_raw)
            let d_theta_total = d_thetas_acc;
            let mut d_k_norm_total = [0.0f32; 4];
            for i in 0..4 {
                let d_th = d_theta_total[i];
                let k_val = ws.k_norm[k_norm_off + i];
                let v_val = ws.v_raw[t * d + h_off + i];
                d_k_norm_total[i] = d_k_norm_rot[i] + SRX_ALPHA * d_th * v_val;
                ws.d_v_raw[t * d + h_off + i] += SRX_ALPHA * d_th * k_val;
                d_thetas_acc[i] = if t > 0 { d_th } else { 0.0 };
            }

            // 7. Backward through L2-normalization of k
            let k_raw_h = &ws.k_raw[t * d + h_off..t * d + h_off + head_dim];
            let n_k = ws.norm_k[t * n_heads + h];
            l2_normalize_backward(
                k_raw_h,
                k_norm_arr,
                n_k,
                &d_k_norm_total,
                &mut ws.d_k_raw[t * d + h_off..t * d + h_off + head_dim],
            );
        }
    }

    // 8. Backward through QKV linear projections:
    // q_raw = W_q * a, k_raw = W_k * a, v_raw = W_v * a
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

                ws.d_a[t * d + j] += dq * layer.attn.w_q[row_off + j]
                    + dk * layer.attn.w_k[row_off + j]
                    + dv * layer.attn.w_v[row_off + j];
            }
        }
    }

    // 9. Backward through Pre-Attention RMSNorm: a = RMSNorm(x_0) * gamma_attn
    for t in 0..seq_len {
        let rms = ws.rms_attn[t];
        if rms <= 1e-12 {
            continue;
        }
        let inv_rms = 1.0 / rms;
        let mut dot = 0.0f32;
        for c in 0..d {
            let da = ws.d_a[t * d + c];
            let x0_norm = ws.x_0_norm[t * d + c];
            grad.attn_norm_gamma[c] += da * x0_norm;

            let d_norm = da * layer.attn_norm_gamma[c];
            dot += d_norm * ws.x_0[t * d + c];
        }

        let mean_dot = dot / (d as f32 * rms * rms);
        for c in 0..d {
            let d_norm = ws.d_a[t * d + c] * layer.attn_norm_gamma[c];
            let dx0 = inv_rms * (d_norm - ws.x_0[t * d + c] * mean_dot);
            ws.d_x_0[t * d + c] += dx0;
        }
    }

    // 10. Accumulate d_x_0 into token embeddings
    for t in 0..seq_len {
        let tok = tokens[t];
        let emb_off = tok * d;
        for c in 0..d {
            grad.token_embeddings[emb_off + c] += ws.d_x_0[t * d + c];
        }
    }
}

/// Splits a flat token stream into discrete sequence slices delimited by EOS.
pub fn split_into_eos_sequences(tokens: &[usize], max_seq_len: usize) -> Vec<Vec<usize>> {
    let mut seqs = Vec::new();
    let mut current = Vec::new();

    for &tok in tokens {
        current.push(tok);
        if tok == EOS_TOKEN_ID || current.len() >= max_seq_len {
            if current.len() >= 2 {
                seqs.push(current.clone());
            }
            current.clear();
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
        let config = TransformerConfig::lang_chinchilla();
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

        // Finite difference check on W_q with precision < 5e-3
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

            let diff = (ana_grad - num_grad).abs();
            assert!(
                diff < 5e-3,
                "W_q[{}] gradient mismatch: ana={}, num={}, diff={}",
                i,
                ana_grad,
                num_grad,
                diff
            );
        }

        // Finite difference check on W_1 with precision < 5e-3
        for i in 0..4 {
            let orig = model.layers[0].mlp.w_1[i];

            model.layers[0].mlp.w_1[i] = orig + eps;
            let loss_plus = forward_loss(&model, &tokens, &mut ws, 1.0);

            model.layers[0].mlp.w_1[i] = orig - eps;
            let loss_minus = forward_loss(&model, &tokens, &mut ws, 1.0);

            model.layers[0].mlp.w_1[i] = orig;

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
            let ana_grad = grad.w_1[i];

            let diff = (ana_grad - num_grad).abs();
            assert!(
                diff < 5e-3,
                "W_1[{}] gradient mismatch: ana={}, num={}, diff={}",
                i,
                ana_grad,
                num_grad,
                diff
            );
        }

        // Finite difference check on W_v with precision < 5e-3
        for i in 0..4 {
            let orig = model.layers[0].attn.w_v[i];

            model.layers[0].attn.w_v[i] = orig + eps;
            let loss_plus = forward_loss(&model, &tokens, &mut ws, 1.0);

            model.layers[0].attn.w_v[i] = orig - eps;
            let loss_minus = forward_loss(&model, &tokens, &mut ws, 1.0);

            model.layers[0].attn.w_v[i] = orig;

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
            let ana_grad = grad.w_v[i];

            let diff = (ana_grad - num_grad).abs();
            assert!(
                diff < 5e-3,
                "W_v[{}] gradient mismatch: ana={}, num={}, diff={}",
                i,
                ana_grad,
                num_grad,
                diff
            );
        }
    }
}
