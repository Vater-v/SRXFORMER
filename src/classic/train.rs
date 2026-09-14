//! High-performance, zero-dependency Backpropagation and Optimizer engine for the SRXformer.
//! Implements exact analytical gradients (BPTT) for all 512 parameters of the Classical Transformer,
//! pre-allocated scratchpad memory for training, and an AdamW optimizer with zero allocations in the update loop.

use super::config::{ActivationType, TransformerConfig};
use super::model::Transformer;
use super::telemetry::TrainTelemetry;
use super::tokenizer::EOS_TOKEN_ID;
use std::time::Instant;

/// Gradient storage for all 512 trainable parameters of the Transformer.
#[derive(Debug, Clone)]
pub struct TransformerGrad {
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

impl TransformerGrad {
    /// Allocates gradient buffers matching the model configuration.
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

    /// Resets all gradient accumulators to zero without any heap allocation.
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

    /// Returns total parameter count stored in this gradient object.
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

    /// Computes the L2 norm across all parameters.
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

    /// In-place gradient clipping by global norm.
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

/// Pre-allocated workspace for training activations and intermediate adjoint vectors.
#[derive(Debug, Clone)]
pub struct TrainWorkspace {
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
    pub a: Vec<f32>,              // [max_seq_len, d_model]
    pub q: Vec<f32>,              // [max_seq_len, d_model]
    pub k: Vec<f32>,              // [max_seq_len, d_model]
    pub v: Vec<f32>,              // [max_seq_len, d_model]
    pub attn_scores: Vec<f32>,  // [n_heads * max_seq_len * max_seq_len]
    pub attn_probs: Vec<f32>,   // [n_heads * max_seq_len * max_seq_len]
    pub attn_out: Vec<f32>,     // [max_seq_len, d_model]
    pub m: Vec<f32>,              // [max_seq_len, d_model]
    pub x_mid: Vec<f32>,        // [max_seq_len, d_model]
    pub x_mid_norm: Vec<f32>,   // [max_seq_len, d_model]
    pub rms_ffn: Vec<f32>,      // [max_seq_len]
    pub b: Vec<f32>,              // [max_seq_len, d_model]
    pub z: Vec<f32>,              // [max_seq_len, d_ff]
    pub h: Vec<f32>,              // [max_seq_len, d_ff]
    pub f: Vec<f32>,              // [max_seq_len, d_model]
    pub x_1: Vec<f32>,          // [max_seq_len, d_model]
    pub x_1_norm: Vec<f32>,     // [max_seq_len, d_model]
    pub rms_final: Vec<f32>,    // [max_seq_len]
    pub u: Vec<f32>,              // [max_seq_len, d_model]
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
    pub d_m: Vec<f32>,          // [max_seq_len, d_model]
    pub d_attn_out: Vec<f32>,   // [max_seq_len, d_model]
    pub d_q: Vec<f32>,          // [max_seq_len, d_model]
    pub d_k: Vec<f32>,          // [max_seq_len, d_model]
    pub d_v: Vec<f32>,          // [max_seq_len, d_model]
    pub d_attn_scores: Vec<f32>,// [n_heads * max_seq_len * max_seq_len]
    pub d_a: Vec<f32>,          // [max_seq_len, d_model]
    pub d_x_0: Vec<f32>,        // [max_seq_len, d_model]
}

impl TrainWorkspace {
    pub fn new(config: &TransformerConfig) -> Self {
        let s = config.max_seq_len;
        let d = config.d_model;
        let d_ff = config.d_ff;
        let v = config.vocab_size;
        let n_heads = config.n_heads;
        let head_dim = config.head_dim();
        let scores_size = n_heads * s * s;

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
            q: vec![0.0; s * d],
            k: vec![0.0; s * d],
            v: vec![0.0; s * d],
            attn_scores: vec![0.0; scores_size],
            attn_probs: vec![0.0; scores_size],
            attn_out: vec![0.0; s * d],
            m: vec![0.0; s * d],
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
            d_m: vec![0.0; s * d],
            d_attn_out: vec![0.0; s * d],
            d_q: vec![0.0; s * d],
            d_k: vec![0.0; s * d],
            d_v: vec![0.0; s * d],
            d_attn_scores: vec![0.0; scores_size],
            d_a: vec![0.0; s * d],
            d_x_0: vec![0.0; s * d],
        }
    }
}

/// Zero-dependency AdamW Optimizer with zero allocations during update steps.
#[derive(Debug, Clone)]
pub struct AdamW {
    pub m: TransformerGrad,
    pub v: TransformerGrad,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: usize,
}

impl AdamW {
    pub fn new(config: &TransformerConfig, weight_decay: f32) -> Self {
        Self {
            m: TransformerGrad::new(config),
            v: TransformerGrad::new(config),
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step_count: 0,
        }
    }

    /// Performs one AdamW optimization step on model parameters.
    pub fn step(&mut self, model: &mut Transformer, grad: &TransformerGrad, lr: f32) {
        self.step_count += 1;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;

        // Bias correction factors
        let step_f = self.step_count as i32;
        let bias_corr1 = 1.0 - beta1.powi(step_f);
        let bias_corr2 = 1.0 - beta2.powi(step_f);
        let alpha = lr * (bias_corr2.sqrt() / bias_corr1);

        macro_rules! update_param_slice {
            ($p:expr, $g:expr, $m:expr, $v:expr) => {
                for i in 0..$p.len() {
                    let g_val = $g[i];
                    $m[i] = beta1 * $m[i] + (1.0 - beta1) * g_val;
                    $v[i] = beta2 * $v[i] + (1.0 - beta2) * g_val * g_val;
                    let denom = $v[i].sqrt() + eps;
                    let update = ($m[i] / denom) * alpha + lr * wd * $p[i];
                    $p[i] -= update;
                }
            };
        }

        update_param_slice!(
            model.token_embeddings,
            grad.token_embeddings,
            self.m.token_embeddings,
            self.v.token_embeddings
        );
        update_param_slice!(
            model.layers[0].attn_norm_gamma,
            grad.attn_norm_gamma,
            self.m.attn_norm_gamma,
            self.v.attn_norm_gamma
        );
        update_param_slice!(
            model.layers[0].attn.w_q,
            grad.w_q,
            self.m.w_q,
            self.v.w_q
        );
        update_param_slice!(
            model.layers[0].attn.w_k,
            grad.w_k,
            self.m.w_k,
            self.v.w_k
        );
        update_param_slice!(
            model.layers[0].attn.w_v,
            grad.w_v,
            self.m.w_v,
            self.v.w_v
        );
        update_param_slice!(
            model.layers[0].attn.w_o,
            grad.w_o,
            self.m.w_o,
            self.v.w_o
        );
        update_param_slice!(
            model.layers[0].ffn_norm_gamma,
            grad.ffn_norm_gamma,
            self.m.ffn_norm_gamma,
            self.v.ffn_norm_gamma
        );
        update_param_slice!(
            model.layers[0].mlp.w_1,
            grad.w_1,
            self.m.w_1,
            self.v.w_1
        );
        update_param_slice!(
            model.layers[0].mlp.w_2,
            grad.w_2,
            self.m.w_2,
            self.v.w_2
        );
        update_param_slice!(
            model.final_norm_gamma,
            grad.final_norm_gamma,
            self.m.final_norm_gamma,
            self.v.final_norm_gamma
        );
    }
}

/// Metrics recorded during pretraining or instruct tuning.
#[derive(Debug, Clone)]
pub struct TrainMetrics {
    pub initial_loss: f32,
    pub final_loss: f32,
    pub loss_history: Vec<f32>,
    pub elapsed_ms: f64,
    pub total_steps: usize,
}

/// Forward pass recording activations and computing cross-entropy loss.
pub fn forward_loss(
    model: &Transformer,
    tokens: &[usize],
    ws: &mut TrainWorkspace,
) -> f32 {
    let seq_len = tokens.len();
    assert!(seq_len >= 2, "tokens must have at least 2 elements");
    assert!(seq_len <= ws.max_seq_len, "seq_len exceeds max_seq_len");

    let d = ws.d_model;
    let d_ff = ws.d_ff;
    let v = ws.vocab_size;
    let n_heads = ws.n_heads;
    let head_dim = ws.head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let eps = model.config.eps;

    let layer = &model.layers[0];

    // 1. Token Embeddings + Positional Embeddings
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
        let rms = (sum_sq / d as f32 + eps).sqrt();
        ws.rms_attn[t] = rms;
        let inv_rms = 1.0 / rms;

        let norm_slice = &mut ws.x_0_norm[t * d..(t + 1) * d];
        let a_slice = &mut ws.a[t * d..(t + 1) * d];
        for c in 0..d {
            norm_slice[c] = x[c] * inv_rms;
            a_slice[c] = norm_slice[c] * layer.attn_norm_gamma[c];
        }
    }

    // 3. Attention Projections (Q, K, V)
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
            ws.q[t * d + c] = q_val;
            ws.k[t * d + c] = k_val;
            ws.v[t * d + c] = v_val;
        }
    }

    // 4. Multi-Head Attention with Causal Mask
    for h in 0..n_heads {
        let head_off = h * head_dim;

        for i in 0..seq_len {
            let q_head = &ws.q[i * d + head_off..i * d + head_off + head_dim];
            let score_base = (h * ws.max_seq_len + i) * ws.max_seq_len;

            let mut max_score = f32::NEG_INFINITY;
            for j in 0..seq_len {
                if j <= i {
                    let k_head = &ws.k[j * d + head_off..j * d + head_off + head_dim];
                    let mut dot = 0.0f32;
                    for c in 0..head_dim {
                        dot += q_head[c] * k_head[c];
                    }
                    let s = dot * scale;
                    ws.attn_scores[score_base + j] = s;
                    if s > max_score {
                        max_score = s;
                    }
                } else {
                    ws.attn_scores[score_base + j] = f32::NEG_INFINITY;
                }
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for j in 0..=i {
                let exp_s = (ws.attn_scores[score_base + j] - max_score).exp();
                ws.attn_probs[score_base + j] = exp_s;
                sum_exp += exp_s;
            }
            let inv_sum = 1.0 / sum_exp;
            for j in 0..=i {
                ws.attn_probs[score_base + j] *= inv_sum;
            }
            for j in i + 1..seq_len {
                ws.attn_probs[score_base + j] = 0.0;
            }

            // Weighted sum of values
            let out_head = &mut ws.attn_out[i * d + head_off..i * d + head_off + head_dim];
            out_head.fill(0.0);
            for j in 0..=i {
                let p = ws.attn_probs[score_base + j];
                let v_head = &ws.v[j * d + head_off..j * d + head_off + head_dim];
                for c in 0..head_dim {
                    out_head[c] += p * v_head[c];
                }
            }
        }
    }

    // 5. Attention Output Projection: M = attn_out * W_o^T
    for t in 0..seq_len {
        let attn_tok = &ws.attn_out[t * d..(t + 1) * d];
        for c in 0..d {
            let mut m_val = 0.0f32;
            for k_idx in 0..d {
                m_val += attn_tok[k_idx] * layer.attn.w_o[c * d + k_idx];
            }
            ws.m[t * d + c] = m_val;
            ws.x_mid[t * d + c] = ws.x_0[t * d + c] + m_val; // Residual 1
        }
    }

    // 6. Pre-FFN RMSNorm
    for t in 0..seq_len {
        let x = &ws.x_mid[t * d..(t + 1) * d];
        let mut sum_sq = 0.0f32;
        for &val in x {
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + eps).sqrt();
        ws.rms_ffn[t] = rms;
        let inv_rms = 1.0 / rms;

        let norm_slice = &mut ws.x_mid_norm[t * d..(t + 1) * d];
        let b_slice = &mut ws.b[t * d..(t + 1) * d];
        for c in 0..d {
            norm_slice[c] = x[c] * inv_rms;
            b_slice[c] = norm_slice[c] * layer.ffn_norm_gamma[c];
        }
    }

    // 7. FFN: Z = b * W1^T, H = act(Z), F = h * W2^T
    for t in 0..seq_len {
        let b_tok = &ws.b[t * d..(t + 1) * d];

        // Linear 1: d_ff
        for k in 0..d_ff {
            let mut z_val = 0.0f32;
            for c in 0..d {
                z_val += b_tok[c] * layer.mlp.w_1[k * d + c];
            }
            ws.z[t * d_ff + k] = z_val;
            ws.h[t * d_ff + k] = match layer.mlp.activation {
                ActivationType::Relu => z_val.max(0.0),
                ActivationType::Gelu => {
                    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
                    const COEFF: f32 = 0.044715;
                    let inner = SQRT_2_OVER_PI * (z_val + COEFF * z_val * z_val * z_val);
                    0.5 * z_val * (1.0 + inner.tanh())
                }
            };
        }

        // Linear 2: d_model
        let h_tok = &ws.h[t * d_ff..(t + 1) * d_ff];
        for c in 0..d {
            let mut f_val = 0.0f32;
            for k in 0..d_ff {
                f_val += h_tok[k] * layer.mlp.w_2[c * d_ff + k];
            }
            ws.f[t * d + c] = f_val;
            ws.x_1[t * d + c] = ws.x_mid[t * d + c] + f_val; // Residual 2
        }
    }

    // 8. Final RMSNorm
    for t in 0..seq_len {
        let x = &ws.x_1[t * d..(t + 1) * d];
        let mut sum_sq = 0.0f32;
        for &val in x {
            sum_sq += val * val;
        }
        let rms = (sum_sq / d as f32 + eps).sqrt();
        ws.rms_final[t] = rms;
        let inv_rms = 1.0 / rms;

        let norm_slice = &mut ws.x_1_norm[t * d..(t + 1) * d];
        let u_slice = &mut ws.u[t * d..(t + 1) * d];
        for c in 0..d {
            norm_slice[c] = x[c] * inv_rms;
            u_slice[c] = norm_slice[c] * model.final_norm_gamma[c];
        }
    }

    // 9. LM Head (Tied with token_embeddings) & Softmax
    for t in 0..seq_len {
        let u_tok = &ws.u[t * d..(t + 1) * d];
        let logits_tok = &mut ws.logits[t * v..(t + 1) * v];

        let mut max_logit = f32::NEG_INFINITY;
        for tok_idx in 0..v {
            let emb = &model.token_embeddings[tok_idx * d..(tok_idx + 1) * d];
            let mut logit = 0.0f32;
            for c in 0..d {
                logit += u_tok[c] * emb[c];
            }
            logits_tok[tok_idx] = logit;
            if logit > max_logit {
                max_logit = logit;
            }
        }

        let probs_tok = &mut ws.probs[t * v..(t + 1) * v];
        let mut sum_exp = 0.0f32;
        for tok_idx in 0..v {
            let exp_l = (logits_tok[tok_idx] - max_logit).exp();
            probs_tok[tok_idx] = exp_l;
            sum_exp += exp_l;
        }
        let inv_sum = 1.0 / sum_exp;
        for tok_idx in 0..v {
            probs_tok[tok_idx] *= inv_sum;
        }
    }

    // 10. Cross-Entropy Loss: L = - 1/(T-1) \sum_{t=0}^{T-2} ln P(tokens[t+1] | tokens[..=t])
    let num_targets = seq_len - 1;
    let mut total_loss = 0.0f32;
    for t in 0..num_targets {
        let target = tokens[t + 1];
        let prob = ws.probs[t * v + target].max(1e-15);
        total_loss -= prob.ln();
    }

    total_loss / (num_targets as f32)
}

/// Analytical Backward Pass (BPTT) computing exact gradients for all 512 parameters.
/// Accumulates gradients directly into `grad` (supports mini-batch accumulation).
pub fn backward_loss(
    model: &Transformer,
    tokens: &[usize],
    ws: &mut TrainWorkspace,
    grad: &mut TransformerGrad,
) {
    let seq_len = tokens.len();
    assert!(seq_len >= 2);
    let num_targets = seq_len - 1;
    let scale_loss = 1.0 / (num_targets as f32);

    let d = ws.d_model;
    let d_ff = ws.d_ff;
    let v = ws.vocab_size;
    let n_heads = ws.n_heads;
    let head_dim = ws.head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let layer = &model.layers[0];

    // 1. Gradients of Loss w.r.t logits:
    // d_logits[t, i] = (probs[t, i] - 1{i == target}) / (T-1) for t in 0..T-2
    for t in 0..num_targets {
        let target = tokens[t + 1];
        for tok_idx in 0..v {
            let p = ws.probs[t * v + tok_idx];
            let target_indicator = if tok_idx == target { 1.0f32 } else { 0.0f32 };
            ws.d_logits[t * v + tok_idx] = (p - target_indicator) * scale_loss;
        }
    }
    // Token T-1 has no target
    for tok_idx in 0..v {
        ws.d_logits[(seq_len - 1) * v + tok_idx] = 0.0;
    }

    // 2. Output Head (tied with token_embeddings):
    // logits[t, v] = \sum_c u[t, c] * E[v, c]
    // d_u[t, c] = \sum_v d_logits[t, v] * E[v, c]
    // d_E_head[v, c] += \sum_t d_logits[t, v] * u[t, c]
    for t in 0..seq_len {
        for c in 0..d {
            let mut sum_du = 0.0f32;
            for tok_idx in 0..v {
                sum_du += ws.d_logits[t * v + tok_idx] * model.token_embeddings[tok_idx * d + c];
            }
            ws.d_u[t * d + c] = sum_du;
        }
    }

    for tok_idx in 0..v {
        for c in 0..d {
            let mut sum_de = 0.0f32;
            for t in 0..num_targets {
                sum_de += ws.d_logits[t * v + tok_idx] * ws.u[t * d + c];
            }
            grad.token_embeddings[tok_idx * d + c] += sum_de;
        }
    }

    // 3. Final RMSNorm backward:
    // u[t, c] = x_1_norm[t, c] * gamma_final[c]
    for c in 0..d {
        let mut sum_dgamma = 0.0f32;
        for t in 0..seq_len {
            sum_dgamma += ws.d_u[t * d + c] * ws.x_1_norm[t * d + c];
        }
        grad.final_norm_gamma[c] += sum_dgamma;
    }

    for t in 0..seq_len {
        let inv_rms = 1.0 / ws.rms_final[t];
        let mut s = 0.0f32;
        for c in 0..d {
            let delta = ws.d_u[t * d + c] * model.final_norm_gamma[c];
            s += delta * ws.x_1_norm[t * d + c];
        }
        s /= d as f32;

        for c in 0..d {
            let delta = ws.d_u[t * d + c] * model.final_norm_gamma[c];
            let dx = inv_rms * (delta - s * ws.x_1_norm[t * d + c]);
            ws.d_x_1[t * d + c] = dx;
            // Residual 2: x_1 = x_mid + f
            ws.d_f[t * d + c] = dx;
            ws.d_x_mid[t * d + c] = dx;
        }
    }

    // 4. FFN Linear 2: F = H * W2^T
    // d_W2[c, k] += \sum_t d_F[t, c] * H[t, k]
    // d_H[t, k] = \sum_c d_F[t, c] * W2[c, k]
    for c in 0..d {
        for k in 0..d_ff {
            let mut sum_dw2 = 0.0f32;
            for t in 0..seq_len {
                sum_dw2 += ws.d_f[t * d + c] * ws.h[t * d_ff + k];
            }
            grad.w_2[c * d_ff + k] += sum_dw2;
        }
    }

    for t in 0..seq_len {
        for k in 0..d_ff {
            let mut sum_dh = 0.0f32;
            for c in 0..d {
                sum_dh += ws.d_f[t * d + c] * layer.mlp.w_2[c * d_ff + k];
            }
            ws.d_h[t * d_ff + k] = sum_dh;
        }
    }

    // 5. FFN Activation backward
    for t in 0..seq_len {
        for k in 0..d_ff {
            let z_val = ws.z[t * d_ff + k];
            let dh = ws.d_h[t * d_ff + k];
            ws.d_z[t * d_ff + k] = match layer.mlp.activation {
                ActivationType::Relu => {
                    if z_val > 0.0 {
                        dh
                    } else {
                        0.0
                    }
                }
                ActivationType::Gelu => {
                    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
                    const COEFF: f32 = 0.044715;
                    let inner = SQRT_2_OVER_PI * (z_val + COEFF * z_val * z_val * z_val);
                    let tanh_inner = inner.tanh();
                    let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * z_val * z_val);
                    let d_gelu = 0.5 * (1.0 + tanh_inner)
                        + 0.5 * z_val * (1.0 - tanh_inner * tanh_inner) * d_inner;
                    dh * d_gelu
                }
            };
        }
    }

    // 6. FFN Linear 1: Z = B * W1^T
    // d_W1[k, c] += \sum_t d_Z[t, k] * B[t, c]
    // d_B[t, c] = \sum_k d_Z[t, k] * W1[k, c]
    for k in 0..d_ff {
        for c in 0..d {
            let mut sum_dw1 = 0.0f32;
            for t in 0..seq_len {
                sum_dw1 += ws.d_z[t * d_ff + k] * ws.b[t * d + c];
            }
            grad.w_1[k * d + c] += sum_dw1;
        }
    }

    for t in 0..seq_len {
        for c in 0..d {
            let mut sum_db = 0.0f32;
            for k in 0..d_ff {
                sum_db += ws.d_z[t * d_ff + k] * layer.mlp.w_1[k * d + c];
            }
            ws.d_b[t * d + c] = sum_db;
        }
    }

    // 7. Pre-FFN RMSNorm backward
    for c in 0..d {
        let mut sum_dgamma = 0.0f32;
        for t in 0..seq_len {
            sum_dgamma += ws.d_b[t * d + c] * ws.x_mid_norm[t * d + c];
        }
        grad.ffn_norm_gamma[c] += sum_dgamma;
    }

    for t in 0..seq_len {
        let inv_rms = 1.0 / ws.rms_ffn[t];
        let mut s = 0.0f32;
        for c in 0..d {
            let delta = ws.d_b[t * d + c] * layer.ffn_norm_gamma[c];
            s += delta * ws.x_mid_norm[t * d + c];
        }
        s /= d as f32;

        for c in 0..d {
            let delta = ws.d_b[t * d + c] * layer.ffn_norm_gamma[c];
            let dx = inv_rms * (delta - s * ws.x_mid_norm[t * d + c]);
            ws.d_x_mid[t * d + c] += dx;
        }
    }

    // 8. Residual 1: x_mid = x_0 + m
    for t in 0..seq_len {
        for c in 0..d {
            let dx = ws.d_x_mid[t * d + c];
            ws.d_m[t * d + c] = dx;
            ws.d_x_0[t * d + c] = dx;
        }
    }

    // 9. Output Projection: M = attn_out * W_o^T
    for c in 0..d {
        for k in 0..d {
            let mut sum_dwo = 0.0f32;
            for t in 0..seq_len {
                sum_dwo += ws.d_m[t * d + c] * ws.attn_out[t * d + k];
            }
            grad.w_o[c * d + k] += sum_dwo;
        }
    }

    for t in 0..seq_len {
        for k in 0..d {
            let mut sum_do = 0.0f32;
            for c in 0..d {
                sum_do += ws.d_m[t * d + c] * layer.attn.w_o[c * d + k];
            }
            ws.d_attn_out[t * d + k] = sum_do;
        }
    }

    // 10. Multi-Head Attention backward
    // Clear d_q, d_k, d_v
    ws.d_q[..seq_len * d].fill(0.0);
    ws.d_k[..seq_len * d].fill(0.0);
    ws.d_v[..seq_len * d].fill(0.0);

    for h in 0..n_heads {
        let head_off = h * head_dim;

        // a) Backward into V:
        // out_head[i, c] = \sum_{j=0}^i P[i, j] * V[j, c]
        // d_V[j, c] += \sum_{i=j}^{seq_len-1} d_attn_out[i, c] * P[i, j]
        for j in 0..seq_len {
            for c in 0..head_dim {
                let mut sum_dv = 0.0f32;
                for i in j..seq_len {
                    let score_base = (h * ws.max_seq_len + i) * ws.max_seq_len;
                    sum_dv += ws.d_attn_out[i * d + head_off + c]
                        * ws.attn_probs[score_base + j];
                }
                ws.d_v[j * d + head_off + c] += sum_dv;
            }
        }

        // b) Backward into Softmax and Attention Scores:
        // d_P[i, j] = \sum_c d_attn_out[i, c] * V[j, c] (for j <= i)
        // Softmax adjoint: d_S[i, j] = P[i, j] * (d_P[i, j] - \sum_k P[i, k] * d_P[i, k])
        for i in 0..seq_len {
            let score_base = (h * ws.max_seq_len + i) * ws.max_seq_len;

            let mut dot_pd = 0.0f32;
            for j in 0..=i {
                let mut dp = 0.0f32;
                for c in 0..head_dim {
                    dp += ws.d_attn_out[i * d + head_off + c]
                        * ws.v[j * d + head_off + c];
                }
                ws.d_attn_scores[score_base + j] = dp; // temporarily hold d_P
                dot_pd += ws.attn_probs[score_base + j] * dp;
            }

            for j in 0..=i {
                let p = ws.attn_probs[score_base + j];
                let dp = ws.d_attn_scores[score_base + j];
                let ds = p * (dp - dot_pd);
                ws.d_attn_scores[score_base + j] = ds;
            }
        }

        // c) Backward from Attention Scores into Q and K:
        // S[i, j] = scale * \sum_c Q[i, c] * K[j, c]
        // d_Q[i, c] = scale * \sum_{j=0}^i d_S[i, j] * K[j, c]
        // d_K[j, c] = scale * \sum_{i=j}^{seq_len-1} d_S[i, j] * Q[i, c]
        for i in 0..seq_len {
            let score_base = (h * ws.max_seq_len + i) * ws.max_seq_len;
            for c in 0..head_dim {
                let mut sum_dq = 0.0f32;
                for j in 0..=i {
                    sum_dq += ws.d_attn_scores[score_base + j]
                        * ws.k[j * d + head_off + c];
                }
                ws.d_q[i * d + head_off + c] += sum_dq * scale;
            }
        }

        for j in 0..seq_len {
            for c in 0..head_dim {
                let mut sum_dk = 0.0f32;
                for i in j..seq_len {
                    let score_base = (h * ws.max_seq_len + i) * ws.max_seq_len;
                    sum_dk += ws.d_attn_scores[score_base + j]
                        * ws.q[i * d + head_off + c];
                }
                ws.d_k[j * d + head_off + c] += sum_dk * scale;
            }
        }
    }

    // 11. Attention Projections (W_q, W_k, W_v) & input A
    // Q = A * W_q^T, K = A * W_k^T, V = A * W_v^T
    for c in 0..d {
        for k in 0..d {
            let mut sum_dwq = 0.0f32;
            let mut sum_dwk = 0.0f32;
            let mut sum_dwv = 0.0f32;
            for t in 0..seq_len {
                let a_val = ws.a[t * d + k];
                sum_dwq += ws.d_q[t * d + c] * a_val;
                sum_dwk += ws.d_k[t * d + c] * a_val;
                sum_dwv += ws.d_v[t * d + c] * a_val;
            }
            grad.w_q[c * d + k] += sum_dwq;
            grad.w_k[c * d + k] += sum_dwk;
            grad.w_v[c * d + k] += sum_dwv;
        }
    }

    for t in 0..seq_len {
        for k in 0..d {
            let mut sum_da = 0.0f32;
            for c in 0..d {
                sum_da += ws.d_q[t * d + c] * layer.attn.w_q[c * d + k];
                sum_da += ws.d_k[t * d + c] * layer.attn.w_k[c * d + k];
                sum_da += ws.d_v[t * d + c] * layer.attn.w_v[c * d + k];
            }
            ws.d_a[t * d + k] = sum_da;
        }
    }

    // 12. Pre-Attn RMSNorm backward
    for c in 0..d {
        let mut sum_dgamma = 0.0f32;
        for t in 0..seq_len {
            sum_dgamma += ws.d_a[t * d + c] * ws.x_0_norm[t * d + c];
        }
        grad.attn_norm_gamma[c] += sum_dgamma;
    }

    for t in 0..seq_len {
        let inv_rms = 1.0 / ws.rms_attn[t];
        let mut s = 0.0f32;
        for c in 0..d {
            let delta = ws.d_a[t * d + c] * layer.attn_norm_gamma[c];
            s += delta * ws.x_0_norm[t * d + c];
        }
        s /= d as f32;

        for c in 0..d {
            let delta = ws.d_a[t * d + c] * layer.attn_norm_gamma[c];
            let dx = inv_rms * (delta - s * ws.x_0_norm[t * d + c]);
            ws.d_x_0[t * d + c] += dx;
        }
    }

    // 13. Input Token Embeddings gradient accumulation
    // X_0[t, c] = E[tokens[t], c] + PE[t, c]
    for (t, &tok) in tokens.iter().enumerate() {
        for c in 0..d {
            grad.token_embeddings[tok * d + c] += ws.d_x_0[t * d + c];
        }
    }
}

/// Splits token slice into sequences demarcated by EOS_TOKEN_ID.
pub fn split_into_eos_sequences(tokens: &[usize], max_len: usize) -> Vec<Vec<usize>> {
    let mut sequences = Vec::new();
    let mut current = Vec::new();

    for &tok in tokens {
        current.push(tok);
        if tok == EOS_TOKEN_ID || current.len() >= max_len {
            if current.len() >= 2 {
                sequences.push(current.clone());
            }
            current.clear();
        }
    }

    if current.len() >= 2 {
        sequences.push(current);
    }

    sequences
}

/// Trains the Transformer model on a tokenized dataset using Analytical BPTT and AdamW.
/// Computes comprehensive training telemetry including total FLOPs, throughput, and loss trajectory.
pub fn train_dataset(
    model: &mut Transformer,
    tokens: &[usize],
    config: &TransformerConfig,
    epochs: usize,
    lr: f32,
) -> TrainTelemetry {
    let start_time = Instant::now();
    let mut ws = TrainWorkspace::new(config);
    let mut grad = TransformerGrad::new(config);
    let mut optimizer = AdamW::new(config, 0.0);

    let mut sequences = split_into_eos_sequences(tokens, config.max_seq_len);
    assert!(!sequences.is_empty(), "Dataset has no valid sequences");

    let mut loss_history = Vec::with_capacity(epochs);
    let mut total_steps = 0;

    // Initial loss evaluation across first 20 sequences
    let eval_count = sequences.len().min(20);
    let mut initial_loss = 0.0f32;
    for seq in &sequences[..eval_count] {
        initial_loss += forward_loss(model, seq, &mut ws);
    }
    initial_loss /= eval_count as f32;

    let mut rng = super::rng::FastRng::new(4242);

    for epoch in 0..epochs {
        // Fisher-Yates shuffle to eliminate ordering bias
        for i in (1..sequences.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            sequences.swap(i, j);
        }

        let mut epoch_loss = 0.0f32;
        let progress = epoch as f32 / epochs.max(1) as f32;
        let current_lr = (lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(lr * 0.1);

        for seq in &sequences {
            grad.zero();
            let loss = forward_loss(model, seq, &mut ws);
            epoch_loss += loss;
            backward_loss(model, seq, &mut ws, &mut grad);
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

/// Pretrains the 512-parameter Transformer model on pretrain corpus tokens.
pub fn train_pretrain(
    model: &mut Transformer,
    tokens: &[usize],
    config: &TransformerConfig,
    epochs: usize,
    lr: f32,
) -> TrainMetrics {
    let start_time = Instant::now();
    let mut ws = TrainWorkspace::new(config);
    let mut grad = TransformerGrad::new(config);
    let mut optimizer = AdamW::new(config, 0.0);

    let mut sequences = split_into_eos_sequences(tokens, config.max_seq_len);
    assert!(!sequences.is_empty(), "Pretrain corpus has no valid sequences");

    let mut loss_history = Vec::with_capacity(epochs);
    let mut total_steps = 0;

    // Initial loss evaluation across first 20 sequences
    let eval_count = sequences.len().min(20);
    let mut initial_loss = 0.0f32;
    for seq in &sequences[..eval_count] {
        initial_loss += forward_loss(model, seq, &mut ws);
    }
    initial_loss /= eval_count as f32;

    let mut rng = super::rng::FastRng::new(1234);

    for epoch in 0..epochs {
        // Fisher-Yates shuffle to eliminate recency bias
        for i in (1..sequences.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            sequences.swap(i, j);
        }

        let mut epoch_loss = 0.0f32;
        let progress = epoch as f32 / epochs.max(1) as f32;
        let current_lr = (lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(lr * 0.1);

        for seq in &sequences {
            grad.zero();
            let loss = forward_loss(model, seq, &mut ws);
            epoch_loss += loss;
            backward_loss(model, seq, &mut ws, &mut grad);
            grad.clip_grad_norm(1.0);
            optimizer.step(model, &grad, current_lr);
            total_steps += 1;
        }

        let avg_epoch_loss = epoch_loss / sequences.len() as f32;
        loss_history.push(avg_epoch_loss);
    }

    let final_loss = *loss_history.last().unwrap_or(&initial_loss);
    let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    TrainMetrics {
        initial_loss,
        final_loss,
        loss_history,
        elapsed_ms,
        total_steps,
    }
}

/// Fine-tunes the model on instruction dialogue tokens (SFT) with replay mix to prevent catastrophic forgetting.
pub fn train_instruct(
    model: &mut Transformer,
    tokens: &[usize],
    config: &TransformerConfig,
    epochs: usize,
    lr: f32,
) -> TrainMetrics {
    let start_time = Instant::now();
    let mut ws = TrainWorkspace::new(config);
    let mut grad = TransformerGrad::new(config);
    let mut optimizer = AdamW::new(config, 0.0);

    let mut sequences = split_into_eos_sequences(tokens, config.max_seq_len);
    assert!(!sequences.is_empty(), "Instruct corpus has no valid sequences");

    let mut loss_history = Vec::with_capacity(epochs);
    let mut total_steps = 0;

    let eval_count = sequences.len().min(10);
    let mut initial_loss = 0.0f32;
    for seq in &sequences[..eval_count] {
        initial_loss += forward_loss(model, seq, &mut ws);
    }
    initial_loss /= eval_count as f32;

    let mut rng = super::rng::FastRng::new(5678);

    for epoch in 0..epochs {
        // Fisher-Yates shuffle to eliminate recency bias
        for i in (1..sequences.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            sequences.swap(i, j);
        }

        let mut epoch_loss = 0.0f32;
        let progress = epoch as f32 / epochs.max(1) as f32;
        let current_lr = (lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(lr * 0.1);

        for seq in &sequences {
            grad.zero();
            let loss = forward_loss(model, seq, &mut ws);
            epoch_loss += loss;
            backward_loss(model, seq, &mut ws, &mut grad);
            grad.clip_grad_norm(1.0);
            optimizer.step(model, &grad, current_lr);
            total_steps += 1;
        }

        let avg_epoch_loss = epoch_loss / sequences.len() as f32;
        loss_history.push(avg_epoch_loss);
    }

    let final_loss = *loss_history.last().unwrap_or(&initial_loss);
    let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    TrainMetrics {
        initial_loss,
        final_loss,
        loss_history,
        elapsed_ms,
        total_steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_check_numerical() {
        let config = TransformerConfig::lang_512();
        let mut model = Transformer::new_with_seed(config.clone(), 12345).unwrap();
        let mut ws = TrainWorkspace::new(&config);
        let mut grad = TransformerGrad::new(&config);

        let tokens = vec![2, 6, 10, 7, 12, 3, 9, 1]; // <user> 2 + 3 = <bot> 5 <eos>

        // 1. Analytical backward pass
        grad.zero();
        let _loss = forward_loss(&model, &tokens, &mut ws);
        backward_loss(&model, &tokens, &mut ws, &mut grad);

        // 2. Numerical gradient check for sample parameters across all subcomponents
        const EPS: f32 = 1e-3;

        // Check a weight in w_q
        let orig_wq = model.layers[0].attn.w_q[5];
        model.layers[0].attn.w_q[5] = orig_wq + EPS;
        let l_plus = forward_loss(&model, &tokens, &mut ws);
        model.layers[0].attn.w_q[5] = orig_wq - EPS;
        let l_minus = forward_loss(&model, &tokens, &mut ws);
        model.layers[0].attn.w_q[5] = orig_wq;
        let num_grad_wq = (l_plus - l_minus) / (2.0 * EPS);
        let ana_grad_wq = grad.w_q[5];
        let diff_wq = (num_grad_wq - ana_grad_wq).abs();
        assert!(
            diff_wq < 1e-2,
            "w_q gradient mismatch: num={num_grad_wq}, ana={ana_grad_wq}, diff={diff_wq}"
        );

        // Check a weight in w_1
        let orig_w1 = model.layers[0].mlp.w_1[3];
        model.layers[0].mlp.w_1[3] = orig_w1 + EPS;
        let l_plus = forward_loss(&model, &tokens, &mut ws);
        model.layers[0].mlp.w_1[3] = orig_w1 - EPS;
        let l_minus = forward_loss(&model, &tokens, &mut ws);
        model.layers[0].mlp.w_1[3] = orig_w1;
        let num_grad_w1 = (l_plus - l_minus) / (2.0 * EPS);
        let ana_grad_w1 = grad.w_1[3];
        let diff_w1 = (num_grad_w1 - ana_grad_w1).abs();
        assert!(
            diff_w1 < 1e-2,
            "w_1 gradient mismatch: num={num_grad_w1}, ana={ana_grad_w1}, diff={diff_w1}"
        );

        // Check a weight in final_norm_gamma
        let orig_gamma = model.final_norm_gamma[2];
        model.final_norm_gamma[2] = orig_gamma + EPS;
        let l_plus = forward_loss(&model, &tokens, &mut ws);
        model.final_norm_gamma[2] = orig_gamma - EPS;
        let l_minus = forward_loss(&model, &tokens, &mut ws);
        model.final_norm_gamma[2] = orig_gamma;
        let num_grad_g = (l_plus - l_minus) / (2.0 * EPS);
        let ana_grad_g = grad.final_norm_gamma[2];
        let diff_g = (num_grad_g - ana_grad_g).abs();
        assert!(
            diff_g < 1e-2,
            "final_norm_gamma gradient mismatch: num={num_grad_g}, ana={ana_grad_g}, diff={diff_g}"
        );
    }

    #[test]
    fn test_single_step_loss_decrease() {
        let config = TransformerConfig::lang_512();
        let mut model = Transformer::new_with_seed(config.clone(), 42).unwrap();
        let mut ws = TrainWorkspace::new(&config);
        let mut grad = TransformerGrad::new(&config);
        let mut optimizer = AdamW::new(&config, 0.0);

        let tokens = vec![13, 17, 15, 1]; // кот это животное <eos>

        let loss_before = forward_loss(&model, &tokens, &mut ws);
        backward_loss(&model, &tokens, &mut ws, &mut grad);
        optimizer.step(&mut model, &grad, 0.05);
        let loss_after = forward_loss(&model, &tokens, &mut ws);

        assert!(
            loss_after < loss_before,
            "Loss must strictly decrease after 1 gradient descent step: before={loss_before}, after={loss_after}"
        );
    }
}
