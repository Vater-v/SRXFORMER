//! # Scaled Classical Transformer Baseline
//!
//! Provides strict 1:1 architectural parity with Scaled SRX:
//! - Multi-Head Attention with causal masking and KV-Cache.
//! - Same parameter dimensions ($H, d_{\text{model}}, d_{\text{ff}}, V$).
//! - Demonstrates $O(N)$ KV-cache expansion and Memory Wall crossover.

use crate::scaled::config::ScaledConfig;
use crate::scaled::nn::{Module, ScaledFFN, ScaledLinear, ScaledRMSNorm};

/// Dynamic KV-Cache for classical autoregressive inference.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaledClassicKvCache {
    pub d_model: usize,
    pub max_seq_len: usize,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
}

impl ScaledClassicKvCache {
    pub fn new(d_model: usize, max_seq_len: usize) -> Self {
        Self {
            d_model,
            max_seq_len,
            k: vec![0.0f32; max_seq_len * d_model],
            v: vec![0.0f32; max_seq_len * d_model],
            len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Appends key and value vectors for the current step.
    pub fn append(&mut self, k_curr: &[f32], v_curr: &[f32]) {
        assert!(self.len < self.max_seq_len, "KV-Cache capacity exceeded");
        let off = self.len * self.d_model;
        self.k[off..off + self.d_model].copy_from_slice(k_curr);
        self.v[off..off + self.d_model].copy_from_slice(v_curr);
        self.len += 1;
    }

    /// Memory footprint in bytes: $2 \times N \times d_{\text{model}} \times 4$.
    pub fn memory_bytes(&self) -> usize {
        2 * self.len * self.d_model * std::mem::size_of::<f32>()
    }
}

/// Classical Multi-Head Attention with Softmax and causal masking.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaledClassicAttention {
    pub n_heads: usize,
    pub d_head: usize,
    pub d_model: usize,
    pub w_q: ScaledLinear,
    pub w_k: ScaledLinear,
    pub w_v: ScaledLinear,
    pub w_o: ScaledLinear,
}

impl ScaledClassicAttention {
    pub fn new(n_heads: usize) -> Self {
        let d_head = 4;
        let d_model = n_heads * d_head;
        Self {
            n_heads,
            d_head,
            d_model,
            w_q: ScaledLinear::new(d_model, d_model),
            w_k: ScaledLinear::new(d_model, d_model),
            w_v: ScaledLinear::new(d_model, d_model),
            w_o: ScaledLinear::new(d_model, d_model),
        }
    }

    /// Single step using KV-cache.
    pub fn step(
        &self,
        x: &[f32],
        cache: &mut ScaledClassicKvCache,
        y_out: &mut [f32],
    ) {
        let dm = self.d_model;
        let h = self.n_heads;
        let dh = self.d_head;
        let scale = 1.0 / (dh as f32).sqrt();

        let mut q = vec![0.0f32; dm];
        let mut k = vec![0.0f32; dm];
        let mut v = vec![0.0f32; dm];

        self.w_q.forward(x, &mut q);
        self.w_k.forward(x, &mut k);
        self.w_v.forward(x, &mut v);

        // Store into KV cache
        cache.append(&k, &v);
        let seq_len = cache.len;

        let mut head_outputs = vec![0.0f32; dm];

        // Softmax attention per head
        for head_idx in 0..h {
            let h_off = head_idx * dh;
            let q_head = &q[h_off..h_off + dh];

            // Compute scores against all cached tokens
            let mut scores = vec![0.0f32; seq_len];
            let mut max_score = f32::NEG_INFINITY;

            for t in 0..seq_len {
                let k_t = &cache.k[t * dm + h_off..t * dm + h_off + dh];
                let mut dot = 0.0f32;
                for i in 0..dh {
                    dot += q_head[i] * k_t[i];
                }
                let s = dot * scale;
                scores[t] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            // Exponentiate & Softmax
            let mut sum_exp = 0.0f32;
            for t in 0..seq_len {
                scores[t] = (scores[t] - max_score).exp();
                sum_exp += scores[t];
            }
            let inv_sum = 1.0 / sum_exp.max(1e-12);
            for t in 0..seq_len {
                scores[t] *= inv_sum;
            }

            // Weighted sum of values
            for t in 0..seq_len {
                let v_t = &cache.v[t * dm + h_off..t * dm + h_off + dh];
                let alpha = scores[t];
                for i in 0..dh {
                    head_outputs[h_off + i] += alpha * v_t[i];
                }
            }
        }

        self.w_o.forward(&head_outputs, y_out);
    }
}

impl Module for ScaledClassicAttention {
    fn in_dim(&self) -> usize { self.d_model }
    fn out_dim(&self) -> usize { self.d_model }
    fn param_count(&self) -> usize {
        self.w_q.param_count() + self.w_k.param_count() + self.w_v.param_count() + self.w_o.param_count()
    }
    fn zero_grad(&mut self) {
        self.w_q.zero_grad();
        self.w_k.zero_grad();
        self.w_v.zero_grad();
        self.w_o.zero_grad();
    }
}

/// Scaled Classical Transformer Model.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaledClassicTransformer {
    pub config: ScaledConfig,
    pub embed: ScaledLinear,
    pub norm_attn: ScaledRMSNorm,
    pub attn: ScaledClassicAttention,
    pub norm_ffn: ScaledRMSNorm,
    pub ffn: ScaledFFN,
    pub norm_final: ScaledRMSNorm,
    pub lm_head: ScaledLinear,
}

impl ScaledClassicTransformer {
    pub fn new(config: ScaledConfig) -> Self {
        let dm = config.d_model;
        let v = config.vocab_size;
        let dff = config.d_ff;

        Self {
            config: config.clone(),
            embed: ScaledLinear::new(v, dm),
            norm_attn: ScaledRMSNorm::new(dm, config.rms_eps),
            attn: ScaledClassicAttention::new(config.n_heads),
            norm_ffn: ScaledRMSNorm::new(dm, config.rms_eps),
            ffn: ScaledFFN::new(dm, dff),
            norm_final: ScaledRMSNorm::new(dm, config.rms_eps),
            lm_head: ScaledLinear::new(dm, v),
        }
    }

    /// Single step using KV-cache.
    pub fn step(
        &self,
        x_emb: &[f32],
        cache: &mut ScaledClassicKvCache,
        logits_out: &mut [f32],
    ) {
        let dm = self.config.d_model;
        let mut norm_buf = vec![0.0f32; dm];
        let mut attn_out = vec![0.0f32; dm];

        self.norm_attn.forward(x_emb, &mut norm_buf);
        self.attn.step(&norm_buf, cache, &mut attn_out);

        // Residual 1
        for i in 0..dm {
            attn_out[i] += x_emb[i];
        }

        // FFN block
        self.norm_ffn.forward(&attn_out, &mut norm_buf);
        let mut hidden_act = vec![0.0f32; self.config.d_ff];
        let mut ffn_out = vec![0.0f32; dm];
        self.ffn.forward(&norm_buf, &mut hidden_act, &mut ffn_out);

        // Residual 2
        for i in 0..dm {
            ffn_out[i] += attn_out[i];
        }

        // Final norm & LM Head
        self.norm_final.forward(&ffn_out, &mut norm_buf);
        self.lm_head.forward(&norm_buf, logits_out);
    }

    /// Total parameter count (identical to ScaledSrxTransformer).
    pub fn param_count(&self) -> usize {
        self.embed.param_count()
            + self.norm_attn.param_count()
            + self.attn.param_count()
            + self.norm_ffn.param_count()
            + self.ffn.param_count()
            + self.norm_final.param_count()
            + self.lm_head.param_count()
    }

    /// Returns mutable slice references to all 11 parameter and gradient layers for AdamW.
    pub fn get_layers_mut(&mut self) -> [(&mut [f32], &mut [f32]); 11] {
        [
            (&mut self.embed.weight[..], &mut self.embed.grad[..]),
            (&mut self.norm_attn.weight[..], &mut self.norm_attn.grad[..]),
            (&mut self.attn.w_q.weight[..], &mut self.attn.w_q.grad[..]),
            (&mut self.attn.w_k.weight[..], &mut self.attn.w_k.grad[..]),
            (&mut self.attn.w_v.weight[..], &mut self.attn.w_v.grad[..]),
            (&mut self.attn.w_o.weight[..], &mut self.attn.w_o.grad[..]),
            (&mut self.norm_ffn.weight[..], &mut self.norm_ffn.grad[..]),
            (&mut self.ffn.w1.weight[..], &mut self.ffn.w1.grad[..]),
            (&mut self.ffn.w2.weight[..], &mut self.ffn.w2.grad[..]),
            (&mut self.norm_final.weight[..], &mut self.norm_final.grad[..]),
            (&mut self.lm_head.weight[..], &mut self.lm_head.grad[..]),
        ]
    }

    /// Autoregressively generates continuation bytes given a text prompt.
    pub fn generate_bytes(&self, prompt: &str, max_bytes: usize) -> Vec<u8> {
        let dm = self.config.d_model;
        let mut cache = ScaledClassicKvCache::new(dm, self.config.max_seq_len);
        let prompt_bytes = prompt.as_bytes();
        let mut logits = vec![0.0f32; 256];

        for &b in prompt_bytes {
            let in_b = b as usize;
            let in_emb = &self.embed.weight[in_b * dm..(in_b + 1) * dm];
            self.step(in_emb, &mut cache, &mut logits);
        }

        let mut generated = Vec::with_capacity(max_bytes);
        for _ in 0..max_bytes {
            let mut best_byte = 0u8;
            let mut max_logit = f32::NEG_INFINITY;
            for b in 0..256 {
                if logits[b] > max_logit {
                    max_logit = logits[b];
                    best_byte = b as u8;
                }
            }
            if best_byte == 0 || best_byte == b'\n' {
                break;
            }
            generated.push(best_byte);
            let in_b = best_byte as usize;
            let in_emb = &self.embed.weight[in_b * dm..(in_b + 1) * dm];
            self.step(in_emb, &mut cache, &mut logits);
        }

        generated
    }

    /// Executes single forward and analytical backward step on a single token prediction.
    /// Accumulates parameter gradients into internal layers' `.grad`.
    /// Returns the cross-entropy loss.
    pub fn train_step(
        &mut self,
        in_b: usize,
        target_b: usize,
        cache: &mut ScaledClassicKvCache,
    ) -> f32 {
        let dm = self.config.d_model;
        let in_emb = self.embed.weight[in_b * dm..(in_b + 1) * dm].to_vec();

        // 1. Pre-Attn RMSNorm
        let mut norm_buf_1 = vec![0.0f32; dm];
        self.norm_attn.forward(&in_emb, &mut norm_buf_1);

        // 2. Attention step
        let h = self.attn.n_heads;
        let dh = self.attn.d_head;
        let scale = 1.0 / (dh as f32).sqrt();

        let mut q = vec![0.0f32; dm];
        let mut k = vec![0.0f32; dm];
        let mut v = vec![0.0f32; dm];
        self.attn.w_q.forward(&norm_buf_1, &mut q);
        self.attn.w_k.forward(&norm_buf_1, &mut k);
        self.attn.w_v.forward(&norm_buf_1, &mut v);

        cache.append(&k, &v);
        let seq_len = cache.len;

        let mut head_outputs = vec![0.0f32; dm];
        let mut all_scores = vec![0.0f32; h * seq_len];

        for head_idx in 0..h {
            let h_off = head_idx * dh;
            let q_head = &q[h_off..h_off + dh];

            let mut max_score = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let k_t = &cache.k[t * dm + h_off..t * dm + h_off + dh];
                let mut dot = 0.0f32;
                for i in 0..dh {
                    dot += q_head[i] * k_t[i];
                }
                let s = dot * scale;
                all_scores[head_idx * seq_len + t] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            let mut sum_exp = 0.0f32;
            for t in 0..seq_len {
                let exp_s = (all_scores[head_idx * seq_len + t] - max_score).exp();
                all_scores[head_idx * seq_len + t] = exp_s;
                sum_exp += exp_s;
            }
            let inv_sum = 1.0 / sum_exp.max(1e-12);
            for t in 0..seq_len {
                all_scores[head_idx * seq_len + t] *= inv_sum;
            }

            for t in 0..seq_len {
                let v_t = &cache.v[t * dm + h_off..t * dm + h_off + dh];
                let alpha = all_scores[head_idx * seq_len + t];
                for i in 0..dh {
                    head_outputs[h_off + i] += alpha * v_t[i];
                }
            }
        }

        let mut attn_out = vec![0.0f32; dm];
        self.attn.w_o.forward(&head_outputs, &mut attn_out);

        // 3. Residual 1
        let mut res1 = vec![0.0f32; dm];
        for i in 0..dm {
            res1[i] = in_emb[i] + attn_out[i];
        }

        // 4. Pre-FFN RMSNorm
        let mut norm_buf_2 = vec![0.0f32; dm];
        self.norm_ffn.forward(&res1, &mut norm_buf_2);

        // 5. FFN block
        let mut hidden_act = vec![0.0f32; self.config.d_ff];
        let mut ffn_out = vec![0.0f32; dm];
        self.ffn.forward(&norm_buf_2, &mut hidden_act, &mut ffn_out);

        // 6. Residual 2
        let mut res2 = vec![0.0f32; dm];
        for i in 0..dm {
            res2[i] = res1[i] + ffn_out[i];
        }

        // 7. Final RMSNorm
        let mut norm_buf_3 = vec![0.0f32; dm];
        self.norm_final.forward(&res2, &mut norm_buf_3);

        // 8. LM Head
        let mut logits = vec![0.0f32; 256];
        self.lm_head.forward(&norm_buf_3, &mut logits);

        // 9. Softmax & Cross-Entropy
        let mut max_l = f32::NEG_INFINITY;
        for &l in &logits {
            if l > max_l {
                max_l = l;
            }
        }
        let mut sum_exp = 0.0f32;
        let mut probs = [0.0f32; 256];
        for b in 0..256 {
            probs[b] = (logits[b] - max_l).exp();
            sum_exp += probs[b];
        }
        let inv_sum = 1.0 / sum_exp.max(1e-12);
        for b in 0..256 {
            probs[b] *= inv_sum;
        }

        let target_b = target_b.min(255);
        let loss = -probs[target_b].max(1e-12).ln();

        // 10. Gradient of cross-entropy
        let mut d_logits = [0.0f32; 256];
        for b in 0..256 {
            d_logits[b] = probs[b];
        }
        d_logits[target_b] -= 1.0;

        // 11. Backward through LM Head
        let mut d_norm_buf_3 = vec![0.0f32; dm];
        self.lm_head.backward(&norm_buf_3, &d_logits, &mut d_norm_buf_3);

        // 12. Backward through Final RMSNorm
        let mut d_res2 = vec![0.0f32; dm];
        self.norm_final.backward(&res2, &d_norm_buf_3, &mut d_res2);

        // 13. Residual 2 & FFN backward
        let mut d_res1 = d_res2.clone();
        let mut d_hidden = vec![0.0f32; self.config.d_ff];
        let mut d_norm_buf_2 = vec![0.0f32; dm];
        self.ffn.backward(&norm_buf_2, &hidden_act, &d_res2, &mut d_hidden, &mut d_norm_buf_2);

        let mut d_res1_from_ffn = vec![0.0f32; dm];
        self.norm_ffn.backward(&res1, &d_norm_buf_2, &mut d_res1_from_ffn);
        for i in 0..dm {
            d_res1[i] += d_res1_from_ffn[i];
        }

        // 14. Attention backward
        let mut d_head_outputs = vec![0.0f32; dm];
        self.attn.w_o.backward(&head_outputs, &d_res1, &mut d_head_outputs);

        let mut d_q = vec![0.0f32; dm];
        let mut d_k = vec![0.0f32; dm];
        let mut d_v = vec![0.0f32; dm];

        for head_idx in 0..h {
            let h_off = head_idx * dh;
            let alpha_curr = all_scores[head_idx * seq_len + seq_len - 1];
            for i in 0..dh {
                d_v[h_off + i] = alpha_curr * d_head_outputs[h_off + i];
                d_q[h_off + i] = scale * d_head_outputs[h_off + i];
                d_k[h_off + i] = scale * d_head_outputs[h_off + i];
            }
        }

        let mut d_norm_buf_1_q = vec![0.0f32; dm];
        let mut d_norm_buf_1_k = vec![0.0f32; dm];
        let mut d_norm_buf_1_v = vec![0.0f32; dm];
        self.attn.w_q.backward(&norm_buf_1, &d_q, &mut d_norm_buf_1_q);
        self.attn.w_k.backward(&norm_buf_1, &d_k, &mut d_norm_buf_1_k);
        self.attn.w_v.backward(&norm_buf_1, &d_v, &mut d_norm_buf_1_v);

        let mut d_norm_buf_1 = vec![0.0f32; dm];
        for i in 0..dm {
            d_norm_buf_1[i] = d_norm_buf_1_q[i] + d_norm_buf_1_k[i] + d_norm_buf_1_v[i];
        }

        // 15. Pre-Attn RMSNorm backward
        let mut d_x_emb = d_res1.clone();
        let mut d_x_emb_from_attn = vec![0.0f32; dm];
        self.norm_attn.backward(&in_emb, &d_norm_buf_1, &mut d_x_emb_from_attn);
        for i in 0..dm {
            d_x_emb[i] += d_x_emb_from_attn[i];
        }

        // 16. Embeddings backward
        for i in 0..dm {
            self.embed.grad[in_b * dm + i] += d_x_emb[i];
        }

        loss
    }
}
