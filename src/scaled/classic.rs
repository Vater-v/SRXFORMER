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
}
