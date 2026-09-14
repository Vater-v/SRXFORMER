use super::cache::{AttentionWorkspace, KvCache};
use super::config::TransformerConfig;
use super::ops::{dot_product, matmul, matvec, softmax, vector_add_scaled};
use super::rng::FastRng;

/// Multi-Head Attention weights for a single Transformer layer.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub w_q: Vec<f32>,            // [d_model, d_model]
    pub b_q: Option<Vec<f32>>,    // [d_model]
    pub w_k: Vec<f32>,            // [d_model, d_model]
    pub b_k: Option<Vec<f32>>,    // [d_model]
    pub w_v: Vec<f32>,            // [d_model, d_model]
    pub b_v: Option<Vec<f32>>,    // [d_model]
    pub w_o: Vec<f32>,            // [d_model, d_model]
    pub b_o: Option<Vec<f32>>,    // [d_model]

    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl MultiHeadAttention {
    /// Creates and initializes weights with Xavier normal initialization.
    pub fn new(config: &TransformerConfig, rng: &mut FastRng) -> Self {
        let d_model = config.d_model;
        let n_heads = config.n_heads;
        let head_dim = config.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Xavier standard deviation: sqrt(2 / (in + out)) = sqrt(2 / (2 * d_model)) = 1 / sqrt(d_model)
        let std = 1.0 / (d_model as f32).sqrt();

        let mut init_mat = || -> Vec<f32> {
            (0..d_model * d_model)
                .map(|_| rng.gen_normal(0.0, std))
                .collect()
        };

        let init_bias = || -> Option<Vec<f32>> {
            if config.use_bias {
                Some(vec![0.0; d_model])
            } else {
                None
            }
        };

        Self {
            w_q: init_mat(),
            b_q: init_bias(),
            w_k: init_mat(),
            b_k: init_bias(),
            w_v: init_mat(),
            b_v: init_bias(),
            w_o: init_mat(),
            b_o: init_bias(),
            d_model,
            n_heads,
            head_dim,
            scale,
        }
    }

    /// Full sequence multi-head attention with causal masking.
    /// Operates on pre-allocated buffers in `workspace`.
    /// `input`: slice of shape [seq_len, d_model]
    /// Writes output into `out` of shape [seq_len, d_model].
    pub fn forward_causal(
        &self,
        input: &[f32],
        seq_len: usize,
        workspace: &mut AttentionWorkspace,
        out: &mut [f32],
    ) {
        let d_model = self.d_model;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let scale = self.scale;

        // 1. Linear projections for Q, K, V
        matmul(
            &mut workspace.q[..seq_len * d_model],
            input,
            &self.w_q,
            self.b_q.as_deref(),
            seq_len,
            d_model,
            d_model,
        );
        matmul(
            &mut workspace.k[..seq_len * d_model],
            input,
            &self.w_k,
            self.b_k.as_deref(),
            seq_len,
            d_model,
            d_model,
        );
        matmul(
            &mut workspace.v[..seq_len * d_model],
            input,
            &self.w_v,
            self.b_v.as_deref(),
            seq_len,
            d_model,
            d_model,
        );

        // 2. Multi-Head Attention computation with Causal Mask
        for h in 0..n_heads {
            let head_offset = h * head_dim;

            for i in 0..seq_len {
                let q_row = &workspace.q[i * d_model + head_offset..i * d_model + head_offset + head_dim];

                let score_offset = (h * seq_len + i) * seq_len;
                let scores = &mut workspace.attn_scores[score_offset..score_offset + seq_len];

                for j in 0..seq_len {
                    if j > i {
                        scores[j] = f32::NEG_INFINITY;
                    } else {
                        let k_row = &workspace.k[j * d_model + head_offset..j * d_model + head_offset + head_dim];
                        scores[j] = dot_product(q_row, k_row) * scale;
                    }
                }

                softmax(scores);

                let out_head = &mut workspace.attn_out[i * d_model + head_offset..i * d_model + head_offset + head_dim];
                out_head.fill(0.0);

                for j in 0..=i {
                    let weight = scores[j];
                    let v_row = &workspace.v[j * d_model + head_offset..j * d_model + head_offset + head_dim];
                    vector_add_scaled(out_head, v_row, weight);
                }
            }
        }

        // 3. Output projection: attn_out * W_o^T (+ b_o)
        matmul(
            out,
            &workspace.attn_out[..seq_len * d_model],
            &self.w_o,
            self.b_o.as_deref(),
            seq_len,
            d_model,
            d_model,
        );
    }

    /// Autoregressive attention step for a single token at position `pos`.
    /// Key and Value vectors are appended to `kv_cache`.
    /// `input`: slice of shape [d_model]
    /// Writes output into `out` of shape [d_model].
    pub fn step(
        &self,
        input: &[f32],
        pos: usize,
        layer_idx: usize,
        kv_cache: &mut KvCache,
        workspace: &mut AttentionWorkspace,
        out: &mut [f32],
    ) {
        let d_model = self.d_model;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let scale = self.scale;

        // 1. Linear projections for the new token: q, k, v
        matvec(
            &mut workspace.step_q,
            &self.w_q,
            input,
            self.b_q.as_deref(),
            d_model,
            d_model,
        );
        matvec(
            &mut workspace.step_k,
            &self.w_k,
            input,
            self.b_k.as_deref(),
            d_model,
            d_model,
        );
        matvec(
            &mut workspace.step_v,
            &self.w_v,
            input,
            self.b_v.as_deref(),
            d_model,
            d_model,
        );

        // 2. Append new k and v to KV cache
        kv_cache.append_layer(layer_idx, pos, &workspace.step_k, &workspace.step_v);

        // 3. Attention over cached keys and values [0..=pos]
        for h in 0..n_heads {
            let q_head = &workspace.step_q[h * head_dim..(h + 1) * head_dim];
            let scores = &mut workspace.step_attn_scores[h * (pos + 1)..(h + 1) * (pos + 1)];

            for j in 0..=pos {
                let k_head = kv_cache.key_head(layer_idx, h, j);
                scores[j] = dot_product(q_head, k_head) * scale;
            }

            softmax(scores);

            let out_head = &mut workspace.step_attn_out[h * head_dim..(h + 1) * head_dim];
            out_head.fill(0.0);

            for j in 0..=pos {
                let v_head = kv_cache.val_head(layer_idx, h, j);
                vector_add_scaled(out_head, v_head, scores[j]);
            }
        }

        // 4. Output projection
        matvec(
            out,
            &self.w_o,
            &workspace.step_attn_out,
            self.b_o.as_deref(),
            d_model,
            d_model,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_causal_mask() {
        let config = TransformerConfig::default_512();
        let mut rng = FastRng::new(999);
        let mha = MultiHeadAttention::new(&config, &mut rng);
        let mut ws = AttentionWorkspace::new(&config);

        let seq_len = 3;
        let mut input = vec![0.0; seq_len * config.d_model];
        for (i, val) in input.iter_mut().enumerate() {
            *val = (i as f32 + 1.0) * 0.1;
        }

        let mut out = vec![0.0; seq_len * config.d_model];
        mha.forward_causal(&input, seq_len, &mut ws, &mut out);

        for &val in out.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_attention_forward_step_equivalence() {
        let config = TransformerConfig::default_512();
        let mut rng = FastRng::new(12345);
        let mha = MultiHeadAttention::new(&config, &mut rng);
        let mut ws = AttentionWorkspace::new(&config);
        let mut kv = KvCache::new(&config);

        let seq_len = 3;
        let mut input = vec![0.0; seq_len * config.d_model];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i * 17 + 5) % 31) as f32 * 0.05;
        }

        // Full forward
        let mut forward_out = vec![0.0; seq_len * config.d_model];
        mha.forward_causal(&input, seq_len, &mut ws, &mut forward_out);

        // Sequential step
        let mut step_out = vec![0.0; config.d_model];
        for pos in 0..seq_len {
            let tok_in = &input[pos * config.d_model..(pos + 1) * config.d_model];
            mha.step(tok_in, pos, 0, &mut kv, &mut ws, &mut step_out);

            let fwd_pos_slice = &forward_out[pos * config.d_model..(pos + 1) * config.d_model];
            for c in 0..config.d_model {
                let diff = (fwd_pos_slice[c] - step_out[c]).abs();
                assert!(
                    diff < 1e-5,
                    "Mismatch at pos {pos}, dim {c}: fwd={}, step={}, diff={diff}",
                    fwd_pos_slice[c],
                    step_out[c]
                );
            }
        }
    }
}
