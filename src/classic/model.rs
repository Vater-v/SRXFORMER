use std::fs;
use std::io;
use std::path::Path;

use super::attention::MultiHeadAttention;
use super::cache::{InferenceWorkspace, KvCache};
use super::config::{ActivationType, NormType, PosEncodingType, TransformerConfig};
use super::mlp::FeedForward;
use super::ops::{
    layernorm, matmul, matvec, rmsnorm, sinusoidal_pe_vector, softmax, vector_add,
};
use super::rng::FastRng;
use super::telemetry::TrainTelemetry;
use super::tokenizer::Tokenizer;

/// Weights and sublayers for a single Transformer decoder layer.
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub attn_norm_gamma: Vec<f32>,
    pub attn_norm_beta: Option<Vec<f32>>,
    pub attn: MultiHeadAttention,
    pub ffn_norm_gamma: Vec<f32>,
    pub ffn_norm_beta: Option<Vec<f32>>,
    pub mlp: FeedForward,
    pub norm_type: NormType,
    pub eps: f32,
    pub d_model: usize,
}

impl TransformerLayer {
    pub fn new(config: &TransformerConfig, rng: &mut FastRng) -> Self {
        let d_model = config.d_model;
        let attn_norm_gamma = vec![1.0; d_model];
        let ffn_norm_gamma = vec![1.0; d_model];

        let attn_norm_beta = match config.norm_type {
            NormType::LayerNorm if config.use_bias => Some(vec![0.0; d_model]),
            _ => None,
        };
        let ffn_norm_beta = match config.norm_type {
            NormType::LayerNorm if config.use_bias => Some(vec![0.0; d_model]),
            _ => None,
        };

        let attn = MultiHeadAttention::new(config, rng);
        let mlp = FeedForward::new(config, rng);

        Self {
            attn_norm_gamma,
            attn_norm_beta,
            attn,
            ffn_norm_gamma,
            ffn_norm_beta,
            mlp,
            norm_type: config.norm_type,
            eps: config.eps,
            d_model,
        }
    }

    #[inline]
    fn apply_norm(
        norm_type: NormType,
        out: &mut [f32],
        x: &[f32],
        gamma: &[f32],
        beta: Option<&[f32]>,
        eps: f32,
    ) {
        match norm_type {
            NormType::LayerNorm => layernorm(out, x, gamma, beta, eps),
            NormType::RMSNorm => rmsnorm(out, x, gamma, eps),
        }
    }

    /// Sequence forward pass through this layer.
    pub fn forward(
        &self,
        seq_len: usize,
        workspace: &mut InferenceWorkspace,
    ) {
        let d_model = self.d_model;

        // 1. Pre-Attention Normalization
        for i in 0..seq_len {
            let x_tok = &workspace.x[i * d_model..(i + 1) * d_model];
            let norm_tok = &mut workspace.norm_out[i * d_model..(i + 1) * d_model];
            Self::apply_norm(
                self.norm_type,
                norm_tok,
                x_tok,
                &self.attn_norm_gamma,
                self.attn_norm_beta.as_deref(),
                self.eps,
            );
        }

        // 2. Self-Attention
        workspace.residual[..seq_len * d_model].copy_from_slice(&workspace.x[..seq_len * d_model]);

        let InferenceWorkspace {
            ref norm_out,
            ref mut attn,
            ref mut x,
            ..
        } = *workspace;

        self.attn.forward_causal(
            &norm_out[..seq_len * d_model],
            seq_len,
            attn,
            &mut x[..seq_len * d_model],
        );

        // Residual connection: x = residual + attn_out
        vector_add(&mut workspace.x[..seq_len * d_model], &workspace.residual[..seq_len * d_model]);

        // 3. Pre-FFN Normalization
        for i in 0..seq_len {
            let x_tok = &workspace.x[i * d_model..(i + 1) * d_model];
            let norm_tok = &mut workspace.norm_out[i * d_model..(i + 1) * d_model];
            Self::apply_norm(
                self.norm_type,
                norm_tok,
                x_tok,
                &self.ffn_norm_gamma,
                self.ffn_norm_beta.as_deref(),
                self.eps,
            );
        }

        // 4. Feed-Forward sublayer
        workspace.residual[..seq_len * d_model].copy_from_slice(&workspace.x[..seq_len * d_model]);

        let InferenceWorkspace {
            ref norm_out,
            ref mut mlp,
            ref mut x,
            ..
        } = *workspace;

        self.mlp.forward(
            &norm_out[..seq_len * d_model],
            seq_len,
            mlp,
            &mut x[..seq_len * d_model],
        );

        // Residual connection: x = residual + mlp_out
        vector_add(&mut workspace.x[..seq_len * d_model], &workspace.residual[..seq_len * d_model]);
    }

    /// Single-token autoregressive step through this layer.
    pub fn step(
        &self,
        pos: usize,
        layer_idx: usize,
        kv_cache: &mut KvCache,
        workspace: &mut InferenceWorkspace,
    ) {
        // 1. Pre-Attention Normalization
        Self::apply_norm(
            self.norm_type,
            &mut workspace.step_norm_out,
            &workspace.step_x,
            &self.attn_norm_gamma,
            self.attn_norm_beta.as_deref(),
            self.eps,
        );

        // 2. Self-Attention
        workspace.step_residual.copy_from_slice(&workspace.step_x);

        let InferenceWorkspace {
            ref step_norm_out,
            ref mut attn,
            ref mut step_x,
            ..
        } = *workspace;

        self.attn.step(
            step_norm_out,
            pos,
            layer_idx,
            kv_cache,
            attn,
            step_x,
        );

        // Residual connection: step_x = step_residual + step_x
        vector_add(&mut workspace.step_x, &workspace.step_residual);

        // 3. Pre-FFN Normalization
        Self::apply_norm(
            self.norm_type,
            &mut workspace.step_norm_out,
            &workspace.step_x,
            &self.ffn_norm_gamma,
            self.ffn_norm_beta.as_deref(),
            self.eps,
        );

        // 4. Feed-Forward sublayer
        workspace.step_residual.copy_from_slice(&workspace.step_x);

        let InferenceWorkspace {
            ref step_norm_out,
            ref mut mlp,
            ref mut step_x,
            ..
        } = *workspace;

        self.mlp.step(step_norm_out, mlp, step_x);

        // Residual connection: step_x = step_residual + step_x
        vector_add(&mut workspace.step_x, &workspace.step_residual);
    }
}

/// The classical Transformer model.
#[derive(Debug, Clone)]
pub struct Transformer {
    pub config: TransformerConfig,
    pub token_embeddings: Vec<f32>,       // [vocab_size, d_model]
    pub pos_embeddings: Option<Vec<f32>>, // [max_seq_len, d_model] (if learned)
    pub sinusoidal_table: Option<Vec<f32>>, // [max_seq_len, d_model] (precomputed)
    pub layers: Vec<TransformerLayer>,
    pub final_norm_gamma: Vec<f32>,       // [d_model]
    pub final_norm_beta: Option<Vec<f32>>, // [d_model]
    pub lm_head: Option<Vec<f32>>,        // [vocab_size, d_model] (if untied)
    pub lm_head_bias: Option<Vec<f32>>,   // [vocab_size]
}

impl Transformer {
    /// Constructs and initializes a new Transformer model with default seed 42.
    pub fn new(config: TransformerConfig) -> Result<Self, String> {
        Self::new_with_seed(config, 42)
    }

    /// Constructs and initializes a new Transformer model with a specified PRNG seed.
    pub fn new_with_seed(config: TransformerConfig, seed: u64) -> Result<Self, String> {
        config.validate()?;
        let mut rng = FastRng::new(seed);

        let d_model = config.d_model;
        let vocab_size = config.vocab_size;
        let max_seq_len = config.max_seq_len;

        // Embedding standard deviation: 1 / sqrt(d_model)
        let embed_std = 1.0 / (d_model as f32).sqrt();

        // 1. Token Embeddings: [vocab_size, d_model]
        let token_embeddings = (0..vocab_size * d_model)
            .map(|_| rng.gen_normal(0.0, embed_std))
            .collect();

        // 2. Positional Encodings
        let pos_embeddings = if config.pos_encoding == PosEncodingType::Learned {
            let p = (0..max_seq_len * d_model)
                .map(|_| rng.gen_normal(0.0, embed_std))
                .collect();
            Some(p)
        } else {
            None
        };

        let sinusoidal_table = if config.pos_encoding == PosEncodingType::Sinusoidal {
            let mut table = vec![0.0; max_seq_len * d_model];
            for pos in 0..max_seq_len {
                sinusoidal_pe_vector(pos, d_model, &mut table[pos * d_model..(pos + 1) * d_model]);
            }
            Some(table)
        } else {
            None
        };

        // 3. Transformer Layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(TransformerLayer::new(&config, &mut rng));
        }

        // 4. Final Normalization
        let final_norm_gamma = vec![1.0; d_model];
        let final_norm_beta = match config.norm_type {
            NormType::LayerNorm if config.use_bias => Some(vec![0.0; d_model]),
            _ => None,
        };

        // 5. LM Head
        let lm_head = if !config.tie_word_embeddings {
            let head_std = 1.0 / (d_model as f32).sqrt();
            let h = (0..vocab_size * d_model)
                .map(|_| rng.gen_normal(0.0, head_std))
                .collect();
            Some(h)
        } else {
            None
        };

        let lm_head_bias = if config.use_bias && !config.tie_word_embeddings {
            Some(vec![0.0; vocab_size])
        } else {
            None
        };

        Ok(Self {
            config,
            token_embeddings,
            pos_embeddings,
            sinusoidal_table,
            layers,
            final_norm_gamma,
            final_norm_beta,
            lm_head,
            lm_head_bias,
        })
    }

    /// Returns configuration reference.
    #[inline]
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Computes total number of trainable parameters in this Transformer instance.
    #[inline]
    pub fn param_count(&self) -> usize {
        self.config.param_count()
    }

    /// Retrieves positional embedding slice for position `pos`.
    #[inline]
    fn get_pos_embedding(&self, pos: usize) -> &[f32] {
        let d_model = self.config.d_model;
        if let Some(ref table) = self.sinusoidal_table {
            &table[pos * d_model..(pos + 1) * d_model]
        } else if let Some(ref learned) = self.pos_embeddings {
            &learned[pos * d_model..(pos + 1) * d_model]
        } else {
            unreachable!()
        }
    }

    /// Full forward pass for prompt tokens.
    /// Returns the logits of the last token: `&[f32]` of length `vocab_size`.
    pub fn forward<'a>(
        &'a self,
        tokens: &[usize],
        workspace: &'a mut InferenceWorkspace,
    ) -> &'a [f32] {
        let all_logits = self.forward_all(tokens, workspace);
        let vocab_size = self.config.vocab_size;
        let last_offset = (tokens.len() - 1) * vocab_size;
        &all_logits[last_offset..last_offset + vocab_size]
    }

    /// Full forward pass returning logits for all sequence positions: [seq_len, vocab_size].
    pub fn forward_all<'a>(
        &'a self,
        tokens: &[usize],
        workspace: &'a mut InferenceWorkspace,
    ) -> &'a [f32] {
        let seq_len = tokens.len();
        assert!(seq_len > 0, "tokens cannot be empty");
        assert!(
            seq_len <= self.config.max_seq_len,
            "seq_len ({seq_len}) exceeds max_seq_len ({})",
            self.config.max_seq_len
        );

        let d_model = self.config.d_model;
        let vocab_size = self.config.vocab_size;

        // 1. Token Embeddings + Positional Embeddings
        for (i, &tok) in tokens.iter().enumerate() {
            assert!(
                tok < vocab_size,
                "token id {tok} exceeds vocab_size {vocab_size}"
            );
            let tok_emb = &self.token_embeddings[tok * d_model..(tok + 1) * d_model];
            let pe_emb = self.get_pos_embedding(i);
            let x_tok = &mut workspace.x[i * d_model..(i + 1) * d_model];
            for c in 0..d_model {
                x_tok[c] = tok_emb[c] + pe_emb[c];
            }
        }

        // 2. Transformer Decoder Layers
        for layer in &self.layers {
            layer.forward(seq_len, workspace);
        }

        // 3. Final Normalization
        for i in 0..seq_len {
            let x_tok = &workspace.x[i * d_model..(i + 1) * d_model];
            let norm_tok = &mut workspace.norm_out[i * d_model..(i + 1) * d_model];
            TransformerLayer::apply_norm(
                self.config.norm_type,
                norm_tok,
                x_tok,
                &self.final_norm_gamma,
                self.final_norm_beta.as_deref(),
                self.config.eps,
            );
        }

        // 4. Output LM Head Projection
        let head_weights = if self.config.tie_word_embeddings {
            &self.token_embeddings
        } else {
            self.lm_head.as_ref().unwrap()
        };

        matmul(
            &mut workspace.logits[..seq_len * vocab_size],
            &workspace.norm_out[..seq_len * d_model],
            head_weights,
            self.lm_head_bias.as_deref(),
            seq_len,
            d_model,
            vocab_size,
        );

        &workspace.logits[..seq_len * vocab_size]
    }

    /// Autoregressive inference step for a single token at position `pos`.
    /// Operates in O(1) new computation per step using KvCache.
    /// Returns next-token logits: `&[f32]` of length `vocab_size`.
    pub fn step<'a>(
        &'a self,
        token: usize,
        pos: usize,
        kv_cache: &mut KvCache,
        workspace: &'a mut InferenceWorkspace,
    ) -> &'a [f32] {
        let vocab_size = self.config.vocab_size;
        let d_model = self.config.d_model;
        assert!(token < vocab_size, "token {token} >= vocab_size {vocab_size}");
        assert!(pos < self.config.max_seq_len, "pos {pos} >= max_seq_len");

        // 1. Token Embedding + Positional Embedding
        let tok_emb = &self.token_embeddings[token * d_model..(token + 1) * d_model];
        let pe_emb = self.get_pos_embedding(pos);
        for c in 0..d_model {
            workspace.step_x[c] = tok_emb[c] + pe_emb[c];
        }

        // 2. Transformer Layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            layer.step(pos, layer_idx, kv_cache, workspace);
        }

        // Update KV cache cached length
        kv_cache.current_len = pos + 1;

        // 3. Final Normalization
        TransformerLayer::apply_norm(
            self.config.norm_type,
            &mut workspace.step_norm_out,
            &workspace.step_x,
            &self.final_norm_gamma,
            self.final_norm_beta.as_deref(),
            self.config.eps,
        );

        // 4. Output LM Head Projection
        let head_weights = if self.config.tie_word_embeddings {
            &self.token_embeddings
        } else {
            self.lm_head.as_ref().unwrap()
        };

        matvec(
            &mut workspace.step_logits,
            head_weights,
            &workspace.step_norm_out,
            self.lm_head_bias.as_deref(),
            vocab_size,
            d_model,
        );

        &workspace.step_logits
    }

    /// Autoregressive token generation.
    /// Takes a prompt of tokens, pre-fills the KV cache, and generates up to `max_new_tokens`.
    /// If `temperature <= 0.0`, performs greedy argmax sampling.
    /// If `temperature > 0.0`, performs temperature softmax sampling with deterministic FastRng.
    pub fn generate(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        kv_cache: &mut KvCache,
        workspace: &mut InferenceWorkspace,
    ) -> Vec<usize> {
        let mut output = prompt.to_vec();
        if prompt.is_empty() || max_new_tokens == 0 {
            return output;
        }

        kv_cache.reset();
        let mut rng = FastRng::new(1337);

        // Prefill prompt tokens into KV cache
        let mut next_logits = &[][..];
        for (pos, &token) in prompt.iter().enumerate() {
            next_logits = self.step(token, pos, kv_cache, workspace);
        }

        let mut sample_buf = vec![0.0f32; self.config.vocab_size];

        for step_idx in 0..max_new_tokens {
            let next_token = if temperature <= 0.0 {
                // Greedy argmax
                next_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            } else {
                // Temperature sampling
                sample_buf.copy_from_slice(next_logits);
                let inv_temp = 1.0 / temperature;
                for val in sample_buf.iter_mut() {
                    *val *= inv_temp;
                }
                softmax(&mut sample_buf);

                // Sample from categorical distribution
                let r = rng.next_f32();
                let mut cumsum = 0.0f32;
                let mut chosen = 0;
                for (idx, &prob) in sample_buf.iter().enumerate() {
                    cumsum += prob;
                    if r <= cumsum {
                        chosen = idx;
                        break;
                    }
                }
                chosen
            };

            output.push(next_token);

            let next_pos = prompt.len() + step_idx;
            if next_pos >= self.config.max_seq_len {
                break;
            }

            next_logits = self.step(next_token, next_pos, kv_cache, workspace);
        }

        output
    }

    /// Convenience wrapper for greedy decoding.
    #[inline]
    pub fn generate_greedy(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        kv_cache: &mut KvCache,
        workspace: &mut InferenceWorkspace,
    ) -> Vec<usize> {
        self.generate(prompt, max_new_tokens, 0.0, kv_cache, workspace)
    }

    /// Autoregressive greedy generation terminating upon emitting `eos_token`.
    pub fn generate_until_eos(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        eos_token: usize,
        kv_cache: &mut KvCache,
        workspace: &mut InferenceWorkspace,
    ) -> Vec<usize> {
        let mut output = prompt.to_vec();
        if prompt.is_empty() || max_new_tokens == 0 {
            return output;
        }

        kv_cache.reset();

        let mut next_logits = &[][..];
        for (pos, &token) in prompt.iter().enumerate() {
            next_logits = self.step(token, pos, kv_cache, workspace);
        }

        for step_idx in 0..max_new_tokens {
            let next_token = next_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            output.push(next_token);
            if next_token == eos_token {
                break;
            }

            let next_pos = prompt.len() + step_idx;
            if next_pos >= self.config.max_seq_len {
                break;
            }

            next_logits = self.step(next_token, next_pos, kv_cache, workspace);
        }

        output
    }

    /// Extracts all trainable parameters into a flat contiguous vector.
    pub fn extract_weights(&self) -> Vec<f32> {
        let mut w = Vec::with_capacity(self.param_count());
        w.extend_from_slice(&self.token_embeddings);
        if let Some(ref pe) = self.pos_embeddings {
            w.extend_from_slice(pe);
        }
        for layer in &self.layers {
            w.extend_from_slice(&layer.attn_norm_gamma);
            if let Some(ref b) = layer.attn_norm_beta {
                w.extend_from_slice(b);
            }
            w.extend_from_slice(&layer.attn.w_q);
            w.extend_from_slice(&layer.attn.w_k);
            w.extend_from_slice(&layer.attn.w_v);
            w.extend_from_slice(&layer.attn.w_o);
            if let Some(ref b) = layer.attn.b_q {
                w.extend_from_slice(b);
            }
            if let Some(ref b) = layer.attn.b_k {
                w.extend_from_slice(b);
            }
            if let Some(ref b) = layer.attn.b_v {
                w.extend_from_slice(b);
            }
            if let Some(ref b) = layer.attn.b_o {
                w.extend_from_slice(b);
            }
            w.extend_from_slice(&layer.ffn_norm_gamma);
            if let Some(ref b) = layer.ffn_norm_beta {
                w.extend_from_slice(b);
            }
            w.extend_from_slice(&layer.mlp.w_1);
            if let Some(ref b) = layer.mlp.b_1 {
                w.extend_from_slice(b);
            }
            w.extend_from_slice(&layer.mlp.w_2);
            if let Some(ref b) = layer.mlp.b_2 {
                w.extend_from_slice(b);
            }
        }
        w.extend_from_slice(&self.final_norm_gamma);
        if let Some(ref b) = self.final_norm_beta {
            w.extend_from_slice(b);
        }
        if let Some(ref head) = self.lm_head {
            w.extend_from_slice(head);
        }
        if let Some(ref b) = self.lm_head_bias {
            w.extend_from_slice(b);
        }
        assert_eq!(w.len(), self.param_count());
        w
    }

    /// Loads all trainable parameters from a flat contiguous vector.
    pub fn load_weights_flat(&mut self, flat: &[f32]) -> Result<(), String> {
        let expected = self.param_count();
        if flat.len() != expected {
            return Err(format!(
                "Weight size mismatch: expected {} parameters, got {}",
                expected,
                flat.len()
            ));
        }

        let mut offset = 0;

        let tok_len = self.token_embeddings.len();
        self.token_embeddings.copy_from_slice(&flat[offset..offset + tok_len]);
        offset += tok_len;

        if let Some(ref mut pe) = self.pos_embeddings {
            let pe_len = pe.len();
            pe.copy_from_slice(&flat[offset..offset + pe_len]);
            offset += pe_len;
        }

        for layer in &mut self.layers {
            let g_len = layer.attn_norm_gamma.len();
            layer.attn_norm_gamma.copy_from_slice(&flat[offset..offset + g_len]);
            offset += g_len;

            if let Some(ref mut b) = layer.attn_norm_beta {
                let b_len = b.len();
                b.copy_from_slice(&flat[offset..offset + b_len]);
                offset += b_len;
            }

            let q_len = layer.attn.w_q.len();
            layer.attn.w_q.copy_from_slice(&flat[offset..offset + q_len]);
            offset += q_len;

            let k_len = layer.attn.w_k.len();
            layer.attn.w_k.copy_from_slice(&flat[offset..offset + k_len]);
            offset += k_len;

            let v_len = layer.attn.w_v.len();
            layer.attn.w_v.copy_from_slice(&flat[offset..offset + v_len]);
            offset += v_len;

            let o_len = layer.attn.w_o.len();
            layer.attn.w_o.copy_from_slice(&flat[offset..offset + o_len]);
            offset += o_len;

            if let Some(ref mut b) = layer.attn.b_q {
                let b_len = b.len();
                b.copy_from_slice(&flat[offset..offset + b_len]);
                offset += b_len;
            }
            if let Some(ref mut b) = layer.attn.b_k {
                let b_len = b.len();
                b.copy_from_slice(&flat[offset..offset + b_len]);
                offset += b_len;
            }
            if let Some(ref mut b) = layer.attn.b_v {
                let b_len = b.len();
                b.copy_from_slice(&flat[offset..offset + b_len]);
                offset += b_len;
            }
            if let Some(ref mut b) = layer.attn.b_o {
                let b_len = b.len();
                b.copy_from_slice(&flat[offset..offset + b_len]);
                offset += b_len;
            }

            let ffn_g_len = layer.ffn_norm_gamma.len();
            layer.ffn_norm_gamma.copy_from_slice(&flat[offset..offset + ffn_g_len]);
            offset += ffn_g_len;

            if let Some(ref mut b) = layer.ffn_norm_beta {
                let b_len = b.len();
                b.copy_from_slice(&flat[offset..offset + b_len]);
                offset += b_len;
            }

            let w1_len = layer.mlp.w_1.len();
            layer.mlp.w_1.copy_from_slice(&flat[offset..offset + w1_len]);
            offset += w1_len;

            if let Some(ref mut b) = layer.mlp.b_1 {
                let b_len = b.len();
                b.copy_from_slice(&flat[offset..offset + b_len]);
                offset += b_len;
            }

            let w2_len = layer.mlp.w_2.len();
            layer.mlp.w_2.copy_from_slice(&flat[offset..offset + w2_len]);
            offset += w2_len;

            if let Some(ref mut b) = layer.mlp.b_2 {
                let b_len = b.len();
                b.copy_from_slice(&flat[offset..offset + b_len]);
                offset += b_len;
            }
        }

        let fn_len = self.final_norm_gamma.len();
        self.final_norm_gamma.copy_from_slice(&flat[offset..offset + fn_len]);
        offset += fn_len;

        if let Some(ref mut b) = self.final_norm_beta {
            let b_len = b.len();
            b.copy_from_slice(&flat[offset..offset + b_len]);
            offset += b_len;
        }

        if let Some(ref mut head) = self.lm_head {
            let h_len = head.len();
            head.copy_from_slice(&flat[offset..offset + h_len]);
            offset += h_len;
        }

        if let Some(ref mut b) = self.lm_head_bias {
            let b_len = b.len();
            b.copy_from_slice(&flat[offset..offset + b_len]);
            offset += b_len;
        }

        assert_eq!(offset, flat.len());
        Ok(())
    }

    /// Saves model weights and configuration metadata to a binary file.
    pub fn save_weights<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let mut bytes = Vec::with_capacity(64 + self.param_count() * 4);
        // 1. Magic bytes
        bytes.extend_from_slice(b"SRXF");
        // 2. Version
        bytes.extend_from_slice(&1u32.to_le_bytes());
        // 3. Config fields
        bytes.extend_from_slice(&(self.config.vocab_size as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.d_model as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.n_heads as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.n_layers as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.d_ff as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.max_seq_len as u32).to_le_bytes());
        bytes.extend_from_slice(&self.config.eps.to_le_bytes());

        let norm_val = match self.config.norm_type {
            NormType::LayerNorm => 0u32,
            NormType::RMSNorm => 1u32,
        };
        bytes.extend_from_slice(&norm_val.to_le_bytes());

        let act_val = match self.config.activation {
            ActivationType::Gelu => 0u32,
            ActivationType::Relu => 1u32,
        };
        bytes.extend_from_slice(&act_val.to_le_bytes());

        let pos_val = match self.config.pos_encoding {
            PosEncodingType::Sinusoidal => 0u32,
            PosEncodingType::Learned => 1u32,
        };
        bytes.extend_from_slice(&pos_val.to_le_bytes());

        bytes.push(if self.config.tie_word_embeddings { 1 } else { 0 });
        bytes.push(if self.config.use_bias { 1 } else { 0 });
        bytes.extend_from_slice(&[0u8; 6]); // reserved padding

        let param_count = self.param_count() as u64;
        bytes.extend_from_slice(&param_count.to_le_bytes());

        assert_eq!(bytes.len(), 64);

        let weights = self.extract_weights();
        for w in weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }

        fs::write(path, bytes)
    }

    /// Loads and validates weights from a binary file into this model instance.
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<(), io::Error> {
        let raw = fs::read(path)?;
        let (cfg, weights) = parse_weights_file(&raw)?;
        if self.config != cfg {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Config mismatch: model {:?} vs file {:?}", self.config, cfg),
            ));
        }
        self.load_weights_flat(&weights)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Constructs a new Transformer instance directly from a saved weights binary file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let raw = fs::read(path)?;
        let (cfg, weights) = parse_weights_file(&raw)?;
        let mut model = Self::new(cfg)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        model.load_weights_flat(&weights)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(model)
    }

    /// High-level training method on a tokenized dataset.
    pub fn train_dataset(&mut self, tokens: &[usize], epochs: usize, lr: f32) -> TrainTelemetry {
        let config = self.config.clone();
        super::train::train_dataset(self, tokens, &config, epochs, lr)
    }

    /// High-level training method directly on a text corpus file.
    pub fn train_file<P: AsRef<Path>>(
        &mut self,
        path: P,
        tokenizer: &Tokenizer,
        epochs: usize,
        lr: f32,
    ) -> Result<TrainTelemetry, String> {
        let text = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read file: {}", e))?;
        let tokens = tokenizer.encode(&text);
        if tokens.is_empty() {
            return Err("File contains no valid tokens".to_string());
        }
        Ok(self.train_dataset(&tokens, epochs, lr))
    }
}

fn parse_weights_file(raw: &[u8]) -> Result<(TransformerConfig, Vec<f32>), io::Error> {
    if raw.len() < 64 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "File too small (< 64 bytes)"));
    }
    if &raw[0..4] != b"SRXF" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic bytes"));
    }
    let version = u32::from_le_bytes(raw[4..8].try_into().unwrap());
    if version != 1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unsupported version: {}", version)));
    }

    let vocab_size = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
    let d_model = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
    let n_heads = u32::from_le_bytes(raw[16..20].try_into().unwrap()) as usize;
    let n_layers = u32::from_le_bytes(raw[20..24].try_into().unwrap()) as usize;
    let d_ff = u32::from_le_bytes(raw[24..28].try_into().unwrap()) as usize;
    let max_seq_len = u32::from_le_bytes(raw[28..32].try_into().unwrap()) as usize;
    let eps = f32::from_le_bytes(raw[32..36].try_into().unwrap());
    let norm_val = u32::from_le_bytes(raw[36..40].try_into().unwrap());
    let act_val = u32::from_le_bytes(raw[40..44].try_into().unwrap());
    let pos_val = u32::from_le_bytes(raw[44..48].try_into().unwrap());
    let tie_words = raw[48] != 0;
    let use_bias = raw[49] != 0;
    let param_count = u64::from_le_bytes(raw[56..64].try_into().unwrap()) as usize;

    let norm_type = match norm_val {
        0 => NormType::LayerNorm,
        1 => NormType::RMSNorm,
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Unknown norm_type")),
    };
    let activation = match act_val {
        0 => ActivationType::Gelu,
        1 => ActivationType::Relu,
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Unknown activation")),
    };
    let pos_encoding = match pos_val {
        0 => PosEncodingType::Sinusoidal,
        1 => PosEncodingType::Learned,
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Unknown pos_encoding")),
    };

    let config = TransformerConfig {
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        max_seq_len,
        eps,
        norm_type,
        activation,
        pos_encoding,
        tie_word_embeddings: tie_words,
        use_bias,
    };

    let expected_len = 64 + param_count * 4;
    if raw.len() != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("File byte length mismatch: expected {}, got {}", expected_len, raw.len()),
        ));
    }

    let mut weights = Vec::with_capacity(param_count);
    for i in 0..param_count {
        let offset = 64 + i * 4;
        let val = f32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap());
        weights.push(val);
    }

    Ok((config, weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_forward_step_equivalence_rms() {
        let config = TransformerConfig::default_512();
        let model = Transformer::new(config.clone()).unwrap();
        let mut ws = InferenceWorkspace::new(&config);
        let mut kv = KvCache::new(&config);

        let tokens = [1, 5, 2, 7];

        // 1. Full sequence forward pass
        let fwd_logits = model.forward(&tokens, &mut ws).to_vec();

        // 2. Sequential step pass
        kv.reset();
        let mut step_logits = vec![0.0; config.vocab_size];
        for (pos, &tok) in tokens.iter().enumerate() {
            let logits = model.step(tok, pos, &mut kv, &mut ws);
            step_logits.copy_from_slice(logits);
        }

        // Verify logits of the final token match
        for v in 0..config.vocab_size {
            let diff = (fwd_logits[v] - step_logits[v]).abs();
            assert!(
                diff < 1e-4,
                "Logits mismatch at vocab index {v}: forward={}, step={}, diff={diff}",
                fwd_logits[v],
                step_logits[v]
            );
        }
    }

    #[test]
    fn test_transformer_forward_step_equivalence_layernorm() {
        let mut config = TransformerConfig::default_512();
        config.norm_type = NormType::LayerNorm;
        config.use_bias = true;
        config.tie_word_embeddings = false;

        let model = Transformer::new(config.clone()).unwrap();
        let mut ws = InferenceWorkspace::new(&config);
        let mut kv = KvCache::new(&config);

        let tokens = [0, 4, 3, 2];

        // Full forward
        let fwd_logits = model.forward(&tokens, &mut ws).to_vec();

        // Sequential step
        kv.reset();
        let mut step_logits = vec![0.0; config.vocab_size];
        for (pos, &tok) in tokens.iter().enumerate() {
            let logits = model.step(tok, pos, &mut kv, &mut ws);
            step_logits.copy_from_slice(logits);
        }

        for v in 0..config.vocab_size {
            let diff = (fwd_logits[v] - step_logits[v]).abs();
            assert!(
                diff < 1e-4,
                "LayerNorm mismatch at vocab index {v}: forward={}, step={}, diff={diff}",
                fwd_logits[v],
                step_logits[v]
            );
        }
    }

    #[test]
    fn test_transformer_generation() {
        let config = TransformerConfig::default_512();
        let model = Transformer::new(config.clone()).unwrap();
        let mut ws = InferenceWorkspace::new(&config);
        let mut kv = KvCache::new(&config);

        let prompt = [1, 2];
        let output_tokens = model.generate_greedy(&prompt, 5, &mut kv, &mut ws);
        assert_eq!(output_tokens.len(), prompt.len() + 5);
        assert_eq!(&output_tokens[..2], &prompt);
        for &tok in &output_tokens {
            assert!(tok < config.vocab_size);
        }
    }

    #[test]
    fn test_weights_serialization_roundtrip() {
        let config = TransformerConfig::lang_512();
        let model1 = Transformer::new_with_seed(config.clone(), 999).unwrap();

        let path = "target/test_roundtrip_weights.bin";
        model1.save_weights(path).expect("Failed to save weights");

        // Load into new instance via load_from_file
        let model2 = Transformer::load_from_file(path).expect("Failed to load from file");

        assert_eq!(model1.config, model2.config);
        let w1 = model1.extract_weights();
        let w2 = model2.extract_weights();
        assert_eq!(w1.len(), 512);
        assert_eq!(w2.len(), 512);
        for i in 0..512 {
            assert_eq!(w1[i], w2[i], "Weight mismatch at index {i}");
        }

        // Test forward pass equality
        let tokens = [2, 6, 10, 7, 12, 3];
        let mut ws1 = InferenceWorkspace::new(&config);
        let mut ws2 = InferenceWorkspace::new(&config);
        let logits1 = model1.forward(&tokens, &mut ws1);
        let logits2 = model2.forward(&tokens, &mut ws2);
        for i in 0..config.vocab_size {
            assert_eq!(logits1[i], logits2[i]);
        }

        // Clean up
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_weights_load_into_existing() {
        let config = TransformerConfig::lang_512();
        let model1 = Transformer::new_with_seed(config.clone(), 111).unwrap();

        let path = "target/test_existing_weights.bin";
        model1.save_weights(path).expect("Failed to save weights");

        let mut model2 = Transformer::new_with_seed(config.clone(), 222).unwrap();
        // Weights should differ before load
        assert_ne!(model1.extract_weights(), model2.extract_weights());

        model2.load_weights(path).expect("Failed to load weights");
        assert_eq!(model1.extract_weights(), model2.extract_weights());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_weights_config_mismatch_fails() {
        let config1 = TransformerConfig::lang_512();
        let model1 = Transformer::new_with_seed(config1, 123).unwrap();

        let path = "target/test_mismatch_weights.bin";
        model1.save_weights(path).expect("Failed to save weights");

        let mut config2 = TransformerConfig::default_512();
        config2.vocab_size = 13; // differs from lang_512's 21
        let mut model2 = Transformer::new_with_seed(config2, 456).unwrap();

        let result = model2.load_weights(path);
        assert!(result.is_err(), "Expected load_weights to fail on config mismatch");

        let _ = std::fs::remove_file(path);
    }
}
