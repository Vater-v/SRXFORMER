//! SrxTransformer v03: Pure Rust Super-Resolvent xFormer Architecture
//! featuring Monarch Butterfly Unitary Factorization, Selective Dynamic Memory Gating,
//! O(1) state memory footprint (160 bytes resident in L1 Cache),
//! and strictly zero heap allocations on hot-path inference and forward steps.

use std::fs;
use std::io;
use std::path::Path;

use crate::classic::config::{ActivationType, NormType, PosEncodingType, TransformerConfig};
use crate::classic::mlp::FeedForward;
use crate::classic::ops::{matmul, matvec, rmsnorm, sinusoidal_pe_vector, softmax, vector_add};
use crate::classic::rng::FastRng;
use crate::classic::telemetry::TrainTelemetry;
use crate::classic::tokenizer::Tokenizer;

use super::attention::{SrxAttention, SRX_EPS_DEFAULT};
use super::state::{SrxState, SrxWorkspace};

/// A single SRX Transformer decoder layer v03.
#[derive(Debug, Clone)]
pub struct SrxLayer {
    pub attn_norm_gamma: Vec<f32>,
    pub attn: SrxAttention,
    pub ffn_norm_gamma: Vec<f32>,
    pub mlp: FeedForward,
    pub eps: f32,
    pub d_model: usize,
}

impl SrxLayer {
    pub fn new(config: &TransformerConfig, rng: &mut FastRng) -> Self {
        let d_model = config.d_model;
        let attn_norm_gamma = vec![1.0; d_model];
        let ffn_norm_gamma = vec![1.0; d_model];

        let attn = SrxAttention::new(config, rng);
        let mlp = FeedForward::new(config, rng);

        Self {
            attn_norm_gamma,
            attn,
            ffn_norm_gamma,
            mlp,
            eps: config.eps,
            d_model,
        }
    }

    /// Single-step inference forward through this layer: strictly ZERO dynamic allocations!
    pub fn step(
        &self,
        state: &mut SrxState,
        workspace: &mut SrxWorkspace,
        eps: f32,
    ) {
        // 1. Pre-Attention RMSNorm
        rmsnorm(
            &mut workspace.step_norm_out,
            &workspace.step_x,
            &self.attn_norm_gamma,
            self.eps,
        );

        // 2. Residual save
        workspace.step_residual.copy_from_slice(&workspace.step_x);

        // 3. SRX Attention Step (Zero Alloc)
        self.attn.step(
            &workspace.step_norm_out,
            state,
            &mut workspace.step_attn_out,
            eps,
        );

        // Residual add
        vector_add(&mut workspace.step_x, &workspace.step_attn_out);

        // 4. Pre-FFN RMSNorm
        rmsnorm(
            &mut workspace.step_norm_out,
            &workspace.step_x,
            &self.ffn_norm_gamma,
            self.eps,
        );

        workspace.step_residual.copy_from_slice(&workspace.step_x);

        // 5. MLP
        self.mlp.step(
            &workspace.step_norm_out,
            &mut workspace.mlp,
            &mut workspace.step_mlp_out,
        );

        // Residual add
        vector_add(&mut workspace.step_x, &workspace.step_mlp_out);
    }

    /// Sequence forward pass through this layer using pre-allocated workspace buffers (Zero Alloc).
    pub fn forward(
        &self,
        seq_len: usize,
        workspace: &mut SrxWorkspace,
        eps: f32,
    ) {
        let d_model = self.d_model;

        // 1. Pre-Attention RMSNorm
        for t in 0..seq_len {
            let x_tok = &workspace.x[t * d_model..(t + 1) * d_model];
            let norm_tok = &mut workspace.norm_out[t * d_model..(t + 1) * d_model];
            rmsnorm(norm_tok, x_tok, &self.attn_norm_gamma, self.eps);
        }

        // 2. Residual save
        workspace.residual[..seq_len * d_model].copy_from_slice(&workspace.x[..seq_len * d_model]);

        // 3. SRX Attention forward causal unroll into pre-allocated workspace.attn_out
        self.attn.forward_causal(
            &workspace.norm_out[..seq_len * d_model],
            seq_len,
            &mut workspace.attn_out[..seq_len * d_model],
            eps,
        );

        // Residual connection: x = residual + attn_out
        for i in 0..seq_len * d_model {
            workspace.x[i] = workspace.residual[i] + workspace.attn_out[i];
        }

        // 4. Pre-FFN RMSNorm
        for t in 0..seq_len {
            let x_tok = &workspace.x[t * d_model..(t + 1) * d_model];
            let norm_tok = &mut workspace.norm_out[t * d_model..(t + 1) * d_model];
            rmsnorm(norm_tok, x_tok, &self.ffn_norm_gamma, self.eps);
        }

        // 5. Residual save
        workspace.residual[..seq_len * d_model].copy_from_slice(&workspace.x[..seq_len * d_model]);

        // 6. MLP forward into pre-allocated workspace.mlp_out
        self.mlp.forward(
            &workspace.norm_out[..seq_len * d_model],
            seq_len,
            &mut workspace.mlp,
            &mut workspace.mlp_out[..seq_len * d_model],
        );

        // Residual connection: x = residual + mlp_out
        for i in 0..seq_len * d_model {
            workspace.x[i] = workspace.residual[i] + workspace.mlp_out[i];
        }
    }
}

/// The Super-Resolvent xFormer (SRXFORMER) Model v03.
#[derive(Debug, Clone)]
pub struct SrxTransformer {
    pub config: TransformerConfig,
    pub token_embeddings: Vec<f32>,        // [vocab_size, d_model]
    pub sinusoidal_table: Option<Vec<f32>>, // [max_seq_len, d_model]
    pub layers: Vec<SrxLayer>,
    pub final_norm_gamma: Vec<f32>,        // [d_model]
}

impl SrxTransformer {
    /// Constructs and initializes a new SrxTransformer v03 model with default seed 42.
    pub fn new(config: TransformerConfig) -> Result<Self, String> {
        Self::new_with_seed(config, 42)
    }

    /// Constructs and initializes a new SrxTransformer v03 model with a specified PRNG seed.
    pub fn new_with_seed(config: TransformerConfig, seed: u64) -> Result<Self, String> {
        config.validate()?;
        let mut rng = FastRng::new(seed);

        let d_model = config.d_model;
        let vocab_size = config.vocab_size;
        let max_seq_len = config.max_seq_len;

        let embed_std = 1.0 / (d_model as f32).sqrt();

        // 1. Token Embeddings: [vocab_size, d_model]
        let token_embeddings = (0..vocab_size * d_model)
            .map(|_| rng.gen_normal(0.0, embed_std))
            .collect();

        // 2. Sinusoidal Positional Encoding table
        let sinusoidal_table = if config.pos_encoding == PosEncodingType::Sinusoidal {
            let mut table = vec![0.0; max_seq_len * d_model];
            for pos in 0..max_seq_len {
                sinusoidal_pe_vector(pos, d_model, &mut table[pos * d_model..(pos + 1) * d_model]);
            }
            Some(table)
        } else {
            None
        };

        // 3. SRX Decoder Layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(SrxLayer::new(&config, &mut rng));
        }

        // 4. Final RMSNorm
        let final_norm_gamma = vec![1.0; d_model];

        Ok(Self {
            config,
            token_embeddings,
            sinusoidal_table,
            layers,
            final_norm_gamma,
        })
    }

    /// Returns configuration reference.
    #[inline]
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Honest parameter accounting: returns exact number of trainable parameters in this model instance.
    #[inline]
    pub fn param_count(&self) -> usize {
        let mut total = self.token_embeddings.len();
        for layer in &self.layers {
            total += layer.attn_norm_gamma.len();
            total += layer.attn.param_count();
            total += layer.ffn_norm_gamma.len();
            total += layer.mlp.w_1.len() + layer.mlp.w_2.len();
        }
        total += self.final_norm_gamma.len();
        total
    }

    /// Retrieves positional embedding slice for position `pos`.
    #[inline]
    fn get_pos_embedding(&self, pos: usize) -> &[f32] {
        let d_model = self.config.d_model;
        if let Some(ref table) = self.sinusoidal_table {
            &table[pos * d_model..(pos + 1) * d_model]
        } else {
            panic!("Sinusoidal table required");
        }
    }

    /// Full forward pass for prompt tokens returning the logits of the last token.
    pub fn forward<'a>(
        &'a self,
        tokens: &[usize],
        workspace: &'a mut SrxWorkspace,
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
        workspace: &'a mut SrxWorkspace,
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

        // 1. Token Embeddings + Sinusoidal Positional Embeddings
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

        // 2. SRX Decoder Layers
        for layer in &self.layers {
            layer.forward(seq_len, workspace, SRX_EPS_DEFAULT);
        }

        // 3. Final RMSNorm
        for i in 0..seq_len {
            let x_tok = &workspace.x[i * d_model..(i + 1) * d_model];
            let norm_tok = &mut workspace.norm_out[i * d_model..(i + 1) * d_model];
            rmsnorm(norm_tok, x_tok, &self.final_norm_gamma, self.config.eps);
        }

        // 4. Output LM Head Projection (tied to token embeddings)
        matmul(
            &mut workspace.logits[..seq_len * vocab_size],
            &workspace.norm_out[..seq_len * d_model],
            &self.token_embeddings,
            None,
            seq_len,
            d_model,
            vocab_size,
        );

        &workspace.logits[..seq_len * vocab_size]
    }

    /// Autoregressive inference step for a single token at position `pos`.
    /// Operates in O(1) state memory and O(d) compute.
    /// Strictly zero heap allocations!
    pub fn step<'a>(
        &'a self,
        token: usize,
        pos: usize,
        state: &mut SrxState,
        workspace: &'a mut SrxWorkspace,
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

        // 2. SRX Layer Step
        for layer in &self.layers {
            layer.step(state, workspace, SRX_EPS_DEFAULT);
        }

        state.current_pos = pos + 1;

        // 3. Final RMSNorm
        rmsnorm(
            &mut workspace.step_norm_out,
            &workspace.step_x,
            &self.final_norm_gamma,
            self.config.eps,
        );

        // 4. Tied LM Head Projection
        matvec(
            &mut workspace.step_logits,
            &self.token_embeddings,
            &workspace.step_norm_out,
            None,
            vocab_size,
            d_model,
        );

        &workspace.step_logits
    }

    /// Autoregressive token generation.
    pub fn generate(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        state: &mut SrxState,
        workspace: &mut SrxWorkspace,
    ) -> Vec<usize> {
        let mut output = prompt.to_vec();
        if prompt.is_empty() || max_new_tokens == 0 {
            return output;
        }

        state.reset();
        let mut rng = FastRng::new(1337);

        // Prefill prompt tokens into SRX state
        let mut next_logits = &[][..];
        for (pos, &token) in prompt.iter().enumerate() {
            next_logits = self.step(token, pos, state, workspace);
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

            next_logits = self.step(next_token, next_pos, state, workspace);
        }

        output
    }

    /// Convenience wrapper for greedy decoding.
    #[inline]
    pub fn generate_greedy(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        state: &mut SrxState,
        workspace: &mut SrxWorkspace,
    ) -> Vec<usize> {
        self.generate(prompt, max_new_tokens, 0.0, state, workspace)
    }

    /// Autoregressive greedy generation terminating upon emitting `eos_token`.
    pub fn generate_until_eos(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        eos_token: usize,
        state: &mut SrxState,
        workspace: &mut SrxWorkspace,
    ) -> Vec<usize> {
        let mut output = prompt.to_vec();
        if prompt.is_empty() || max_new_tokens == 0 {
            return output;
        }

        state.reset();

        let mut next_logits = &[][..];
        for (pos, &token) in prompt.iter().enumerate() {
            next_logits = self.step(token, pos, state, workspace);
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

            next_logits = self.step(next_token, next_pos, state, workspace);
        }

        output
    }

    /// Extracts all trainable parameters into a flat contiguous vector.
    pub fn extract_weights(&self) -> Vec<f32> {
        let mut w = Vec::with_capacity(self.param_count());
        w.extend_from_slice(&self.token_embeddings);
        for layer in &self.layers {
            w.extend_from_slice(&layer.attn_norm_gamma);
            w.extend_from_slice(&layer.attn.w_q);
            w.extend_from_slice(&layer.attn.w_k);
            w.extend_from_slice(&layer.attn.w_v);
            w.extend_from_slice(&layer.attn.w_o);
            w.extend_from_slice(&layer.attn.w_gamma);
            w.extend_from_slice(&layer.attn.b_gamma);
            w.extend_from_slice(&layer.ffn_norm_gamma);
            w.extend_from_slice(&layer.mlp.w_1);
            w.extend_from_slice(&layer.mlp.w_2);
        }
        w.extend_from_slice(&self.final_norm_gamma);
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

        for layer in &mut self.layers {
            let g_len = layer.attn_norm_gamma.len();
            layer.attn_norm_gamma.copy_from_slice(&flat[offset..offset + g_len]);
            offset += g_len;

            let wq_len = layer.attn.w_q.len();
            layer.attn.w_q.copy_from_slice(&flat[offset..offset + wq_len]);
            offset += wq_len;

            let wk_len = layer.attn.w_k.len();
            layer.attn.w_k.copy_from_slice(&flat[offset..offset + wk_len]);
            offset += wk_len;

            let wv_len = layer.attn.w_v.len();
            layer.attn.w_v.copy_from_slice(&flat[offset..offset + wv_len]);
            offset += wv_len;

            let wo_len = layer.attn.w_o.len();
            layer.attn.w_o.copy_from_slice(&flat[offset..offset + wo_len]);
            offset += wo_len;

            let wg_len = layer.attn.w_gamma.len();
            layer.attn.w_gamma.copy_from_slice(&flat[offset..offset + wg_len]);
            offset += wg_len;

            let bg_len = layer.attn.b_gamma.len();
            layer.attn.b_gamma.copy_from_slice(&flat[offset..offset + bg_len]);
            offset += bg_len;

            let fg_len = layer.ffn_norm_gamma.len();
            layer.ffn_norm_gamma.copy_from_slice(&flat[offset..offset + fg_len]);
            offset += fg_len;

            let w1_len = layer.mlp.w_1.len();
            layer.mlp.w_1.copy_from_slice(&flat[offset..offset + w1_len]);
            offset += w1_len;

            let w2_len = layer.mlp.w_2.len();
            layer.mlp.w_2.copy_from_slice(&flat[offset..offset + w2_len]);
            offset += w2_len;
        }

        let fn_len = self.final_norm_gamma.len();
        self.final_norm_gamma.copy_from_slice(&flat[offset..offset + fn_len]);
        offset += fn_len;

        assert_eq!(offset, flat.len());
        Ok(())
    }

    /// Default binary weights path for SRX v03 model.
    pub const DEFAULT_WEIGHTS_PATH: &'static str = "data/srx_v03_model_weights.bin";

    /// Saves SRX model weights and configuration metadata to a binary file (magic SRX3 v3).
    pub fn save_weights<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let mut bytes = Vec::with_capacity(64 + self.param_count() * 4);
        // 1. Magic bytes: SRX3
        bytes.extend_from_slice(b"SRX3");
        // 2. Version: 3
        bytes.extend_from_slice(&3u32.to_le_bytes());
        // 3. Config fields
        bytes.extend_from_slice(&(self.config.vocab_size as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.d_model as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.n_heads as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.n_layers as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.d_ff as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.max_seq_len as u32).to_le_bytes());
        bytes.extend_from_slice(&self.config.eps.to_le_bytes());
        let norm_code: u32 = match self.config.norm_type {
            NormType::LayerNorm => 0,
            NormType::RMSNorm => 1,
        };
        bytes.extend_from_slice(&norm_code.to_le_bytes());
        let act_code: u32 = match self.config.activation {
            ActivationType::Gelu => 0,
            ActivationType::Relu => 1,
        };
        bytes.extend_from_slice(&act_code.to_le_bytes());
        let pos_code: u32 = match self.config.pos_encoding {
            PosEncodingType::Sinusoidal => 0,
            PosEncodingType::Learned => 1,
        };
        bytes.extend_from_slice(&pos_code.to_le_bytes());
        bytes.extend_from_slice(&(self.config.tie_word_embeddings as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.use_bias as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.param_count() as u32).to_le_bytes());

        // 4. Weight floats
        let flat_w = self.extract_weights();
        for &w_val in &flat_w {
            bytes.extend_from_slice(&w_val.to_le_bytes());
        }

        fs::write(path, bytes)
    }

    /// Loads an SRX v03 model from a binary weights file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("Failed to read weights file: {e}"))?;
        if bytes.len() < 56 {
            return Err("Binary weights file is too short".to_string());
        }

        // Check magic bytes: SRX3
        if &bytes[0..4] != b"SRX3" {
            return Err(format!(
                "Invalid magic header: expected SRX3, got {:?}",
                &bytes[0..4]
            ));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != 3 {
            return Err(format!("Unsupported weights version: {version}, expected 3"));
        }

        let vocab_size = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let d_model = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let n_heads = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let n_layers = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        let d_ff = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
        let max_seq_len = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;
        let eps = f32::from_le_bytes(bytes[32..36].try_into().unwrap());

        let norm_code = u32::from_le_bytes(bytes[36..40].try_into().unwrap());
        let norm_type = match norm_code {
            0 => NormType::LayerNorm,
            1 => NormType::RMSNorm,
            _ => return Err(format!("Unknown norm type code: {norm_code}")),
        };

        let act_code = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        let activation = match act_code {
            0 => ActivationType::Gelu,
            1 => ActivationType::Relu,
            _ => return Err(format!("Unknown activation code: {act_code}")),
        };

        let pos_code = u32::from_le_bytes(bytes[44..48].try_into().unwrap());
        let pos_encoding = match pos_code {
            0 => PosEncodingType::Sinusoidal,
            1 => PosEncodingType::Learned,
            _ => return Err(format!("Unknown pos encoding code: {pos_code}")),
        };

        let tie_word_embeddings = u32::from_le_bytes(bytes[48..52].try_into().unwrap()) != 0;
        let use_bias = u32::from_le_bytes(bytes[52..56].try_into().unwrap()) != 0;
        let param_count = u32::from_le_bytes(bytes[56..60].try_into().unwrap()) as usize;

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
            tie_word_embeddings,
            use_bias,
        };

        let mut model = Self::new(config)?;

        let expected_bytes = 60 + param_count * 4;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "File size mismatch: expected {} bytes, found {}",
                expected_bytes,
                bytes.len()
            ));
        }

        let mut flat = Vec::with_capacity(param_count);
        let mut offset = 60;
        for _ in 0..param_count {
            let val = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            flat.push(val);
            offset += 4;
        }

        model.load_weights_flat(&flat)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srx_v03_param_count() {
        let mut config = TransformerConfig::lang_512();
        config.d_ff = 8;
        let model = SrxTransformer::new(config).unwrap();

        // Token embeddings: 21 * 8 = 168
        // Attn norm: 8
        // Attn: Wq(64) + Wk(64) + Wv(64) + Wo(64) + W_gamma(16) + b_gamma(2) = 274
        // FFN norm: 8
        // FFN: W1(8*8=64) + W2(8*8=64) = 128
        // Final norm: 8
        // Grand total: 168 + 8 + 274 + 8 + 128 + 8 = 594 parameters (2376 bytes)
        assert_eq!(model.param_count(), 594);
        assert_eq!(model.extract_weights().len(), 594);
    }

    #[test]
    fn test_srx_v03_forward_step_equivalence() {
        let mut config = TransformerConfig::lang_512();
        config.d_ff = 8;
        let model = SrxTransformer::new_with_seed(config.clone(), 77).unwrap();
        let tokens = [1, 5, 2, 8, 3];

        let mut ws_forward = SrxWorkspace::new(&config);
        let logits_forward = model.forward_all(&tokens, &mut ws_forward);
        let last_forward_logits = &logits_forward[(tokens.len() - 1) * config.vocab_size..tokens.len() * config.vocab_size];

        let mut ws_step = SrxWorkspace::new(&config);
        let mut state = SrxState::new(&config);
        let mut last_step_logits = vec![0.0f32; config.vocab_size];

        for (pos, &token) in tokens.iter().enumerate() {
            let logits = model.step(token, pos, &mut state, &mut ws_step);
            last_step_logits.copy_from_slice(logits);
        }

        for (i, (&fwd, &stp)) in last_forward_logits.iter().zip(last_step_logits.iter()).enumerate() {
            assert!(
                (fwd - stp).abs() < 1e-4,
                "Logit mismatch at index {i}: forward={fwd}, step={stp}"
            );
        }
    }

    #[test]
    fn test_srx_v03_weights_serialization_roundtrip() {
        let mut config = TransformerConfig::lang_512();
        config.d_ff = 8;
        let model = SrxTransformer::new_with_seed(config.clone(), 99).unwrap();
        let path = "target/test_srx_v03_weights.bin";

        model.save_weights(path).unwrap();
        let loaded = SrxTransformer::load_from_file(path).unwrap();

        assert_eq!(model.param_count(), loaded.param_count());
        assert_eq!(model.extract_weights(), loaded.extract_weights());

        let _ = fs::remove_file(path);
    }
}
