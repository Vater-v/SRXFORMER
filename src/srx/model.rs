//! SrxTransformer: Pure Rust Super-Resolvent xFormer Architecture.
//! Features O(1) state memory footprint (152 bytes resident in L1 Cache),
//! unit-stride AVX FP32 vectorization, zero heap allocations on the hot path,
//! and bit-identical 512-parameter structure matching the classical transformer baseline.

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

/// A single SRX Transformer decoder layer.
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

    /// Single-step inference forward through this layer.
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

        // 3. SRX Attention Step
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

    /// Sequence forward pass through this layer.
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

        // 3. SRX Attention forward causal unroll
        let mut attn_out = vec![0.0f32; seq_len * d_model];
        self.attn.forward_causal(
            &workspace.norm_out[..seq_len * d_model],
            seq_len,
            &mut attn_out,
            eps,
        );

        // Residual connection: x = residual + attn_out
        for i in 0..seq_len * d_model {
            workspace.x[i] = workspace.residual[i] + attn_out[i];
        }

        // 4. Pre-FFN RMSNorm
        for t in 0..seq_len {
            let x_tok = &workspace.x[t * d_model..(t + 1) * d_model];
            let norm_tok = &mut workspace.norm_out[t * d_model..(t + 1) * d_model];
            rmsnorm(norm_tok, x_tok, &self.ffn_norm_gamma, self.eps);
        }

        // 5. Residual save
        workspace.residual[..seq_len * d_model].copy_from_slice(&workspace.x[..seq_len * d_model]);

        // 6. MLP forward
        let mut mlp_out = vec![0.0f32; seq_len * d_model];
        self.mlp.forward(
            &workspace.norm_out[..seq_len * d_model],
            seq_len,
            &mut workspace.mlp,
            &mut mlp_out,
        );

        // Residual connection: x = residual + mlp_out
        for i in 0..seq_len * d_model {
            workspace.x[i] = workspace.residual[i] + mlp_out[i];
        }
    }
}

/// The Super-Resolvent xFormer (SRXFORMER) model.
#[derive(Debug, Clone)]
pub struct SrxTransformer {
    pub config: TransformerConfig,
    pub token_embeddings: Vec<f32>,        // [vocab_size, d_model]
    pub sinusoidal_table: Option<Vec<f32>>, // [max_seq_len, d_model]
    pub layers: Vec<SrxLayer>,
    pub final_norm_gamma: Vec<f32>,        // [d_model]
}

impl SrxTransformer {
    /// Constructs and initializes a new SrxTransformer model with default seed 42.
    pub fn new(config: TransformerConfig) -> Result<Self, String> {
        Self::new_with_seed(config, 42)
    }

    /// Constructs and initializes a new SrxTransformer model with a specified PRNG seed.
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

    /// Computes total number of trainable parameters in this SrxTransformer instance (exactly 512 in lang_512).
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
        } else {
            panic!("Sinusoidal table required");
        }
    }

    /// Full forward pass for prompt tokens returning the logits of the last token: `&[f32]` of length `vocab_size`.
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
    /// Returns next-token logits: `&[f32]` of length `vocab_size`.
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

    /// Saves SRX model weights and configuration metadata to a binary file (magic SRXX v1).
    pub fn save_weights<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let mut bytes = Vec::with_capacity(64 + self.param_count() * 4);
        // 1. Magic bytes: SRXX
        bytes.extend_from_slice(b"SRXX");
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

    /// Loads and validates weights from a binary file (magic SRXX v1) into this model instance.
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<(), io::Error> {
        let raw = fs::read(path)?;
        let (cfg, weights) = parse_srx_weights_file(&raw)?;
        if self.config != cfg {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Config mismatch: model {:?} vs file {:?}", self.config, cfg),
            ));
        }
        self.load_weights_flat(&weights)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Constructs a new SrxTransformer instance directly from a saved weights binary file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let raw = fs::read(path)?;
        let (cfg, weights) = parse_srx_weights_file(&raw)?;
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

fn parse_srx_weights_file(raw: &[u8]) -> Result<(TransformerConfig, Vec<f32>), io::Error> {
    if raw.len() < 64 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "File too small (< 64 bytes)"));
    }
    if &raw[0..4] != b"SRXX" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid SRXX magic bytes"));
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

    let expected_bytes = 64 + param_count * 4;
    if raw.len() != expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("File payload size mismatch: expected {} bytes, got {}", expected_bytes, raw.len()),
        ));
    }

    let mut weights = Vec::with_capacity(param_count);
    for chunk in raw[64..].chunks_exact(4) {
        weights.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }

    Ok((config, weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srx_param_count_exact_512() {
        let config = TransformerConfig::lang_512();
        let model = SrxTransformer::new(config).unwrap();
        assert_eq!(model.param_count(), 512);
        assert_eq!(model.extract_weights().len(), 512);
    }

    #[test]
    fn test_srx_forward_step_equivalence() {
        let config = TransformerConfig::lang_512();
        let model = SrxTransformer::new_with_seed(config.clone(), 777).unwrap();
        let mut ws = SrxWorkspace::new(&config);

        let tokens = [1, 5, 2, 8, 3];

        // 1. Full sequence forward pass
        let forward_logits = model.forward(&tokens, &mut ws).to_vec();

        // 2. Sequential step pass
        let mut state = SrxState::new(&config);
        let mut step_logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            let logits = model.step(tok, pos, &mut state, &mut ws);
            if pos == tokens.len() - 1 {
                step_logits = logits.to_vec();
            }
        }

        assert_eq!(forward_logits.len(), step_logits.len());
        for (i, (&f_val, &s_val)) in forward_logits.iter().zip(step_logits.iter()).enumerate() {
            assert!(
                (f_val - s_val).abs() < 1e-4,
                "Forward vs Step mismatch at logit {}: forward={}, step={}",
                i,
                f_val,
                s_val
            );
        }
    }

    #[test]
    fn test_srx_weights_serialization_roundtrip() {
        let config = TransformerConfig::lang_512();
        let model = SrxTransformer::new_with_seed(config.clone(), 999).unwrap();

        let path = "target/test_srx_weights_roundtrip.bin";
        model.save_weights(path).unwrap();

        let loaded = SrxTransformer::load_from_file(path).unwrap();
        let _ = fs::remove_file(path);

        assert_eq!(model.extract_weights(), loaded.extract_weights());
    }
}
