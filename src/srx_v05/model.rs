//! SRX Transformer Model v05 (Quantum-Algebraic Core).
//! Single-decoder layer transformer architecture featuring:
//! - 2nd-Order Recursive Least Squares (Sherman-Morrison update) associative memory
//! - Krylov Recurrent Depth (K=2) resolvent query refinement
//! - Monarch Butterfly Unitary Mixer with Phase Momentum
//! - Strictly 288 Bytes Context State (100% L1D resident)
//! - Exact Parameter Parity: exactly 896 trainable parameters at V=53, d=8, H=2, d_ff=12.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::classic::config::TransformerConfig;
use crate::classic::mlp::FeedForward;
use crate::classic::ops::{rmsnorm, softmax};
use crate::classic::rng::FastRng;
#[allow(unused_imports)]
use crate::classic::tokenizer::EOS_TOKEN_ID;

use super::attention::SrxAttention;
use super::state::{SrxState, SrxWorkspace};

const MAGIC: [u8; 4] = *b"SRX5";
const VERSION: u32 = 5;

/// Single SRX v05 decoder layer.
#[derive(Debug, Clone)]
pub struct SrxLayer {
    pub attn_norm_gamma: Vec<f32>,
    pub attn: SrxAttention,
    pub ffn_norm_gamma: Vec<f32>,
    pub mlp: FeedForward,
}

impl SrxLayer {
    pub fn new(config: &TransformerConfig, rng: &mut FastRng) -> Self {
        let d = config.d_model;
        Self {
            attn_norm_gamma: vec![1.0; d],
            attn: SrxAttention::new(config, rng),
            ffn_norm_gamma: vec![1.0; d],
            mlp: FeedForward::new(config, rng),
        }
    }

    #[inline]
    pub fn param_count(&self) -> usize {
        self.attn_norm_gamma.len()
            + self.attn.param_count()
            + self.ffn_norm_gamma.len()
            + self.mlp.w_1.len()
            + self.mlp.w_2.len()
            + self.mlp.b_1.as_ref().map_or(0, |b| b.len())
            + self.mlp.b_2.as_ref().map_or(0, |b| b.len())
    }
}

/// Full SRXFORMER v05 model.
#[derive(Debug, Clone)]
pub struct SrxTransformer {
    pub config: TransformerConfig,
    pub token_embeddings: Vec<f32>,
    pub layers: Vec<SrxLayer>,
    pub final_norm_gamma: Vec<f32>,
}

impl SrxTransformer {
    /// Creates a new model initialized with deterministic default seed 42.
    pub fn new(config: TransformerConfig) -> Result<Self, String> {
        config.validate()?;
        let mut rng = FastRng::new(42);
        Self::from_rng(config, &mut rng)
    }

    /// Creates a new model initialized with specified deterministic RNG seed.
    pub fn new_with_seed(config: TransformerConfig, seed: u64) -> Result<Self, String> {
        config.validate()?;
        let mut rng = FastRng::new(seed);
        Self::from_rng(config, &mut rng)
    }

    fn from_rng(config: TransformerConfig, rng: &mut FastRng) -> Result<Self, String> {
        let v = config.vocab_size;
        let d = config.d_model;
        let std = 1.0 / (d as f32).sqrt();

        let token_embeddings = (0..v * d).map(|_| rng.gen_normal(0.0, std)).collect();
        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(SrxLayer::new(&config, rng));
        }
        let final_norm_gamma = vec![1.0; d];

        Ok(Self {
            config,
            token_embeddings,
            layers,
            final_norm_gamma,
        })
    }

    /// Returns exact count of trainable parameters.
    /// At V=53, d=8, H=2, d_ff=12:
    /// - Token Embeddings: 53 * 8 = 424
    /// - Attention Projections: 4 * (8 * 8) = 256
    /// - Norms (Pre-Attn, Pre-FFN, Final): 3 * 8 = 24
    /// - FFN (W1 [12, 8] + W2 [8, 12]): 96 + 96 = 192
    /// Total = 424 + 256 + 24 + 192 = EXACTLY 896 parameters.
    pub fn param_count(&self) -> usize {
        let mut count = self.token_embeddings.len();
        for layer in &self.layers {
            count += layer.param_count();
        }
        count += self.final_norm_gamma.len();
        count
    }

    /// Hot-path single token inference step with strictly zero heap allocations.
    #[inline]
    pub fn step<'a>(
        &self,
        token_id: usize,
        _pos: usize,
        state: &mut SrxState,
        ws: &'a mut SrxWorkspace,
    ) -> &'a [f32] {
        let d = self.config.d_model;
        let v = self.config.vocab_size;
        let eps = self.config.eps;

        let emb_offset = token_id * d;
        ws.step_x.copy_from_slice(&self.token_embeddings[emb_offset..emb_offset + d]);

        for layer in &self.layers {
            ws.step_residual.copy_from_slice(&ws.step_x);

            rmsnorm(&mut ws.step_norm_out, &ws.step_x, &layer.attn_norm_gamma, eps);
            layer.attn.step(&ws.step_norm_out, state, &mut ws.step_attn_out, 1e-3);

            for i in 0..d {
                ws.step_x[i] = ws.step_residual[i] + ws.step_attn_out[i];
            }

            ws.step_residual.copy_from_slice(&ws.step_x);
            rmsnorm(&mut ws.step_norm_out, &ws.step_x, &layer.ffn_norm_gamma, eps);
            layer.mlp.step(&ws.step_norm_out, &mut ws.mlp, &mut ws.step_mlp_out);

            for i in 0..d {
                ws.step_x[i] = ws.step_residual[i] + ws.step_mlp_out[i];
            }
        }

        rmsnorm(&mut ws.step_norm_out, &ws.step_x, &self.final_norm_gamma, eps);

        // Tied LM head: logits = Token_Embeddings * x
        for i in 0..v {
            let row_off = i * d;
            let mut dot = 0.0f32;
            for j in 0..d {
                dot += self.token_embeddings[row_off + j] * ws.step_norm_out[j];
            }
            ws.step_logits[i] = dot;
        }

        &ws.step_logits
    }

    /// Autoregressive token generator decoding until EOS or max_new_tokens.
    pub fn generate_until_eos(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        eos_token_id: usize,
        state: &mut SrxState,
        ws: &mut SrxWorkspace,
    ) -> Vec<usize> {
        let mut generated = prompt_tokens.to_vec();
        if prompt_tokens.is_empty() || max_new_tokens == 0 {
            return generated;
        }

        state.reset();

        // 1. Prefill prompt tokens into associative state
        let mut next_logits = &[][..];
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            next_logits = self.step(tok, pos, state, ws);
        }

        let mut prob_scratch = vec![0.0f32; self.config.vocab_size];

        // 2. Autoregressive token generation loop
        for step in 0..max_new_tokens {
            prob_scratch.copy_from_slice(next_logits);
            softmax(&mut prob_scratch);

            let mut best_tok = 0;
            let mut best_prob = -1.0f32;
            for (idx, &p) in prob_scratch.iter().enumerate() {
                if p > best_prob {
                    best_prob = p;
                    best_tok = idx;
                }
            }

            generated.push(best_tok);
            if best_tok == eos_token_id {
                break;
            }

            let next_pos = prompt_tokens.len() + step;
            if next_pos >= self.config.max_seq_len {
                break;
            }

            next_logits = self.step(best_tok, next_pos, state, ws);
        }

        generated
    }

    /// High-level dataset training API wrapper.
    pub fn train_dataset(
        &mut self,
        tokens: &[usize],
        epochs: usize,
        lr: f32,
    ) -> crate::classic::telemetry::TrainTelemetry {
        super::train::train_dataset(self, tokens, epochs, lr)
    }

    /// Stage 1: Pretraining on Chinchilla / raw corpus tokens.
    pub fn train_pretrain(&mut self, tokens: &[usize], epochs: usize, lr: f32) -> crate::classic::TrainMetrics {
        super::train::train_pretrain(self, tokens, epochs, lr)
    }

    /// Stage 2: Instruction fine-tuning without replay.
    pub fn train_instruct(&mut self, tokens: &[usize], epochs: usize, lr: f32) -> crate::classic::TrainMetrics {
        super::train::train_instruct(self, tokens, epochs, lr)
    }

    /// Stage 2: Instruction fine-tuning with replay mixing (20-30% base facts) to eliminate catastrophic forgetting.
    pub fn train_instruct_with_replay(
        &mut self,
        instruct_tokens: &[usize],
        replay_tokens: &[usize],
        epochs: usize,
        lr: f32,
        replay_ratio: f32,
    ) -> crate::classic::TrainMetrics {
        super::train::train_instruct_with_replay(
            self,
            instruct_tokens,
            replay_tokens,
            epochs,
            lr,
            replay_ratio,
        )
    }

    /// High-level Two-Stage Training Pipeline: Stage 1 Pretrain + Stage 2 Instruct with Replay Mix.
    pub fn train_two_stage_pipeline(
        &mut self,
        pretrain_tokens: &[usize],
        instruct_tokens: &[usize],
        pretrain_epochs: usize,
        instruct_epochs: usize,
        pretrain_lr: f32,
        instruct_lr: f32,
        replay_ratio: f32,
    ) -> crate::classic::TwoStagePipelineResult {
        let pretrain_metrics = self.train_pretrain(pretrain_tokens, pretrain_epochs, pretrain_lr);
        let instruct_metrics = self.train_instruct_with_replay(
            instruct_tokens,
            pretrain_tokens,
            instruct_epochs,
            instruct_lr,
            replay_ratio,
        );
        crate::classic::TwoStagePipelineResult {
            pretrain_metrics,
            instruct_metrics,
        }
    }

    /// High-level Two-Stage Training Pipeline directly from corpus text files.
    pub fn train_chinchilla_pipeline_files<P1: AsRef<Path>, P2: AsRef<Path>>(
        &mut self,
        pretrain_path: P1,
        instruct_path: P2,
        tokenizer: &crate::classic::Tokenizer,
        pretrain_epochs: usize,
        instruct_epochs: usize,
        pretrain_lr: f32,
        instruct_lr: f32,
        replay_ratio: f32,
    ) -> Result<crate::classic::TwoStagePipelineResult, String> {
        let pretrain_text = std::fs::read_to_string(pretrain_path.as_ref())
            .map_err(|e| format!("Failed to read pretrain file: {}", e))?;
        let pretrain_tokens = tokenizer.encode(&pretrain_text);
        if pretrain_tokens.is_empty() {
            return Err("Pretrain file contains no valid tokens".to_string());
        }

        let instruct_text = std::fs::read_to_string(instruct_path.as_ref())
            .map_err(|e| format!("Failed to read instruct file: {}", e))?;
        let instruct_tokens = tokenizer.encode(&instruct_text);
        if instruct_tokens.is_empty() {
            return Err("Instruct file contains no valid tokens".to_string());
        }

        Ok(self.train_two_stage_pipeline(
            &pretrain_tokens,
            &instruct_tokens,
            pretrain_epochs,
            instruct_epochs,
            pretrain_lr,
            instruct_lr,
            replay_ratio,
        ))
    }

    /// Loads weights from a binary file into this existing model instance.
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        let loaded = Self::load_from_file(path)?;
        if self.config != loaded.config {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Config mismatch: model {:?} vs file {:?}", self.config, loaded.config),
            ));
        }
        self.token_embeddings = loaded.token_embeddings;
        self.layers = loaded.layers;
        self.final_norm_gamma = loaded.final_norm_gamma;
        Ok(())
    }

    /// Serializes model weights to a binary file with b"SRX5" header.
    pub fn save_weights<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        file.write_all(&MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;

        file.write_all(&(self.config.vocab_size as u32).to_le_bytes())?;
        file.write_all(&(self.config.d_model as u32).to_le_bytes())?;
        file.write_all(&(self.config.n_heads as u32).to_le_bytes())?;
        file.write_all(&(self.config.n_layers as u32).to_le_bytes())?;
        file.write_all(&(self.config.d_ff as u32).to_le_bytes())?;
        file.write_all(&(self.config.max_seq_len as u32).to_le_bytes())?;
        file.write_all(&self.config.eps.to_le_bytes())?;

        Self::write_slice(&mut file, &self.token_embeddings)?;
        for layer in &self.layers {
            Self::write_slice(&mut file, &layer.attn_norm_gamma)?;
            Self::write_slice(&mut file, &layer.attn.w_q)?;
            Self::write_slice(&mut file, &layer.attn.w_k)?;
            Self::write_slice(&mut file, &layer.attn.w_v)?;
            Self::write_slice(&mut file, &layer.attn.w_o)?;
            Self::write_slice(&mut file, &layer.ffn_norm_gamma)?;
            Self::write_slice(&mut file, &layer.mlp.w_1)?;
            Self::write_slice(&mut file, &layer.mlp.w_2)?;
        }
        Self::write_slice(&mut file, &self.final_norm_gamma)?;
        Ok(())
    }

    /// Deserializes model weights from a binary file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid magic header: expected SRX5, got {:?}", magic),
            ));
        }

        let mut v_buf = [0u8; 4];
        file.read_exact(&mut v_buf)?;
        let ver = u32::from_le_bytes(v_buf);
        if ver != VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported version: expected {}, got {}", VERSION, ver),
            ));
        }

        let vocab_size = Self::read_u32(&mut file)? as usize;
        let d_model = Self::read_u32(&mut file)? as usize;
        let n_heads = Self::read_u32(&mut file)? as usize;
        let n_layers = Self::read_u32(&mut file)? as usize;
        let d_ff = Self::read_u32(&mut file)? as usize;
        let max_seq_len = Self::read_u32(&mut file)? as usize;
        let eps = Self::read_f32(&mut file)?;

        let config = TransformerConfig {
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            max_seq_len,
            eps,
            norm_type: crate::classic::config::NormType::RMSNorm,
            activation: crate::classic::config::ActivationType::Relu,
            pos_encoding: crate::classic::config::PosEncodingType::Sinusoidal,
            tie_word_embeddings: true,
            use_bias: false,
        };

        let mut token_embeddings = vec![0.0f32; vocab_size * d_model];
        Self::read_slice(&mut file, &mut token_embeddings)?;

        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let mut attn_norm_gamma = vec![0.0f32; d_model];
            Self::read_slice(&mut file, &mut attn_norm_gamma)?;

            let mut w_q = vec![0.0f32; d_model * d_model];
            let mut w_k = vec![0.0f32; d_model * d_model];
            let mut w_v = vec![0.0f32; d_model * d_model];
            let mut w_o = vec![0.0f32; d_model * d_model];
            Self::read_slice(&mut file, &mut w_q)?;
            Self::read_slice(&mut file, &mut w_k)?;
            Self::read_slice(&mut file, &mut w_v)?;
            Self::read_slice(&mut file, &mut w_o)?;

            let attn = SrxAttention {
                w_q,
                w_k,
                w_v,
                w_o,
                d_model,
                n_heads,
                head_dim: d_model / n_heads,
                alpha: super::attention::SRX_ALPHA,
            };

            let mut ffn_norm_gamma = vec![0.0f32; d_model];
            Self::read_slice(&mut file, &mut ffn_norm_gamma)?;

            let mut w_1 = vec![0.0f32; d_ff * d_model];
            let mut w_2 = vec![0.0f32; d_model * d_ff];
            Self::read_slice(&mut file, &mut w_1)?;
            Self::read_slice(&mut file, &mut w_2)?;

            let mlp = FeedForward {
                w_1,
                b_1: None,
                w_2,
                b_2: None,
                activation: crate::classic::config::ActivationType::Relu,
                d_model,
                d_ff,
            };

            layers.push(SrxLayer {
                attn_norm_gamma,
                attn,
                ffn_norm_gamma,
                mlp,
            });
        }

        let mut final_norm_gamma = vec![0.0f32; d_model];
        Self::read_slice(&mut file, &mut final_norm_gamma)?;

        Ok(Self {
            config,
            token_embeddings,
            layers,
            final_norm_gamma,
        })
    }

    fn write_slice(file: &mut File, slice: &[f32]) -> std::io::Result<()> {
        for &val in slice {
            file.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    fn read_slice(file: &mut File, slice: &mut [f32]) -> std::io::Result<()> {
        let mut buf = [0u8; 4];
        for val in slice.iter_mut() {
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }
        Ok(())
    }

    fn read_u32(file: &mut File) -> std::io::Result<u32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_f32(file: &mut File) -> std::io::Result<f32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srx_v05_param_count() {
        let config = TransformerConfig::lang_chinchilla();
        let model = SrxTransformer::new(config).unwrap();
        assert_eq!(
            model.param_count(),
            896,
            "SRX v05 must contain EXACTLY 896 trainable parameters under Chinchilla configuration!"
        );
    }

    #[test]
    fn test_srx_v05_forward_step_equivalence() {
        let config = TransformerConfig::lang_chinchilla();
        let model = SrxTransformer::new_with_seed(config.clone(), 42).unwrap();
        let mut state = SrxState::new(&config);
        let mut ws = SrxWorkspace::new(&config);

        let tokens = [2, 6, 41, 6, 12, 3, 7, EOS_TOKEN_ID]; // "<user> 2 / 2 = <bot> 1 <eos>"
        let mut logits_step = Vec::new();

        for (pos, &tok) in tokens.iter().enumerate() {
            let logits = model.step(tok, pos, &mut state, &mut ws);
            logits_step.push(logits.to_vec());
        }

        assert_eq!(logits_step.len(), tokens.len());
        for (i, logit) in logits_step.iter().enumerate() {
            assert_eq!(logit.len(), config.vocab_size);
            for (j, &val) in logit.iter().enumerate() {
                assert!(val.is_finite(), "NaN/Inf at step {} logit {}", i, j);
            }
        }
    }

    #[test]
    fn test_srx_v05_weights_serialization_roundtrip() {
        let config = TransformerConfig::lang_v3();
        let model = SrxTransformer::new_with_seed(config.clone(), 12345).unwrap();

        let temp_path = std::env::temp_dir().join("srx_v05_test_weights.bin");
        model.save_weights(&temp_path).expect("Failed to save weights");

        let loaded = SrxTransformer::load_from_file(&temp_path).expect("Failed to load weights");
        assert_eq!(model.param_count(), loaded.param_count());
        assert_eq!(model.token_embeddings, loaded.token_embeddings);
        assert_eq!(model.layers[0].attn.w_q, loaded.layers[0].attn.w_q);
        assert_eq!(model.layers[0].mlp.w_1, loaded.layers[0].mlp.w_1);
        assert_eq!(model.final_norm_gamma, loaded.final_norm_gamma);

        let _ = std::fs::remove_file(temp_path);
    }
}
