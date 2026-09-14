/// Normalization type used in Transformer blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// Standard Layer Normalization (mean subtraction, variance normalization, scale and optional bias)
    LayerNorm,
    /// Root Mean Square Normalization (no mean subtraction, scale only)
    RMSNorm,
}

/// Activation function used in Feed-Forward / MLP blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// Gaussian Error Linear Unit (fast tanh approximation)
    Gelu,
    /// Rectified Linear Unit max(0, x)
    Relu,
}

/// Positional encoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PosEncodingType {
    /// Vaswani et al. fixed sinusoidal wave encoding (0 trainable parameters)
    Sinusoidal,
    /// Trainable positional embedding lookup table
    Learned,
}

/// Configuration parameters for the classical Transformer.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerConfig {
    /// Vocabulary size (number of distinct tokens)
    pub vocab_size: usize,
    /// Embedding and model hidden dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of transformer decoder layers
    pub n_layers: usize,
    /// Feed-forward intermediate hidden dimension
    pub d_ff: usize,
    /// Maximum sequence length (context window)
    pub max_seq_len: usize,
    /// Epsilon for numerical stability in LayerNorm / RMSNorm
    pub eps: f32,
    /// Normalization layer type (LayerNorm or RMSNorm)
    pub norm_type: NormType,
    /// Activation function in MLP
    pub activation: ActivationType,
    /// Positional encoding type
    pub pos_encoding: PosEncodingType,
    /// Whether to tie LM Head weights to the input token embedding table
    pub tie_word_embeddings: bool,
    /// Whether linear projection layers include additive biases
    pub use_bias: bool,
}

impl TransformerConfig {
    /// Base default configuration tailored to ~512 parameters (exactly 512 trainable parameters).
    /// Excellent for ultra-fast verification, regression testing, and micro-benchmarks.
    pub fn default_512() -> Self {
        Self {
            vocab_size: 13,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 8,
            max_seq_len: 32,
            eps: 1e-5,
            norm_type: NormType::RMSNorm,
            activation: ActivationType::Gelu,
            pos_encoding: PosEncodingType::Sinusoidal,
            tie_word_embeddings: true,
            use_bias: false,
        }
    }

    /// Language configuration tailored to 21 vocabulary tokens and exactly 512 parameters.
    /// (Vocab = 21, d_model = 8, n_heads = 2, n_layers = 1, d_ff = 4, max_seq_len = 32,
    /// RMSNorm, Relu, Sinusoidal, tie_word_embeddings = true, use_bias = false).
    /// Embeddings: 21*8 = 168
    /// MHA: 4*(8*8) = 256
    /// Norms: 3*8 = 24
    /// FFN: 4*8 + 8*4 = 64
    /// Total = 168 + 256 + 24 + 64 = 512 parameters.
    pub fn lang_512() -> Self {
        Self {
            vocab_size: 21,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 4,
            max_seq_len: 32,
            eps: 1e-5,
            norm_type: NormType::RMSNorm,
            activation: ActivationType::Relu,
            pos_encoding: PosEncodingType::Sinusoidal,
            tie_word_embeddings: true,
            use_bias: false,
        }
    }

    /// Language configuration v2 tailored to 41 vocabulary tokens (unified corpus v2).
    /// (Vocab = 41, d_model = 8, n_heads = 2, n_layers = 1, d_ff = 16, max_seq_len = 32,
    /// RMSNorm, Relu, Sinusoidal, tie_word_embeddings = true, use_bias = false).
    /// Parameter breakdown for Classical Transformer:
    /// - Token Embeddings: 41 * 8 = 328
    /// - Attention Projections: 4 * (8 * 8) = 256
    /// - Norms (Pre-Attn, Pre-FFN, Final): 3 * 8 = 24
    /// - FFN (W1 [16, 8] + W2 [8, 16]): 128 + 128 = 256
    /// Total = 864 parameters (3,456 bytes -> strictly L1D Cache resident < 32 KB).
    pub fn lang_v2() -> Self {
        Self {
            vocab_size: 41,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 16,
            max_seq_len: 32,
            eps: 1e-5,
            norm_type: NormType::RMSNorm,
            activation: ActivationType::Relu,
            pos_encoding: PosEncodingType::Sinusoidal,
            tie_word_embeddings: true,
            use_bias: false,
        }
    }

    /// Strict parameter parity baseline configuration for Classical Transformer (896 parameters).
    /// (Vocab = 41, d_model = 8, n_heads = 2, n_layers = 1, d_ff = 18, max_seq_len = 32,
    /// RMSNorm, Relu, Sinusoidal, tie_word_embeddings = true, use_bias = false).
    /// Parameter breakdown:
    /// - Token Embeddings: 41 * 8 = 328
    /// - Attention Projections: 4 * (8 * 8) = 256
    /// - Norms (Pre-Attn, Pre-FFN, Final): 3 * 8 = 24
    /// - FFN (W1 [18, 8] + W2 [8, 18]): 144 + 144 = 288
    /// Total = 608 + 288 = EXACTLY 896 parameters (3,584 bytes -> 100% L1D Cache resident).
    pub fn lang_v2_parity() -> Self {
        Self {
            vocab_size: 41,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 18,
            max_seq_len: 32,
            eps: 1e-5,
            norm_type: NormType::RMSNorm,
            activation: ActivationType::Relu,
            pos_encoding: PosEncodingType::Sinusoidal,
            tie_word_embeddings: true,
            use_bias: false,
        }
    }

    /// Strict parameter parity configuration for SRX v04 Physics-Spectral Core (898 parameters).
    /// (Vocab = 41, d_model = 8, n_heads = 2, head_dim = 4, n_layers = 1, d_ff = 17, max_seq_len = 32,
    /// RMSNorm, Relu, Sinusoidal, tie_word_embeddings = true, use_bias = false).
    /// Parameter breakdown:
    /// - Token Embeddings: 41 * 8 = 328
    /// - Attention Projections: 4 * (8 * 8) = 256
    /// - Selective Memory Gate: W_gamma [2, 8] + b_gamma [2] = 16 + 2 = 18
    /// - Norms (Pre-Attn, Pre-FFN, Final): 3 * 8 = 24
    /// - FFN (W1 [17, 8] + W2 [8, 17]): 136 + 136 = 272
    /// Total = 608 + 18 + 272 = EXACTLY 898 parameters (3,592 bytes -> 100% L1D Cache resident).
    /// Delta vs Classical Baseline: |898 - 896| = 2 parameters (0.22% delta, corridor 896 ± 4).
    pub fn srx_v04_parity() -> Self {
        Self {
            vocab_size: 41,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 17,
            max_seq_len: 32,
            eps: 1e-5,
            norm_type: NormType::RMSNorm,
            activation: ActivationType::Relu,
            pos_encoding: PosEncodingType::Sinusoidal,
            tie_word_embeddings: true,
            use_bias: false,
        }
    }

    /// Strict parameter parity configuration for Corpus v3 (EXACTLY 896 parameters).
    /// Used by both Classical Transformer and SRX v05 Quantum-Algebraic Core.
    /// (Vocab = 53, d_model = 8, n_heads = 2, n_layers = 1, d_ff = 12, max_seq_len = 32,
    /// RMSNorm, Relu, Sinusoidal, tie_word_embeddings = true, use_bias = false).
    /// Parameter breakdown:
    /// - Token Embeddings: 53 * 8 = 424
    /// - Attention Projections (W_q, W_k, W_v, W_o): 4 * (8 * 8) = 256
    /// - Norms (Pre-Attn, Pre-FFN, Final): 3 * 8 = 24
    /// - FFN (W1 [12, 8] + W2 [8, 12]): 96 + 96 = 192
    /// Total = 424 + 256 + 24 + 192 = EXACTLY 896 parameters (3,584 bytes -> 100% L1D Cache resident).
    pub fn lang_v3() -> Self {
        Self {
            vocab_size: 53,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 12,
            max_seq_len: 32,
            eps: 1e-5,
            norm_type: NormType::RMSNorm,
            activation: ActivationType::Relu,
            pos_encoding: PosEncodingType::Sinusoidal,
            tie_word_embeddings: true,
            use_bias: false,
        }
    }

    /// Strict parameter parity configuration for Chinchilla Sprint 1 (EXACTLY 896 parameters).
    /// Tailored to Chinchilla 20:1 optimal compute-scaling law (17,920 pretrain tokens).
    /// (Vocab = 65, d_model = 8, n_heads = 2, n_layers = 1, d_ff = 6, max_seq_len = 32,
    /// RMSNorm, Relu, Sinusoidal, tie_word_embeddings = true, use_bias = false).
    /// Parameter breakdown:
    /// - Token Embeddings: 65 * 8 = 520
    /// - Attention Projections (W_q, W_k, W_v, W_o): 4 * (8 * 8) = 256
    /// - Norms (Pre-Attn, Pre-FFN, Final): 3 * 8 = 24
    /// - FFN (W1 [6, 8] + W2 [8, 6]): 48 + 48 = 96
    /// Total = 520 + 256 + 24 + 96 = EXACTLY 896 parameters (3,584 bytes -> 100% L1D Cache resident).
    pub fn chinchilla() -> Self {
        Self {
            vocab_size: 65,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 6,
            max_seq_len: 32,
            eps: 1e-5,
            norm_type: NormType::RMSNorm,
            activation: ActivationType::Relu,
            pos_encoding: PosEncodingType::Sinusoidal,
            tie_word_embeddings: true,
            use_bias: false,
        }
    }

    /// Strict parameter parity configuration for Chinchilla language modeling (EXACTLY 896 parameters).
    /// Alias to chinchilla() providing explicit naming parity.
    #[inline]
    pub fn lang_chinchilla() -> Self {
        Self::chinchilla()
    }

    /// Micro configuration (~3.5k parameters) with 2 layers and untied head.
    pub fn micro() -> Self {
        Self {
            vocab_size: 32,
            d_model: 16,
            n_heads: 2,
            n_layers: 2,
            d_ff: 32,
            max_seq_len: 64,
            eps: 1e-5,
            norm_type: NormType::RMSNorm,
            activation: ActivationType::Gelu,
            pos_encoding: PosEncodingType::Sinusoidal,
            tie_word_embeddings: false,
            use_bias: false,
        }
    }

    /// Small configuration (~150k parameters) suitable for real word-level or byte-level modeling.
    pub fn small() -> Self {
        Self {
            vocab_size: 256,
            d_model: 64,
            n_heads: 4,
            n_layers: 4,
            d_ff: 128,
            max_seq_len: 128,
            eps: 1e-5,
            norm_type: NormType::RMSNorm,
            activation: ActivationType::Gelu,
            pos_encoding: PosEncodingType::Sinusoidal,
            tie_word_embeddings: false,
            use_bias: false,
        }
    }

    /// Dimension of each attention head.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Validates configuration invariant constraints.
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 {
            return Err("vocab_size must be greater than 0".to_string());
        }
        if self.d_model == 0 {
            return Err("d_model must be greater than 0".to_string());
        }
        if self.n_heads == 0 {
            return Err("n_heads must be greater than 0".to_string());
        }
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            ));
        }
        if self.n_layers == 0 {
            return Err("n_layers must be greater than 0".to_string());
        }
        if self.d_ff == 0 {
            return Err("d_ff must be greater than 0".to_string());
        }
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be greater than 0".to_string());
        }
        if self.eps <= 0.0 || !self.eps.is_finite() {
            return Err("eps must be a positive finite float".to_string());
        }
        Ok(())
    }

    /// Computes the exact number of trainable parameters in the model.
    pub fn param_count(&self) -> usize {
        let mut count = 0;

        // 1. Token Embeddings: [vocab_size, d_model]
        count += self.vocab_size * self.d_model;

        // 2. Positional Embeddings (if learned): [max_seq_len, d_model]
        if self.pos_encoding == PosEncodingType::Learned {
            count += self.max_seq_len * self.d_model;
        }

        // 3. Normalization parameter count per norm layer:
        // RMSNorm: gamma [d_model]
        // LayerNorm: gamma [d_model] + beta [d_model] (if use_bias or standard)
        let norm_params = match self.norm_type {
            NormType::RMSNorm => self.d_model,
            NormType::LayerNorm => {
                if self.use_bias {
                    2 * self.d_model
                } else {
                    self.d_model
                }
            }
        };

        // 4. Per layer parameters:
        let mut per_layer = 0;

        // Attention Q, K, V, O projections: 4 * (d_model * d_model)
        per_layer += 4 * (self.d_model * self.d_model);
        if self.use_bias {
            // Biases for Q, K, V, O: 4 * d_model
            per_layer += 4 * self.d_model;
        }

        // Pre-Attention Norm
        per_layer += norm_params;

        // Feed-Forward Network:
        // W1: [d_ff, d_model]
        per_layer += self.d_ff * self.d_model;
        if self.use_bias {
            per_layer += self.d_ff;
        }
        // W2: [d_model, d_ff]
        per_layer += self.d_model * self.d_ff;
        if self.use_bias {
            per_layer += self.d_model;
        }

        // Pre-FFN Norm
        per_layer += norm_params;

        count += self.n_layers * per_layer;

        // 5. Final Norm
        count += norm_params;

        // 6. LM Head
        if !self.tie_word_embeddings {
            // [vocab_size, d_model]
            count += self.vocab_size * self.d_model;
            if self.use_bias {
                count += self.vocab_size;
            }
        }

        count
    }
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self::default_512()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_512_param_count() {
        let config = TransformerConfig::default_512();
        assert_eq!(config.validate(), Ok(()));
        let count = config.param_count();
        assert_eq!(count, 512, "Default 512 config must have exactly 512 parameters!");
    }

    #[test]
    fn test_lang_512_param_count() {
        let config = TransformerConfig::lang_512();
        assert_eq!(config.validate(), Ok(()));
        let count = config.param_count();
        assert_eq!(count, 512, "Lang 512 config must have exactly 512 parameters!");
    }

    #[test]
    fn test_lang_v2_param_count() {
        let config = TransformerConfig::lang_v2();
        assert_eq!(config.validate(), Ok(()));
        let count = config.param_count();
        assert_eq!(count, 864, "Lang v2 config must have exactly 864 parameters!");
    }

    #[test]
    fn test_lang_v2_parity_param_count() {
        let config = TransformerConfig::lang_v2_parity();
        assert_eq!(config.validate(), Ok(()));
        let count = config.param_count();
        assert_eq!(count, 896, "Classical parity config must have exactly 896 parameters!");
    }

    #[test]
    fn test_lang_v3_param_count() {
        let config = TransformerConfig::lang_v3();
        assert_eq!(config.validate(), Ok(()));
        let count = config.param_count();
        assert_eq!(count, 896, "Lang v3 config must have exactly 896 parameters!");
    }

    #[test]
    fn test_chinchilla_param_count() {
        let config = TransformerConfig::chinchilla();
        assert_eq!(config.validate(), Ok(()));
        let count = config.param_count();
        assert_eq!(count, 896, "Chinchilla config must have exactly 896 parameters!");
    }

    #[test]
    fn test_lang_chinchilla_param_count() {
        let config = TransformerConfig::lang_chinchilla();
        assert_eq!(config.validate(), Ok(()));
        let count = config.param_count();
        assert_eq!(count, 896, "Lang Chinchilla config must have exactly 896 parameters!");
    }

    #[test]
    fn test_validation() {
        let mut cfg = TransformerConfig::default();
        assert!(cfg.validate().is_ok());

        cfg.d_model = 9; // Not divisible by n_heads=2
        assert!(cfg.validate().is_err());

        cfg.d_model = 8;
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }
}
