//! # Scaled Architecture Configuration & Mathematical Scaling Calculator
//!
//! Replaces manual parameter guessing with a rigorous quantum scaling law:
//! - Base Quantum Lie/Butterfly invariant: $d_{\text{head}} = 4$ (Lie group SO(4)).
//! - Hidden dimension: $d_{\text{model}} = H \times 4$.
//! - Number of heads $H$ scaled by powers of 2 or AVX register width (e.g. 8 heads $\times$ 4 = 32 floats = 4 AVX registers `ymm0..ymm3`).
//! - State memory: strictly $H \times 80$ bytes ($O(1)$ L1D resident forever).

/// Architecture tier defining model scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// H = 2, d_model = 8, d_ff = 12 (State: 160 B)
    Micro,
    /// H = 4, d_model = 16, d_ff = 24 (State: 320 B)
    Standard,
    /// H = 8, d_model = 32, d_ff = 48 (State: 640 B, exact AVX 256-bit ymm0..ymm3 alignment)
    Pro,
    /// H = 16, d_model = 64, d_ff = 96 (State: 1,280 B)
    Ultra,
    /// Custom configuration for research experimentation.
    Custom { n_heads: usize, d_ff: usize },
}

impl Tier {
    /// Number of attention heads $H$.
    pub fn n_heads(&self) -> usize {
        match self {
            Tier::Micro => 2,
            Tier::Standard => 4,
            Tier::Pro => 8,
            Tier::Ultra => 16,
            Tier::Custom { n_heads, .. } => *n_heads,
        }
    }

    /// Feed-forward intermediate dimension $d_{\text{ff}}$.
    pub fn d_ff(&self) -> usize {
        match self {
            Tier::Micro => 12,
            Tier::Standard => 24,
            Tier::Pro => 48,
            Tier::Ultra => 96,
            Tier::Custom { d_ff, .. } => *d_ff,
        }
    }
}

/// Architectural configuration for Scaled SRX and Classical models.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaledConfig {
    /// Architectural tier.
    pub tier: Tier,
    /// Number of heads $H$.
    pub n_heads: usize,
    /// Head dimension $d_{\text{head}} = 4$ (SO(4) quantum invariant).
    pub d_head: usize,
    /// Model hidden dimension $d_{\text{model}} = H \times 4$.
    pub d_model: usize,
    /// Feed-forward hidden dimension $d_{\text{ff}}$.
    pub d_ff: usize,
    /// Number of layers (default: 1).
    pub n_layers: usize,
    /// Vocabulary size (e.g. 256 for pure bytes, 65 for Chinchilla).
    pub vocab_size: usize,
    /// Maximum sequence length for KV-cache bounds (default: 1024).
    pub max_seq_len: usize,
    /// Epsilon for RMSNorm stability (default: 1e-5).
    pub rms_eps: f32,
}

impl ScaledConfig {
    /// Constructs a configuration from a tier and vocabulary size.
    pub fn from_tier(tier: Tier, vocab_size: usize) -> Self {
        let n_heads = tier.n_heads();
        let d_head = 4;
        let d_model = n_heads * d_head;
        let d_ff = tier.d_ff();

        Self {
            tier,
            n_heads,
            d_head,
            d_model,
            d_ff,
            n_layers: 1,
            vocab_size,
            max_seq_len: 1024,
            rms_eps: 1e-5,
        }
    }

    /// Returns the exact number of trainable parameters.
    pub fn param_count(&self) -> usize {
        let emb = self.vocab_size * self.d_model;
        let attn = 4 * self.d_model * self.d_model; // W_q, W_k, W_v, W_o
        let norms = 3 * self.d_model; // pre-attn, pre-ffn, final
        let ffn = 2 * self.d_model * self.d_ff; // W_1, W_2
        (emb + attn + norms + ffn) * self.n_layers
    }

    /// Returns SRX context state memory footprint in bytes: strictly $H \times 80$ bytes.
    pub fn srx_state_bytes(&self) -> usize {
        self.n_heads * 80 * self.n_layers
    }

    /// Returns Classical Transformer KV-cache memory in bytes for sequence length $N$.
    pub fn classic_kv_cache_bytes(&self, seq_len: usize) -> usize {
        2 * self.n_layers * seq_len * self.d_model * std::mem::size_of::<f32>()
    }
}

/// Mathematical scaling calculator for hardware-informed architecture sizing.
pub struct ScalingCalculator;

impl ScalingCalculator {
    /// Computes configuration for a given tier and vocabulary size.
    pub fn compute_config(tier: Tier, vocab_size: usize) -> ScaledConfig {
        ScaledConfig::from_tier(tier, vocab_size)
    }

    /// Returns state memory in bytes for an arbitrary head count $H$: strictly $H \times 80$ bytes.
    pub fn srx_state_bytes(n_heads: usize) -> usize {
        n_heads * 80
    }

    /// Returns Classical KV cache size in bytes for given sequence length and model dimension.
    pub fn classic_kv_bytes(d_model: usize, seq_len: usize) -> usize {
        2 * seq_len * d_model * std::mem::size_of::<f32>()
    }

    /// Classifies memory footprint according to Intel Xeon E5-2650 v2 cache hierarchy.
    pub fn cache_tier(bytes: usize) -> &'static str {
        if bytes <= 32 * 1024 {
            "L1D Cache (<= 32 KB, Resident)"
        } else if bytes <= 256 * 1024 {
            "L2 Cache (32 KB - 256 KB, Spill)"
        } else if bytes <= 20 * 1024 * 1024 {
            "L3 Cache (256 KB - 20 MB, Shared Bus)"
        } else {
            "DRAM Memory Wall (> 20 MB, Memory Thrashing)"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_calculator_tiers() {
        let v = 256;
        let micro = ScalingCalculator::compute_config(Tier::Micro, v);
        assert_eq!(micro.n_heads, 2);
        assert_eq!(micro.d_model, 8);
        assert_eq!(micro.d_ff, 12);
        assert_eq!(micro.srx_state_bytes(), 160);

        let standard = ScalingCalculator::compute_config(Tier::Standard, v);
        assert_eq!(standard.n_heads, 4);
        assert_eq!(standard.d_model, 16);
        assert_eq!(standard.d_ff, 24);
        assert_eq!(standard.srx_state_bytes(), 320);

        let pro = ScalingCalculator::compute_config(Tier::Pro, v);
        assert_eq!(pro.n_heads, 8);
        assert_eq!(pro.d_model, 32);
        assert_eq!(pro.d_ff, 48);
        assert_eq!(pro.srx_state_bytes(), 640);

        let ultra = ScalingCalculator::compute_config(Tier::Ultra, v);
        assert_eq!(ultra.n_heads, 16);
        assert_eq!(ultra.d_model, 64);
        assert_eq!(ultra.d_ff, 96);
        assert_eq!(ultra.srx_state_bytes(), 1280);
    }
}
