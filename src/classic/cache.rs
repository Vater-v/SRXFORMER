use super::config::TransformerConfig;

/// Pre-allocated contiguous Key-Value Cache for autoregressive decoding.
/// Eliminates heap allocations during decoding and enables O(1) step computation.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Contiguous buffer for Keys: [n_layers, max_seq_len, d_model]
    pub k: Vec<f32>,
    /// Contiguous buffer for Values: [n_layers, max_seq_len, d_model]
    pub v: Vec<f32>,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Maximum sequence length supported by this cache
    pub max_seq_len: usize,
    /// Model hidden dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dimension per attention head (d_model / n_heads)
    pub head_dim: usize,
    /// Number of tokens currently cached
    pub current_len: usize,
}

impl KvCache {
    /// Allocates and pre-zeros the KV cache according to configuration.
    pub fn new(config: &TransformerConfig) -> Self {
        let total_floats = config.n_layers * config.max_seq_len * config.d_model;
        Self {
            k: vec![0.0; total_floats],
            v: vec![0.0; total_floats],
            n_layers: config.n_layers,
            max_seq_len: config.max_seq_len,
            d_model: config.d_model,
            n_heads: config.n_heads,
            head_dim: config.head_dim(),
            current_len: 0,
        }
    }

    /// Resets cached length back to 0 without deallocating or modifying heap buffers.
    #[inline]
    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    /// Returns the number of tokens currently stored in the cache.
    #[inline]
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// Checks if cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Appends key and value vectors for a single token at position `pos` for a given layer.
    #[inline]
    pub fn append_layer(&mut self, layer: usize, pos: usize, k_token: &[f32], v_token: &[f32]) {
        debug_assert!(layer < self.n_layers);
        debug_assert!(pos < self.max_seq_len);
        debug_assert_eq!(k_token.len(), self.d_model);
        debug_assert_eq!(v_token.len(), self.d_model);

        let offset = (layer * self.max_seq_len + pos) * self.d_model;
        self.k[offset..offset + self.d_model].copy_from_slice(k_token);
        self.v[offset..offset + self.d_model].copy_from_slice(v_token);
    }

    /// Retrieves a contiguous slice of keys for a specific head at position `pos` in layer `layer`.
    #[inline]
    pub fn key_head(&self, layer: usize, head: usize, pos: usize) -> &[f32] {
        debug_assert!(layer < self.n_layers);
        debug_assert!(head < self.n_heads);
        debug_assert!(pos < self.max_seq_len);

        let offset = (layer * self.max_seq_len + pos) * self.d_model + head * self.head_dim;
        &self.k[offset..offset + self.head_dim]
    }

    /// Retrieves a contiguous slice of values for a specific head at position `pos` in layer `layer`.
    #[inline]
    pub fn val_head(&self, layer: usize, head: usize, pos: usize) -> &[f32] {
        debug_assert!(layer < self.n_layers);
        debug_assert!(head < self.n_heads);
        debug_assert!(pos < self.max_seq_len);

        let offset = (layer * self.max_seq_len + pos) * self.d_model + head * self.head_dim;
        &self.v[offset..offset + self.head_dim]
    }
}

/// Scratchpad buffers specifically dedicated to Multi-Head Attention operations.
#[derive(Debug, Clone)]
pub struct AttentionWorkspace {
    pub q: Vec<f32>,            // [max_seq_len * d_model]
    pub k: Vec<f32>,            // [max_seq_len * d_model]
    pub v: Vec<f32>,            // [max_seq_len * d_model]
    pub attn_scores: Vec<f32>,  // [n_heads * max_seq_len * max_seq_len]
    pub attn_out: Vec<f32>,     // [max_seq_len * d_model]

    pub step_q: Vec<f32>,            // [d_model]
    pub step_k: Vec<f32>,            // [d_model]
    pub step_v: Vec<f32>,            // [d_model]
    pub step_attn_scores: Vec<f32>,  // [n_heads * max_seq_len]
    pub step_attn_out: Vec<f32>,     // [d_model]
}

impl AttentionWorkspace {
    pub fn new(config: &TransformerConfig) -> Self {
        let max_seq_len = config.max_seq_len;
        let d_model = config.d_model;
        let n_heads = config.n_heads;
        let seq_scores_len = n_heads * max_seq_len.min(1024) * max_seq_len.min(1024);

        Self {
            q: vec![0.0; max_seq_len * d_model],
            k: vec![0.0; max_seq_len * d_model],
            v: vec![0.0; max_seq_len * d_model],
            attn_scores: vec![0.0; seq_scores_len],
            attn_out: vec![0.0; max_seq_len * d_model],

            step_q: vec![0.0; d_model],
            step_k: vec![0.0; d_model],
            step_v: vec![0.0; d_model],
            step_attn_scores: vec![0.0; n_heads * max_seq_len],
            step_attn_out: vec![0.0; d_model],
        }
    }
}

/// Scratchpad buffers dedicated to Feed-Forward / MLP operations.
#[derive(Debug, Clone)]
pub struct MlpWorkspace {
    pub hidden: Vec<f32>,      // [max_seq_len * d_ff]
    pub step_hidden: Vec<f32>, // [d_ff]
}

impl MlpWorkspace {
    pub fn new(config: &TransformerConfig) -> Self {
        Self {
            hidden: vec![0.0; config.max_seq_len * config.d_ff],
            step_hidden: vec![0.0; config.d_ff],
        }
    }
}

/// Pre-allocated scratchpad buffers for forward and step inference passes.
/// Pre-allocated once up-front to guarantee zero dynamic allocations in the hot path.
#[derive(Debug, Clone)]
pub struct InferenceWorkspace {
    // Sequence-level buffers (used in forward pass)
    pub x: Vec<f32>,         // [max_seq_len * d_model]
    pub residual: Vec<f32>,  // [max_seq_len * d_model]
    pub norm_out: Vec<f32>,  // [max_seq_len * d_model]
    pub logits: Vec<f32>,    // [max_seq_len * vocab_size]

    // Step-level buffers (used in single-token autoregressive step)
    pub step_x: Vec<f32>,         // [d_model]
    pub step_residual: Vec<f32>,  // [d_model]
    pub step_norm_out: Vec<f32>,  // [d_model]
    pub step_logits: Vec<f32>,    // [vocab_size]

    // Sublayer scratchpads
    pub attn: AttentionWorkspace,
    pub mlp: MlpWorkspace,

    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
}

impl InferenceWorkspace {
    /// Creates and pre-allocates all memory buffers required for inference.
    pub fn new(config: &TransformerConfig) -> Self {
        let max_seq_len = config.max_seq_len;
        let d_model = config.d_model;
        let n_heads = config.n_heads;
        let d_ff = config.d_ff;
        let vocab_size = config.vocab_size;

        Self {
            x: vec![0.0; max_seq_len * d_model],
            residual: vec![0.0; max_seq_len * d_model],
            norm_out: vec![0.0; max_seq_len * d_model],
            logits: vec![0.0; max_seq_len * vocab_size],

            step_x: vec![0.0; d_model],
            step_residual: vec![0.0; d_model],
            step_norm_out: vec![0.0; d_model],
            step_logits: vec![0.0; vocab_size],

            attn: AttentionWorkspace::new(config),
            mlp: MlpWorkspace::new(config),

            max_seq_len,
            d_model,
            n_heads,
            d_ff,
            vocab_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_append_and_retrieve() {
        let config = TransformerConfig::default_512();
        let mut cache = KvCache::new(&config);
        assert_eq!(cache.len(), 0);

        let k_tok = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v_tok = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        cache.append_layer(0, 0, &k_tok, &v_tok);
        cache.current_len = 1;

        // Head 0 has 4 floats: [1, 2, 3, 4]
        let h0_k = cache.key_head(0, 0, 0);
        assert_eq!(h0_k, &[1.0, 2.0, 3.0, 4.0]);

        // Head 1 has 4 floats: [5, 6, 7, 8]
        let h1_k = cache.key_head(0, 1, 0);
        assert_eq!(h1_k, &[5.0, 6.0, 7.0, 8.0]);

        let h0_v = cache.val_head(0, 0, 0);
        assert_eq!(h0_v, &[8.0, 7.0, 6.0, 5.0]);

        cache.reset();
        assert_eq!(cache.len(), 0);
    }
}
