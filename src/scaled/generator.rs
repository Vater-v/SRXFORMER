//! # Scaled Generation Engine: Temperature, Top-k, and Repetition Penalty Sampling
//!
//! Eliminates argmax unigram mode collapse ("ооооо") by providing:
//! - **Repetition Penalty ($\rho \approx 1.2$):** Suppresses recently emitted tokens in a sliding window.
//! - **Temperature Scaling ($T \in [0.6, 0.8]$):** Flattens entropy spikes and allows varied continuations.
//! - **Top-$k$ Truncation ($k \in [3, 8]$):** Discards low-probability noise tail.
//! - **Zero-dependency XorShift64 RNG:** Fast deterministic sampling on Intel Xeon Ivy Bridge.

/// Configuration for stochastic autoregressive token sampling.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    /// Temperature scaling parameter ($T > 0$). Lower is sharper, higher is more diverse.
    pub temperature: f32,
    /// Top-k truncation limit. Only the top $k$ tokens with highest logits are considered.
    pub top_k: usize,
    /// Repetition penalty factor $\rho \ge 1.0$.
    pub repetition_penalty: f32,
    /// Window size for repetition penalty.
    pub penalty_window: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 5,
            repetition_penalty: 1.2,
            penalty_window: 16,
        }
    }
}

impl SamplingConfig {
    /// Deterministic greedy decoding configuration.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            repetition_penalty: 1.0,
            penalty_window: 0,
        }
    }

    /// Creative balanced sampling configuration.
    pub fn balanced() -> Self {
        Self::default()
    }
}

/// A fast 64-bit XorShift pseudo-random number generator (`std`-only).
#[derive(Debug, Clone)]
pub struct FastRng {
    state: u64,
}

impl FastRng {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x853c49e6748fea9b } else { seed };
        Self { state }
    }

    /// Generates a pseudo-random u64.
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generates a pseudo-random float uniformly distributed in $[0.0, 1.0)$.
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16777216.0
    }
}

/// Applies repetition penalty to logits in-place for tokens present in `recent_tokens`.
pub fn apply_repetition_penalty(
    logits: &mut [f32],
    recent_tokens: &[usize],
    penalty: f32,
    window: usize,
) {
    if penalty <= 1.0 || recent_tokens.is_empty() || window == 0 {
        return;
    }

    let start = recent_tokens.len().saturating_sub(window);
    let slice = &recent_tokens[start..];

    for &tok in slice {
        if tok < logits.len() {
            if logits[tok] > 0.0 {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

/// Samples the next token index from logits given sampling configuration and RNG.
pub fn sample_token(
    logits: &mut [f32],
    recent_tokens: &[usize],
    cfg: &SamplingConfig,
    rng: &mut FastRng,
) -> usize {
    let vocab_size = logits.len();
    if vocab_size == 0 {
        return 0;
    }

    // 1. Apply repetition penalty
    apply_repetition_penalty(logits, recent_tokens, cfg.repetition_penalty, cfg.penalty_window);

    // 2. Greedy shortcut
    if cfg.temperature <= 1e-4 || cfg.top_k <= 1 {
        let mut best_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if l > max_val {
                max_val = l;
                best_idx = i;
            }
        }
        return best_idx;
    }

    // 3. Apply temperature scaling
    let inv_t = 1.0 / cfg.temperature;
    for l in logits.iter_mut() {
        *l *= inv_t;
    }

    // 4. Find top-k threshold
    let k = cfg.top_k.clamp(1, vocab_size);
    // Find top-k values using partial sorting / scanning
    let mut top_indices: Vec<usize> = (0..vocab_size).collect();
    top_indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    top_indices.truncate(k);

    // 5. Compute softmax over top-k
    let max_logit = logits[top_indices[0]];
    let mut sum_exp = 0.0f32;
    let mut probs: Vec<f32> = Vec::with_capacity(k);

    for &idx in &top_indices {
        let exp_val = (logits[idx] - max_logit).exp();
        probs.push(exp_val);
        sum_exp += exp_val;
    }

    let inv_sum = 1.0 / sum_exp.max(1e-12);
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    // 6. Cumulative probability sampling
    let r = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return top_indices[i];
        }
    }

    // Fallback to top-1 if rounding edge case
    top_indices[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![2.0, -1.0, 0.5];
        let recent = vec![0, 1];
        apply_repetition_penalty(&mut logits, &recent, 2.0, 10);

        assert!((logits[0] - 1.0).abs() < 1e-5, "Positive logit should be divided by penalty");
        assert!((logits[1] - (-2.0)).abs() < 1e-5, "Negative logit should be multiplied by penalty");
        assert!((logits[2] - 0.5).abs() < 1e-5, "Unseen token logit should remain unchanged");
    }

    #[test]
    fn test_greedy_sampling() {
        let mut logits = vec![0.1, 0.9, 0.2];
        let mut rng = FastRng::new(42);
        let tok = sample_token(&mut logits, &[], &SamplingConfig::greedy(), &mut rng);
        assert_eq!(tok, 1);
    }

    #[test]
    fn test_top_k_sampling_within_bounds() {
        let logits = vec![10.0, 9.5, 0.1, -5.0];
        let mut rng = FastRng::new(12345);
        let cfg = SamplingConfig {
            temperature: 0.8,
            top_k: 2,
            repetition_penalty: 1.0,
            penalty_window: 0,
        };

        for _ in 0..50 {
            let mut l_copy = logits.clone();
            let tok = sample_token(&mut l_copy, &[], &cfg, &mut rng);
            assert!(tok == 0 || tok == 1, "Top-2 must only select token 0 or 1, got {}", tok);
        }
    }
}
