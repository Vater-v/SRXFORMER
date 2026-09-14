/// Deterministic PRNG based on SplitMix64 / XorShift64.
/// Used for weight initialization and categorical sampling with zero dependencies.
#[derive(Debug, Clone)]
pub struct FastRng {
    state: u64,
}

impl FastRng {
    /// Creates a new `FastRng` with a 64-bit seed.
    /// If seed is 0, a non-zero default is used to prevent degenerate state.
    #[inline]
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x853c49e6748fea9b
        } else {
            seed
        };
        Self { state }
    }

    /// Generates next 64-bit pseudo-random unsigned integer.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // SplitMix64 step
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generates a uniform pseudo-random f32 in [0.0, 1.0).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // 24 bits of precision for f32 mantissa
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 * (1.0 / 16777216.0)
    }

    /// Generates a uniform pseudo-random f32 in [low, high).
    #[inline]
    pub fn gen_range(&mut self, low: f32, high: f32) -> f32 {
        low + self.next_f32() * (high - low)
    }

    /// Generates a normally distributed f32 ~ N(mean, std^2) using Box-Muller transform.
    pub fn gen_normal(&mut self, mean: f32, std: f32) -> f32 {
        let mut u1 = self.next_f32();
        while u1 <= 1e-7 {
            u1 = self.next_f32();
        }
        let u2 = self.next_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        let z = radius * theta.cos();
        mean + std * z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_determinism() {
        let mut rng1 = FastRng::new(12345);
        let mut rng2 = FastRng::new(12345);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_range() {
        let mut rng = FastRng::new(42);
        for _ in 0..1000 {
            let val = rng.gen_range(-2.5, 3.5);
            assert!(val >= -2.5 && val < 3.5);
        }
    }
}
