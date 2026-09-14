//! # Quantum Renormalization & Operator-Field Tokenizer (Q-RENO): Basis Wave Field
//!
//! Provides the fundamental quantum field basis:
//! Each byte $b \in [0, 255]$ possesses:
//! - Chemical potential $\epsilon(b) \in \mathbb{R}$
//! - Phase frequency $\omega(b) \in [0, 2\pi)$
//! - Continuous charge vector $c(b) \in \mathbb{R}^{d_f}$ (default $d_f = 8$)

use std::f32::consts::PI;

/// Continuous quantum field parameters for all 256 possible bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct QrenoField {
    /// Charge vector dimension $d_f$ (default: 8).
    pub field_dim: usize,
    /// Continuous charge vectors for all 256 bytes, flattened: shape `[256, field_dim]`.
    pub charges: Vec<f32>,
    /// Chemical potential $\epsilon(b)$ for each byte $b \in [0, 255]$.
    pub epsilon: [f32; 256],
    /// Phase frequency $\omega(b)$ for each byte $b \in [0, 255]$.
    pub omega: [f32; 256],
}

impl QrenoField {
    /// Creates a new Q-RENO basis field with default harmonic-physical initialization.
    pub fn new(field_dim: usize) -> Self {
        assert!(field_dim > 0, "field_dim must be positive");
        let mut field = Self {
            field_dim,
            charges: vec![0.0; 256 * field_dim],
            epsilon: [0.0; 256],
            omega: [0.0; 256],
        };

        // Initialize physical basis:
        // 1. Phase frequency omega(b) uniformly distributed in [0, 2pi)
        for b in 0..256 {
            field.omega[b] = (2.0 * PI * (b as f32)) / 256.0;
        }

        // 2. Chemical potential epsilon(b): negative base potential favoring bound cluster states
        for b in 0..256 {
            field.epsilon[b] = -0.5 + 0.1 * ((2.0 * PI * (b as f32) / 256.0).sin());
        }

        // 3. Continuous charge vector c(b): deterministic distinct pseudo-random unit vectors
        for b in 0..256 {
            let offset = b * field_dim;
            let mut rng = (b as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xBF58476D1CE4E5B9;
            let mut norm_sq = 0.0f32;
            for k in 0..field_dim {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let val = ((rng >> 40) as f32) / 16777216.0 - 0.5;
                field.charges[offset + k] = val;
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt().max(1e-7);
            // In UTF-8, multi-byte lead bytes (0xC0..=0xFF) act as structural transport/gauge carriers
            // without lexical payload. Renormalize their carrier amplitude so semantic bytes dominate.
            let scale = if b >= 0xC0 { 0.12 } else { 1.0 };
            for k in 0..field_dim {
                field.charges[offset + k] = (field.charges[offset + k] / norm) * scale;
            }
        }

        field
    }

    /// Creates a field with seeded pseudo-random Gaussian perturbations for training.
    pub fn with_seed(field_dim: usize, seed: u64) -> Self {
        let mut field = Self::new(field_dim);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let mut next_f32 = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            ((rng_state >> 40) as f32) / 16777216.0 - 0.5
        };

        for b in 0..256 {
            field.epsilon[b] += 0.05 * next_f32();
            field.omega[b] = (field.omega[b] + 0.05 * next_f32()).rem_euclid(2.0 * PI);

            let offset = b * field_dim;
            let mut norm_sq = 0.0f32;
            for k in 0..field_dim {
                field.charges[offset + k] += 0.1 * next_f32();
                norm_sq += field.charges[offset + k] * field.charges[offset + k];
            }
            let norm = norm_sq.sqrt().max(1e-7);
            for k in 0..field_dim {
                field.charges[offset + k] /= norm;
            }
        }

        field
    }

    /// Returns a slice to the continuous charge vector $c(b) \in \mathbb{R}^{d_f}$.
    #[inline(always)]
    pub fn charge(&self, byte: u8) -> &[f32] {
        let offset = (byte as usize) * self.field_dim;
        &self.charges[offset..offset + self.field_dim]
    }

    /// Returns a mutable slice to the continuous charge vector $c(b) \in \mathbb{R}^{d_f}$.
    #[inline(always)]
    pub fn charge_mut(&mut self, byte: u8) -> &mut [f32] {
        let offset = (byte as usize) * self.field_dim;
        &mut self.charges[offset..offset + self.field_dim]
    }

    /// Returns the chemical potential $\epsilon(b)$.
    #[inline(always)]
    pub fn epsilon(&self, byte: u8) -> f32 {
        self.epsilon[byte as usize]
    }

    /// Returns the phase frequency $\omega(b)$.
    #[inline(always)]
    pub fn omega(&self, byte: u8) -> f32 {
        self.omega[byte as usize]
    }

    /// Returns total trainable parameter count in this field.
    pub fn param_count(&self) -> usize {
        self.charges.len() + self.epsilon.len() + self.omega.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_initialization_and_charges() {
        let field = QrenoField::new(8);
        assert_eq!(field.field_dim, 8);
        assert_eq!(field.charges.len(), 256 * 8);

        // Verify norm of charge vectors (1.0 for payload bytes, 0.12 for lead bytes)
        for b in 0..256 {
            let c = field.charge(b as u8);
            assert_eq!(c.len(), 8);
            let norm_sq: f32 = c.iter().map(|&x| x * x).sum();
            let expected_norm = if b >= 0xC0 { 0.12 } else { 1.0 };
            assert!((norm_sq.sqrt() - expected_norm).abs() < 1e-5);
        }

        // Verify omega in [0, 2pi)
        for b in 0..256 {
            let w = field.omega(b as u8);
            assert!(w >= 0.0 && w < 2.0 * PI);
        }
    }
}
