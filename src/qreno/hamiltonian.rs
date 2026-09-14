//! # Quantum Renormalization & Operator-Field Tokenizer (Q-RENO): Chemical Hamiltonian & Covalent Bonds
//!
//! Models inter-byte interactions via tight-binding overlap integrals:
//! - Overlap resonance integral: $t_{i, i+1} = \text{sigmoid}(W_{\text{bond}} \cdot (c(b_i) \odot c(b_{i+1})) + b_{\text{bond}}) \in (0, 1)$
//! - Probability current: $J_{i, i+1} = t_{i, i+1} \cdot \sin(\omega(b_{i+1}) - \omega(b_i))$
//! - Dynamic cluster segmentation based on bond strength, delimiter status, and `max_cluster_len`.

use crate::qreno::field::QrenoField;

/// Parameters governing covalent bond formation between adjacent bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct BondParams {
    /// Field dimension $d_f$.
    pub field_dim: usize,
    /// Bond weight vector $W_{\text{bond}} \in \mathbb{R}^{d_f}$.
    pub w_bond: Vec<f32>,
    /// Bond bias scalar $b_{\text{bond}} \in \mathbb{R}$.
    pub b_bond: f32,
    /// Threshold below which covalent bond breaks ($t_{i, i+1} < \text{threshold}$). Default: 0.5.
    pub threshold: f32,
}

impl BondParams {
    /// Creates bond parameters with physically tuned defaults.
    pub fn new(field_dim: usize) -> Self {
        Self {
            field_dim,
            // Uniform positive weights to reward aligned charge features
            w_bond: vec![1.2; field_dim],
            b_bond: 1.5,
            threshold: 0.5,
        }
    }

    /// Computes the overlap resonance integral $t_{i, i+1} \in (0, 1)$.
    #[inline(always)]
    pub fn compute_bond(&self, c1: &[f32], c2: &[f32]) -> f32 {
        debug_assert_eq!(c1.len(), self.field_dim);
        debug_assert_eq!(c2.len(), self.field_dim);

        let mut z = self.b_bond;
        for k in 0..self.field_dim {
            z += self.w_bond[k] * c1[k] * c2[k];
        }
        // Sigmoid activation
        1.0 / (1.0 + (-z).exp())
    }

    /// Computes probability current $J_{i, i+1} = t_{i, i+1} \cdot \sin(\omega(b_{i+1}) - \omega(b_i))$.
    #[inline(always)]
    pub fn compute_current(&self, t: f32, omega1: f32, omega2: f32) -> f32 {
        t * (omega2 - omega1).sin()
    }

    /// Total trainable parameter count for bond parameters ($d_f + 1$).
    pub fn param_count(&self) -> usize {
        self.w_bond.len() + 1
    }
}

/// A contiguous cluster of bytes forming a quasi-particle quantum state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cluster {
    /// Starting byte index in the source buffer.
    pub start: usize,
    /// Length of the cluster in bytes ($1 \le L_m \le \text{max\_cluster\_len}$).
    pub len: usize,
    /// Whether this cluster is a delimiter (whitespace, punctuation, etc.).
    pub is_delimiter: bool,
}

/// Checks if a byte is a semantic delimiter / van der Waals boundary.
#[inline(always)]
pub fn is_delimiter_byte(b: u8) -> bool {
    matches!(
        b,
        b' ' | b'\t' | b'\n' | b'\r' | b',' | b'.' | b'!' | b'?' | b';' | b':' | b'(' | b')'
            | b'[' | b']' | b'{' | b'}' | b'"' | b'\'' | b'/' | b'\\' | b'\0'
    )
}

/// Partitions a byte sequence into bound clusters according to bond threshold and delimiters.
pub fn partition_into_clusters(
    bytes: &[u8],
    field: &QrenoField,
    bond: &BondParams,
    max_cluster_len: usize,
) -> Vec<Cluster> {
    if bytes.is_empty() {
        return Vec::new();
    }

    let max_len = max_cluster_len.clamp(1, 16);
    let mut clusters = Vec::new();
    let n = bytes.len();
    let mut i = 0;

    while i < n {
        let b_i = bytes[i];
        if is_delimiter_byte(b_i) {
            // Group consecutive identical whitespace/delimiters up to max_len
            let start = i;
            let mut len = 1;
            while (start + len) < n && len < max_len && bytes[start + len] == b_i {
                len += 1;
            }
            clusters.push(Cluster {
                start,
                len,
                is_delimiter: true,
            });
            i += len;
            continue;
        }

        // Informative word/morpheme cluster
        let start = i;
        let mut len = 1;
        while (start + len) < n && len < max_len {
            let next_byte = bytes[start + len];
            if is_delimiter_byte(next_byte) {
                break;
            }

            let curr_byte = bytes[start + len - 1];
            let c_curr = field.charge(curr_byte);
            let c_next = field.charge(next_byte);
            let t = bond.compute_bond(c_curr, c_next);

            if t < bond.threshold {
                // Van der Waals gap: bond broken
                break;
            }

            len += 1;
        }

        clusters.push(Cluster {
            start,
            len,
            is_delimiter: false,
        });
        i += len;
    }

    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bond_computation_and_current() {
        let bond = BondParams::new(8);
        let c1 = vec![0.5; 8];
        let c2 = vec![0.5; 8];

        let t = bond.compute_bond(&c1, &c2);
        assert!(t > 0.5, "Strong positive overlap should exceed threshold");

        let current = bond.compute_current(t, 0.0, std::f32::consts::FRAC_PI_2);
        assert!((current - t).abs() < 1e-5);
    }

    #[test]
    fn test_cluster_partitioning_whitespace() {
        let field = QrenoField::new(8);
        let bond = BondParams::new(8);
        let text = "hello world";
        let clusters = partition_into_clusters(text.as_bytes(), &field, &bond, 16);

        assert_eq!(clusters.len(), 3);
        assert_eq!(&text.as_bytes()[clusters[0].start..clusters[0].start + clusters[0].len], b"hello");
        assert_eq!(clusters[0].is_delimiter, false);

        assert_eq!(&text.as_bytes()[clusters[1].start..clusters[1].start + clusters[1].len], b" ");
        assert_eq!(clusters[1].is_delimiter, true);

        assert_eq!(&text.as_bytes()[clusters[2].start..clusters[2].start + clusters[2].len], b"world");
        assert_eq!(clusters[2].is_delimiter, false);
    }

    #[test]
    fn test_cluster_max_len_cap() {
        let field = QrenoField::new(8);
        let bond = BondParams::new(8);
        let text = "abcdefghijklmnopqrstuvwxyz"; // 26 chars
        let clusters = partition_into_clusters(text.as_bytes(), &field, &bond, 16);

        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].len, 16);
        assert_eq!(clusters[1].len, 10);
    }
}
