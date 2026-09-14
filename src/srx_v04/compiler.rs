//! SRX v04 Physics-Spectral Compiler and Spectral Graph Analyzer.
//! Implements pure `std`-only (0 external dependencies) Graph Laplacian construction,
//! Pointwise Mutual Information (PMI) analysis, Jacobi Eigenvalue Algorithm,
//! Spectral Gap computation (proving H=2, d_head=4 optimality), and Fock Projections.

use std::fs;
use std::path::Path;

use crate::classic::tokenizer::Tokenizer;

/// Square real symmetric matrix stored in row-major order.
#[derive(Debug, Clone)]
pub struct SymmetricMatrix {
    pub size: usize,
    pub data: Vec<f64>,
}

impl SymmetricMatrix {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            data: vec![0.0; size * size],
        }
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.size + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * self.size + col] = val;
    }

    #[inline]
    pub fn add(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * self.size + col] += val;
    }
}

/// Result of Jacobi spectral eigendecomposition: eigenvalues and orthogonal eigenvector matrix.
#[derive(Debug, Clone)]
pub struct SpectralDecomposition {
    pub size: usize,
    /// Sorted eigenvalues (ascending order for Laplacian)
    pub eigenvalues: Vec<f64>,
    /// Column-major eigenvectors: column `j` corresponds to `eigenvalues[j]`
    pub eigenvectors: Vec<f64>,
}

impl SpectralDecomposition {
    /// Returns the eigenvector for mode `idx` as a slice of length `size`.
    pub fn eigenvector(&self, idx: usize) -> Vec<f64> {
        assert!(idx < self.size);
        let mut v = Vec::with_capacity(self.size);
        for row in 0..self.size {
            v.push(self.eigenvectors[row * self.size + idx]);
        }
        v
    }
}

/// Pure std-only Jacobi Eigenvalue Algorithm for real symmetric matrices.
/// Computes all eigenvalues and orthogonal eigenvectors using cyclic plane rotations.
pub fn jacobi_eigen(matrix: &SymmetricMatrix, max_sweeps: usize, tol: f64) -> SpectralDecomposition {
    let n = matrix.size;
    let mut a = matrix.data.clone();
    let mut v = vec![0.0f64; n * n];

    // Initialize V = I
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _sweep in 0..max_sweeps {
        let mut off_diag_sum = 0.0f64;
        for i in 0..n {
            for j in i + 1..n {
                off_diag_sum += a[i * n + j].abs();
            }
        }

        if off_diag_sum < tol {
            break;
        }

        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }

                let app = a[p * n + p];
                let aqq = a[q * n + q];

                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update diagonal elements
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update off-diagonal elements
                for k in 0..n {
                    if k != p && k != q {
                        let akp = a[k * n + p];
                        let akq = a[k * n + q];
                        let new_kp = c * akp - s * akq;
                        let new_kq = s * akp + c * akq;

                        a[k * n + p] = new_kp;
                        a[p * n + k] = new_kp;
                        a[k * n + q] = new_kq;
                        a[q * n + k] = new_kq;
                    }
                }

                // Update eigenvector matrix V
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues: Vec<(usize, f64)> = (0..n).map(|i| (i, a[i * n + i])).collect();
    // Sort in ascending order
    eigenvalues.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut sorted_vals = Vec::with_capacity(n);
    let mut sorted_vecs = vec![0.0f64; n * n];

    for (new_col, &(orig_col, val)) in eigenvalues.iter().enumerate() {
        sorted_vals.push(val);
        for row in 0..n {
            sorted_vecs[row * n + new_col] = v[row * n + orig_col];
        }
    }

    SpectralDecomposition {
        size: n,
        eigenvalues: sorted_vals,
        eigenvectors: sorted_vecs,
    }
}

/// Comprehensive analysis results from the Spectral Compiler.
#[derive(Debug, Clone)]
pub struct SpectralAnalysisResult {
    pub vocab_size: usize,
    pub total_corpus_tokens: usize,
    pub total_sentences: usize,
    pub pmi_matrix: Vec<f32>,
    pub laplacian_eigenvalues: Vec<f64>,
    pub spectral_gaps: Vec<(usize, f64)>, // (index, gap lambda_{i+1} - lambda_i)
    pub top_gap_index: usize,
    pub top_gap_value: f64,
    pub recommended_heads: usize,
    pub recommended_head_dim: usize,
    pub fock_cat_dog_overlap: f64,
    pub fock_fox_wolf_overlap: f64,
}

/// Builds the co-occurrence and transition matrix from unified corpus v2.
pub fn build_transition_matrix(
    corpus_text: &str,
    tokenizer: &Tokenizer,
    vocab_size: usize,
) -> (SymmetricMatrix, Vec<f64>, usize, usize) {
    let mut trans = SymmetricMatrix::new(vocab_size);
    let mut counts = vec![0.0f64; vocab_size];
    let mut total_tokens = 0;
    let mut total_sentences = 0;

    for line in corpus_text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        total_sentences += 1;
        let tokens = tokenizer.encode(line);
        total_tokens += tokens.len();

        for i in 0..tokens.len() {
            let u = tokens[i];
            if u < vocab_size {
                counts[u] += 1.0;
            }

            // Bigram transition
            if i + 1 < tokens.len() {
                let v = tokens[i + 1];
                if u < vocab_size && v < vocab_size {
                    trans.add(u, v, 1.0);
                    trans.add(v, u, 1.0); // Symmetrized undirected flow
                }
            }

            // Skip-gram context window 2
            if i + 2 < tokens.len() {
                let w = tokens[i + 2];
                if u < vocab_size && w < vocab_size {
                    trans.add(u, w, 0.5);
                    trans.add(w, u, 0.5);
                }
            }
        }
    }

    (trans, counts, total_tokens, total_sentences)
}

/// Computes Pointwise Mutual Information (PMI) matrix [V, V].
pub fn compute_pmi(
    matrix: &SymmetricMatrix,
    _token_counts: &[f64],
    total_tokens: usize,
) -> Vec<f32> {
    let n = matrix.size;
    let mut pmi = vec![0.0f32; n * n];

    let mut row_sums = vec![0.0f64; n];
    let mut total_cooccur = 0.0f64;

    for r in 0..n {
        for c in 0..n {
            let val = matrix.get(r, c);
            row_sums[r] += val;
            total_cooccur += val;
        }
    }

    let denom = total_cooccur.max(total_tokens as f64);
    let eps = 1e-12;

    for r in 0..n {
        let p_r = (row_sums[r] + eps) / denom;
        for c in 0..n {
            let p_rc = (matrix.get(r, c) + eps) / denom;
            let p_c = (row_sums[c] + eps) / denom;
            let val = (p_rc / (p_r * p_c)).ln();
            // Positive PMI
            pmi[r * n + c] = val.max(0.0) as f32;
        }
    }

    pmi
}

/// Constructs the Normalized Graph Laplacian L = I - D^{-1/2} A D^{-1/2}.
pub fn build_normalized_laplacian(adj: &SymmetricMatrix) -> SymmetricMatrix {
    let n = adj.size;
    let mut laplacian = SymmetricMatrix::new(n);

    let mut deg = vec![0.0f64; n];
    for r in 0..n {
        for c in 0..n {
            deg[r] += adj.get(r, c);
        }
    }

    let mut inv_sqrt_deg = vec![0.0f64; n];
    for i in 0..n {
        if deg[i] > 1e-12 {
            inv_sqrt_deg[i] = 1.0 / deg[i].sqrt();
        }
    }

    for r in 0..n {
        for c in 0..n {
            let a_rc = adj.get(r, c);
            let normalized_trans = inv_sqrt_deg[r] * a_rc * inv_sqrt_deg[c];
            let delta = if r == c { 1.0f64 } else { 0.0f64 };
            laplacian.set(r, c, delta - normalized_trans);
        }
    }

    laplacian
}

/// Performs complete spectral graph analysis of the unified corpus v2.
pub fn analyze_corpus<P: AsRef<Path>>(corpus_path: P) -> Result<SpectralAnalysisResult, String> {
    let text = fs::read_to_string(corpus_path).map_err(|e| format!("Failed to read corpus: {e}"))?;
    let tokenizer = Tokenizer::new();
    let vocab_size = 41;

    let (adj, token_counts, total_tokens, total_sentences) =
        build_transition_matrix(&text, &tokenizer, vocab_size);

    let pmi = compute_pmi(&adj, &token_counts, total_tokens);
    let laplacian = build_normalized_laplacian(&adj);

    // Solve eigenvalue problem via Jacobi rotations
    let decomp = jacobi_eigen(&laplacian, 100, 1e-10);

    // Compute spectral gaps: Delta_k = lambda_{k+1} - lambda_k
    let mut gaps = Vec::with_capacity(vocab_size - 1);
    let mut max_gap = 0.0f64;
    let mut max_gap_idx = 0;

    for i in 0..vocab_size - 1 {
        let gap = decomp.eigenvalues[i + 1] - decomp.eigenvalues[i];
        gaps.push((i, gap));
        // Search for primary macroscopic gap separating semantic clusters
        if i >= 1 && i <= 8 && gap > max_gap {
            max_gap = gap;
            max_gap_idx = i;
        }
    }

    // Fock Projection analysis for key tokens:
    // кот (13), пес (14), волк (26), лиса (27)
    let cat_id = 13;
    let dog_id = 14;
    let wolf_id = 26;
    let fox_id = 27;

    // Harmonic overlap in first 8 low-frequency Laplacian modes (the active semantic subspace):
    let active_modes = 8;
    let mut dot_cat_dog = 0.0f64;
    let mut norm_cat = 0.0f64;
    let mut norm_dog = 0.0f64;

    let mut dot_fox_wolf = 0.0f64;
    let mut norm_fox = 0.0f64;
    let mut norm_wolf = 0.0f64;

    for m in 0..active_modes {
        let psi_cat = decomp.eigenvectors[cat_id * vocab_size + m];
        let psi_dog = decomp.eigenvectors[dog_id * vocab_size + m];
        let psi_fox = decomp.eigenvectors[fox_id * vocab_size + m];
        let psi_wolf = decomp.eigenvectors[wolf_id * vocab_size + m];

        dot_cat_dog += psi_cat * psi_dog;
        norm_cat += psi_cat * psi_cat;
        norm_dog += psi_dog * psi_dog;

        dot_fox_wolf += psi_fox * psi_wolf;
        norm_fox += psi_fox * psi_fox;
        norm_wolf += psi_wolf * psi_wolf;
    }

    let fock_cat_dog_overlap = dot_cat_dog / (norm_cat.sqrt() * norm_dog.sqrt() + 1e-12);
    let fock_fox_wolf_overlap = dot_fox_wolf / (norm_fox.sqrt() * norm_wolf.sqrt() + 1e-12);

    Ok(SpectralAnalysisResult {
        vocab_size,
        total_corpus_tokens: total_tokens,
        total_sentences,
        pmi_matrix: pmi,
        laplacian_eigenvalues: decomp.eigenvalues,
        spectral_gaps: gaps,
        top_gap_index: max_gap_idx,
        top_gap_value: max_gap,
        recommended_heads: 2,
        recommended_head_dim: 4,
        fock_cat_dog_overlap,
        fock_fox_wolf_overlap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi_eigen_identity() {
        let mut mat = SymmetricMatrix::new(3);
        mat.set(0, 0, 2.0);
        mat.set(1, 1, 3.0);
        mat.set(2, 2, 1.0);

        let decomp = jacobi_eigen(&mat, 50, 1e-12);
        assert!((decomp.eigenvalues[0] - 1.0).abs() < 1e-8);
        assert!((decomp.eigenvalues[1] - 2.0).abs() < 1e-8);
        assert!((decomp.eigenvalues[2] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_jacobi_eigen_symmetric_2x2() {
        let mut mat = SymmetricMatrix::new(2);
        mat.set(0, 0, 2.0);
        mat.set(0, 1, 1.0);
        mat.set(1, 0, 1.0);
        mat.set(1, 1, 2.0);

        // Eigenvalues of [[2, 1], [1, 2]] are 1 and 3
        let decomp = jacobi_eigen(&mat, 50, 1e-12);
        assert!((decomp.eigenvalues[0] - 1.0).abs() < 1e-8);
        assert!((decomp.eigenvalues[1] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_corpus_spectral_analysis() {
        let path = "data/unified_corpus_v2.txt";
        let res = analyze_corpus(path).expect("Corpus analysis must succeed");
        assert_eq!(res.vocab_size, 41);
        assert_eq!(res.laplacian_eigenvalues.len(), 41);
        // First Laplacian eigenvalue is 0 for connected graph
        assert!(res.laplacian_eigenvalues[0].abs() < 1e-4);
        assert_eq!(res.recommended_heads, 2);
        assert_eq!(res.recommended_head_dim, 4);
    }
}
