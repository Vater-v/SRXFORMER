//! # Quantum Renormalization & Operator-Field Tokenizer (Q-RENO): Spectral Solver
//!
//! Solves the 1D tight-binding Hamiltonian $H_m \in \mathbb{R}^{L_m \times L_m}$ ($L_m \le 16$):
//! - Diagonal: $\epsilon_1, \dots, \epsilon_L$
//! - Subdiagonal: $-t_1, \dots, -t_{L-1}$
//!
//! Computes the ground state $\Psi_0$ ($H_m \Psi_0 = E_0 \Psi_0, \|\Psi_0\|_2 = 1$) via:
//! 1. Sturm sequence bisection for exact eigenvalue $E_0$
//! 2. Inverse iteration using the Thomas algorithm (tridiagonal solver) in $O(L)$ steps
//! 3. Zero heap allocations on the hot path (all buffers are stack arrays `[f32; 16]`).
//! 4. Wilson RG coarse-graining: $\Phi_m = \sum_{j=0}^{L_m-1} \Psi_0[j] \cdot c(b_{\text{start} + j})$.

use crate::qreno::field::QrenoField;

/// Maximum cluster length supported by stack-allocated buffers.
pub const MAX_CLUSTER_LEN: usize = 16;

/// Ground state quantum eigensystem for a cluster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundState {
    /// Ground state energy eigenvalue $E_0$.
    pub energy: f32,
    /// Normalized ground state eigenvector $\Psi_0$, with $\|\Psi_0\|_2 = 1.0$.
    pub psi: [f32; MAX_CLUSTER_LEN],
    /// Actual dimension of this cluster $L_m \le 16$.
    pub len: usize,
}

impl GroundState {
    /// Returns the active slice of the ground state wavefunction $\Psi_0 \in \mathbb{R}^{L_m}$.
    #[inline(always)]
    pub fn psi_slice(&self) -> &[f32] {
        &self.psi[..self.len]
    }
}

/// Evaluates the Sturm sequence to count the number of eigenvalues strictly less than $\lambda$.
#[inline(always)]
fn count_eigenvalues_less(diag: &[f32], subdiag: &[f32], lambda: f32) -> usize {
    let n = diag.len();
    if n == 0 {
        return 0;
    }

    let mut count = 0;
    let mut q = diag[0] - lambda;
    if q < 0.0 {
        count += 1;
    }

    for k in 1..n {
        let denom = if q.abs() < 1e-12 {
            1e-12 * if q >= 0.0 { 1.0 } else { -1.0 }
        } else {
            q
        };
        let s = subdiag[k - 1];
        q = (diag[k] - lambda) - (s * s) / denom;
        if q < 0.0 {
            count += 1;
        }
    }

    count
}

/// Solves for the minimal eigenvalue $E_0$ using Sturm sequence bisection.
fn bisection_minimal_eigenvalue(diag: &[f32], subdiag: &[f32]) -> f32 {
    let n = diag.len();
    if n == 1 {
        return diag[0];
    }

    // Gershgorin circle bounds
    let mut min_bound = diag[0] - subdiag[0].abs();
    let mut max_bound = diag[0] + subdiag[0].abs();

    for i in 1..n {
        let r = subdiag[i - 1].abs() + if i < n - 1 { subdiag[i].abs() } else { 0.0 };
        min_bound = min_bound.min(diag[i] - r);
        max_bound = max_bound.max(diag[i] + r);
    }

    let mut low = min_bound - 0.05;
    let mut high = max_bound + 0.05;

    // 40 bisection iterations guarantee machine precision on [low, high]
    for _ in 0..40 {
        let mid = 0.5 * (low + high);
        if count_eigenvalues_less(diag, subdiag, mid) == 0 {
            low = mid;
        } else {
            high = mid;
        }
    }

    0.5 * (low + high)
}

/// Solves the tridiagonal system $(H - \mu I) x = y$ using the Thomas algorithm on stack buffers.
#[inline(always)]
fn thomas_solve(
    diag: &[f32],
    subdiag: &[f32],
    shift: f32,
    y: &[f32],
    x_out: &mut [f32; MAX_CLUSTER_LEN],
) {
    let n = diag.len();
    let mut c_prime = [0.0f32; MAX_CLUSTER_LEN];
    let mut d_prime = [0.0f32; MAX_CLUSTER_LEN];

    let mut a0 = diag[0] - shift;
    if a0.abs() < 1e-10 {
        a0 = 1e-10 * if a0 >= 0.0 { 1.0 } else { -1.0 };
    }

    c_prime[0] = subdiag[0] / a0;
    d_prime[0] = y[0] / a0;

    for i in 1..n {
        let mut denom = (diag[i] - shift) - subdiag[i - 1] * c_prime[i - 1];
        if denom.abs() < 1e-10 {
            denom = 1e-10 * if denom >= 0.0 { 1.0 } else { -1.0 };
        }

        if i < n - 1 {
            c_prime[i] = subdiag[i] / denom;
        }
        d_prime[i] = (y[i] - subdiag[i - 1] * d_prime[i - 1]) / denom;
    }

    // Back-substitution
    x_out[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x_out[i] = d_prime[i] - c_prime[i] * x_out[i + 1];
    }
}

/// Solves for the ground state eigensystem $(E_0, \Psi_0)$ of a symmetric tridiagonal Hamiltonian.
///
/// Matrix structure:
/// - Diagonal: `diag` $\epsilon_0, \dots, \epsilon_{L-1}$
/// - Subdiagonal / superdiagonal: `subdiag` $-t_0, \dots, -t_{L-2}$
pub fn solve_ground_state(diag: &[f32], subdiag: &[f32]) -> GroundState {
    let n = diag.len();
    assert!(
        n > 0 && n <= MAX_CLUSTER_LEN,
        "Cluster length must be in 1..={MAX_CLUSTER_LEN}, got {n}"
    );
    assert_eq!(
        subdiag.len(),
        n - 1,
        "Subdiagonal length must equal diag.len() - 1"
    );

    let mut psi = [0.0f32; MAX_CLUSTER_LEN];

    if n == 1 {
        psi[0] = 1.0;
        return GroundState {
            energy: diag[0],
            psi,
            len: 1,
        };
    }

    // Step 1: Find minimal eigenvalue E0 by Sturm bisection
    let e0_est = bisection_minimal_eigenvalue(diag, subdiag);

    // Step 2: Inverse iteration with Thomas algorithm
    let mut rhs = [1.0f32; MAX_CLUSTER_LEN];
    let mut v = [0.0f32; MAX_CLUSTER_LEN];

    // First solve with slight offset to avoid singular pivot
    thomas_solve(diag, subdiag, e0_est - 1e-5, &rhs[..n], &mut v);

    // Normalize v
    let mut norm_sq = 0.0f32;
    for i in 0..n {
        norm_sq += v[i] * v[i];
    }
    let norm = norm_sq.sqrt().max(1e-12);
    for i in 0..n {
        rhs[i] = v[i] / norm;
    }

    // Compute Rayleigh quotient
    let mut mu = 0.0f32;
    for i in 0..n {
        mu += diag[i] * rhs[i] * rhs[i];
    }
    for i in 0..n - 1 {
        mu += 2.0 * subdiag[i] * rhs[i] * rhs[i + 1];
    }

    // Second refined solve
    thomas_solve(diag, subdiag, mu - 1e-6, &rhs[..n], &mut psi);

    // Final normalization
    let mut final_norm_sq = 0.0f32;
    for i in 0..n {
        final_norm_sq += psi[i] * psi[i];
    }
    let final_norm = final_norm_sq.sqrt().max(1e-12);
    for i in 0..n {
        psi[i] /= final_norm;
    }

    // Enforce gauge invariance / sign convention: positive sum of amplitudes
    let sum_psi: f32 = psi[..n].iter().sum();
    if sum_psi < 0.0 {
        for i in 0..n {
            psi[i] = -psi[i];
        }
    }

    // Exact energy E0 = psi^T H psi
    let mut e0 = 0.0f32;
    for i in 0..n {
        e0 += diag[i] * psi[i] * psi[i];
    }
    for i in 0..n - 1 {
        e0 += 2.0 * subdiag[i] * psi[i] * psi[i + 1];
    }

    GroundState {
        energy: e0,
        psi,
        len: n,
    }
}

/// Wilson RG coarse-graining: projects cluster bytes into a single continuous quasi-particle charge vector.
///
/// $\Phi_m = \sum_{j=0}^{L_m-1} \Psi_0[j] \cdot c(b_{\text{start} + j}) \in \mathbb{R}^{d_f}$
#[inline(always)]
pub fn coarse_grain_cluster(
    cluster_bytes: &[u8],
    psi: &[f32],
    field: &QrenoField,
    phi_out: &mut [f32],
) {
    let n = cluster_bytes.len();
    let d_f = field.field_dim;
    debug_assert_eq!(psi.len(), n);
    debug_assert_eq!(phi_out.len(), d_f);

    phi_out.fill(0.0);

    for j in 0..n {
        let p = psi[j];
        let c = field.charge(cluster_bytes[j]);
        for k in 0..d_f {
            phi_out[k] += p * c[k];
        }
    }
}

/// Solves $(H - E_0 I + \Psi_0 \Psi_0^\top) v = g$ via Gaussian elimination on stack buffers
/// to compute the exact analytical VJP gradient of the ground state $\Psi_0$.
///
/// Zero heap allocations. Stack footprint: $< 1.2$ KB.
pub fn solve_ground_state_vjp(
    diag: &[f32],
    subdiag: &[f32],
    gs: &GroundState,
    psi_grad: &[f32],
    diag_grad: &mut [f32],
    subdiag_grad: &mut [f32],
) {
    let n = gs.len;
    debug_assert_eq!(diag.len(), n);
    debug_assert_eq!(subdiag.len(), n - 1);
    debug_assert_eq!(psi_grad.len(), n);
    debug_assert_eq!(diag_grad.len(), n);
    debug_assert_eq!(subdiag_grad.len(), n - 1);

    if n == 1 {
        // psi[0] is identically 1.0 (constant unit vector), so d(psi)/d(H) = 0
        return;
    }

    // 1. Project gradient into the orthogonal complement of psi_0:
    // g = psi_grad - (psi_0^T psi_grad) psi_0
    let mut dot = 0.0f32;
    for i in 0..n {
        dot += gs.psi[i] * psi_grad[i];
    }

    let mut g = [0.0f32; MAX_CLUSTER_LEN];
    for i in 0..n {
        g[i] = psi_grad[i] - dot * gs.psi[i];
    }

    // 2. Build M = H - E0 * I + psi_0 * psi_0^T on stack [f32; 16 * 16]
    let mut mat = [0.0f32; MAX_CLUSTER_LEN * MAX_CLUSTER_LEN];
    for i in 0..n {
        for j in 0..n {
            let mut val = gs.psi[i] * gs.psi[j];
            if i == j {
                val += diag[i] - gs.energy;
            } else if j == i + 1 {
                val += subdiag[i];
            } else if i == j + 1 {
                val += subdiag[j];
            }
            mat[i * n + j] = val;
        }
    }

    // 3. Gaussian elimination with partial pivoting to solve M v = g
    let mut v = g;
    for col in 0..n {
        // Find pivot
        let mut max_val = mat[col * n + col].abs();
        let mut pivot_row = col;
        for row in col + 1..n {
            let val = mat[row * n + col].abs();
            if val > max_val {
                max_val = val;
                pivot_row = row;
            }
        }

        // Swap rows if necessary
        if pivot_row != col {
            for k in 0..n {
                mat.swap(col * n + k, pivot_row * n + k);
            }
            v.swap(col, pivot_row);
        }

        let pivot = mat[col * n + col];
        let pivot_safe = if pivot.abs() < 1e-12 {
            1e-12 * if pivot >= 0.0 { 1.0 } else { -1.0 }
        } else {
            pivot
        };

        for row in col + 1..n {
            let factor = mat[row * n + col] / pivot_safe;
            for k in col..n {
                mat[row * n + k] -= factor * mat[col * n + k];
            }
            v[row] -= factor * v[col];
        }
    }

    // Back-substitution
    for row in (0..n).rev() {
        let mut sum = v[row];
        for k in row + 1..n {
            sum -= mat[row * n + k] * v[k];
        }
        let pivot = mat[row * n + row];
        let pivot_safe = if pivot.abs() < 1e-12 {
            1e-12 * if pivot >= 0.0 { 1.0 } else { -1.0 }
        } else {
            pivot
        };
        v[row] = sum / pivot_safe;
    }

    // 4. Analytical Hellmann-Feynman / Rayleigh-Schroedinger gradient propagation:
    // dL / d(diag[j]) = - v[j] * psi_0[j]
    // dL / d(subdiag[j]) = v[j] * psi_0[j + 1] + v[j + 1] * psi_0[j]
    for j in 0..n {
        diag_grad[j] += -v[j] * gs.psi[j];
    }
    for j in 0..n - 1 {
        // Since H_{j, j+1} = subdiag[j] = -t_j, and dL/d(H_{j, j+1}) = -0.5*(v_j psi_{j+1} + v_{j+1} psi_j),
        // dL / d(subdiag[j]) = - (dL/dH_{j, j+1} + dL/dH_{j+1, j}) = v_j psi_{j+1} + v_{j+1} psi_j
        subdiag_grad[j] += - (v[j] * gs.psi[j + 1] + v[j + 1] * gs.psi[j]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ground_state_accuracy() {
        // Test symmetric tridiagonal Hamiltonian
        let diag = [-0.5, -0.6, -0.4, -0.7];
        let subdiag = [-0.8, -0.85, -0.75]; // -t_i

        let gs = solve_ground_state(&diag, &subdiag);
        assert_eq!(gs.len, 4);

        // Verify ||psi||_2 = 1.0
        let mut norm_sq = 0.0f32;
        for &p in gs.psi_slice() {
            norm_sq += p * p;
        }
        assert!((norm_sq.sqrt() - 1.0).abs() < 1e-5, "Wavefunction must be normalized");

        // Verify H psi = E0 psi (residual check)
        let n = 4;
        let mut h_psi = vec![0.0f32; n];
        for i in 0..n {
            h_psi[i] += diag[i] * gs.psi[i];
            if i > 0 {
                h_psi[i] += subdiag[i - 1] * gs.psi[i - 1];
            }
            if i < n - 1 {
                h_psi[i] += subdiag[i] * gs.psi[i + 1];
            }
        }

        let mut res_sq = 0.0f32;
        for i in 0..n {
            let diff = h_psi[i] - gs.energy * gs.psi[i];
            res_sq += diff * diff;
        }
        let res = res_sq.sqrt();
        assert!(res < 1e-4, "Eigenvalue residual ||H psi - E0 psi|| must be < 1e-4, got {res}");
    }
}
