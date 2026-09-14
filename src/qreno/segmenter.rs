//! # Quantum Renormalization & Operator-Field Tokenizer (Q-RENO): Physical SSH Lattice Segmenter
//!
//! Models text as a 1D interacting quantum crystal lattice:
//! - **Su-Schrieffer-Heeger (SSH) Hamiltonian:** Dimerization parameter $\delta t_n = t_{n, n+1} - t_{n-1, n}$.
//! - **Topological Solitons:** Boundaries detected by sign-reversal / zero-crossing of dimerization order $\delta t_n \le 0$
//!   indicating a topological domain wall (zero-mode $E = 0$).
//! - **Morse Bond Potential & Quantum Dissociation:**
//!   $$V_{\text{Morse}}(r) = D_e \left(1 - e^{-a (r - r_0)}\right)^2$$
//!   Covalent bond breaking occurs when $E_{\text{bond}} < \alpha \bar{E}_{\text{local}} - \beta \text{KL}(p_n \parallel p_{n+1})$.
//! - **Boundary Mode Suppression:** Edge solitons at open boundary conditions ($n=0$, $n=N-1$) are suppressed
//!   to prevent edge words from fragmenting into unigrams.
//! - **Zero-Allocation Hot Path:** Scratch buffers and indices are preallocated and reused.

use crate::qreno::field::QrenoField;
use crate::qreno::hamiltonian::Cluster;

/// Physical parameters for 1D SSH lattice and Morse bond dissociation.
#[derive(Debug, Clone, PartialEq)]
pub struct SshLatticeParams {
    /// Field dimension $d_f$ (default 8).
    pub field_dim: usize,
    /// Morse potential depth $D_e$ (covalent bond dissociation limit).
    pub d_e: f32,
    /// Morse stiffness parameter $a$.
    pub a: f32,
    /// Equilibrium inter-byte distance $r_0$.
    pub r_0: f32,
    /// Coupling weight $\alpha$ for local chemical potential $\bar{E}_{\text{local}}$.
    pub alpha_local: f32,
    /// Coupling weight $\beta$ for charge KL-divergence.
    pub beta_kl: f32,
    /// Maximum allowed cluster length in bytes (default 16).
    pub max_cluster_len: usize,
    /// Minimum cluster length in bytes (suppresses micro-fragmentation).
    pub min_cluster_len: usize,
}

impl Default for SshLatticeParams {
    fn default() -> Self {
        Self {
            field_dim: 8,
            d_e: 2.5,
            a: 1.8,
            r_0: 0.25,
            alpha_local: 0.65,
            beta_kl: 0.15,
            max_cluster_len: 16,
            min_cluster_len: 1,
        }
    }
}

/// Zero-allocation scratchpad for physical lattice segmentation.
#[derive(Debug, Clone, Default)]
pub struct SshLatticeWorkspace {
    /// Hopping integrals $t_{n, n+1}$ along the lattice.
    pub t_bonds: Vec<f32>,
    /// Morse bond energies $E_{\text{bond}}(n, n+1)$.
    pub e_bonds: Vec<f32>,
    /// Cut boundaries detected along the 1D chain.
    pub cut_indices: Vec<usize>,
}

impl SshLatticeWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            t_bonds: Vec::with_capacity(capacity),
            e_bonds: Vec::with_capacity(capacity),
            cut_indices: Vec::with_capacity(capacity / 4 + 1),
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.t_bonds.clear();
        self.e_bonds.clear();
        self.cut_indices.clear();
    }
}

/// Computes the Euclidean distance in continuous charge space between two byte states.
#[inline(always)]
pub fn charge_distance(c1: &[f32], c2: &[f32]) -> f32 {
    let mut sum_sq = 0.0f32;
    for k in 0..c1.len() {
        let diff = c1[k] - c2[k];
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Computes Morse bond energy between adjacent particles:
/// $V(r) = D_e (1 - e^{-a(r - r_0)})^2$
/// $E_{\text{bond}} = D_e - V(r) \in (-\infty, D_e]$.
#[inline(always)]
pub fn morse_bond_energy(r: f32, d_e: f32, a: f32, r_0: f32) -> f32 {
    let exp_term = (-a * (r - r_0)).exp();
    let v = d_e * (1.0 - exp_term) * (1.0 - exp_term);
    d_e - v
}

/// Computes KL-divergence between two normalized charge densities:
/// $p_k = (c_k)^2 / \sum_j (c_j)^2$.
#[inline(always)]
pub fn charge_kl_divergence(c1: &[f32], c2: &[f32]) -> f32 {
    const EPS: f32 = 1e-6;
    let mut norm1 = 0.0f32;
    let mut norm2 = 0.0f32;
    for k in 0..c1.len() {
        norm1 += c1[k] * c1[k];
        norm2 += c2[k] * c2[k];
    }
    norm1 = norm1.max(EPS);
    norm2 = norm2.max(EPS);

    let mut kl = 0.0f32;
    for k in 0..c1.len() {
        let p = (c1[k] * c1[k] / norm1).max(EPS);
        let q = (c2[k] * c2[k] / norm2).max(EPS);
        kl += p * (p / q).ln();
    }
    kl.max(0.0)
}

/// Physical lattice segmentation based on SSH topological solitons and Morse dissociation.
///
/// Converts a raw byte sequence into quantum molecular clusters without hardcoded delimiter lookup tables.
pub fn physical_segment(
    bytes: &[u8],
    field: &QrenoField,
    params: &SshLatticeParams,
    ws: &mut SshLatticeWorkspace,
) -> Vec<Cluster> {
    if bytes.is_empty() {
        return Vec::new();
    }

    if bytes.len() == 1 {
        return vec![Cluster {
            start: 0,
            len: 1,
            is_delimiter: bytes[0].is_ascii_whitespace(),
        }];
    }

    ws.clear();
    let n = bytes.len();
    let num_bonds = n - 1;

    // 1. Calculate bond lengths, Morse energies, and hopping integrals along the 1D lattice
    let mut e_sum = 0.0f32;
    for i in 0..num_bonds {
        let b1 = bytes[i];
        let b2 = bytes[i + 1];

        // Whitespace and control bytes inherently have high potential barrier and phase jump
        let is_ws1 = b1.is_ascii_whitespace() || b1 == b'\0';
        let is_ws2 = b2.is_ascii_whitespace() || b2 == b'\0';

        let c1 = field.charge(b1);
        let c2 = field.charge(b2);
        let r = charge_distance(c1, c2);

        // Effective inter-particle separation: whitespace expands lattice spacing
        let eff_r = if is_ws1 || is_ws2 { r + 2.0 * params.r_0 } else { r };

        let e_bond = morse_bond_energy(eff_r, params.d_e, params.a, params.r_0);
        let t_hop = 1.0 / (1.0 + (-(e_bond)).exp()); // Sigmoid activation for hopping integral

        ws.e_bonds.push(e_bond);
        ws.t_bonds.push(t_hop);
        e_sum += e_bond;
    }

    let global_e_mean = e_sum / (num_bonds as f32);

    // 2. Identify physical cuts:
    // - Dimerization zero-crossing (SSH soliton domain wall): delta_t_i <= 0
    // - Morse bond dissociation: E_bond < alpha * E_local - beta * KL
    // - Suppression of parasitic boundary solitons on edges (i = 0, i = num_bonds - 1)
    let mut local_e_mean = global_e_mean;
    let mut last_cut = 0usize;

    for i in 0..num_bonds {
        let curr_len = (i + 1) - last_cut;
        let e_curr = ws.e_bonds[i];

        // Exponential moving average for local chemical potential
        local_e_mean = 0.8 * local_e_mean + 0.2 * e_curr;

        // Force cut if maximum cluster length reached
        if curr_len >= params.max_cluster_len {
            ws.cut_indices.push(i + 1);
            last_cut = i + 1;
            continue;
        }

        // Suppress edge modes: open boundary conditions cause artificial edge cuts
        if i == 0 || i + 1 == num_bonds {
            // Only allow cut on boundary if it is an explicit whitespace/delimiter separation
            let b1 = bytes[i];
            let b2 = bytes[i + 1];
            if (b1.is_ascii_whitespace() != b2.is_ascii_whitespace()) && curr_len >= params.min_cluster_len {
                ws.cut_indices.push(i + 1);
                last_cut = i + 1;
            }
            continue;
        }

        // SSH dimerization: delta_t = t_{i, i+1} - t_{i-1, i}
        let t_curr = ws.t_bonds[i];
        let t_prev = ws.t_bonds[i - 1];
        let delta_t = t_curr - t_prev;

        // Charge KL-divergence
        let c1 = field.charge(bytes[i]);
        let c2 = field.charge(bytes[i + 1]);
        let kl = charge_kl_divergence(c1, c2);

        // Adaptive dissociation barrier
        let e_barrier = params.alpha_local * local_e_mean - params.beta_kl * kl;

        // Criterion for topological domain wall / covalent dissociation:
        // 1. Dimerization flips from positive to non-positive (soliton formation) AND bond weakens
        // 2. OR bond energy plunges below adaptive dissociation threshold
        // 3. OR whitespace transition (phase boundary)
        let is_ws_transition = bytes[i].is_ascii_whitespace() != bytes[i + 1].is_ascii_whitespace();
        let is_ssh_soliton = delta_t <= -0.15 && e_curr < e_barrier;
        let is_dissociation = e_curr < (e_barrier - 0.5);

        if (is_ws_transition || is_ssh_soliton || is_dissociation) && curr_len >= params.min_cluster_len {
            ws.cut_indices.push(i + 1);
            last_cut = i + 1;
        }
    }

    // 3. Materialize clusters from cut boundaries
    let mut clusters = Vec::with_capacity(ws.cut_indices.len() + 1);
    let mut start = 0usize;

    for &cut in &ws.cut_indices {
        if cut > start {
            let len = cut - start;
            let is_del = bytes[start..cut].iter().all(|&b| b.is_ascii_whitespace());
            clusters.push(Cluster {
                start,
                len,
                is_delimiter: is_del,
            });
            start = cut;
        }
    }

    if start < n {
        let len = n - start;
        let is_del = bytes[start..n].iter().all(|&b| b.is_ascii_whitespace());
        clusters.push(Cluster {
            start,
            len,
            is_delimiter: is_del,
        });
    }

    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morse_potential_physics() {
        let d_e = 2.5f32;
        let a = 1.8f32;
        let r_0 = 0.25f32;

        // At equilibrium distance r = r_0, energy is maximum (E = D_e)
        let e_eq = morse_bond_energy(r_0, d_e, a, r_0);
        assert!((e_eq - d_e).abs() < 1e-5);

        // At large distance, bond dissociates (energy falls towards zero)
        let e_far = morse_bond_energy(r_0 + 2.0, d_e, a, r_0);
        assert!(e_far < e_eq * 0.2, "Bond energy must drop significantly at large separation: {} vs {}", e_far, e_eq);
        assert!(e_far >= 0.0, "Morse bound energy stays non-negative for r > r_0");
    }

    #[test]
    fn test_ssh_physical_segmentation_words() {
        let field = QrenoField::new(8);
        let params = SshLatticeParams::default();
        let mut ws = SshLatticeWorkspace::new();

        let text = "хочешь сей а хочешь куй";
        let clusters = physical_segment(text.as_bytes(), &field, &params, &mut ws);

        assert!(clusters.len() >= 5, "Should segment into words and spaces: {:?}", clusters);
        // Ensure no single-byte fragmentation at string edges (OBC protection)
        assert!(clusters[0].len > 1, "First cluster must not be fragmented into 1 byte");
        let last = &clusters[clusters.len() - 1];
        assert!(last.len > 1, "Last cluster must not be fragmented into 1 byte");
    }

    #[test]
    fn test_boundary_soliton_suppression() {
        let field = QrenoField::new(8);
        let params = SshLatticeParams::default();
        let mut ws = SshLatticeWorkspace::new();

        let text = "математика";
        let clusters = physical_segment(text.as_bytes(), &field, &params, &mut ws);

        // Word should remain a cohesive molecule or split into at most 2 bound morphemes
        assert!(clusters.len() <= 2, "Word 'математика' should be cohesive, got {:?}", clusters);
        assert_eq!(clusters[0].start, 0);
    }
}
