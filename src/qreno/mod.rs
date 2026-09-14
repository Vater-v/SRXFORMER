//! # Quantum Renormalization & Operator-Field Tokenizer (Q-RENO)
//!
//! Replaces lookup tables ($W_E$) and discrete BPE tokenizers with a continuous
//! quantum field theory and tight-binding Hamiltonian framework:
//!
//! - **Basis Wave Field** ([`field`]): Continuous charge vectors $c(b)$, chemical potentials $\epsilon(b)$, and phase frequencies $\omega(b)$.
//! - **Chemical Hamiltonian & Bonds** ([`hamiltonian`]): Overlap resonance integrals $t_{i, i+1}$ and probability currents $J_{i, i+1}$.
//! - **Spectral Solver & Wilson RG** ([`solver`]): Ground state $\Psi_0$ discovery via Sturm sequence and Thomas tridiagonal algorithm with zero allocations.
//! - **Operator Measurement** ([`measure`]): Semantic operator projection $W_O$ with RMSNorm into continuous model representations.
//! - **Top-Level Tokenizer & Model** ([`model`]): End-to-end tokenizer, gauge-invariant embeddings, analytical gradients, and AdamW optimizer.

pub mod field;
pub mod hamiltonian;
pub mod measure;
pub mod model;
pub mod solver;

pub use field::QrenoField;
pub use hamiltonian::{is_delimiter_byte, partition_into_clusters, BondParams, Cluster};
pub use measure::OperatorMeasure;
pub use model::{QrenoAdamW, QrenoConfig, QrenoGrad, QrenoTokenizer, QrenoWeights};
pub use solver::{
    coarse_grain_cluster, solve_ground_state, solve_ground_state_vjp, GroundState, MAX_CLUSTER_LEN,
};
