//! # SRX (Super-Resolvent xFormer) Architecture v04: Physics-Spectral Core
//!
//! Features the 5 Core Pillars:
//! 1. Pillar 1: Widrow-Hoff Delta Rule (Novelty Error Residual Memory):
//!    - Current prediction: $v_{hat} = M_{t-1}^T k_{rot}$
//!    - Novelty residual: $e_t = v_{raw} - v_{hat}$
//!    - Memory update: $M_t = \lambda_t M_{t-1} + \gamma_t (k_{rot} e_t^T)$
//!    - Mathematical cancellation: $(I - k_{rot} k_{rot}^T)$ annihilates memory projection along $k_{rot}$,
//!      achieving 100% distortion-free storage of orthogonal facts (resolving cat/dog and fox/wolf interference).
//! 2. Pillar 2: Monarch Butterfly Unitary Mixer:
//!    $U(\Theta) = B_2(\Theta_2) \cdot P \cdot B_1(\Theta_1)$
//!    Strictly unitary ($U U^\dagger = I$), non-commutative Lie group, full global coordinate coupling.
//! 3. Pillar 3: Zero-Allocation Hot Path:
//!    All vectors in `step()`, `forward()`, and `generate()` live on stack arrays or pre-allocated `SrxWorkspace`.
//!    O(1) state memory footprint (160 bytes in L1D cache).
//! 4. Pillar 4: Analytical Reversible BPTT & Exact Gradient Descent:
//!    Quadratic adaptive epsilon annealing from 1.0 to 1e-3, Post-MUSIC RMSNorm, and gain clipping.
//! 5. Pillar 5: Spectral Graph Compiler & Fock Projections:
//!    Pure `std`-only Jacobi rotation eigenvalue solver, Laplacian spectral gap proving H=2 and d_head=4.

pub mod attention;
pub mod compiler;
pub mod model;
pub mod ops;
pub mod state;
pub mod telemetry;
pub mod train;

pub use attention::{SrxAttention, SRX_ALPHA, SRX_EPS_DEFAULT, SRX_GAMMA_BASE, SRX_W_MAX};
pub use compiler::{
    analyze_corpus, build_normalized_laplacian, build_transition_matrix, compute_pmi,
    jacobi_eigen, SpectralAnalysisResult, SpectralDecomposition, SymmetricMatrix,
};
pub use model::{SrxLayer, SrxTransformer};
pub use ops::{
    apply_butterfly_4, apply_butterfly_4_backward, apply_butterfly_4_inplace, fast_sin_cos,
    l2_normalize, l2_normalize_backward,
};
pub use state::{SrxState, SrxWorkspace};
pub use telemetry::SrxTelemetryReport;
pub use train::{
    backward_loss, forward_loss, split_into_eos_sequences, train_dataset, SrxAdamW, SrxGrad,
    SrxTrainWorkspace,
};
