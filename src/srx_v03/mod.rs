//! # SRX (Super-Resolvent xFormer) Innovation Module v03: Golden Core
//!
//! Features the 4 Golden Core Pillars:
//! 1. Pillar 1: Selective Dynamic Memory Gate:
//!    $\gamma_t = \text{sigmoid}(W_\gamma x_t + b_\gamma)$
//!    $M_t = \lambda_t M_{t-1} + \gamma_t (U_t k_t) v_t^T$ where $\lambda_t = 1 - (1 - \gamma_{base})\gamma_t$.
//! 2. Pillar 2: Monarch Butterfly Unitary Mixer:
//!    $U(\Theta) = B_2(\Theta_2) \cdot P \cdot B_1(\Theta_1)$
//!    Strictly unitary ($U U^\dagger = I$), non-commutative Lie group, full global coordinate coupling.
//! 3. Pillar 3: Zero-Allocation Hot Path:
//!    All vectors in `step()`, `forward()`, and `generate()` live on stack arrays or pre-allocated `SrxWorkspace`.
//!    Sub-microsecond latency (< 1.0 µs on Intel Xeon E5-2650 v2).
//! 4. Pillar 4: Reversible Analytical BPTT:
//!    $U^{-1} = U^\dagger$ for O(1) intermediate state reconstruction during backward pass,
//!    quadratic adaptive epsilon annealing from 1.0 to 1e-3, Post-MUSIC RMSNorm, and gain clipping.

pub mod attention;
pub mod model;
pub mod ops;
pub mod state;
pub mod telemetry;
pub mod train;

pub use attention::{SrxAttention, SRX_ALPHA, SRX_EPS_DEFAULT, SRX_GAMMA_BASE, SRX_W_MAX};
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
