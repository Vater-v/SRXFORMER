//! # SRX (Super-Resolvent xFormer) Architecture v05: Quantum-Algebraic Core
//!
//! Features the 5 Core Mathematical Pillars:
//! 1. Pillar 1: 2nd-Order Associative RLS Memory (Recursive Least Squares):
//!    - Inverse covariance matrix P_t in R^{4x4} per head (P_0 = delta^{-1} I_{4x4}, lambda = 0.999).
//!    - Online Sherman-Morrison rank-1 update:
//!      v_p = P_{t-1} * k_rot
//!      denom = lambda + k_rot^T * v_p
//!      k_gain = v_p / denom
//!      e_t = v_raw - M_{t-1}^T * k_rot
//!      M_t = lambda * M_{t-1} + k_gain * e_t^T
//!      P_t = (1 / lambda) * (P_{t-1} - k_gain * (k_rot^T * P_{t-1}))
//!    Zero learned gating parameters: 100% algebraic and mathematically closed.
//! 2. Pillar 2: Krylov Recurrent Depth (K=2) Resolvent Subspace:
//!    - q^{(0)} = q_norm
//!    - q^{(1)} = L2_Norm(0.5 * q^{(0)} + 0.5 * (U_t * q^{(0)}))
//!    Refines query vector through unitary resolvent subspace before MUSIC projection and retrieval.
//! 3. Pillar 3: Monarch Butterfly Unitary Mixer with Phase Momentum:
//!    - U(\Theta) = B_2(\Theta_2) * P * B_1(\Theta_1)
//!    - Physical phase momentum: p_{\theta, t} = mu * p_{\theta, t-1} + alpha * (k_norm \odot v_raw[:4])
//!    - \theta_t = \theta_{t-1} + p_{\theta, t} (mu=0.85, alpha=0.1)
//! 4. Pillar 4: Zero-Allocation Hot Path & Strictly 288 Bytes Context State:
//!    Thetas [2, 4] (32 B) + M [2, 4, 4] (128 B) + P [2, 4, 4] (128 B) = EXACTLY 288 bytes (100% L1D cache resident).
//! 5. Pillar 5: Exact Parameter Parity:
//!    At V=53, d=8, H=2, head_dim=4, d_ff=12: EXACTLY 896 parameters for both Classical Transformer and SRX v05.

pub mod attention;
pub mod model;
pub mod ops;
pub mod state;
pub mod telemetry;
pub mod train;

pub use attention::{
    SrxAttention, SRX_ALPHA, SRX_EPS_DEFAULT, SRX_MU, SRX_RLS_DELTA, SRX_RLS_LAMBDA, SRX_W_MAX,
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
