//! # SRX (Super-Resolvent xFormer) Architecture v05: Quantum-Algebraic Core («Автомат Калашникова»)
//!
//! Features the Core Mathematical Pillars:
//! 1. Pillar 1: Pure Orthogonal Complement Projector Memory:
//!    - Projector onto orthogonal complement of the key:
//!      \Pi_{k^\perp} = I - k_rot * k_rot^T (where ||k_rot||_2 = 1)
//!    - Associative memory update:
//!      e_t = v_raw - M_{t-1}^T * k_rot
//!      M_t = M_{t-1} \Pi_{k_rot^\perp} + k_rot * v_raw^T = M_{t-1} + k_rot * e_t^T
//!    - Exact response identity:
//!      M_t^T * k_rot = (I - k_rot * k_rot^T) M_{t-1}^T * k_rot + v_raw * (k_rot^T * k_rot) = 0 + v_raw = v_raw
//!    - Zero heuristics, zero learned gates, zero manual decay hyperparameters lambda.
//!      If key repeats and v matches: e_t = 0 => M_t = M_{t-1} (strictly zero memory drift!).
//! 2. Pillar 2: Undistorted MUSIC Subspace Pseudo-Spectrum:
//!    - q_inv = U^\dagger(\Theta_t) * q_norm
//!    - E_noise = q_inv[2]^2 + q_inv[3]^2
//!    - w(q) = min(1 / (E_noise + eps), 15.0)
//!    - y_ret = M_t^T * (U(\Theta_t) * q_norm) * w(q)
//!    - When q = k_rot: U^\dagger * U * k_sig = k_sig = [k_0, k_1, 0, 0] => E_noise = 0 => w = 15.0!
//! 3. Pillar 3: Monarch Butterfly Unitary Factorization:
//!    - U(\Theta) = B_2(\Theta_2) * P * B_1(\Theta_1)
//!    - Physical phase coupling: \theta_t = \theta_{t-1} + \alpha * (k_norm \odot v_raw) (\alpha = 0.1)
//! 4. Pillar 4: Zero-Allocation Hot Path & Strictly 160 Bytes Context State:
//!    Thetas [2, 4] (32 B) + M [2, 4, 4] (128 B) = EXACTLY 160 bytes (100% L1D cache resident, < 0.5% of 32 KB).
//! 5. Pillar 5: Exact Parameter Parity:
//!    At V=65 (Chinchilla), d=8, H=2, head_dim=4, d_ff=6: EXACTLY 896 parameters (0.00% delta with Classical Transformer).

pub mod attention;
pub mod model;
pub mod ops;
pub mod state;
pub mod telemetry;
pub mod train;

pub use attention::{SrxAttention, SRX_ALPHA, SRX_EPS_DEFAULT, SRX_W_MAX};
pub use model::{SrxLayer, SrxTransformer};
pub use ops::{
    apply_butterfly_4, apply_butterfly_4_backward, apply_butterfly_4_inplace, fast_sin_cos,
    l2_normalize, l2_normalize_backward,
};
pub use state::{SrxState, SrxWorkspace};
pub use telemetry::SrxTelemetryReport;
pub use train::{
    backward_loss, forward_loss, split_into_eos_sequences, train_dataset, train_instruct,
    train_instruct_with_replay, train_pretrain, SrxAdamW, SrxGrad, SrxTrainWorkspace,
};
