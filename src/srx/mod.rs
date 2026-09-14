//! # SRX (Super-Resolvent xFormer) Innovation Module
//!
//! Subspace Resonance Attention architecture replacing Softmax attention with:
//! - Elementary Givens Unitary Operator Factorization: $U(\Theta) = \prod_{i=0}^{d-2} G_i(\theta_i)$
//! - Non-linear Phase Writing Law: $\Theta_t = \Theta_{t-1} + \alpha \tanh(k_t \odot v_t)$
//! - Associative Value Memory Accumulator: $M_t = \gamma M_{t-1} + (U_t k_t) v_t^T$
//! - MUSIC Noise Subspace Projector and Pseudo-spectral Dirac-like Gain: $w(q) = 1 / (||\Pi_\perp q||^2 + \epsilon)$
//! - Strictly $O(1)$ memory state (152 bytes in default lang_512, 100% L1D cache resident).

pub mod attention;
pub mod model;
pub mod ops;
pub mod state;
pub mod telemetry;
pub mod train;

pub use attention::{SrxAttention, SRX_ALPHA, SRX_EPS_DEFAULT, SRX_GAMMA};
pub use model::{SrxLayer, SrxTransformer};
pub use ops::{apply_givens, apply_givens_backward, apply_givens_forward_with_intermediates, l2_normalize, l2_normalize_backward};
pub use state::{SrxState, SrxWorkspace};
pub use telemetry::SrxTelemetryReport;
pub use train::{
    backward_loss, forward_loss, split_into_eos_sequences, train_dataset, SrxAdamW, SrxGrad,
    SrxTrainWorkspace,
};
