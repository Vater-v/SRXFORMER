//! # SRXformer
//!
//! High-performance, zero-dependency (`std`-only) Classical Transformer baseline,
//! frozen SRX v01 reference architecture, and state-of-the-art Super-Resolvent xFormer v02 (SRX v02)
//! with MUSIC Subspace Resonance, Post-MUSIC RMSNorm, hard gain clipping, and fast Ivy Bridge AVX Givens rotations.
//!
//! Features:
//! - Strictly 152 bytes state memory footprint (100% resident in L1 D-Cache).
//! - Fast Givens Rotations with Taylor polynomial approximation (no scalar libc transcendental calls).
//! - Gain clipping & Post-MUSIC RMSNorm eliminating spectral clicks and generation artifacts.
//! - Exact 512 parameters (1:1 bitwise structural parity with classical baseline).

pub mod classic;
pub mod srx_v01;
pub mod srx_v02;

// Re-export classic transformer module
pub use classic::{
    backward_loss, forward_loss, train_dataset, train_instruct, train_pretrain, ActivationType,
    AdamW, FastRng, FeedForward, InferenceTelemetry, InferenceWorkspace, KvCache,
    MultiHeadAttention, NormType, PosEncodingType, TestCaseResult, TelemetryReport, Tokenizer,
    TrainMetrics, TrainTelemetry, TrainWorkspace, Transformer, TransformerConfig, TransformerGrad,
    TransformerLayer, VOCAB, BOT_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, USER_TOKEN_ID,
};

// Re-export SRX v01 (Frozen)
pub use srx_v01::{
    apply_givens as srx_v01_apply_givens, apply_givens_backward as srx_v01_apply_givens_backward,
    apply_givens_forward_with_intermediates as srx_v01_apply_givens_forward_with_intermediates,
    backward_loss as srx_v01_backward_loss, forward_loss as srx_v01_forward_loss,
    l2_normalize as srx_v01_l2_normalize, l2_normalize_backward as srx_v01_l2_normalize_backward,
    train_dataset as srx_v01_train_dataset, SrxAdamW as SrxAdamWV01,
    SrxAttention as SrxAttentionV01, SrxGrad as SrxGradV01, SrxLayer as SrxLayerV01,
    SrxState as SrxStateV01, SrxTelemetryReport as SrxTelemetryReportV01,
    SrxTrainWorkspace as SrxTrainWorkspaceV01, SrxTransformer as SrxTransformerV01,
    SrxWorkspace as SrxWorkspaceV01, SRX_ALPHA as SRX_V01_ALPHA,
    SRX_EPS_DEFAULT as SRX_V01_EPS_DEFAULT, SRX_GAMMA as SRX_V01_GAMMA,
};

// Re-export SRX v02 (Active / Primary)
pub use srx_v02::{
    apply_givens as srx_v02_apply_givens, apply_givens_backward as srx_v02_apply_givens_backward,
    apply_givens_forward_with_intermediates as srx_v02_apply_givens_forward_with_intermediates,
    backward_loss as srx_v02_backward_loss, fast_sin_cos as srx_v02_fast_sin_cos,
    forward_loss as srx_v02_forward_loss, l2_normalize as srx_v02_l2_normalize,
    l2_normalize_backward as srx_v02_l2_normalize_backward,
    train_dataset as srx_v02_train_dataset, SrxAdamW as SrxAdamWV02,
    SrxAttention as SrxAttentionV02, SrxGrad as SrxGradV02, SrxLayer as SrxLayerV02,
    SrxState as SrxStateV02, SrxTelemetryReport as SrxTelemetryReportV02,
    SrxTrainWorkspace as SrxTrainWorkspaceV02, SrxTransformer as SrxTransformerV02,
    SrxWorkspace as SrxWorkspaceV02, SRX_ALPHA as SRX_V02_ALPHA,
    SRX_EPS_DEFAULT as SRX_V02_EPS_DEFAULT, SRX_GAMMA as SRX_V02_GAMMA,
    SRX_W_MAX as SRX_V02_W_MAX,
};

// Primary SRX exports map to v02
pub use srx_v02::{
    apply_givens, apply_givens_backward, apply_givens_forward_with_intermediates,
    backward_loss as srx_backward_loss, fast_sin_cos, forward_loss as srx_forward_loss,
    l2_normalize, l2_normalize_backward, train_dataset as srx_train_dataset, SrxAdamW,
    SrxAttention, SrxGrad, SrxLayer, SrxState, SrxTelemetryReport, SrxTrainWorkspace,
    SrxTransformer, SrxWorkspace, SRX_ALPHA, SRX_EPS_DEFAULT, SRX_GAMMA, SRX_W_MAX,
};

// Compatibility module alias
pub mod srx {
    pub use crate::srx_v02::*;
}
