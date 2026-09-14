//! # SRXformer
//!
//! High-performance, zero-dependency (`std`-only) Classical Transformer baseline,
//! frozen SRX v01 reference architecture, frozen SRX v02 architecture,
//! and state-of-the-art Super-Resolvent xFormer v03 (SRX v03 Golden Core)
//! featuring Monarch Butterfly Unitary Factorization, Selective Dynamic Memory Gating,
//! strictly O(1) state footprint (160 bytes in L1D cache), and zero-allocation hot-path inference.

pub mod classic;
pub mod qreno;
pub mod scaled;
pub mod srx_v01;
pub mod srx_v02;
pub mod srx_v03;
pub mod srx_v04;
pub mod srx_v05;

// Re-export Scaled architecture module
pub use scaled::{
    AutoregressiveModel, Module, QrenoSrxLM, ScaledClassicAttention, ScaledClassicKvCache,
    ScaledClassicTransformer, ScaledConfig, ScaledFFN, ScaledLinear, ScaledRMSNorm,
    ScaledSrxAttention, ScaledSrxState, ScaledSrxTransformer, ScaledWorkspace, ScalingCalculator,
    Tier,
};

// Re-export Q-RENO module
pub use qreno::{
    coarse_grain_cluster, is_delimiter_byte, partition_into_clusters, solve_ground_state,
    solve_ground_state_vjp, BondParams, Cluster, GroundState, OperatorMeasure, QrenoAdamW,
    QrenoConfig, QrenoField, QrenoGrad, QrenoTokenizer, QrenoWeights, MAX_CLUSTER_LEN as QRENO_MAX_CLUSTER_LEN,
};

// Re-export classic transformer module
pub use classic::{
    backward_loss, forward_loss, train_dataset, train_instruct, train_instruct_with_replay,
    train_pretrain, ActivationType, AdamW, FastRng, FeedForward, InferenceTelemetry,
    InferenceWorkspace, KvCache, MultiHeadAttention, NormType, PosEncodingType, TestCaseResult,
    TelemetryReport, Tokenizer, TrainMetrics, TrainTelemetry, TrainWorkspace, Transformer,
    TransformerConfig, TransformerGrad, TransformerLayer, TwoStagePipelineResult, VOCAB,
    VOCAB_CHINCHILLA, VOCAB_V2, VOCAB_V3, VOCAB_V4, BOT_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID,
    USER_TOKEN_ID,
};

// Re-export SRX v01 (Frozen Reference)
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

// Re-export SRX v02 (Frozen Reference)
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

// Re-export SRX v03 (Active / Golden Core)
pub use srx_v03::{
    apply_butterfly_4 as srx_v03_apply_butterfly_4,
    apply_butterfly_4_backward as srx_v03_apply_butterfly_4_backward,
    apply_butterfly_4_inplace as srx_v03_apply_butterfly_4_inplace,
    backward_loss as srx_v03_backward_loss, fast_sin_cos as srx_v03_fast_sin_cos,
    forward_loss as srx_v03_forward_loss, l2_normalize as srx_v03_l2_normalize,
    l2_normalize_backward as srx_v03_l2_normalize_backward,
    train_dataset as srx_v03_train_dataset, split_into_eos_sequences as srx_v03_split_into_eos_sequences,
    SrxAdamW as SrxAdamWV03, SrxAttention as SrxAttentionV03, SrxGrad as SrxGradV03,
    SrxLayer as SrxLayerV03, SrxState as SrxStateV03, SrxTelemetryReport as SrxTelemetryReportV03,
    SrxTrainWorkspace as SrxTrainWorkspaceV03, SrxTransformer as SrxTransformerV03,
    SrxWorkspace as SrxWorkspaceV03, SRX_ALPHA as SRX_V03_ALPHA,
    SRX_EPS_DEFAULT as SRX_V03_EPS_DEFAULT, SRX_GAMMA_BASE as SRX_V03_GAMMA_BASE,
    SRX_W_MAX as SRX_V03_W_MAX,
};

// Re-export SRX v04 (Physics-Spectral Core & Widrow-Hoff Delta Rule)
pub use srx_v04::{
    analyze_corpus as srx_v04_analyze_corpus,
    apply_butterfly_4 as srx_v04_apply_butterfly_4,
    apply_butterfly_4_backward as srx_v04_apply_butterfly_4_backward,
    apply_butterfly_4_inplace as srx_v04_apply_butterfly_4_inplace,
    backward_loss as srx_v04_backward_loss, fast_sin_cos as srx_v04_fast_sin_cos,
    forward_loss as srx_v04_forward_loss, l2_normalize as srx_v04_l2_normalize,
    l2_normalize_backward as srx_v04_l2_normalize_backward,
    split_into_eos_sequences as srx_v04_split_into_eos_sequences,
    train_dataset as srx_v04_train_dataset, SrxAdamW as SrxAdamWV04,
    SrxAttention as SrxAttentionV04, SrxGrad as SrxGradV04, SrxLayer as SrxLayerV04,
    SrxState as SrxStateV04, SrxTelemetryReport as SrxTelemetryReportV04,
    SrxTrainWorkspace as SrxTrainWorkspaceV04, SrxTransformer as SrxTransformerV04,
    SrxWorkspace as SrxWorkspaceV04, SRX_ALPHA as SRX_V04_ALPHA,
    SRX_EPS_DEFAULT as SRX_V04_EPS_DEFAULT, SRX_GAMMA_BASE as SRX_V04_GAMMA_BASE,
    SRX_W_MAX as SRX_V04_W_MAX,
};

// Re-export SRX v05 (Quantum-Algebraic Core: Pure Orthogonal Projector, Undistorted MUSIC, 160-byte state)
pub use srx_v05::{
    apply_butterfly_4 as srx_v05_apply_butterfly_4,
    apply_butterfly_4_backward as srx_v05_apply_butterfly_4_backward,
    apply_butterfly_4_inplace as srx_v05_apply_butterfly_4_inplace,
    backward_loss as srx_v05_backward_loss, fast_sin_cos as srx_v05_fast_sin_cos,
    forward_loss as srx_v05_forward_loss, l2_normalize as srx_v05_l2_normalize,
    l2_normalize_backward as srx_v05_l2_normalize_backward,
    split_into_eos_sequences as srx_v05_split_into_eos_sequences,
    train_dataset as srx_v05_train_dataset,
    train_instruct as srx_v05_train_instruct,
    train_instruct_with_replay as srx_v05_train_instruct_with_replay,
    train_pretrain as srx_v05_train_pretrain,
    SrxAdamW as SrxAdamWV05,
    SrxAttention as SrxAttentionV05, SrxGrad as SrxGradV05, SrxLayer as SrxLayerV05,
    SrxState as SrxStateV05, SrxTelemetryReport as SrxTelemetryReportV05,
    SrxTrainWorkspace as SrxTrainWorkspaceV05, SrxTransformer as SrxTransformerV05,
    SrxWorkspace as SrxWorkspaceV05, SRX_ALPHA as SRX_V05_ALPHA,
    SRX_EPS_DEFAULT as SRX_V05_EPS_DEFAULT, SRX_W_MAX as SRX_V05_W_MAX,
};

// Primary SRX exports map to v02 (preserves 100% test compatibility)
pub use srx_v02::{
    apply_givens, apply_givens_backward, apply_givens_forward_with_intermediates,
    backward_loss as srx_backward_loss, fast_sin_cos, forward_loss as srx_forward_loss,
    l2_normalize, l2_normalize_backward, train_dataset as srx_train_dataset, SrxAdamW,
    SrxAttention, SrxGrad, SrxLayer, SrxState, SrxTelemetryReport, SrxTrainWorkspace,
    SrxTransformer, SrxWorkspace, SRX_ALPHA, SRX_EPS_DEFAULT, SRX_GAMMA, SRX_W_MAX,
};

// Re-export SRX v03 butterfly operations at root
pub use srx_v03::{apply_butterfly_4, apply_butterfly_4_backward, apply_butterfly_4_inplace};

// Compatibility module alias
pub mod srx {
    pub use crate::srx_v02::*;
}
