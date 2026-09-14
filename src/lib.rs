//! # SRXformer
//!
//! High-performance, zero-dependency (`std`-only) Classical Transformer baseline
//! and innovative Super-Resolvent xFormer (SRX) architecture with MUSIC Subspace Resonance
//! engineered with low-level hardware awareness for Intel Ivy Bridge-EP (AVX FP32 vectorization,
//! unit-stride contiguous buffers, L1D/L2 cache preservation, pre-allocated inference scratchpads).

pub mod classic;
pub mod srx;

pub use classic::{
    backward_loss, forward_loss, train_dataset, train_instruct, train_pretrain, ActivationType,
    AdamW, FastRng, FeedForward, InferenceTelemetry, InferenceWorkspace, KvCache,
    MultiHeadAttention, NormType, PosEncodingType, TestCaseResult, TelemetryReport, Tokenizer,
    TrainMetrics, TrainTelemetry, TrainWorkspace, Transformer, TransformerConfig, TransformerGrad,
    TransformerLayer, VOCAB, BOT_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, USER_TOKEN_ID,
};

pub use srx::{
    apply_givens, apply_givens_backward, apply_givens_forward_with_intermediates,
    backward_loss as srx_backward_loss, forward_loss as srx_forward_loss, l2_normalize,
    l2_normalize_backward, train_dataset as srx_train_dataset, SrxAdamW, SrxAttention, SrxGrad,
    SrxLayer, SrxState, SrxTelemetryReport, SrxTrainWorkspace, SrxTransformer, SrxWorkspace,
    SRX_ALPHA, SRX_EPS_DEFAULT, SRX_GAMMA,
};
