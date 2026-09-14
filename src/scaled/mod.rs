//! # Scaled Architecture & PyTorch-like Ergonomic Framework
//!
//! Provides mathematically scalable SRX cores and classical parity baselines:
//! - **Mathematical Scaling Calculator** ([`config`]): Rigorous $d_{\text{head}} = 4$ SO(4) quantum scaling across Micro, Standard, Pro, and Ultra tiers.
//! - **PyTorch-like Ergonomic Primitives** ([`nn`]): `Module`, `AutoregressiveModel`, `ScaledRMSNorm`, `ScaledLinear`, `ScaledFFN`.
//! - **Arbitrary Head Scaled SRX Core** ([`srx`]): `ScaledSrxAttention`, `ScaledSrxState` ($H \times 80$ B $O(1)$ L1D), `ScaledSrxTransformer`.
//! - **Parity Classical Baseline** ([`classic`]): `ScaledClassicTransformer` with KV-cache.
//! - **End-to-End Quantum LM** ([`qreno_srx`]): `QrenoSrxLM` integrating Q-RENO field frontend with Scaled SRX.

pub mod bench;
pub mod classic;
pub mod config;
pub mod generator;
pub mod nn;
pub mod qreno_srx;
pub mod srx;

pub use bench::{
    load_corpus_lines, load_typo_pairs, run_isotime_benchmark, run_memory_wall_challenge,
    CheckpointRecord, IsoTimeConfig, MemoryWallPoint, ModelBenchmarkResult, TypoEvalResult,
};
pub use classic::{ScaledClassicAttention, ScaledClassicKvCache, ScaledClassicTransformer};
pub use config::{ScaledConfig, ScalingCalculator, Tier};
pub use generator::{apply_repetition_penalty, sample_token, FastRng, SamplingConfig};
pub use nn::{clip_grad_norm_layers, AutoregressiveModel, Module, ScaledAdamW, ScaledFFN, ScaledLinear, ScaledRMSNorm};
pub use qreno_srx::QrenoSrxLM;
pub use srx::{ScaledSequenceWorkspace, ScaledSrxAttention, ScaledSrxState, ScaledSrxTransformer, ScaledWorkspace};

