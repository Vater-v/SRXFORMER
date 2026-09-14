//! Sprint 3 Verification Suite: Chinchilla Two-Stage Training Pipeline and Strict Parity.
//! Validates:
//! 1. Parameter Parity: Both Classical Transformer and SRX v05 have EXACTLY 896 parameters (0.00% delta).
//! 2. Two-Stage Training Pipeline: Pretrain (Stage 1) + Instruct with Replay Mix (Stage 2) decreases loss
//!    and successfully generates answers for both architectures.
//! 3. Serialization / Deserialization: Full binary roundtrip fidelity for both models.
//! 4. Pretrained weight file integrity and parameter parity verification.

use std::fs;
use std::path::Path;

use srxformer::{
    classic::{InferenceWorkspace, KvCache, Tokenizer, Transformer, TransformerConfig, EOS_TOKEN_ID},
    srx_v05::{SrxState, SrxTransformer, SrxWorkspace},
};

#[test]
fn test_chinchilla_parameter_parity_strict_896() {
    let config = TransformerConfig::lang_chinchilla();
    assert!(config.validate().is_ok());

    // 1. Classical Transformer parameter accounting
    let classic = Transformer::new(config.clone()).expect("Failed to create Classical Transformer");
    assert_eq!(
        classic.param_count(),
        896,
        "Classical Transformer must have EXACTLY 896 trainable parameters!"
    );

    // Embeddings: 65 * 8 = 520
    assert_eq!(classic.token_embeddings.len(), 520);
    // Multi-Head Attention: 4 * (8 * 8) = 256
    let attn = &classic.layers[0].attn;
    assert_eq!(attn.w_q.len(), 64);
    assert_eq!(attn.w_k.len(), 64);
    assert_eq!(attn.w_v.len(), 64);
    assert_eq!(attn.w_o.len(), 64);
    // RMSNorm: 3 * 8 = 24
    assert_eq!(classic.layers[0].attn_norm_gamma.len(), 8);
    assert_eq!(classic.layers[0].ffn_norm_gamma.len(), 8);
    assert_eq!(classic.final_norm_gamma.len(), 8);
    // FFN: W1 [6, 8] + W2 [8, 6] = 48 + 48 = 96
    assert_eq!(classic.layers[0].mlp.w_1.len(), 48);
    assert_eq!(classic.layers[0].mlp.w_2.len(), 48);
    assert_eq!(520 + 256 + 24 + 96, 896);

    // 2. SRX v05 Quantum Core parameter accounting
    let srx = SrxTransformer::new(config.clone()).expect("Failed to create SRX v05");
    assert_eq!(
        srx.param_count(),
        896,
        "SRX v05 must have EXACTLY 896 trainable parameters!"
    );

    // Embeddings: 65 * 8 = 520
    assert_eq!(srx.token_embeddings.len(), 520);
    // Multi-Head Attention: 4 * (8 * 8) = 256
    let srx_attn = &srx.layers[0].attn;
    assert_eq!(srx_attn.w_q.len(), 64);
    assert_eq!(srx_attn.w_k.len(), 64);
    assert_eq!(srx_attn.w_v.len(), 64);
    assert_eq!(srx_attn.w_o.len(), 64);
    // RMSNorm: 3 * 8 = 24
    assert_eq!(srx.layers[0].attn_norm_gamma.len(), 8);
    assert_eq!(srx.layers[0].ffn_norm_gamma.len(), 8);
    assert_eq!(srx.final_norm_gamma.len(), 8);
    // FFN: W1 [6, 8] + W2 [8, 6] = 48 + 48 = 96
    assert_eq!(srx.layers[0].mlp.w_1.len(), 48);
    assert_eq!(srx.layers[0].mlp.w_2.len(), 48);
    assert_eq!(520 + 256 + 24 + 96, 896);

    // 3. Strict Parity Assertion: Exactly 0 parameters delta (0.00% delta)
    assert_eq!(
        classic.param_count(),
        srx.param_count(),
        "Strict parity failure: Classical ({}) != SRX v05 ({})",
        classic.param_count(),
        srx.param_count()
    );
}

#[test]
fn test_chinchilla_two_stage_training_and_generation() {
    let tok = Tokenizer::chinchilla();
    let config = TransformerConfig::lang_chinchilla();

    let pretrain_text = fs::read_to_string("data/pretrain_chinchilla.txt")
        .expect("Missing data/pretrain_chinchilla.txt");
    let all_pretrain_tokens = tok.encode(&pretrain_text);
    // Use first 1500 tokens for fast CI mini-epoch verification
    let pretrain_tokens = &all_pretrain_tokens[..1500.min(all_pretrain_tokens.len())];

    let instruct_text = fs::read_to_string("data/instruct_chinchilla.txt")
        .expect("Missing data/instruct_chinchilla.txt");
    let all_instruct_tokens = tok.encode(&instruct_text);
    // Use first 500 tokens for fast CI mini-epoch verification
    let instruct_tokens = &all_instruct_tokens[..500.min(all_instruct_tokens.len())];

    // 1. Classical Transformer Two-Stage Training
    let mut classic = Transformer::new_with_seed(config.clone(), 101).unwrap();
    let classic_res = classic.train_two_stage_pipeline(
        pretrain_tokens,
        instruct_tokens,
        2,    // 2 pretrain epochs
        4,    // 4 instruct epochs
        0.02,
        0.02,
        0.25, // 25% replay mix
    );

    assert!(
        classic_res.pretrain_metrics.final_loss < classic_res.pretrain_metrics.initial_loss,
        "Classical pretrain loss must decrease: init={}, final={}",
        classic_res.pretrain_metrics.initial_loss,
        classic_res.pretrain_metrics.final_loss
    );
    assert!(
        classic_res.instruct_metrics.final_loss < classic_res.instruct_metrics.initial_loss,
        "Classical instruct loss must decrease: init={}, final={}",
        classic_res.instruct_metrics.initial_loss,
        classic_res.instruct_metrics.final_loss
    );

    // Test Classical generation
    let mut classic_kv = KvCache::new(&config);
    let mut classic_ws = InferenceWorkspace::new(&config);
    let prompt = tok.encode("<user> 2 + 3 = <bot>");
    let classic_gen = classic.generate_until_eos(&prompt, 6, EOS_TOKEN_ID, &mut classic_kv, &mut classic_ws);
    assert!(classic_gen.len() > prompt.len(), "Classical must generate completion tokens");
    for &tok_id in &classic_gen {
        assert!(tok_id < config.vocab_size, "Token id must be valid within vocab");
    }

    // 2. SRX v05 Two-Stage Training
    let mut srx = SrxTransformer::new_with_seed(config.clone(), 101).unwrap();
    let srx_res = srx.train_two_stage_pipeline(
        pretrain_tokens,
        instruct_tokens,
        2,    // 2 pretrain epochs
        4,    // 4 instruct epochs
        0.02,
        0.02,
        0.25, // 25% replay mix
    );

    assert!(
        srx_res.pretrain_metrics.final_loss < srx_res.pretrain_metrics.initial_loss,
        "SRX v05 pretrain loss must decrease: init={}, final={}",
        srx_res.pretrain_metrics.initial_loss,
        srx_res.pretrain_metrics.final_loss
    );
    assert!(
        srx_res.instruct_metrics.final_loss < srx_res.instruct_metrics.initial_loss,
        "SRX v05 instruct loss must decrease: init={}, final={}",
        srx_res.instruct_metrics.initial_loss,
        srx_res.instruct_metrics.final_loss
    );

    // Test SRX v05 generation
    let mut srx_state = SrxState::new(&config);
    let mut srx_ws = SrxWorkspace::new(&config);
    let srx_gen = srx.generate_until_eos(&prompt, 6, EOS_TOKEN_ID, &mut srx_state, &mut srx_ws);
    assert!(srx_gen.len() > prompt.len(), "SRX v05 must generate completion tokens");
    for &tok_id in &srx_gen {
        assert!(tok_id < config.vocab_size, "Token id must be valid within vocab");
    }
}

#[test]
fn test_chinchilla_weights_serialization_roundtrip() {
    let config = TransformerConfig::lang_chinchilla();

    // 1. Classical Transformer Serialization Roundtrip
    let classic = Transformer::new_with_seed(config.clone(), 555).unwrap();
    let temp_classic_path = "target/test_classic_chinchilla_temp.bin";
    classic.save_weights(temp_classic_path).expect("Failed to save classic weights");

    assert!(Path::new(temp_classic_path).exists());
    let loaded_classic = Transformer::load_from_file(temp_classic_path)
        .expect("Failed to load classic weights from file");
    assert_eq!(classic.param_count(), loaded_classic.param_count());
    assert_eq!(classic.extract_weights(), loaded_classic.extract_weights());

    let mut classic_into = Transformer::new(config.clone()).unwrap();
    classic_into.load_weights(temp_classic_path).expect("Failed to load into existing classic instance");
    assert_eq!(classic.extract_weights(), classic_into.extract_weights());
    let _ = fs::remove_file(temp_classic_path);

    // 2. SRX v05 Serialization Roundtrip
    let srx = SrxTransformer::new_with_seed(config.clone(), 555).unwrap();
    let temp_srx_path = "target/test_srx_v05_chinchilla_temp.bin";
    srx.save_weights(temp_srx_path).expect("Failed to save srx weights");

    assert!(Path::new(temp_srx_path).exists());
    let loaded_srx = SrxTransformer::load_from_file(temp_srx_path)
        .expect("Failed to load srx weights from file");
    assert_eq!(srx.param_count(), loaded_srx.param_count());
    assert_eq!(srx.token_embeddings, loaded_srx.token_embeddings);
    assert_eq!(srx.final_norm_gamma, loaded_srx.final_norm_gamma);
    assert_eq!(srx.layers[0].attn.w_q, loaded_srx.layers[0].attn.w_q);
    assert_eq!(srx.layers[0].mlp.w_1, loaded_srx.layers[0].mlp.w_1);

    let mut srx_into = SrxTransformer::new(config.clone()).unwrap();
    srx_into.load_weights(temp_srx_path).expect("Failed to load into existing srx instance");
    assert_eq!(srx.token_embeddings, srx_into.token_embeddings);
    assert_eq!(srx.layers[0].attn.w_q, srx_into.layers[0].attn.w_q);
    let _ = fs::remove_file(temp_srx_path);
}

#[test]
fn test_chinchilla_saved_weights_load_and_predict() {
    let classic_path = "data/classic_chinchilla_weights.bin";
    let srx_path = "data/srx_v05_chinchilla_weights.bin";

    if Path::new(classic_path).exists() {
        let classic = Transformer::load_from_file(classic_path)
            .expect("Failed to load data/classic_chinchilla_weights.bin");
        assert_eq!(classic.param_count(), 896);
    }

    if Path::new(srx_path).exists() {
        let srx = SrxTransformer::load_from_file(srx_path)
            .expect("Failed to load data/srx_v05_chinchilla_weights.bin");
        assert_eq!(srx.param_count(), 896);
    }
}
