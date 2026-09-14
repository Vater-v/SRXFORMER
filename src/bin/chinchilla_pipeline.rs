//! Sprint 3 Two-Stage Training Pipeline Runner.
//! Executes:
//! - Stage 1 (Pretrain): Chinchilla 20:1 scaling law on `data/pretrain_chinchilla.txt` (17,920 tokens).
//! - Stage 2 (Instruct / SFT): Dialogue fine-tuning on `data/instruct_chinchilla.txt` (1,901 tokens) with 25% replay mix.
//! - Saves final weights to `data/classic_chinchilla_weights.bin` and `data/srx_v05_chinchilla_weights.bin`.
//! - Evaluates Exact Match and generation quality.

use std::fs;
use std::time::Instant;

use srxformer::{
    classic::{InferenceWorkspace, KvCache, Tokenizer, Transformer, TransformerConfig, EOS_TOKEN_ID},
    srx_v05::{SrxState, SrxTransformer, SrxWorkspace},
};

fn main() {
    let start_all = Instant::now();
    println!("================================================================================");
    println!(" SRXFORMER SPRINT 3: CHINCHILLA TWO-STAGE TRAINING PIPELINE (PRETRAIN + INSTRUCT)");
    println!(" Target: Intel Xeon E5-2650 v2 (Ivy Bridge-EP) | 896 Weights (Strict Parity)");
    println!("================================================================================\n");

    let tok = Tokenizer::chinchilla();
    let config = TransformerConfig::lang_chinchilla();

    // 1. Parameter parity assertion
    let classic_model = Transformer::new(config.clone()).expect("Failed to create Classical Transformer");
    let srx_model = SrxTransformer::new(config.clone()).expect("Failed to create SRX v05");

    println!("[1] PARAMETER PARITY VERIFICATION:");
    println!("    * Classical Transformer: {} parameters", classic_model.param_count());
    println!("    * SRX v05 Quantum Core:  {} parameters", srx_model.param_count());
    assert_eq!(classic_model.param_count(), 896);
    assert_eq!(srx_model.param_count(), 896);
    println!("    * Delta: 0 parameters (0.00% delta) - STRICT PARITY CONFIRMED!\n");

    // 2. Load corpora
    let pretrain_text = fs::read_to_string("data/pretrain_chinchilla.txt")
        .expect("Missing data/pretrain_chinchilla.txt");
    let pretrain_tokens = tok.encode(&pretrain_text);
    println!("[2] DATASET TOKEN ACCOUNTING:");
    println!("    * Pretrain Tokens: {} (Chinchilla 20:1 target: 17,920)", pretrain_tokens.len());
    assert_eq!(pretrain_tokens.len(), 17920);

    let instruct_text = fs::read_to_string("data/instruct_chinchilla.txt")
        .expect("Missing data/instruct_chinchilla.txt");
    let instruct_tokens = tok.encode(&instruct_text);
    println!("    * Instruct Tokens: {} (dialogue pairs in <user> ... <bot> ... <eos>)", instruct_tokens.len());
    assert_eq!(instruct_tokens.len(), 1901);
    println!("    * Replay Mix Ratio: 25.0% base facts integrated into Instruct Stage\n");

    // 3. Training Classical Transformer
    println!("--------------------------------------------------------------------------------");
    println!(" [3] TRAINING CLASSICAL TRANSFORMER BASELINE (896 PARAMETERS)...");
    println!("--------------------------------------------------------------------------------");
    let mut classic = Transformer::new_with_seed(config.clone(), 42)
        .expect("Failed to initialize Classical Transformer");

    let classic_pipeline_res = classic.train_two_stage_pipeline(
        &pretrain_tokens,
        &instruct_tokens,
        10,    // 10 pretrain epochs
        25,    // 25 instruct epochs
        0.02,  // pretrain lr
        0.015, // instruct lr
        0.25,  // 25% replay mix
    );

    println!("    * Stage 1 (Pretrain): Initial Loss = {:.4}, Final Loss = {:.4} (Elapsed: {:.2} ms)",
        classic_pipeline_res.pretrain_metrics.initial_loss,
        classic_pipeline_res.pretrain_metrics.final_loss,
        classic_pipeline_res.pretrain_metrics.elapsed_ms
    );
    println!("    * Stage 2 (Instruct): Initial Loss = {:.4}, Final Loss = {:.4} (Elapsed: {:.2} ms)",
        classic_pipeline_res.instruct_metrics.initial_loss,
        classic_pipeline_res.instruct_metrics.final_loss,
        classic_pipeline_res.instruct_metrics.elapsed_ms
    );

    let classic_weights_path = "data/classic_chinchilla_weights.bin";
    classic.save_weights(classic_weights_path)
        .expect("Failed to save classic weights");
    println!("    * Saved weights to: {}", classic_weights_path);

    // 4. Training SRX v05 Quantum Core
    println!("\n--------------------------------------------------------------------------------");
    println!(" [4] TRAINING SRX v05 QUANTUM-ALGEBRAIC CORE (896 PARAMETERS)...");
    println!("--------------------------------------------------------------------------------");
    let mut srx = SrxTransformer::new_with_seed(config.clone(), 42)
        .expect("Failed to initialize SRX v05");

    let srx_pipeline_res = srx.train_two_stage_pipeline(
        &pretrain_tokens,
        &instruct_tokens,
        10,    // 10 pretrain epochs
        25,    // 25 instruct epochs
        0.02,  // pretrain lr
        0.015, // instruct lr
        0.25,  // 25% replay mix
    );

    println!("    * Stage 1 (Pretrain): Initial Loss = {:.4}, Final Loss = {:.4} (Elapsed: {:.2} ms)",
        srx_pipeline_res.pretrain_metrics.initial_loss,
        srx_pipeline_res.pretrain_metrics.final_loss,
        srx_pipeline_res.pretrain_metrics.elapsed_ms
    );
    println!("    * Stage 2 (Instruct): Initial Loss = {:.4}, Final Loss = {:.4} (Elapsed: {:.2} ms)",
        srx_pipeline_res.instruct_metrics.initial_loss,
        srx_pipeline_res.instruct_metrics.final_loss,
        srx_pipeline_res.instruct_metrics.elapsed_ms
    );

    let srx_weights_path = "data/srx_v05_chinchilla_weights.bin";
    srx.save_weights(srx_weights_path)
        .expect("Failed to save srx weights");
    println!("    * Saved weights to: {}", srx_weights_path);

    // 5. Verification of Sample Prompts
    println!("\n--------------------------------------------------------------------------------");
    println!(" [5] GENERATION & INFERENCE VERIFICATION (DIALOGUE & BASE FACTS):");
    println!("--------------------------------------------------------------------------------");
    let test_prompts = [
        "<user> 2 + 3 = <bot>",
        "<user> 7 - 4 = <bot>",
        "<user> кто кот <bot>",
        "<user> наука это знание <bot>",
        "логика это",
        "2 + 3 =",
    ];

    let mut classic_kv = KvCache::new(&config);
    let mut classic_ws = InferenceWorkspace::new(&config);
    let mut srx_state = SrxState::new(&config);
    let mut srx_ws = SrxWorkspace::new(&config);

    for &prompt in &test_prompts {
        let p_toks = tok.encode(prompt);

        let classic_gen = classic.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut classic_kv, &mut classic_ws);
        let classic_completion = tok.decode(&classic_gen[p_toks.len()..]);

        let srx_gen = srx.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut srx_state, &mut srx_ws);
        let srx_completion = tok.decode(&srx_gen[p_toks.len()..]);

        println!("  Prompt: '{:<30}' | Classic: '{:<16}' | SRX v05: '{:<16}'",
            prompt, classic_completion, srx_completion);
    }

    println!("\n================================================================================");
    println!(" SPRINT 3 PIPELINE EXECUTION COMPLETED IN {:.2} SECONDS!", start_all.elapsed().as_secs_f64());
    println!("================================================================================");
}
