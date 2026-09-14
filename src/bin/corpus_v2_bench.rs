use std::fs;
use std::time::Instant;

use srxformer::{
    classic::{
        InferenceTelemetry, InferenceWorkspace, KvCache, TestCaseResult, TelemetryReport,
        Tokenizer, Transformer, TransformerConfig, EOS_TOKEN_ID,
    },
    srx_v03::{
        SrxState as SrxStateV03, SrxTelemetryReport as SrxTelemetryReportV03,
        SrxTransformer as SrxTransformerV03, SrxWorkspace as SrxWorkspaceV03,
    },
};

fn main() {
    println!("=================================================================================================================");
    println!(" SRXformer: Scaled Corpus v2 Benchmark (Classical Baseline vs SRX v03 Golden Core)                              ");
    println!(" Target Hardware: Intel Xeon E5-2650 v2 (Ivy Bridge-EP, AVX FP32, L1D 32KB per core)                           ");
    println!(" Dataset: data/unified_corpus_v2.txt (2,940 tokens, 440 unique lines, strictly 0 duplicates)                     ");
    println!(" Control Suite: 30 Heterogeneous Control Tasks                                                                   ");
    println!("=================================================================================================================\n");

    // -------------------------------------------------------------------------
    // 1. CONFIGURATION
    // -------------------------------------------------------------------------
    let config = TransformerConfig::lang_v2();
    let tokenizer = Tokenizer::new();

    println!("[1] Architecture & Parameter Accounting (Config: lang_v2):");
    println!("    * Vocab Size (V):    {} tokens (data/vocab_v2.txt)", config.vocab_size);
    println!("    * Hidden Dim (d):     {}", config.d_model);
    println!("    * Attention Heads:    {} (head_dim = {})", config.n_heads, config.head_dim());
    println!("    * Decoder Layers:     {}", config.n_layers);
    println!("    * FFN Dim (d_ff):     {}", config.d_ff);
    println!("    * Context Window:     {} tokens", config.max_seq_len);
    println!("    * Normalization:      {:?} (eps = {:.1e})", config.norm_type, config.eps);
    println!("    * Activation:         {:?}", config.activation);
    println!("    * Positional Enc:     {:?} (0 params)", config.pos_encoding);
    println!("    * Tied LM Head:       {}", config.tie_word_embeddings);
    println!("    ----------------------------------------------------------------");
    println!("    * Classical Transformer Parameters: 864 params (3,456 bytes -> 100% L1D Cache Resident < 32 KB)");
    println!("    * SRX v03 Golden Core Parameters:   882 params (3,528 bytes -> 100% L1D Cache Resident < 32 KB)");
    println!("      (Includes Selective Gate: 2 * 8 + 2 = 18 params)");
    println!();

    // -------------------------------------------------------------------------
    // 2. CORPUS v2 INTEGRITY VERIFICATION
    // -------------------------------------------------------------------------
    println!("[2] Loading Unified Corpus v2:");
    let unified_path = "data/unified_corpus_v2.txt";
    let corpus_text = fs::read_to_string(unified_path).unwrap_or_else(|_| {
        panic!("Failed to read {}. Run `cargo run --bin generate_data` first.", unified_path)
    });

    let lines: Vec<&str> = corpus_text.lines().filter(|l| !l.is_empty()).collect();
    let mut seen = std::collections::HashSet::new();
    for l in &lines {
        assert!(l.ends_with("<eos>"), "Line must end with <eos>: {}", l);
        assert!(seen.insert(*l), "Duplicate line detected: {}", l);
    }

    let tokens = tokenizer.encode(&corpus_text);
    println!("    * Loaded from {}:", unified_path);
    println!("      - Total sentences:  {} (strictly unique, 0 duplicates)", lines.len());
    println!("      - Total tokens:     {} tokens (TARGET: 2,940)", tokens.len());
    println!("      - Corpus integrity: PASSED (every sentence terminated with <eos>)");
    assert_eq!(tokens.len(), 2940, "Corpus v2 must contain exactly 2,940 tokens");
    println!();

    // -------------------------------------------------------------------------
    // 3. 30 HETEROGENEOUS CONTROL TASKS
    // -------------------------------------------------------------------------
    let test_cases = [
        // 1. Basic Addition (2)
        ("<user> 2 + 3 = <bot>", "5 <eos>", "Addition Arithmetic (Basic)"),
        ("<user> 1 + 2 = <bot>", "3 <eos>", "Addition Arithmetic (Basic)"),
        // 2. Extended Addition (3)
        ("<user> 3 + 4 = <bot>", "7 <eos>", "Addition Arithmetic (Extended)"),
        ("<user> 5 + 3 = <bot>", "8 <eos>", "Addition Arithmetic (Extended)"),
        ("<user> 4 + 5 = <bot>", "9 <eos>", "Addition Arithmetic (Extended)"),
        // 3. Basic Subtraction (2)
        ("<user> 4 - 1 = <bot>", "3 <eos>", "Subtraction Arithmetic (Basic)"),
        ("<user> 5 - 2 = <bot>", "3 <eos>", "Subtraction Arithmetic (Basic)"),
        // 4. Extended Subtraction (3)
        ("<user> 9 - 4 = <bot>", "5 <eos>", "Subtraction Arithmetic (Extended)"),
        ("<user> 8 - 3 = <bot>", "5 <eos>", "Subtraction Arithmetic (Extended)"),
        ("<user> 7 - 2 = <bot>", "5 <eos>", "Subtraction Arithmetic (Extended)"),
        // 5. Multiplication (3)
        ("<user> 2 * 3 = <bot>", "6 <eos>", "Multiplication Arithmetic"),
        ("<user> 2 * 2 = <bot>", "4 <eos>", "Multiplication Arithmetic"),
        ("<user> 3 * 3 = <bot>", "9 <eos>", "Multiplication Arithmetic"),
        // 6. Entity Definitions (4)
        ("<user> кто кот <bot>", "кот это животное <eos>", "Entity Fact / Definition"),
        ("<user> кто пес <bot>", "пес это друг <eos>", "Entity Fact / Definition"),
        ("<user> кто волк <bot>", "волк это зверь <eos>", "Entity Fact / Definition"),
        ("<user> кто лиса <bot>", "лиса это хищник <eos>", "Entity Fact / Definition"),
        // 7. Spatial Reasoning (4)
        ("<user> где волк <bot>", "лес <eos>", "Spatial Reasoning"),
        ("<user> где рыба <bot>", "река <eos>", "Spatial Reasoning"),
        ("<user> где кот <bot>", "дом <eos>", "Spatial Reasoning"),
        ("<user> где птица <bot>", "небо <eos>", "Spatial Reasoning"),
        // 8. Logical Negations (2)
        ("<user> кот это пес <bot>", "нет <eos>", "Boolean Logic Negation"),
        ("<user> волк это пес <bot>", "нет <eos>", "Boolean Logic Negation"),
        // 9. Logical Assertions (2)
        ("<user> кот это животное <bot>", "да <eos>", "Boolean Logic Affirmation"),
        ("<user> волк это зверь <bot>", "да <eos>", "Boolean Logic Affirmation"),
        // 10. Base Anti-Forgetting Formulations (5)
        ("2 + 3 =", "5 <eos>", "Anti-Forgetting Base Addition"),
        ("4 - 1 =", "3 <eos>", "Anti-Forgetting Base Subtraction"),
        ("2 * 3 =", "6 <eos>", "Anti-Forgetting Base Multiplication"),
        ("волк это зверь =", "да <eos>", "Anti-Forgetting Base Logic"),
        ("где рыба =", "река <eos>", "Anti-Forgetting Base Spatial"),
    ];

    const EPOCHS: usize = 280;
    const LR_CLASSIC: f32 = 0.015;
    const LR_SRX: f32 = 0.022;
    const BENCH_STEPS: usize = 50_000;

    // =========================================================================
    // 4. CLASSICAL TRANSFORMER BASELINE ON CORPUS v2
    // =========================================================================
    println!("================================================================================");
    println!(" [3] RUNNING CLASSICAL TRANSFORMER BASELINE (CORPUS v2)");
    println!("================================================================================");
    let mut classic_model = Transformer::new_with_seed(config.clone(), 100)
        .expect("Failed to initialize Classical Transformer");

    println!("    * Training Classical Transformer for {} epochs...", EPOCHS);
    let classic_train_telemetry = classic_model.train_dataset(&tokens, EPOCHS, LR_CLASSIC);
    println!("      - Initial Loss: {:.4} (PPL: {:.2})", classic_train_telemetry.initial_loss, classic_train_telemetry.initial_perplexity);
    println!("      - Final Loss:   {:.4} (PPL: {:.2})", classic_train_telemetry.final_loss, classic_train_telemetry.final_perplexity);
    println!("      - Elapsed Time: {:.2} ms", classic_train_telemetry.elapsed_ms);

    let mut classic_ws = InferenceWorkspace::new(&config);
    let mut kv = KvCache::new(&config);
    let mut classic_test_results = Vec::new();
    let mut classic_passed = 0;

    println!("    * Evaluating Quality (30 control tasks):");
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = classic_model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut kv, &mut classic_ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        let passed = gen_text == expected;
        if passed { classic_passed += 1; }
        println!(
            "      [{}] {:<34}: \"{:<24}\" -> \"{:<22}\"",
            if passed { "PASS" } else { "FAIL" }, category, prompt, gen_text
        );
        classic_test_results.push(TestCaseResult {
            prompt: prompt.to_string(),
            generated: gen_text,
            expected: expected.to_string(),
            category: category.to_string(),
            passed,
        });
    }

    kv.reset();
    let b_start = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % config.max_seq_len;
        if pos == 0 { kv.reset(); }
        classic_model.step(i % config.vocab_size, pos, &mut kv, &mut classic_ws);
    }
    let b_elapsed = b_start.elapsed();
    let classic_step_ns = b_elapsed.as_nanos() as f64 / BENCH_STEPS as f64;
    let classic_step_us = classic_step_ns / 1000.0;
    let classic_tok_sec = BENCH_STEPS as f64 / b_elapsed.as_secs_f64();
    let classic_flops_per_token = 2 * config.param_count() as u64;
    let classic_inf_gflops = (classic_flops_per_token as f64 * classic_tok_sec) / 1e9;

    let classic_inf_telemetry = InferenceTelemetry {
        flops_per_token: classic_flops_per_token,
        bench_steps: BENCH_STEPS,
        step_latency_ns: classic_step_ns,
        step_latency_us: classic_step_us,
        tokens_per_sec: classic_tok_sec,
        gflops_per_sec: classic_inf_gflops,
    };

    let classic_report = TelemetryReport::new(
        classic_train_telemetry.clone(),
        classic_inf_telemetry.clone(),
        classic_test_results.clone(),
    );
    classic_report.save_to_file("telemetry_classic_corpus_v2.txt").expect("Failed to write telemetry_classic_corpus_v2.txt");
    println!("    * Classical report saved to: telemetry_classic_corpus_v2.txt\n");

    // =========================================================================
    // 5. SRXFORMER v03 (GOLDEN CORE) ON CORPUS v2
    // =========================================================================
    println!("================================================================================");
    println!(" [4] RUNNING SUPER-RESOLVENT XFORMER v03 (CORPUS v2)");
    println!("================================================================================");
    let mut srx_v03_model = SrxTransformerV03::new_with_seed(config.clone(), 42)
        .expect("Failed to initialize SRX v03");

    println!("    * Training SRX v03 for {} epochs...", EPOCHS);
    let srx_v03_train_telemetry = srx_v03_model.train_dataset(&tokens, EPOCHS, LR_SRX);
    println!("      - Initial Loss: {:.4} (PPL: {:.2})", srx_v03_train_telemetry.initial_loss, srx_v03_train_telemetry.initial_perplexity);
    println!("      - Final Loss:   {:.4} (PPL: {:.2})", srx_v03_train_telemetry.final_loss, srx_v03_train_telemetry.final_perplexity);
    println!("      - Elapsed Time: {:.2} ms", srx_v03_train_telemetry.elapsed_ms);

    let srx_v03_weights_path = "data/srx_v03_corpus_v2_weights.bin";
    srx_v03_model.save_weights(srx_v03_weights_path).expect("Failed to save SRX v03 corpus v2 weights");
    let loaded_srx_v03 = SrxTransformerV03::load_from_file(srx_v03_weights_path).expect("Failed to load SRX v03 model");

    let mut srx_v03_ws = SrxWorkspaceV03::new(&config);
    let mut srx_v03_state = SrxStateV03::new(&config);
    let mut srx_v03_test_results = Vec::new();
    let mut srx_v03_passed = 0;

    println!("    * Evaluating Quality (30 control tasks):");
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = loaded_srx_v03.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut srx_v03_state, &mut srx_v03_ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        let passed = gen_text == expected;
        if passed { srx_v03_passed += 1; }
        println!(
            "      [{}] {:<34}: \"{:<24}\" -> \"{:<22}\"",
            if passed { "PASS" } else { "FAIL" }, category, prompt, gen_text
        );
        srx_v03_test_results.push(TestCaseResult {
            prompt: prompt.to_string(),
            generated: gen_text,
            expected: expected.to_string(),
            category: category.to_string(),
            passed,
        });
    }

    srx_v03_state.reset();
    let srx_b_start = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % config.max_seq_len;
        if pos == 0 { srx_v03_state.reset(); }
        loaded_srx_v03.step(i % config.vocab_size, pos, &mut srx_v03_state, &mut srx_v03_ws);
    }
    let srx_b_elapsed = srx_b_start.elapsed();
    let srx_step_ns = srx_b_elapsed.as_nanos() as f64 / BENCH_STEPS as f64;
    let srx_step_us = srx_step_ns / 1000.0;
    let srx_tok_sec = BENCH_STEPS as f64 / srx_b_elapsed.as_secs_f64();
    let srx_flops_per_token = 2 * loaded_srx_v03.param_count() as u64;
    let srx_inf_gflops = (srx_flops_per_token as f64 * srx_tok_sec) / 1e9;

    let srx_v03_inf_telemetry = InferenceTelemetry {
        flops_per_token: srx_flops_per_token,
        bench_steps: BENCH_STEPS,
        step_latency_ns: srx_step_ns,
        step_latency_us: srx_step_us,
        tokens_per_sec: srx_tok_sec,
        gflops_per_sec: srx_inf_gflops,
    };

    let srx_v03_report = SrxTelemetryReportV03::new(
        srx_v03_train_telemetry.clone(),
        srx_v03_inf_telemetry.clone(),
        srx_v03_test_results.clone(),
        srx_v03_state.memory_bytes(),
        config.max_seq_len,
    );
    srx_v03_report.save_to_file("telemetry_srx_v03_corpus_v2.txt").expect("Failed to write telemetry_srx_v03_corpus_v2.txt");
    println!("    * SRX v03 report saved to: telemetry_srx_v03_corpus_v2.txt\n");

    // =========================================================================
    // 6. COMPARATIVE SUMMARY TABLE
    // =========================================================================
    println!("=================================================================================================");
    println!(" [5] COMPARATIVE SUMMARY TABLE: CORPUS v2 (2,940 TOKENS, 30 CONTROL TASKS)");
    println!("=================================================================================================");
    println!(" | {:<32} | {:<25} | {:<25} |", "Метрика", "Classical Baseline", "SRX v03 Golden Core");
    println!(" |----------------------------------|---------------------------|---------------------------|");
    println!(" | {:<32} | {:<25} | {:<25} |", "Trainable Parameters", format!("{} params (3,456 B)", config.param_count()), format!("{} params (3,528 B)", loaded_srx_v03.param_count()));
    println!(" | {:<32} | {:<25} | {:<25} |", "Memory State Footprint", format!("{} B (KV Cache)", 2 * config.n_layers * config.max_seq_len * config.d_model * 4), format!("{} B (O(1) L1D resident)", srx_v03_state.memory_bytes()));
    println!(" | {:<32} | {:<25} | {:<25} |", "Initial Loss -> Final Loss", format!("{:.4} -> {:.4}", classic_train_telemetry.initial_loss, classic_train_telemetry.final_loss), format!("{:.4} -> {:.4}", srx_v03_train_telemetry.initial_loss, srx_v03_train_telemetry.final_loss));
    println!(" | {:<32} | {:<25} | {:<25} |", "Training FLOPs", format!("{:.2} MFLOPs", classic_train_telemetry.total_training_flops as f64 / 1e6), format!("{:.2} MFLOPs", srx_v03_train_telemetry.total_training_flops as f64 / 1e6));
    println!(" | {:<32} | {:<25} | {:<25} |", "Step Latency (single token)", format!("{:.2} ns ({:.3} µs)", classic_step_ns, classic_step_us), format!("{:.2} ns ({:.3} µs)", srx_step_ns, srx_step_us));
    println!(" | {:<32} | {:<25} | {:<25} |", "Throughput (tokens/sec)", format!("{:.0} tok/s", classic_tok_sec), format!("{:.0} tok/s", srx_tok_sec));
    println!(" | {:<32} | {:<25} | {:<25} |", "Exact Match Accuracy (30 tests)", format!("{}/30 ({:.1}%)", classic_passed, (classic_passed as f64 / 30.0) * 100.0), format!("{}/30 ({:.1}%)", srx_v03_passed, (srx_v03_passed as f64 / 30.0) * 100.0));
    println!("=================================================================================================\n");
}
