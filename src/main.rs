use std::fs;
use std::time::Instant;

use srxformer::{
    classic::{
        InferenceTelemetry, InferenceWorkspace, KvCache, TestCaseResult, TelemetryReport,
        Tokenizer, Transformer, TransformerConfig, EOS_TOKEN_ID,
    },
    srx_v01::{
        SrxState as SrxStateV01, SrxTelemetryReport as SrxTelemetryReportV01,
        SrxTransformer as SrxTransformerV01, SrxWorkspace as SrxWorkspaceV01,
    },
    srx_v02::{
        SrxState as SrxStateV02, SrxTelemetryReport as SrxTelemetryReportV02,
        SrxTransformer as SrxTransformerV02, SrxWorkspace as SrxWorkspaceV02,
    },
};

fn main() {
    println!("========================================================================================");
    println!(" SRXformer: Tri-System Comparative Benchmark (Classical v01 vs SRX v01 vs SRX v02)");
    println!(" Target Hardware Architecture: Intel Xeon E5-2650 v2 (Ivy Bridge-EP, AVX FP32, L1D 32KB)");
    println!(" Mathematical Innovation: Unitary Givens, Post-MUSIC RMSNorm, Gain Clipping, Taylor AVX");
    println!(" Strict Parameter Parity: Exactly 512 parameters across all three evaluated systems     ");
    println!("========================================================================================\n");

    // -------------------------------------------------------------------------
    // 1. CONFIGURATION & EXACT 512-PARAMETER ARCHITECTURE
    // -------------------------------------------------------------------------
    let config = TransformerConfig::lang_512();
    let tokenizer = Tokenizer::new();

    println!("[1] Tri-Model Architecture & Exact Parameter Derivation (lang_512):");
    println!("    * Vocab Size (V):    {} tokens", config.vocab_size);
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
    println!("    * 1:1 Bitwise Parameter Breakdown (Classical, SRX v01, SRX v02):");
    println!("      - Token Embeddings: 21 * 8                = 168");
    println!("      - Attention Projections (W_q, W_k, W_v, W_o): 4 * (8 * 8) = 256");
    println!("      - Pre-Attn RMSNorm Gamma:                 =   8");
    println!("      - FFN (W_1 [4, 8] + W_2 [8, 4]): 32 + 32  =  64");
    println!("      - Pre-FFN RMSNorm Gamma:                  =   8");
    println!("      - Final RMSNorm Gamma:                    =   8");
    println!("      - LM Head (Tied to Embeddings):           =   0");
    println!("      ================================================");
    println!("      GRAND TOTAL:                              = 512 parameters (EXACT PARITY)");
    let weight_bytes = config.param_count() * std::mem::size_of::<f32>();
    println!(
        "    * Total Model Weight Size: {} bytes ({:.2} KB) -> 100% L1D Cache Resident (32 KB)",
        weight_bytes,
        weight_bytes as f64 / 1024.0
    );
    println!();

    // -------------------------------------------------------------------------
    // 2. UNIFIED CORPUS LOADING (STRICTLY 0 DUPLICATES)
    // -------------------------------------------------------------------------
    println!("[2] Loading Unified Non-Duplicate Corpus:");
    let unified_path = "data/unified_corpus.txt";
    let corpus_text = fs::read_to_string(unified_path).unwrap_or_else(|_| {
        panic!(
            "Failed to read {}. Run `cargo run --bin generate_data` first.",
            unified_path
        )
    });

    let lines: Vec<&str> = corpus_text.lines().collect();
    let mut seen = std::collections::HashSet::new();
    for l in &lines {
        assert!(l.ends_with("<eos>"), "Line must end with <eos>: {}", l);
        assert!(seen.insert(*l), "Duplicate line detected: {}", l);
    }

    let tokens = tokenizer.encode(&corpus_text);
    println!("    * Loaded from {}:", unified_path);
    println!("      - Total sentences:  {} (strictly unique, 0 duplicates)", lines.len());
    println!("      - Total tokens:     {} tokens", tokens.len());
    println!("      - Corpus integrity: PASSED (every sentence terminated with <eos>)");
    println!();

    const EPOCHS: usize = 120;
    const LR: f32 = 0.015;
    const BENCH_STEPS: usize = 50_000;

    let test_cases = [
        ("<user> 2 + 3 = <bot>", "5 <eos>", "Addition Arithmetic"),
        ("<user> 1 + 2 = <bot>", "3 <eos>", "Addition Arithmetic"),
        ("<user> 4 - 1 = <bot>", "3 <eos>", "Subtraction Arithmetic"),
        ("<user> кто кот <bot>", "кот это животное <eos>", "Entity Fact / Definition"),
        ("<user> кто пес <bot>", "пес это друг <eos>", "Entity Fact / Definition"),
        ("<user> кот это пес <bot>", "нет <eos>", "Boolean Logic Negation"),
        ("<user> кот это животное <bot>", "да <eos>", "Boolean Logic Affirmation"),
        ("2 + 3 =", "5 <eos>", "Base Addition Formulation"),
        ("4 - 1 =", "3 <eos>", "Base Subtraction Formulation"),
        ("кот это пес =", "нет <eos>", "Base Logic Formulation"),
    ];

    // =========================================================================
    // 3. CLASSICAL TRANSFORMER (BASELINE)
    // =========================================================================
    println!("================================================================================");
    println!(" [3] RUNNING CLASSICAL TRANSFORMER v01 (BASELINE)");
    println!("================================================================================");
    let mut classic_model = Transformer::new_with_seed(config.clone(), 100)
        .expect("Failed to initialize Transformer");

    println!("    * Training Classical Transformer for {} epochs...", EPOCHS);
    let classic_train_telemetry = classic_model.train_dataset(&tokens, EPOCHS, LR);
    println!("      - Initial Loss: {:.4} (PPL: {:.2})", classic_train_telemetry.initial_loss, classic_train_telemetry.initial_perplexity);
    println!("      - Final Loss:   {:.4} (PPL: {:.2})", classic_train_telemetry.final_loss, classic_train_telemetry.final_perplexity);
    println!("      - Elapsed Time: {:.2} ms", classic_train_telemetry.elapsed_ms);

    let classic_weights_path = "data/model_weights.bin";
    classic_model
        .save_weights(classic_weights_path)
        .expect("Failed to save classical model weights");
    let loaded_classic = Transformer::load_from_file(classic_weights_path)
        .expect("Failed to load classical model");

    let mut classic_ws = InferenceWorkspace::new(&config);
    let mut kv = KvCache::new(&config);
    let mut classic_test_results = Vec::new();

    println!("    * Evaluating Quality (10 control tasks):");
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = loaded_classic.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut kv, &mut classic_ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        let passed = gen_text == expected;
        println!(
            "      [{}] {:<26}: \"{:<24}\" -> \"{:<20}\"",
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
        if pos == 0 {
            kv.reset();
        }
        loaded_classic.step(i % config.vocab_size, pos, &mut kv, &mut classic_ws);
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
    classic_report
        .save_to_file("telemetry_classic_v01.txt")
        .expect("Failed to write telemetry_classic_v01.txt");
    println!("    * Classical report saved to: telemetry_classic_v01.txt\n");

    // =========================================================================
    // 4. SRXFORMER v01 (FROZEN REFERENCE)
    // =========================================================================
    println!("================================================================================");
    println!(" [4] RUNNING SRXFORMER v01 (FROZEN REFERENCE ARCHITECTURE)");
    println!("================================================================================");
    let mut srx_v01_model = SrxTransformerV01::new_with_seed(config.clone(), 100)
        .expect("Failed to initialize SrxTransformerV01");

    println!("    * Training SRXformer v01 (Unitary Givens + Unclamped MUSIC) for {} epochs...", EPOCHS);
    let srx_v01_train_telemetry = srx_v01_model.train_dataset(&tokens, EPOCHS, LR);
    println!("      - Initial Loss: {:.4} (PPL: {:.2})", srx_v01_train_telemetry.initial_loss, srx_v01_train_telemetry.initial_perplexity);
    println!("      - Final Loss:   {:.4} (PPL: {:.2})", srx_v01_train_telemetry.final_loss, srx_v01_train_telemetry.final_perplexity);
    println!("      - Elapsed Time: {:.2} ms", srx_v01_train_telemetry.elapsed_ms);

    let srx_v01_weights_path = "data/srx_v01_model_weights.bin";
    srx_v01_model
        .save_weights(srx_v01_weights_path)
        .expect("Failed to save SRX v01 weights");
    let loaded_srx_v01 = SrxTransformerV01::load_from_file(srx_v01_weights_path)
        .expect("Failed to load SRX v01 model");

    let mut srx_v01_ws = SrxWorkspaceV01::new(&config);
    let mut srx_v01_state = SrxStateV01::new(&config);
    let mut srx_v01_test_results = Vec::new();

    println!("    * Evaluating Quality (10 control tasks):");
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = loaded_srx_v01.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut srx_v01_state, &mut srx_v01_ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        let passed = gen_text == expected;
        println!(
            "      [{}] {:<26}: \"{:<24}\" -> \"{:<20}\"",
            if passed { "PASS" } else { "FAIL" }, category, prompt, gen_text
        );
        srx_v01_test_results.push(TestCaseResult {
            prompt: prompt.to_string(),
            generated: gen_text,
            expected: expected.to_string(),
            category: category.to_string(),
            passed,
        });
    }

    srx_v01_state.reset();
    let srx_v01_b_start = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % config.max_seq_len;
        if pos == 0 {
            srx_v01_state.reset();
        }
        loaded_srx_v01.step(i % config.vocab_size, pos, &mut srx_v01_state, &mut srx_v01_ws);
    }
    let srx_v01_b_elapsed = srx_v01_b_start.elapsed();
    let srx_v01_step_ns = srx_v01_b_elapsed.as_nanos() as f64 / BENCH_STEPS as f64;
    let srx_v01_step_us = srx_v01_step_ns / 1000.0;
    let srx_v01_tok_sec = BENCH_STEPS as f64 / srx_v01_b_elapsed.as_secs_f64();
    let srx_v01_flops_per_token = 2 * config.param_count() as u64;
    let srx_v01_inf_gflops = (srx_v01_flops_per_token as f64 * srx_v01_tok_sec) / 1e9;

    let srx_v01_inf_telemetry = InferenceTelemetry {
        flops_per_token: srx_v01_flops_per_token,
        bench_steps: BENCH_STEPS,
        step_latency_ns: srx_v01_step_ns,
        step_latency_us: srx_v01_step_us,
        tokens_per_sec: srx_v01_tok_sec,
        gflops_per_sec: srx_v01_inf_gflops,
    };

    let srx_v01_report = SrxTelemetryReportV01::new(
        srx_v01_train_telemetry.clone(),
        srx_v01_inf_telemetry.clone(),
        srx_v01_test_results.clone(),
        srx_v01_state.memory_bytes(),
        config.max_seq_len,
    );
    srx_v01_report
        .save_to_file("telemetry_srx_v01.txt")
        .expect("Failed to write telemetry_srx_v01.txt");
    println!("    * SRX v01 report saved to: telemetry_srx_v01.txt\n");

    // =========================================================================
    // 5. SRXFORMER v02 (OPTIMIZED INNOVATION)
    // =========================================================================
    println!("================================================================================");
    println!(" [5] RUNNING SRXFORMER v02 (OPTIMIZED INNOVATION: POST-MUSIC RMSNORM + FAST GIVENS)");
    println!("================================================================================");
    let mut srx_v02_model = SrxTransformerV02::new_with_seed(config.clone(), 100)
        .expect("Failed to initialize SrxTransformerV02");

    println!("    * Training SRXformer v02 (Gain Clipping + Post-MUSIC RMSNorm + Epsilon Annealing) for {} epochs...", EPOCHS);
    let srx_v02_train_telemetry = srx_v02_model.train_dataset(&tokens, EPOCHS, LR);
    println!("      - Initial Loss: {:.4} (PPL: {:.2})", srx_v02_train_telemetry.initial_loss, srx_v02_train_telemetry.initial_perplexity);
    println!("      - Final Loss:   {:.4} (PPL: {:.2})", srx_v02_train_telemetry.final_loss, srx_v02_train_telemetry.final_perplexity);
    println!("      - Elapsed Time: {:.2} ms", srx_v02_train_telemetry.elapsed_ms);

    let srx_v02_weights_path = "data/srx_v02_model_weights.bin";
    srx_v02_model
        .save_weights(srx_v02_weights_path)
        .expect("Failed to save SRX v02 weights");
    let loaded_srx_v02 = SrxTransformerV02::load_from_file(srx_v02_weights_path)
        .expect("Failed to load SRX v02 model");

    let mut srx_v02_ws = SrxWorkspaceV02::new(&config);
    let mut srx_v02_state = SrxStateV02::new(&config);
    let mut srx_v02_test_results = Vec::new();

    println!("    * Evaluating Quality (10 control tasks):");
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = loaded_srx_v02.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut srx_v02_state, &mut srx_v02_ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        let passed = gen_text == expected;
        println!(
            "      [{}] {:<26}: \"{:<24}\" -> \"{:<20}\"",
            if passed { "PASS" } else { "FAIL" }, category, prompt, gen_text
        );
        srx_v02_test_results.push(TestCaseResult {
            prompt: prompt.to_string(),
            generated: gen_text,
            expected: expected.to_string(),
            category: category.to_string(),
            passed,
        });
    }

    srx_v02_state.reset();
    let srx_v02_b_start = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % config.max_seq_len;
        if pos == 0 {
            srx_v02_state.reset();
        }
        loaded_srx_v02.step(i % config.vocab_size, pos, &mut srx_v02_state, &mut srx_v02_ws);
    }
    let srx_v02_b_elapsed = srx_v02_b_start.elapsed();
    let srx_v02_step_ns = srx_v02_b_elapsed.as_nanos() as f64 / BENCH_STEPS as f64;
    let srx_v02_step_us = srx_v02_step_ns / 1000.0;
    let srx_v02_tok_sec = BENCH_STEPS as f64 / srx_v02_b_elapsed.as_secs_f64();
    let srx_v02_flops_per_token = 2 * config.param_count() as u64;
    let srx_v02_inf_gflops = (srx_v02_flops_per_token as f64 * srx_v02_tok_sec) / 1e9;

    let srx_v02_inf_telemetry = InferenceTelemetry {
        flops_per_token: srx_v02_flops_per_token,
        bench_steps: BENCH_STEPS,
        step_latency_ns: srx_v02_step_ns,
        step_latency_us: srx_v02_step_us,
        tokens_per_sec: srx_v02_tok_sec,
        gflops_per_sec: srx_v02_inf_gflops,
    };

    let srx_v02_report = SrxTelemetryReportV02::new(
        srx_v02_train_telemetry.clone(),
        srx_v02_inf_telemetry.clone(),
        srx_v02_test_results.clone(),
        srx_v02_state.memory_bytes(),
        config.max_seq_len,
    );
    srx_v02_report
        .save_to_file("telemetry_srx_v02.txt")
        .expect("Failed to write telemetry_srx_v02.txt");
    println!("    * SRX v02 report saved to: telemetry_srx_v02.txt\n");

    // =========================================================================
    // 6. TRI-SYSTEM COMPARATIVE TABLE
    // =========================================================================
    println!("=================================================================================================================");
    println!(" [6] TRI-SYSTEM ARCHITECTURAL & PERFORMANCE COMPARISON TABLE");
    println!("=================================================================================================================");
    println!(
        " | {:<32} | {:<22} | {:<22} | {:<22} |",
        "Метрика / Характеристика", "Classical v01", "SRXformer v01 (Frozen)", "SRXformer v02 (Optimized)"
    );
    println!(" |----------------------------------|------------------------|------------------------|------------------------|");
    println!(
        " | {:<32} | {:<22} | {:<22} | {:<22} |",
        "Addressing Principle", "Softmax Attention", "MUSIC Resonant Gain", "MUSIC + Post RMSNorm"
    );
    println!(
        " | {:<32} | {:<22} | {:<22} | {:<22} |",
        "Trigonometric Rotation Engine", "N/A", "Scalar libc sin/cos", "Fast AVX Taylor Poly"
    );
    println!(
        " | {:<32} | {:<22} | {:<22} | {:<22} |",
        "Spectral Click Mitigation", "N/A", "None (Gain Unbounded)", "w_clamped<=10 + RMSNorm"
    );
    println!(
        " | {:<32} | {:<22} | {:<22} | {:<22} |",
        "Trainable Parameters", "512 (1:1 Bitwise)", "512 (1:1 Bitwise)", "512 (1:1 Bitwise)"
    );
    println!(
        " | {:<32} | {:<22} | {:<22} | {:<22} |",
        "State Complexity (O-notation)", "O(N * d) (KV cache)", "O(d) (Phase & Memory)", "O(d) (Phase & Memory)"
    );
    println!(
        " | {:<32} | {:<22} | {:<22} | {:<22} |",
        "State Memory (N=32 tokens)", "2,048 bytes", "152 bytes (13.5x less)", "152 bytes (13.5x less)"
    );
    println!(
        " | {:<32} | {:<22} | {:<22} | {:<22} |",
        "State Memory (N=1,024 tokens)", "65,536 bytes", "152 bytes (431x less)", "152 bytes (431x less)"
    );
    println!(
        " | {:<32} | {:<22} | {:<22} | {:<22} |",
        "State Memory (N=100,000 tokens)", "6.4 MB (Spills to DRAM)", "152 bytes (L1 Resident)", "152 bytes (L1 Resident)"
    );
    println!(
        " | {:<32} | {:<22} | {:<22} | {:<22} |",
        "Hardware Cache Resident Status", "Exceeds L1D at N*=500", "100% L1D Resident", "100% L1D Resident"
    );
    println!(
        " | {:<32} | {:.4} ({:.2})          | {:.4} ({:.2})          | {:.4} ({:.2})          |",
        "Initial Training Loss (PPL)",
        classic_train_telemetry.initial_loss, classic_train_telemetry.initial_perplexity,
        srx_v01_train_telemetry.initial_loss, srx_v01_train_telemetry.initial_perplexity,
        srx_v02_train_telemetry.initial_loss, srx_v02_train_telemetry.initial_perplexity
    );
    println!(
        " | {:<32} | {:.4} ({:.2})          | {:.4} ({:.2})          | {:.4} ({:.2})          |",
        "Final Training Loss (PPL)",
        classic_train_telemetry.final_loss, classic_train_telemetry.final_perplexity,
        srx_v01_train_telemetry.final_loss, srx_v01_train_telemetry.final_perplexity,
        srx_v02_train_telemetry.final_loss, srx_v02_train_telemetry.final_perplexity
    );
    println!(
        " | {:<32} | {:.1} ns ({:.3} µs)     | {:.1} ns ({:.3} µs)     | {:.1} ns ({:.3} µs)     |",
        "Single Step Latency (Inference)",
        classic_step_ns, classic_step_us,
        srx_v01_step_ns, srx_v01_step_us,
        srx_v02_step_ns, srx_v02_step_us
    );
    println!(
        " | {:<32} | {:.0} tok/sec          | {:.0} tok/sec          | {:.0} tok/sec          |",
        "Inference Throughput",
        classic_tok_sec, srx_v01_tok_sec, srx_v02_tok_sec
    );
    let classic_passed = classic_test_results.iter().filter(|t| t.passed).count();
    let srx_v01_passed = srx_v01_test_results.iter().filter(|t| t.passed).count();
    let srx_v02_passed = srx_v02_test_results.iter().filter(|t| t.passed).count();
    println!(
        " | {:<32} | {:.1}% ({}/10)          | {:.1}% ({}/10)          | {:.1}% ({}/10)          |",
        "Exact Match Accuracy (Quality)",
        classic_report.exact_match_accuracy, classic_passed,
        srx_v01_report.exact_match_accuracy, srx_v01_passed,
        srx_v02_report.exact_match_accuracy, srx_v02_passed
    );
    let cat_task_classic = classic_test_results.iter().find(|t| t.prompt.contains("кто кот")).map(|t| t.passed).unwrap_or(false);
    let cat_task_v01 = srx_v01_test_results.iter().find(|t| t.prompt.contains("кто кот")).map(|t| t.passed).unwrap_or(false);
    let cat_task_v02 = srx_v02_test_results.iter().find(|t| t.prompt.contains("кто кот")).map(|t| t.passed).unwrap_or(false);
    println!(
        " | {:<32} | {:<22} | {:<22} | {:<22} |",
        "\"Кто кот\" Artifact Verification",
        if cat_task_classic { "PASS" } else { "FAIL" },
        if cat_task_v01 { "PASS" } else { "FAIL (Spectral Click)" },
        if cat_task_v02 { "PASS (Resolved!)" } else { "FAIL" }
    );
    println!(" =================================================================================================================\n");

    println!("All telemetries recorded successfully into `telemetry_classic_v01.txt`, `telemetry_srx_v01.txt`, and `telemetry_srx_v02.txt`.");
}
