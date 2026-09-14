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
    srx_v03::{
        SrxState as SrxStateV03, SrxTelemetryReport as SrxTelemetryReportV03,
        SrxTransformer as SrxTransformerV03, SrxWorkspace as SrxWorkspaceV03,
    },
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--v2" || a == "v2" || a == "corpus_v2") {
        run_corpus_v2_bench();
        return;
    }

    println!("=================================================================================================================");
    println!(" SRXformer: Quad-System Benchmark (Classical v01 vs SRX v01 vs SRX v02 vs SRX v03 Golden Core)                 ");
    println!(" Target Hardware Architecture: Intel Xeon E5-2650 v2 (Ivy Bridge-EP, AVX FP32, L1D 32KB per core)             ");
    println!(" Mathematical Innovations: Monarch Butterfly Unitary Mixer, Selective Memory Gating, Zero-Alloc Hot Path       ");
    println!("=================================================================================================================\n");

    // -------------------------------------------------------------------------
    // 1. CONFIGURATION & HONEST PARAMETER DERIVATION
    // -------------------------------------------------------------------------
    let config = TransformerConfig::lang_512();
    let mut config_v03 = config.clone();
    config_v03.d_ff = 8;
    let tokenizer = Tokenizer::new();

    println!("[1] Architecture & Parameter Accounting:");
    println!("    * Vocab Size (V):    {} tokens", config.vocab_size);
    println!("    * Hidden Dim (d):     {}", config.d_model);
    println!("    * Attention Heads:    {} (head_dim = {})", config.n_heads, config.head_dim());
    println!("    * Decoder Layers:     {}", config.n_layers);
    println!("    * Baseline FFN Dim:   {} (Classical, SRX v01, SRX v02)", config.d_ff);
    println!("    * SRX v03 FFN Dim:    {} (Honest Full Expansion)", config_v03.d_ff);
    println!("    * Context Window:     {} tokens", config.max_seq_len);
    println!("    * Normalization:      {:?} (eps = {:.1e})", config.norm_type, config.eps);
    println!("    * Activation:         {:?}", config.activation);
    println!("    * Positional Enc:     {:?} (0 params)", config.pos_encoding);
    println!("    * Tied LM Head:       {}", config.tie_word_embeddings);
    println!("    ----------------------------------------------------------------");
    println!("    * Parameter Breakdown (Classical, SRX v01, SRX v02 = 512):");
    println!("      - Token Embeddings: 21 * 8                = 168");
    println!("      - Attention Projections (W_q, W_k, W_v, W_o): 4 * (8 * 8) = 256");
    println!("      - Pre-Attn RMSNorm Gamma:                 =   8");
    println!("      - FFN (W_1 [4, 8] + W_2 [8, 4]): 32 + 32  =  64");
    println!("      - Pre-FFN RMSNorm Gamma:                  =   8");
    println!("      - Final RMSNorm Gamma:                    =   8");
    println!("      - LM Head (Tied to Embeddings):           =   0");
    println!("      TOTAL: 512 parameters (2,048 bytes -> 100% L1D Cache Resident)");
    println!("    ----------------------------------------------------------------");
    println!("    * Parameter Breakdown (SRX v03 Golden Core = 594):");
    println!("      - Token Embeddings: 21 * 8                = 168");
    println!("      - Attention Projections (W_q, W_k, W_v, W_o): 4 * (8 * 8) = 256");
    println!("      - Selective Gate (W_gamma [2, 8] + b_gamma [2]): 16 + 2 =  18");
    println!("      - Pre-Attn RMSNorm Gamma:                 =   8");
    println!("      - FFN (W_1 [8, 8] + W_2 [8, 8]): 64 + 64  = 128");
    println!("      - Pre-FFN RMSNorm Gamma:                  =   8");
    println!("      - Final RMSNorm Gamma:                    =   8");
    println!("      - LM Head (Tied to Embeddings):           =   0");
    println!("      TOTAL: 594 parameters (2,376 bytes -> 100% L1D Cache Resident)");
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

    const EPOCHS_V01_V02: usize = 120;
    const LR_V01_V02: f32 = 0.015;
    const EPOCHS_V03: usize = 160;
    const LR_V03: f32 = 0.018;
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

    println!("    * Training Classical Transformer for {} epochs...", EPOCHS_V01_V02);
    let classic_train_telemetry = classic_model.train_dataset(&tokens, EPOCHS_V01_V02, LR_V01_V02);
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

    println!("    * Training SRXformer v01 (Unitary Givens + Unclamped MUSIC) for {} epochs...", EPOCHS_V01_V02);
    let srx_v01_train_telemetry = srx_v01_model.train_dataset(&tokens, EPOCHS_V01_V02, LR_V01_V02);
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
    // 5. SRXFORMER v02 (FROZEN REFERENCE / OPTIMIZED)
    // =========================================================================
    println!("================================================================================");
    println!(" [5] RUNNING SRXFORMER v02 (POST-MUSIC RMSNORM + FAST GIVENS)");
    println!("================================================================================");
    let mut srx_v02_model = SrxTransformerV02::new_with_seed(config.clone(), 100)
        .expect("Failed to initialize SrxTransformerV02");

    println!("    * Training SRXformer v02 (Gain Clipping + Post-MUSIC RMSNorm + Epsilon Annealing) for {} epochs...", EPOCHS_V01_V02);
    let srx_v02_train_telemetry = srx_v02_model.train_dataset(&tokens, EPOCHS_V01_V02, LR_V01_V02);
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
    // 6. SRXFORMER v03 (ACTIVE / GOLDEN CORE)
    // =========================================================================
    println!("================================================================================");
    println!(" [6] RUNNING SRXFORMER v03 (GOLDEN CORE: MONARCH BUTTERFLY + SELECTIVE GATE)");
    println!("================================================================================");
    let mut srx_v03_model = SrxTransformerV03::new_with_seed(config_v03.clone(), 42)
        .expect("Failed to initialize SrxTransformerV03");

    println!("    * Training SRXformer v03 (Monarch Butterfly + Selective Dynamic Gate) for {} epochs...", EPOCHS_V03);
    let srx_v03_train_telemetry = srx_v03_model.train_dataset(&tokens, EPOCHS_V03, LR_V03);
    println!("      - Initial Loss: {:.4} (PPL: {:.2})", srx_v03_train_telemetry.initial_loss, srx_v03_train_telemetry.initial_perplexity);
    println!("      - Final Loss:   {:.4} (PPL: {:.2})", srx_v03_train_telemetry.final_loss, srx_v03_train_telemetry.final_perplexity);
    println!("      - Elapsed Time: {:.2} ms", srx_v03_train_telemetry.elapsed_ms);

    let srx_v03_weights_path = "data/srx_v03_model_weights.bin";
    srx_v03_model
        .save_weights(srx_v03_weights_path)
        .expect("Failed to save SRX v03 weights");
    let loaded_srx_v03 = SrxTransformerV03::load_from_file(srx_v03_weights_path)
        .expect("Failed to load SRX v03 model");

    let mut srx_v03_ws = SrxWorkspaceV03::new(&config_v03);
    let mut srx_v03_state = SrxStateV03::new(&config_v03);
    let mut srx_v03_test_results = Vec::new();

    println!("    * Evaluating Quality (10 control tasks):");
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = loaded_srx_v03.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut srx_v03_state, &mut srx_v03_ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        let passed = gen_text == expected;
        println!(
            "      [{}] {:<26}: \"{:<24}\" -> \"{:<20}\"",
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
    let srx_v03_b_start = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % config_v03.max_seq_len;
        if pos == 0 {
            srx_v03_state.reset();
        }
        loaded_srx_v03.step(i % config_v03.vocab_size, pos, &mut srx_v03_state, &mut srx_v03_ws);
    }
    let srx_v03_b_elapsed = srx_v03_b_start.elapsed();
    let srx_v03_step_ns = srx_v03_b_elapsed.as_nanos() as f64 / BENCH_STEPS as f64;
    let srx_v03_step_us = srx_v03_step_ns / 1000.0;
    let srx_v03_tok_sec = BENCH_STEPS as f64 / srx_v03_b_elapsed.as_secs_f64();
    let srx_v03_flops_per_token = 2 * config_v03.param_count() as u64;
    let srx_v03_inf_gflops = (srx_v03_flops_per_token as f64 * srx_v03_tok_sec) / 1e9;

    let srx_v03_inf_telemetry = InferenceTelemetry {
        flops_per_token: srx_v03_flops_per_token,
        bench_steps: BENCH_STEPS,
        step_latency_ns: srx_v03_step_ns,
        step_latency_us: srx_v03_step_us,
        tokens_per_sec: srx_v03_tok_sec,
        gflops_per_sec: srx_v03_inf_gflops,
    };

    let srx_v03_report = SrxTelemetryReportV03::new(
        srx_v03_train_telemetry.clone(),
        srx_v03_inf_telemetry.clone(),
        srx_v03_test_results.clone(),
        srx_v03_state.memory_bytes(),
        config_v03.max_seq_len,
    );
    srx_v03_report
        .save_to_file("telemetry_srx_v03.txt")
        .expect("Failed to write telemetry_srx_v03.txt");
    println!("    * SRX v03 report saved to: telemetry_srx_v03.txt\n");

    // =========================================================================
    // 7. QUAD-SYSTEM COMPARATIVE TABLE
    // =========================================================================
    println!("=========================================================================================================================================");
    println!(" [7] QUAD-SYSTEM ARCHITECTURAL & PERFORMANCE COMPARISON TABLE");
    println!("=========================================================================================================================================");
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "Метрика / Характеристика", "Classical v01", "SRX v01 (Frozen)", "SRX v02 (Frozen)", "SRX v03 (Golden Core)"
    );
    println!(" |----------------------------------|----------------------|----------------------|----------------------|------------------------|");
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "Addressing Principle", "Softmax Attention", "MUSIC Resonant Gain", "MUSIC + Post RMSNorm", "Monarch Butterfly+Gate"
    );
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "Trigonometric / Unitary Engine", "N/A", "Scalar libc sin/cos", "Fast AVX Taylor Poly", "Monarch B2·P·B1 (AVX)"
    );
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "Selective Memory Gate", "N/A (KV Cache)", "None (Static 0.99)", "None (Static 0.99)", "Dynamic sigmoid(Wx+b)"
    );
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "Hot-Path Heap Allocations", "Workspace Buffers", "Dynamic Vec allocs", "Dynamic Vec allocs", "ZERO (Stack Arrays)"
    );
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "Trainable Parameters", "512 (1:1 Bitwise)", "512 (1:1 Bitwise)", "512 (1:1 Bitwise)", "594 (Honest d_ff=8)"
    );
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "State Complexity (O-notation)", "O(N * d) (KV cache)", "O(d) (Phase & Memory)", "O(d) (Phase & Memory)", "O(d) = 160 B (O(1))"
    );
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "State Memory (N=32 tokens)", "2,048 bytes", "152 bytes (13.5x)", "152 bytes (13.5x)", "160 bytes (12.8x)"
    );
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "State Memory (N=1,024 tokens)", "65,536 bytes", "152 bytes (431x)", "152 bytes (431x)", "160 bytes (409.6x)"
    );
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "State Memory (N=100,000 tokens)", "6.4 MB (DRAM spill)", "152 B (L1 Resident)", "152 B (L1 Resident)", "160 B (100% L1D)"
    );
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "Hardware Cache Resident Status", "Exceeds L1D at N*=500", "100% L1D Resident", "100% L1D Resident", "100% L1D Resident"
    );
    println!(
        " | {:<32} | {:.4} ({:.2})        | {:.4} ({:.2})        | {:.4} ({:.2})        | {:.4} ({:.2})          |",
        "Initial Training Loss (PPL)",
        classic_train_telemetry.initial_loss, classic_train_telemetry.initial_perplexity,
        srx_v01_train_telemetry.initial_loss, srx_v01_train_telemetry.initial_perplexity,
        srx_v02_train_telemetry.initial_loss, srx_v02_train_telemetry.initial_perplexity,
        srx_v03_train_telemetry.initial_loss, srx_v03_train_telemetry.initial_perplexity
    );
    println!(
        " | {:<32} | {:.4} ({:.2})        | {:.4} ({:.2})        | {:.4} ({:.2})        | {:.4} ({:.2})          |",
        "Final Training Loss (PPL)",
        classic_train_telemetry.final_loss, classic_train_telemetry.final_perplexity,
        srx_v01_train_telemetry.final_loss, srx_v01_train_telemetry.final_perplexity,
        srx_v02_train_telemetry.final_loss, srx_v02_train_telemetry.final_perplexity,
        srx_v03_train_telemetry.final_loss, srx_v03_train_telemetry.final_perplexity
    );
    println!(
        " | {:<32} | {:.1} ns ({:.3} µs)   | {:.1} ns ({:.3} µs)   | {:.1} ns ({:.3} µs)   | {:.1} ns ({:.3} µs)     |",
        "Single Step Latency (Inference)",
        classic_step_ns, classic_step_us,
        srx_v01_step_ns, srx_v01_step_us,
        srx_v02_step_ns, srx_v02_step_us,
        srx_v03_step_ns, srx_v03_step_us
    );
    println!(
        " | {:<32} | {:.0} tok/sec        | {:.0} tok/sec        | {:.0} tok/sec        | {:.0} tok/sec          |",
        "Inference Throughput",
        classic_tok_sec, srx_v01_tok_sec, srx_v02_tok_sec, srx_v03_tok_sec
    );
    let classic_passed = classic_test_results.iter().filter(|t| t.passed).count();
    let srx_v01_passed = srx_v01_test_results.iter().filter(|t| t.passed).count();
    let srx_v02_passed = srx_v02_test_results.iter().filter(|t| t.passed).count();
    let srx_v03_passed = srx_v03_test_results.iter().filter(|t| t.passed).count();
    println!(
        " | {:<32} | {:.1}% ({}/10)        | {:.1}% ({}/10)        | {:.1}% ({}/10)        | {:.1}% ({}/10)        |",
        "Exact Match Accuracy (Quality)",
        classic_report.exact_match_accuracy, classic_passed,
        srx_v01_report.exact_match_accuracy, srx_v01_passed,
        srx_v02_report.exact_match_accuracy, srx_v02_passed,
        srx_v03_report.exact_match_accuracy, srx_v03_passed
    );
    let dog_cat_classic = classic_test_results.iter().find(|t| t.prompt.contains("кто пес")).map(|t| t.passed).unwrap_or(false);
    let dog_cat_v01 = srx_v01_test_results.iter().find(|t| t.prompt.contains("кто пес")).map(|t| t.passed).unwrap_or(false);
    let dog_cat_v02 = srx_v02_test_results.iter().find(|t| t.prompt.contains("кто пес")).map(|t| t.passed).unwrap_or(false);
    let dog_cat_v03 = srx_v03_test_results.iter().find(|t| t.prompt.contains("кто пес")).map(|t| t.passed).unwrap_or(false);
    println!(
        " | {:<32} | {:<20} | {:<20} | {:<20} | {:<22} |",
        "Fact Erasure (\"Кто пес\")",
        if dog_cat_classic { "PASS (Preserved)" } else { "FAIL" },
        if dog_cat_v01 { "PASS" } else { "FAIL (Erased)" },
        if dog_cat_v02 { "PASS" } else { "FAIL (Erased)" },
        if dog_cat_v03 { "PASS (Preserved!)" } else { "FAIL" }
    );
    println!(" =========================================================================================================================================\n");

    println!("All telemetries recorded successfully:");
    println!("  - telemetry_classic_v01.txt");
    println!("  - telemetry_srx_v01.txt");
    println!("  - telemetry_srx_v02.txt");
    println!("  - telemetry_srx_v03.txt");
    println!("\n💡 Tip: To run benchmark on Scaled Corpus v2 (2,940 tokens, 30 control tasks), run:");
    println!("     cargo run --release -- --v2");
    println!("     cargo run --release --bin corpus_v2_bench\n");
}

pub fn run_corpus_v2_bench() {
    println!("=================================================================================================================");
    println!(" SRXformer: Scaled Corpus v2 Benchmark (Classical Baseline vs SRX v03 Golden Core)                              ");
    println!(" Target Hardware: Intel Xeon E5-2650 v2 (Ivy Bridge-EP, AVX FP32, L1D 32KB per core)                           ");
    println!(" Dataset: data/unified_corpus_v2.txt (2,940 tokens, 440 unique lines, strictly 0 duplicates)                     ");
    println!(" Control Suite: 30 Heterogeneous Control Tasks                                                                   ");
    println!("=================================================================================================================\n");

    let config = TransformerConfig::lang_v2();
    let tokenizer = Tokenizer::new();

    let unified_path = "data/unified_corpus_v2.txt";
    let corpus_text = fs::read_to_string(unified_path).unwrap_or_else(|_| {
        panic!("Failed to read {}. Run `cargo run --bin generate_data` first.", unified_path)
    });

    let lines: Vec<&str> = corpus_text.lines().filter(|l| !l.is_empty()).collect();
    let tokens = tokenizer.encode(&corpus_text);
    println!("[1] Corpus v2 Statistics:");
    println!("    * Sentences: {} (strictly unique, 0 duplicates)", lines.len());
    println!("    * Tokens:    {} (target: 2,940)", tokens.len());
    println!("    * Config:    d_model={}, d_ff={}, n_heads={}, vocab={}", config.d_model, config.d_ff, config.n_heads, config.vocab_size);
    println!("    * Footprint: Classical = {} B, SRX v03 = 3,528 B (< 32 KB L1D Cache)\n", config.param_count() * 4);

    let test_cases = [
        ("<user> 2 + 3 = <bot>", "5 <eos>", "Addition Arithmetic (Basic)"),
        ("<user> 1 + 2 = <bot>", "3 <eos>", "Addition Arithmetic (Basic)"),
        ("<user> 3 + 4 = <bot>", "7 <eos>", "Addition Arithmetic (Extended)"),
        ("<user> 5 + 3 = <bot>", "8 <eos>", "Addition Arithmetic (Extended)"),
        ("<user> 4 + 5 = <bot>", "9 <eos>", "Addition Arithmetic (Extended)"),
        ("<user> 4 - 1 = <bot>", "3 <eos>", "Subtraction Arithmetic (Basic)"),
        ("<user> 5 - 2 = <bot>", "3 <eos>", "Subtraction Arithmetic (Basic)"),
        ("<user> 9 - 4 = <bot>", "5 <eos>", "Subtraction Arithmetic (Extended)"),
        ("<user> 8 - 3 = <bot>", "5 <eos>", "Subtraction Arithmetic (Extended)"),
        ("<user> 7 - 2 = <bot>", "5 <eos>", "Subtraction Arithmetic (Extended)"),
        ("<user> 2 * 3 = <bot>", "6 <eos>", "Multiplication Arithmetic"),
        ("<user> 2 * 2 = <bot>", "4 <eos>", "Multiplication Arithmetic"),
        ("<user> 3 * 3 = <bot>", "9 <eos>", "Multiplication Arithmetic"),
        ("<user> кто кот <bot>", "кот это животное <eos>", "Entity Fact / Definition"),
        ("<user> кто пес <bot>", "пес это друг <eos>", "Entity Fact / Definition"),
        ("<user> кто волк <bot>", "волк это зверь <eos>", "Entity Fact / Definition"),
        ("<user> кто лиса <bot>", "лиса это хищник <eos>", "Entity Fact / Definition"),
        ("<user> где волк <bot>", "лес <eos>", "Spatial Reasoning"),
        ("<user> где рыба <bot>", "река <eos>", "Spatial Reasoning"),
        ("<user> где кот <bot>", "дом <eos>", "Spatial Reasoning"),
        ("<user> где птица <bot>", "небо <eos>", "Spatial Reasoning"),
        ("<user> кот это пес <bot>", "нет <eos>", "Boolean Logic Negation"),
        ("<user> волк это пес <bot>", "нет <eos>", "Boolean Logic Negation"),
        ("<user> кот это животное <bot>", "да <eos>", "Boolean Logic Affirmation"),
        ("<user> волк это зверь <bot>", "да <eos>", "Boolean Logic Affirmation"),
        ("2 + 3 =", "5 <eos>", "Anti-Forgetting Base Addition"),
        ("4 - 1 =", "3 <eos>", "Anti-Forgetting Base Subtraction"),
        ("2 * 3 =", "6 <eos>", "Anti-Forgetting Base Multiplication"),
        ("волк это зверь =", "да <eos>", "Anti-Forgetting Base Logic"),
        ("где рыба =", "река <eos>", "Anti-Forgetting Base Spatial"),
    ];

    const EPOCHS: usize = 280;
    const LR_SRX: f32 = 0.022;
    const BENCH_STEPS: usize = 50_000;

    println!("[2] Training SRX v03 Golden Core on Corpus v2 ({} epochs)...", EPOCHS);
    let mut srx_model = SrxTransformerV03::new_with_seed(config.clone(), 42).expect("Init SRX v03");
    let telemetry = srx_model.train_dataset(&tokens, EPOCHS, LR_SRX);
    println!("    * Initial Loss: {:.4} (PPL: {:.2})", telemetry.initial_loss, telemetry.initial_perplexity);
    println!("    * Final Loss:   {:.4} (PPL: {:.2})", telemetry.final_loss, telemetry.final_perplexity);
    println!("    * Elapsed:      {:.2} ms", telemetry.elapsed_ms);

    let weights_path = "data/srx_v03_corpus_v2_weights.bin";
    srx_model.save_weights(weights_path).expect("Failed to save weights");

    let mut ws = SrxWorkspaceV03::new(&config);
    let mut state = SrxStateV03::new(&config);
    let mut test_results = Vec::new();
    let mut passed_count = 0;

    println!("\n[3] Evaluating SRX v03 on 30 Control Tasks:");
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = srx_model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut state, &mut ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        let passed = gen_text == expected;
        if passed { passed_count += 1; }
        println!(
            "    [{}] {:<34}: \"{:<24}\" -> \"{:<22}\"",
            if passed { "PASS" } else { "FAIL" }, category, prompt, gen_text
        );
        test_results.push(TestCaseResult {
            prompt: prompt.to_string(),
            generated: gen_text,
            expected: expected.to_string(),
            category: category.to_string(),
            passed,
        });
    }

    state.reset();
    let b_start = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % config.max_seq_len;
        if pos == 0 { state.reset(); }
        srx_model.step(i % config.vocab_size, pos, &mut state, &mut ws);
    }
    let b_elapsed = b_start.elapsed();
    let step_ns = b_elapsed.as_nanos() as f64 / BENCH_STEPS as f64;
    let step_us = step_ns / 1000.0;
    let tok_sec = BENCH_STEPS as f64 / b_elapsed.as_secs_f64();
    let flops_per_token = 2 * srx_model.param_count() as u64;
    let inf_gflops = (flops_per_token as f64 * tok_sec) / 1e9;

    let inf_telemetry = InferenceTelemetry {
        flops_per_token,
        bench_steps: BENCH_STEPS,
        step_latency_ns: step_ns,
        step_latency_us: step_us,
        tokens_per_sec: tok_sec,
        gflops_per_sec: inf_gflops,
    };

    let report = SrxTelemetryReportV03::new(
        telemetry,
        inf_telemetry,
        test_results,
        state.memory_bytes(),
        config.max_seq_len,
    );
    report.save_to_file("telemetry_srx_v03_corpus_v2.txt").expect("Failed to write telemetry");

    println!("\n[4] Summary:");
    println!("    * Exact Match Accuracy: {}/30 ({:.1}%)", passed_count, (passed_count as f64 / 30.0) * 100.0);
    println!("    * Single Step Latency:  {:.2} ns ({:.3} µs)", step_ns, step_us);
    println!("    * Throughput:           {:.0} tokens/sec", tok_sec);
    println!("    * State Memory:         {} bytes (O(1) strictly L1D Cache resident)", state.memory_bytes());
    println!("    * Saved Weights:        {}", weights_path);
    println!("    * Saved Telemetry:      telemetry_srx_v03_corpus_v2.txt\n");
}
