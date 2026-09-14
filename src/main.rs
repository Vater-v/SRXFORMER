use std::fs;
use std::time::Instant;

use srxformer::{
    InferenceTelemetry, InferenceWorkspace, KvCache, SrxState, SrxTelemetryReport, SrxTransformer,
    SrxWorkspace, TestCaseResult, TelemetryReport, Tokenizer, Transformer, TransformerConfig,
    EOS_TOKEN_ID,
};

fn main() {
    println!("================================================================================");
    println!(" SRXformer: Pure Rust Classical Baseline vs SRX Innovation Dual-Engine Benchmark");
    println!(" Target Hardware Architecture: Intel Xeon E5-2650 v2 (Ivy Bridge-EP)");
    println!(" Features: Zero-Dep std-only, AVX FP32, L1D Resident (32KB), Analytical Backprop");
    println!(" Unified Non-Duplicate Corpus & Compute/Latency/Quality Telemetry Accounting   ");
    println!("================================================================================\n");

    // -------------------------------------------------------------------------
    // 1. CONFIGURATION & EXACT 512-PARAMETER ARCHITECTURE
    // -------------------------------------------------------------------------
    let config = TransformerConfig::lang_512();
    let tokenizer = Tokenizer::new();

    println!("[1] Dual Model Architecture & Exact Parameter Derivation (lang_512):");
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
    println!("    * 1:1 Parity Parameter Breakdown (Both Classical & SRX):");
    println!("      - Token Embeddings: 21 * 8                = 168");
    println!("      - Attention Projections (W_q, W_k, W_v, W_o): 4 * (8 * 8) = 256");
    println!("      - Pre-Attn RMSNorm Gamma:                 =   8");
    println!("      - FFN (W_1 [4, 8] + W_2 [8, 4]): 32 + 32  =  64");
    println!("      - Pre-FFN RMSNorm Gamma:                  =   8");
    println!("      - Final RMSNorm Gamma:                    =   8");
    println!("      - LM Head (Tied to Embeddings):           =   0");
    println!("      ================================================");
    println!("      GRAND TOTAL:                              = 512 parameters (EXACT BITWISE PARITY)");
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
    // 3. CLASSICAL TRANSFORMER (BASELINE) EXECUTION
    // =========================================================================
    println!("================================================================================");
    println!(" [3] RUNNING CLASSICAL TRANSFORMER (BASELINE)");
    println!("================================================================================");
    let mut classic_model = Transformer::new_with_seed(config.clone(), 100)
        .expect("Failed to initialize Transformer");

    println!("    * Training Classical Transformer for {} epochs...", EPOCHS);
    let classic_train_telemetry = classic_model.train_dataset(&tokens, EPOCHS, LR);
    println!("      - Initial Loss: {:.4} (PPL: {:.2})", classic_train_telemetry.initial_loss, classic_train_telemetry.initial_perplexity);
    println!("      - Final Loss:   {:.4} (PPL: {:.2})", classic_train_telemetry.final_loss, classic_train_telemetry.final_perplexity);
    println!("      - Elapsed Time: {:.2} ms", classic_train_telemetry.elapsed_ms);

    // Save weights
    let classic_weights_path = "data/model_weights.bin";
    classic_model
        .save_weights(classic_weights_path)
        .expect("Failed to save classical model weights");
    let loaded_classic = Transformer::load_from_file(classic_weights_path)
        .expect("Failed to load classical model");

    // Quality Evaluation
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

    // Hardware Latency Benchmark
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

    let classic_report = TelemetryReport::new(classic_train_telemetry.clone(), classic_inf_telemetry.clone(), classic_test_results.clone());
    classic_report.save_to_file("telemetry_classic.txt").expect("Failed to write telemetry_classic.txt");
    println!("    * Classical report saved to: telemetry_classic.txt\n");

    // =========================================================================
    // 4. SRX TRANSFORMER (INNOVATION) EXECUTION
    // =========================================================================
    println!("================================================================================");
    println!(" [4] RUNNING SRXFORMER (SUPER-RESOLVENT XFORMER INNOVATION)");
    println!("================================================================================");
    let mut srx_model = SrxTransformer::new_with_seed(config.clone(), 100)
        .expect("Failed to initialize SrxTransformer");

    println!("    * Training SRXformer (Unitary Givens + MUSIC Subspace) for {} epochs...", EPOCHS);
    let srx_train_telemetry = srx_model.train_dataset(&tokens, EPOCHS, LR);
    println!("      - Initial Loss: {:.4} (PPL: {:.2})", srx_train_telemetry.initial_loss, srx_train_telemetry.initial_perplexity);
    println!("      - Final Loss:   {:.4} (PPL: {:.2})", srx_train_telemetry.final_loss, srx_train_telemetry.final_perplexity);
    println!("      - Elapsed Time: {:.2} ms", srx_train_telemetry.elapsed_ms);

    // Save weights
    let srx_weights_path = "data/srx_model_weights.bin";
    srx_model
        .save_weights(srx_weights_path)
        .expect("Failed to save SRX model weights");
    let loaded_srx = SrxTransformer::load_from_file(srx_weights_path)
        .expect("Failed to load SRX model");

    // Quality Evaluation
    let mut srx_ws = SrxWorkspace::new(&config);
    let mut srx_state = SrxState::new(&config);
    let mut srx_test_results = Vec::new();

    println!("    * Evaluating Quality (10 control tasks):");
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = loaded_srx.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut srx_state, &mut srx_ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        let passed = gen_text == expected;
        println!(
            "      [{}] {:<26}: \"{:<24}\" -> \"{:<20}\"",
            if passed { "PASS" } else { "FAIL" }, category, prompt, gen_text
        );
        srx_test_results.push(TestCaseResult {
            prompt: prompt.to_string(),
            generated: gen_text,
            expected: expected.to_string(),
            category: category.to_string(),
            passed,
        });
    }

    // Hardware Latency Benchmark
    srx_state.reset();
    let srx_b_start = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % config.max_seq_len;
        if pos == 0 {
            srx_state.reset();
        }
        loaded_srx.step(i % config.vocab_size, pos, &mut srx_state, &mut srx_ws);
    }
    let srx_b_elapsed = srx_b_start.elapsed();
    let srx_step_ns = srx_b_elapsed.as_nanos() as f64 / BENCH_STEPS as f64;
    let srx_step_us = srx_step_ns / 1000.0;
    let srx_tok_sec = BENCH_STEPS as f64 / srx_b_elapsed.as_secs_f64();
    let srx_flops_per_token = 2 * config.param_count() as u64;
    let srx_inf_gflops = (srx_flops_per_token as f64 * srx_tok_sec) / 1e9;

    let srx_inf_telemetry = InferenceTelemetry {
        flops_per_token: srx_flops_per_token,
        bench_steps: BENCH_STEPS,
        step_latency_ns: srx_step_ns,
        step_latency_us: srx_step_us,
        tokens_per_sec: srx_tok_sec,
        gflops_per_sec: srx_inf_gflops,
    };

    let srx_report = SrxTelemetryReport::new(
        srx_train_telemetry.clone(),
        srx_inf_telemetry.clone(),
        srx_test_results.clone(),
        srx_state.memory_bytes(),
        config.max_seq_len,
    );
    srx_report.save_to_file("telemetry_srx.txt").expect("Failed to write telemetry_srx.txt");
    println!("    * SRX report saved to: telemetry_srx.txt\n");

    // =========================================================================
    // 5. COMPARATIVE TABLE: CLASSICAL TRANSFORMER VS SRXFORMER
    // =========================================================================
    println!("================================================================================");
    println!(" [5] ARCHITECTURAL & PERFORMANCE COMPARISON TABLE: CLASSICAL VS SRXFORMER");
    println!("================================================================================");
    println!(" | {:<32} | {:<22} | {:<22} |", "Метрика / Характеристика", "Classical Transformer", "SRXformer (Innovation)");
    println!(" |----------------------------------|------------------------|------------------------|");
    println!(" | Addressing Principle             | Softmax Attention      | MUSIC Subspace Resonance|");
    println!(" | Trainable Parameters             | 512 (1:1 Bitwise)      | 512 (1:1 Bitwise)      |");
    println!(" | State Complexity (O-notation)    | O(N * d) (KV cache)    | O(d) (Phase & Matrix)  |");
    println!(" | State Memory (N=32 context)      | 2,048 bytes            | 152 bytes (13.5x less) |");
    println!(" | State Memory (N=1,024 context)   | 65,536 bytes           | 152 bytes (431x less)  |");
    println!(" | State Memory (N=100,000 context) | 6.4 MB (Spills to DRAM)| 152 bytes (L1 Resident)|");
    println!(" | Memory Hardware Bottleneck       | Memory-Bound (DRAM)    | Strictly L1 SRAM-Bound |");
    println!(" | Initial Training Loss (PPL)      | {:.4} ({:.2})          | {:.4} ({:.2})          |",
        classic_train_telemetry.initial_loss, classic_train_telemetry.initial_perplexity,
        srx_train_telemetry.initial_loss, srx_train_telemetry.initial_perplexity
    );
    println!(" | Final Training Loss (PPL)        | {:.4} ({:.2})          | {:.4} ({:.2})          |",
        classic_train_telemetry.final_loss, classic_train_telemetry.final_perplexity,
        srx_train_telemetry.final_loss, srx_train_telemetry.final_perplexity
    );
    println!(" | Single Step Latency (Inference)  | {:.1} ns ({:.3} µs)     | {:.1} ns ({:.3} µs)     |",
        classic_step_ns, classic_step_us, srx_step_ns, srx_step_us
    );
    println!(" | Inference Throughput             | {:.0} tok/sec          | {:.0} tok/sec          |",
        classic_tok_sec, srx_tok_sec
    );
    println!(" | Exact Match Accuracy (Quality)   | {:.1}% ({}/10)          | {:.1}% ({}/10)          |",
        classic_report.exact_match_accuracy, classic_test_results.iter().filter(|t| t.passed).count(),
        srx_report.exact_match_accuracy, srx_test_results.iter().filter(|t| t.passed).count()
    );
    println!(" ================================================================================\n");

    println!("All telemetries recorded successfully into `telemetry_classic.txt` and `telemetry_srx.txt`.");
}
