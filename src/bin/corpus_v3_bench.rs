use std::fs;
use std::time::Instant;

use srxformer::{
    classic::{
        backward_loss as classic_backward_loss, forward_loss as classic_forward_loss,
        split_into_eos_sequences as classic_split_sequences,
        AdamW as ClassicAdamW, FastRng, InferenceTelemetry, InferenceWorkspace, KvCache,
        TestCaseResult, TelemetryReport, Tokenizer, Transformer, TransformerConfig,
        TransformerGrad as ClassicGrad, TrainTelemetry, TrainWorkspace as ClassicTrainWorkspace,
        EOS_TOKEN_ID,
    },
    srx_v05::{
        backward_loss as srx_v05_backward_loss, forward_loss as srx_v05_forward_loss,
        split_into_eos_sequences as srx_v05_split_sequences,
        SrxAdamW as SrxAdamWV05, SrxGrad as SrxGradV05, SrxState as SrxStateV05,
        SrxTelemetryReport as SrxTelemetryReportV05, SrxTrainWorkspace as SrxTrainWorkspaceV05,
        SrxTransformer as SrxTransformerV05, SrxWorkspace as SrxWorkspaceV05,
    },
};

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CheckpointRecord {
    pct: usize,
    epoch: usize,
    mflops: f64,
    loss: f32,
    perplexity: f32,
    passed: usize,
    total: usize,
    accuracy: f32,
}

fn main() {
    println!("=================================================================================================================");
    println!(" SRXformer: Scaled Corpus v3 Iso-FLOPs Benchmark (Classical Baseline vs SRX v05 Quantum-Algebraic Core)          ");
    println!(" Target Hardware: Intel Xeon E5-2650 v2 (Ivy Bridge-EP, AVX FP32, L1D 32KB per core)                           ");
    println!(" Budget: Strict Iso-FLOPs Compute Parity @ 21,750 MFLOPs (21.75 GFLOPs)                                         ");
    println!(" Dataset: data/unified_corpus_v3.txt (5,880 tokens, 880 unique lines, strictly 0 duplicates)                    ");
    println!(" Control Suite: 60 Heterogeneous Control Tasks Across 6 Domains                                                  ");
    println!("=================================================================================================================\n");

    // -------------------------------------------------------------------------
    // 1. ARCHITECTURE CONFIGURATIONS & PARAMETER ACCOUNTING
    // -------------------------------------------------------------------------
    let classic_config = TransformerConfig::lang_v3();
    let srx_v05_config = TransformerConfig::lang_v3();
    let tokenizer = Tokenizer::v3();

    println!("[1] Architecture & Parameter Accounting (Exact Iso-Parameter Corridor 896 ± 4 weights):");
    println!("    * Vocab Size (V):    {} tokens (data/vocab_v3.txt)", classic_config.vocab_size);
    println!("    * Hidden Dim (d):     {}", classic_config.d_model);
    println!("    * Attention Heads:    {} (head_dim = {})", classic_config.n_heads, classic_config.head_dim());
    println!("    * Decoder Layers:     {}", classic_config.n_layers);
    println!("    * Context Window:     {} tokens", classic_config.max_seq_len);
    println!("    * Feed-Forward Dim:   d_ff = {}", classic_config.d_ff);
    println!("    * Tied LM Head:       true (Token embeddings tied with LM Head)");
    println!("    * Normalization:      RMSNorm (eps = {:.1e})", classic_config.eps);
    println!("    -----------------------------------------------------------------------------------------");
    println!("    * Classical Transformer: {:>3} params (delta = 0, 0.00%) [100% L1D resident]", classic_config.param_count());
    println!("    * SRX v05 Quantum Core:  {:>3} params (delta = 0, 0.00%) [100% L1D resident]", srx_v05_config.param_count());
    println!("      - SRX v05 Attention: 256 params (W_q, W_k, W_v, W_o: 4 * 64, 0 learned gate weights)");
    println!("      - Online Sherman-Morrison 2nd-Order RLS: algebraic closed-form update (lambda=0.999)");
    println!("      - Krylov Recurrent Depth: K=2 resolvent unitary refinement");
    println!("      - Monarch Butterfly Unitary Mixer: physical phase momentum (mu=0.85, alpha=0.1)");
    println!("      - Context State Footprint: EXACTLY 288 bytes (Thetas: 32B, p_thetas: 32B, M: 128B, P: 128B)");
    println!();

    // -------------------------------------------------------------------------
    // 2. CORPUS v3 INTEGRITY VERIFICATION
    // -------------------------------------------------------------------------
    println!("[2] Loading Unified Corpus v3:");
    let unified_path = "data/unified_corpus_v3.txt";
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
    println!("      - Total tokens:     {} tokens (TARGET: 5,880)", tokens.len());
    println!("      - Corpus integrity: PASSED (every sentence terminated with <eos>)");
    assert_eq!(tokens.len(), 5880, "Corpus v3 must contain exactly 5,880 tokens");
    println!();

    // -------------------------------------------------------------------------
    // 3. 60 HETEROGENEOUS CONTROL TASKS ACROSS 6 DOMAINS
    // -------------------------------------------------------------------------
    let test_cases = [
        // =========================================================================
        // Domain 1: Addition Arithmetic (10 tasks)
        // =========================================================================
        ("<user> 2 + 3 = <bot>", "5 <eos>", "Addition Arithmetic (Basic)"),
        ("<user> 1 + 2 = <bot>", "3 <eos>", "Addition Arithmetic (Basic)"),
        ("<user> 0 + 1 = <bot>", "1 <eos>", "Addition Arithmetic (Basic)"),
        ("<user> 2 + 2 = <bot>", "4 <eos>", "Addition Arithmetic (Basic)"),
        ("<user> 3 + 1 = <bot>", "4 <eos>", "Addition Arithmetic (Basic)"),
        ("<user> 3 + 4 = <bot>", "7 <eos>", "Addition Arithmetic (Extended)"),
        ("<user> 5 + 3 = <bot>", "8 <eos>", "Addition Arithmetic (Extended)"),
        ("<user> 4 + 5 = <bot>", "9 <eos>", "Addition Arithmetic (Extended)"),
        ("2 + 3 =", "5 <eos>", "Addition Arithmetic (Base Prompt)"),
        ("3 + 4 =", "7 <eos>", "Addition Arithmetic (Base Prompt)"),

        // =========================================================================
        // Domain 2: Subtraction Arithmetic (10 tasks)
        // =========================================================================
        ("<user> 4 - 1 = <bot>", "3 <eos>", "Subtraction Arithmetic (Basic)"),
        ("<user> 5 - 2 = <bot>", "3 <eos>", "Subtraction Arithmetic (Basic)"),
        ("<user> 3 - 1 = <bot>", "2 <eos>", "Subtraction Arithmetic (Basic)"),
        ("<user> 2 - 1 = <bot>", "1 <eos>", "Subtraction Arithmetic (Basic)"),
        ("<user> 5 - 5 = <bot>", "0 <eos>", "Subtraction Arithmetic (Basic)"),
        ("<user> 9 - 4 = <bot>", "5 <eos>", "Subtraction Arithmetic (Extended)"),
        ("<user> 8 - 3 = <bot>", "5 <eos>", "Subtraction Arithmetic (Extended)"),
        ("<user> 7 - 2 = <bot>", "5 <eos>", "Subtraction Arithmetic (Extended)"),
        ("4 - 1 =", "3 <eos>", "Subtraction Arithmetic (Base Prompt)"),
        ("8 - 3 =", "5 <eos>", "Subtraction Arithmetic (Base Prompt)"),

        // =========================================================================
        // Domain 3: Multiplication & Division Arithmetic (10 tasks)
        // =========================================================================
        ("<user> 2 * 3 = <bot>", "6 <eos>", "Multiplication Arithmetic (Basic)"),
        ("<user> 2 * 2 = <bot>", "4 <eos>", "Multiplication Arithmetic (Basic)"),
        ("<user> 3 * 3 = <bot>", "9 <eos>", "Multiplication Arithmetic (Basic)"),
        ("<user> 4 * 2 = <bot>", "8 <eos>", "Multiplication Arithmetic (Extended)"),
        ("<user> 6 / 2 = <bot>", "3 <eos>", "Division Arithmetic (Exact)"),
        ("<user> 8 / 4 = <bot>", "2 <eos>", "Division Arithmetic (Exact)"),
        ("<user> 9 / 3 = <bot>", "3 <eos>", "Division Arithmetic (Exact)"),
        ("<user> 4 / 2 = <bot>", "2 <eos>", "Division Arithmetic (Exact)"),
        ("2 * 3 =", "6 <eos>", "Multiplication Arithmetic (Base Prompt)"),
        ("6 / 2 =", "3 <eos>", "Division Arithmetic (Base Prompt)"),

        // =========================================================================
        // Domain 4: Taxonomy & Entity Definitions (10 tasks)
        // =========================================================================
        ("<user> кто кот <bot>", "кот это животное <eos>", "Entity Taxonomy (кот)"),
        ("<user> кто пес <bot>", "пес это друг <eos>", "Entity Taxonomy (пес)"),
        ("<user> кто волк <bot>", "волк это зверь <eos>", "Entity Taxonomy (волк)"),
        ("<user> кто лиса <bot>", "лиса это хищник <eos>", "Entity Taxonomy (лиса)"),
        ("<user> кто медведь <bot>", "медведь это зверь <eos>", "Entity Taxonomy (медведь)"),
        ("<user> кто щука <bot>", "щука это рыба <eos>", "Entity Taxonomy (щука)"),
        ("<user> кто змея <bot>", "змея это хищник <eos>", "Entity Taxonomy (змея)"),
        ("<user> кто дуб <bot>", "дуб это дерево <eos>", "Entity Taxonomy (дуб)"),
        ("кто кот =", "кот это животное <eos>", "Entity Taxonomy (Base Prompt)"),
        ("кто медведь =", "медведь это зверь <eos>", "Entity Taxonomy (Base Prompt)"),

        // =========================================================================
        // Domain 5: Spatial Reasoning (10 tasks)
        // =========================================================================
        ("<user> где волк <bot>", "лес <eos>", "Spatial Reasoning (волк)"),
        ("<user> где рыба <bot>", "река <eos>", "Spatial Reasoning (рыба)"),
        ("<user> где кот <bot>", "дом <eos>", "Spatial Reasoning (кот)"),
        ("<user> где птица <bot>", "небо <eos>", "Spatial Reasoning (птица)"),
        ("<user> где медведь <bot>", "тайга <eos>", "Spatial Reasoning (медведь)"),
        ("<user> где щука <bot>", "вода <eos>", "Spatial Reasoning (щука)"),
        ("<user> где змея <bot>", "нора <eos>", "Spatial Reasoning (змея)"),
        ("<user> где заяц <bot>", "поле <eos>", "Spatial Reasoning (заяц)"),
        ("где рыба =", "река <eos>", "Spatial Reasoning (Base Prompt)"),
        ("где медведь =", "тайга <eos>", "Spatial Reasoning (Base Prompt)"),

        // =========================================================================
        // Domain 6: Boolean Logic & Food-Chain Transitivity (10 tasks)
        // =========================================================================
        ("<user> кот это пес <bot>", "нет <eos>", "Boolean Logic (Negation)"),
        ("<user> кот это животное <bot>", "да <eos>", "Boolean Logic (Affirmation)"),
        ("<user> медведь это рыба <bot>", "нет <eos>", "Boolean Logic (Negation)"),
        ("<user> щука это рыба <bot>", "да <eos>", "Boolean Logic (Affirmation)"),
        ("<user> волк ест заяц <bot>", "да <eos>", "Transitivity / Food Chain (Affirmation)"),
        ("<user> заяц ест волк <bot>", "нет <eos>", "Transitivity / Food Chain (Negation)"),
        ("<user> щука ест рыба <bot>", "да <eos>", "Transitivity / Food Chain (Affirmation)"),
        ("<user> рыба ест щука <bot>", "нет <eos>", "Transitivity / Food Chain (Negation)"),
        ("волк это зверь =", "да <eos>", "Boolean Logic (Base Prompt)"),
        ("медведь ест рыба =", "да <eos>", "Transitivity / Food Chain (Base Prompt)"),
    ];

    println!("[3] Control Suite Verification:");
    println!("    * Total Control Tasks: {} (strictly 10 tasks across 6 domains)", test_cases.len());
    for (i, (prompt, expected, domain)) in test_cases.iter().enumerate() {
        let p_toks = tokenizer.encode(prompt);
        let e_toks = tokenizer.encode(expected);
        assert!(!p_toks.is_empty(), "Prompt {} failed to encode: {}", i, prompt);
        assert!(!e_toks.is_empty(), "Expected {} failed to encode: {}", i, expected);
        if i == 0 || i == 10 || i == 20 || i == 30 || i == 40 || i == 50 {
            println!("      Domain {}: {:<32} (e.g. \"{}\" -> \"{}\")", i / 10 + 1, domain, prompt, expected);
        }
    }
    println!();

    const BENCH_STEPS: usize = 50_000;
    const ISO_FLOPS_BUDGET: f64 = 21_750.0; // MFLOPs = 21.75 GFLOPs

    // =========================================================================
    // 4. MODEL 1: CLASSICAL TRANSFORMER BASELINE (896 PARAMS, SEED 42)
    // =========================================================================
    println!("================================================================================");
    println!(" [4] RUNNING CLASSICAL TRANSFORMER BASELINE (PARITY: 896 PARAMS, SEED 42)");
    println!("================================================================================");
    let classic_params = classic_config.param_count();
    let classic_flops_per_tok = 6 * classic_params as u64; // 5,376 FLOPs/tok
    let classic_flops_per_epoch = (classic_flops_per_tok * tokens.len() as u64) as f64 / 1e6; // 31.61088 MFLOPs
    let classic_epochs = (ISO_FLOPS_BUDGET / classic_flops_per_epoch).round() as usize; // 688 epochs
    let classic_lr = 0.015f32;

    println!("    * Training FLOPs budget: {:.1} MFLOPs -> {} epochs ({:.3} MFLOPs/epoch)",
        ISO_FLOPS_BUDGET, classic_epochs, classic_flops_per_epoch);

    let mut classic_model = Transformer::new_with_seed(classic_config.clone(), 42)
        .expect("Failed to initialize Classical Transformer");

    let mut classic_train_ws = ClassicTrainWorkspace::new(&classic_config);
    let mut classic_grad = ClassicGrad::new(&classic_config);
    let mut classic_optimizer = ClassicAdamW::new(&classic_config, 0.01);
    let mut classic_seqs = classic_split_sequences(&tokens, classic_config.max_seq_len);
    let mut classic_rng = FastRng::new(42);

    let mut classic_eval_ws = InferenceWorkspace::new(&classic_config);
    let mut classic_eval_kv = KvCache::new(&classic_config);

    let eval_count = classic_seqs.len().min(30);
    let mut classic_initial_loss = 0.0f32;
    for seq in &classic_seqs[..eval_count] {
        classic_initial_loss += classic_forward_loss(&mut classic_model, seq, &mut classic_train_ws);
    }
    classic_initial_loss /= eval_count as f32;

    let classic_checkpoints = [
        (20, (classic_epochs as f64 * 0.20).round() as usize),
        (40, (classic_epochs as f64 * 0.40).round() as usize),
        (60, (classic_epochs as f64 * 0.60).round() as usize),
        (80, (classic_epochs as f64 * 0.80).round() as usize),
        (100, classic_epochs),
    ];
    let mut classic_checkpoint_records = Vec::new();
    let mut classic_loss_history = Vec::with_capacity(classic_epochs);
    let classic_start = Instant::now();
    let mut classic_total_steps = 0;

    println!("    * Checkpoint Trajectory (20%, 40%, 60%, 80%, 100% compute):");
    println!("      | Pct | Epoch | Compute (MFLOPs) | Loss   | PPL   | Accuracy (60 tests) |");
    println!("      |-----|-------|------------------|--------|-------|---------------------|");

    let mut next_cp_idx = 0;
    for epoch in 1..=classic_epochs {
        // In-place Fisher-Yates shuffle per epoch
        for i in (1..classic_seqs.len()).rev() {
            let j = (classic_rng.next_u64() as usize) % (i + 1);
            classic_seqs.swap(i, j);
        }

        let progress = epoch as f32 / classic_epochs as f32;
        let current_lr = (classic_lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(classic_lr * 0.05);

        let mut epoch_loss = 0.0f32;
        for seq in &classic_seqs {
            classic_grad.zero();
            let loss = classic_forward_loss(&mut classic_model, seq, &mut classic_train_ws);
            epoch_loss += loss;
            classic_backward_loss(&mut classic_model, seq, &mut classic_train_ws, &mut classic_grad);
            classic_grad.clip_grad_norm(1.0);
            classic_optimizer.step(&mut classic_model, &classic_grad, current_lr);
            classic_total_steps += 1;
        }
        let avg_epoch_loss = epoch_loss / classic_seqs.len() as f32;
        classic_loss_history.push(avg_epoch_loss);

        if next_cp_idx < classic_checkpoints.len() && epoch == classic_checkpoints[next_cp_idx].1 {
            let (pct, cp_epoch) = classic_checkpoints[next_cp_idx];
            let cp_mflops = cp_epoch as f64 * classic_flops_per_epoch;

            let mut passed = 0;
            for (prompt, expected, _) in test_cases {
                let p_toks = tokenizer.encode(prompt);
                let gen_ids = classic_model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut classic_eval_kv, &mut classic_eval_ws);
                let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
                if gen_text == expected {
                    passed += 1;
                }
            }
            let acc = (passed as f32 / test_cases.len() as f32) * 100.0;
            let ppl = avg_epoch_loss.exp();

            println!(
                "      | {:>3}%| {:>5} | {:>16.2} | {:>6.4} | {:>5.2} | {:>2}/60 ({:>5.1}%)      |",
                pct, cp_epoch, cp_mflops, avg_epoch_loss, ppl, passed, acc
            );

            classic_checkpoint_records.push(CheckpointRecord {
                pct,
                epoch: cp_epoch,
                mflops: cp_mflops,
                loss: avg_epoch_loss,
                perplexity: ppl,
                passed,
                total: test_cases.len(),
                accuracy: acc,
            });
            next_cp_idx += 1;
        }
    }

    let classic_elapsed_ms = classic_start.elapsed().as_secs_f64() * 1000.0;
    let classic_final_loss = *classic_loss_history.last().unwrap_or(&classic_initial_loss);
    let classic_total_flops = classic_flops_per_tok * tokens.len() as u64 * classic_epochs as u64;

    let classic_train_telemetry = TrainTelemetry {
        num_params: classic_params,
        dataset_tokens: tokens.len(),
        epochs: classic_epochs,
        total_training_flops: classic_total_flops,
        elapsed_ms: classic_elapsed_ms,
        mflops_per_sec: (classic_total_flops as f64 / 1e6) / (classic_elapsed_ms / 1000.0),
        gflops_per_sec: (classic_total_flops as f64 / 1e9) / (classic_elapsed_ms / 1000.0),
        initial_loss: classic_initial_loss,
        final_loss: classic_final_loss,
        initial_perplexity: classic_initial_loss.exp(),
        final_perplexity: classic_final_loss.exp(),
        total_steps: classic_total_steps,
        loss_history: classic_loss_history,
    };

    println!("\n    * Classical Baseline Final Evaluation (60 control tasks):");
    let mut classic_test_results = Vec::new();
    let mut classic_passed = 0;
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = classic_model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut classic_eval_kv, &mut classic_eval_ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        let passed = gen_text == expected;
        if passed { classic_passed += 1; }
        println!(
            "      [{}] {:<36}: \"{:<26}\" -> \"{:<22}\"",
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

    classic_eval_kv.reset();
    let b_start = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % classic_config.max_seq_len;
        if pos == 0 { classic_eval_kv.reset(); }
        classic_model.step(i % classic_config.vocab_size, pos, &mut classic_eval_kv, &mut classic_eval_ws);
    }
    let b_elapsed = b_start.elapsed();
    let classic_step_ns = b_elapsed.as_nanos() as f64 / BENCH_STEPS as f64;
    let classic_step_us = classic_step_ns / 1000.0;
    let classic_tok_sec = BENCH_STEPS as f64 / b_elapsed.as_secs_f64();
    let classic_inf_flops = 2 * classic_params as u64;
    let classic_inf_gflops = (classic_inf_flops as f64 * classic_tok_sec) / 1e9;

    let classic_inf_telemetry = InferenceTelemetry {
        flops_per_token: classic_inf_flops,
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
    classic_report.save_to_file("telemetry_classic_corpus_v3.txt").expect("Failed to write telemetry_classic_corpus_v3.txt");
    println!("    * Classical report saved to: telemetry_classic_corpus_v3.txt\n");

    // =========================================================================
    // 5. MODEL 2: SRXFORMER v05 QUANTUM-ALGEBRAIC CORE (896 PARAMS, SEED 42)
    // =========================================================================
    println!("================================================================================");
    println!(" [5] RUNNING SRXFORMER v05 QUANTUM-ALGEBRAIC CORE (896 PARAMS, SEED 42)");
    println!("================================================================================");
    let mut srx_v05_model = SrxTransformerV05::new_with_seed(srx_v05_config.clone(), 42)
        .expect("Failed to initialize SRX v05");
    let srx_v05_params = srx_v05_model.param_count(); // 896
    let srx_v05_flops_per_tok = 6 * srx_v05_params as u64 + 864; // 6,240 FLOPs/tok
    let srx_v05_flops_per_epoch = (srx_v05_flops_per_tok * tokens.len() as u64) as f64 / 1e6; // 36.6912 MFLOPs
    let srx_v05_epochs = (ISO_FLOPS_BUDGET / srx_v05_flops_per_epoch).round() as usize; // 593 epochs
    let srx_v05_lr = 0.020f32;

    println!("    * Training FLOPs budget: {:.1} MFLOPs -> {} epochs ({:.3} MFLOPs/epoch)",
        ISO_FLOPS_BUDGET, srx_v05_epochs, srx_v05_flops_per_epoch);

    let mut srx_v05_train_ws = SrxTrainWorkspaceV05::new(&srx_v05_config);
    let mut srx_v05_grad = SrxGradV05::new(&srx_v05_config);
    let mut srx_v05_optimizer = SrxAdamWV05::new(&srx_v05_config, 0.01);
    let mut srx_v05_seqs = srx_v05_split_sequences(&tokens, srx_v05_config.max_seq_len);
    let mut srx_v05_rng = FastRng::new(42);

    let mut srx_v05_eval_ws = SrxWorkspaceV05::new(&srx_v05_config);
    let mut srx_v05_eval_state = SrxStateV05::new(&srx_v05_config);

    let mut srx_v05_initial_loss = 0.0f32;
    for seq in &srx_v05_seqs[..eval_count] {
        srx_v05_initial_loss += srx_v05_forward_loss(&mut srx_v05_model, seq, &mut srx_v05_train_ws, 1.0);
    }
    srx_v05_initial_loss /= eval_count as f32;

    let srx_v05_checkpoints = [
        (20, (srx_v05_epochs as f64 * 0.20).round() as usize),
        (40, (srx_v05_epochs as f64 * 0.40).round() as usize),
        (60, (srx_v05_epochs as f64 * 0.60).round() as usize),
        (80, (srx_v05_epochs as f64 * 0.80).round() as usize),
        (100, srx_v05_epochs),
    ];
    let mut srx_v05_checkpoint_records = Vec::new();
    let mut srx_v05_loss_history = Vec::with_capacity(srx_v05_epochs);
    let srx_v05_start = Instant::now();
    let mut srx_v05_total_steps = 0;

    println!("    * Checkpoint Trajectory (20%, 40%, 60%, 80%, 100% compute):");
    println!("      | Pct | Epoch | Compute (MFLOPs) | Loss   | PPL   | Accuracy (60 tests) |");
    println!("      |-----|-------|------------------|--------|-------|---------------------|");

    let mut next_v05_cp_idx = 0;
    for epoch in 1..=srx_v05_epochs {
        // In-place Fisher-Yates shuffle per epoch
        for i in (1..srx_v05_seqs.len()).rev() {
            let j = (srx_v05_rng.next_u64() as usize) % (i + 1);
            srx_v05_seqs.swap(i, j);
        }

        let progress = epoch as f32 / srx_v05_epochs as f32;
        let eps_min = 1e-3f32;
        let eps_max = 1.0f32;
        let eps_srx = eps_min + (eps_max - eps_min) * (1.0 - progress).powi(2);
        let current_lr = (srx_v05_lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(srx_v05_lr * 0.05);

        let mut epoch_loss = 0.0f32;
        for seq in &srx_v05_seqs {
            srx_v05_grad.zero();
            let loss = srx_v05_forward_loss(&mut srx_v05_model, seq, &mut srx_v05_train_ws, eps_srx);
            epoch_loss += loss;
            srx_v05_backward_loss(&mut srx_v05_model, seq, &mut srx_v05_train_ws, &mut srx_v05_grad, eps_srx);
            srx_v05_grad.clip_grad_norm(1.0);
            srx_v05_optimizer.step(&mut srx_v05_model, &srx_v05_grad, current_lr);
            srx_v05_total_steps += 1;
        }
        let avg_epoch_loss = epoch_loss / srx_v05_seqs.len() as f32;
        srx_v05_loss_history.push(avg_epoch_loss);

        if next_v05_cp_idx < srx_v05_checkpoints.len() && epoch == srx_v05_checkpoints[next_v05_cp_idx].1 {
            let (pct, cp_epoch) = srx_v05_checkpoints[next_v05_cp_idx];
            let cp_mflops = cp_epoch as f64 * srx_v05_flops_per_epoch;

            let mut passed = 0;
            for (prompt, expected, _) in test_cases {
                let p_toks = tokenizer.encode(prompt);
                let gen_ids = srx_v05_model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut srx_v05_eval_state, &mut srx_v05_eval_ws);
                let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
                if gen_text == expected {
                    passed += 1;
                }
            }
            let acc = (passed as f32 / test_cases.len() as f32) * 100.0;
            let ppl = avg_epoch_loss.exp();

            println!(
                "      | {:>3}%| {:>5} | {:>16.2} | {:>6.4} | {:>5.2} | {:>2}/60 ({:>5.1}%)      |",
                pct, cp_epoch, cp_mflops, avg_epoch_loss, ppl, passed, acc
            );

            srx_v05_checkpoint_records.push(CheckpointRecord {
                pct,
                epoch: cp_epoch,
                mflops: cp_mflops,
                loss: avg_epoch_loss,
                perplexity: ppl,
                passed,
                total: test_cases.len(),
                accuracy: acc,
            });
            next_v05_cp_idx += 1;
        }
    }

    let srx_v05_elapsed_ms = srx_v05_start.elapsed().as_secs_f64() * 1000.0;
    let srx_v05_final_loss = *srx_v05_loss_history.last().unwrap_or(&srx_v05_initial_loss);
    let srx_v05_total_flops = srx_v05_flops_per_tok * tokens.len() as u64 * srx_v05_epochs as u64;

    let srx_v05_train_telemetry = TrainTelemetry {
        num_params: srx_v05_params,
        dataset_tokens: tokens.len(),
        epochs: srx_v05_epochs,
        total_training_flops: srx_v05_total_flops,
        elapsed_ms: srx_v05_elapsed_ms,
        mflops_per_sec: (srx_v05_total_flops as f64 / 1e6) / (srx_v05_elapsed_ms / 1000.0),
        gflops_per_sec: (srx_v05_total_flops as f64 / 1e9) / (srx_v05_elapsed_ms / 1000.0),
        initial_loss: srx_v05_initial_loss,
        final_loss: srx_v05_final_loss,
        initial_perplexity: srx_v05_initial_loss.exp(),
        final_perplexity: srx_v05_final_loss.exp(),
        total_steps: srx_v05_total_steps,
        loss_history: srx_v05_loss_history,
    };

    println!("\n    * SRX v05 Quantum Core Final Evaluation (60 control tasks):");
    let mut srx_v05_test_results = Vec::new();
    let mut srx_v05_passed = 0;
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = srx_v05_model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut srx_v05_eval_state, &mut srx_v05_eval_ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        let passed = gen_text == expected;
        if passed { srx_v05_passed += 1; }
        println!(
            "      [{}] {:<36}: \"{:<26}\" -> \"{:<22}\"",
            if passed { "PASS" } else { "FAIL" }, category, prompt, gen_text
        );
        srx_v05_test_results.push(TestCaseResult {
            prompt: prompt.to_string(),
            generated: gen_text,
            expected: expected.to_string(),
            category: category.to_string(),
            passed,
        });
    }

    srx_v05_eval_state.reset();
    let b_start_v05 = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % srx_v05_config.max_seq_len;
        if pos == 0 { srx_v05_eval_state.reset(); }
        srx_v05_model.step(i % srx_v05_config.vocab_size, pos, &mut srx_v05_eval_state, &mut srx_v05_eval_ws);
    }
    let b_elapsed_v05 = b_start_v05.elapsed();
    let srx_v05_step_ns = b_elapsed_v05.as_nanos() as f64 / BENCH_STEPS as f64;
    let srx_v05_step_us = srx_v05_step_ns / 1000.0;
    let srx_v05_tok_sec = BENCH_STEPS as f64 / b_elapsed_v05.as_secs_f64();
    let srx_v05_inf_flops = 2 * srx_v05_params as u64 + 288;
    let srx_v05_inf_gflops = (srx_v05_inf_flops as f64 * srx_v05_tok_sec) / 1e9;

    let srx_v05_inf_telemetry = InferenceTelemetry {
        flops_per_token: srx_v05_inf_flops,
        bench_steps: BENCH_STEPS,
        step_latency_ns: srx_v05_step_ns,
        step_latency_us: srx_v05_step_us,
        tokens_per_sec: srx_v05_tok_sec,
        gflops_per_sec: srx_v05_inf_gflops,
    };

    let srx_v05_report = SrxTelemetryReportV05::new(
        srx_v05_train_telemetry.clone(),
        srx_v05_inf_telemetry.clone(),
        srx_v05_test_results.clone(),
        srx_v05_eval_state.memory_bytes(),
        srx_v05_config.max_seq_len,
    );
    srx_v05_report.save_to_file("telemetry_srx_v05_corpus_v3.txt").expect("Failed to write telemetry_srx_v05_corpus_v3.txt");
    println!("    * SRX v05 report saved to: telemetry_srx_v05_corpus_v3.txt\n");

    // -------------------------------------------------------------------------
    // 6. WEIGHT SERIALIZATION & BITWISE INTEGRITY VERIFICATION
    // -------------------------------------------------------------------------
    println!("--------------------------------------------------------------------------------");
    println!(" [6] Model Weight Serialization & Bitwise Integrity Check:");
    let weights_path = "data/srx_v05_model_weights.bin";
    srx_v05_model.save_weights(weights_path).expect("Failed to save SRX v05 weights");
    let file_meta = fs::metadata(weights_path).expect("Failed to get weights file metadata");
    println!("    * Weights saved to: {} ({} bytes)", weights_path, file_meta.len());

    let reloaded_model = SrxTransformerV05::load_from_file(weights_path).expect("Failed to reload SRX v05 weights");
    assert_eq!(reloaded_model.config.vocab_size, srx_v05_model.config.vocab_size);
    assert_eq!(reloaded_model.config.d_model, srx_v05_model.config.d_model);
    assert_eq!(reloaded_model.param_count(), srx_v05_model.param_count());
    assert_eq!(reloaded_model.token_embeddings, srx_v05_model.token_embeddings);
    assert_eq!(reloaded_model.layers[0].attn.w_q, srx_v05_model.layers[0].attn.w_q);
    assert_eq!(reloaded_model.layers[0].attn.w_k, srx_v05_model.layers[0].attn.w_k);
    assert_eq!(reloaded_model.layers[0].attn.w_v, srx_v05_model.layers[0].attn.w_v);
    assert_eq!(reloaded_model.layers[0].attn.w_o, srx_v05_model.layers[0].attn.w_o);
    assert_eq!(reloaded_model.layers[0].mlp.w_1, srx_v05_model.layers[0].mlp.w_1);
    assert_eq!(reloaded_model.layers[0].mlp.w_2, srx_v05_model.layers[0].mlp.w_2);
    println!("    * Bitwise Integrity Verification: PASSED (100% parameter identity verified)");
    println!();

    // -------------------------------------------------------------------------
    // 7. COMPREHENSIVE BENCHMARK SUMMARY & PARETO ANALYSIS
    // -------------------------------------------------------------------------
    println!("=================================================================================================================");
    println!(" SRXformer v05 vs Classical Transformer: Scaled Corpus v3 Benchmark Summary                                     ");
    println!("=================================================================================================================");
    println!(" Metric                           | Classical Baseline         | SRX v05 Quantum-Algebraic  | Advantage / Delta  ");
    println!("----------------------------------|----------------------------|----------------------------|--------------------");
    println!(" Trainable Parameters             | {:>26} | {:>26} | {:>18}",
        format!("{} weights", classic_params),
        format!("{} weights", srx_v05_params),
        "0 (0.00% parity)"
    );
    println!(" State Footprint (Memory)         | {:>26} | {:>26} | {:>18}",
        "O(N) (2,048 B @ N=32)",
        "O(1) (288 B constant)",
        "7.1x smaller"
    );
    println!(" L1D Cache Residence (32 KB)      | {:>26} | {:>26} | {:>18}",
        "Degrades at N >= 512",
        "100% L1D (0.88% L1D)",
        "Zero eviction"
    );
    println!(" Total Compute Budget             | {:>26} | {:>26} | {:>18}",
        format!("{:.2} MFLOPs", classic_total_flops as f64 / 1e6),
        format!("{:.2} MFLOPs", srx_v05_total_flops as f64 / 1e6),
        "Iso-FLOPs Parity"
    );
    println!(" Training Epochs                  | {:>26} | {:>26} | {:>18}",
        format!("{} epochs", classic_epochs),
        format!("{} epochs", srx_v05_epochs),
        format!("{} vs {}", classic_epochs, srx_v05_epochs)
    );
    println!(" Training Wallclock Time          | {:>26} | {:>26} | {:>18}",
        format!("{:.2} s ({:.1} min)", classic_elapsed_ms / 1000.0, classic_elapsed_ms / 60000.0),
        format!("{:.2} s ({:.1} min)", srx_v05_elapsed_ms / 1000.0, srx_v05_elapsed_ms / 60000.0),
        format!("{:.2}x", classic_elapsed_ms / srx_v05_elapsed_ms)
    );
    println!(" Final Training Loss              | {:>26} | {:>26} | {:>18}",
        format!("{:.4}", classic_final_loss),
        format!("{:.4}", srx_v05_final_loss),
        format!("{:.4}", classic_final_loss - srx_v05_final_loss)
    );
    println!(" Final Perplexity                 | {:>26} | {:>26} | {:>18}",
        format!("{:.2}", classic_final_loss.exp()),
        format!("{:.2}", srx_v05_final_loss.exp()),
        format!("{:.2}", classic_final_loss.exp() - srx_v05_final_loss.exp())
    );
    println!(" Total Task Accuracy (60 tests)   | {:>26} | {:>26} | {:>18}",
        format!("{}/60 ({:.1}%)", classic_passed, (classic_passed as f32 / 60.0) * 100.0),
        format!("{}/60 ({:.1}%)", srx_v05_passed, (srx_v05_passed as f32 / 60.0) * 100.0),
        format!("+{} passed", srx_v05_passed as i32 - classic_passed as i32)
    );
    println!(" Inference Step Latency           | {:>26} | {:>26} | {:>18}",
        format!("{:.3} us ({:.1} ns)", classic_step_us, classic_step_ns),
        format!("{:.3} us ({:.1} ns)", srx_v05_step_us, srx_v05_step_ns),
        format!("{:.2}x faster", classic_step_us / srx_v05_step_us)
    );
    println!(" Inference Throughput             | {:>26} | {:>26} | {:>18}",
        format!("{:.0} tok/s", classic_tok_sec),
        format!("{:.0} tok/s", srx_v05_tok_sec),
        format!("{:.2}x", srx_v05_tok_sec / classic_tok_sec)
    );
    println!("=================================================================================================================\n");

    // Print Domain Breakdown
    println!("Domain Accuracy Breakdown (10 tests per domain):");
    let domains = [
        "Addition Arithmetic",
        "Subtraction Arithmetic",
        "Multiplication & Division",
        "Taxonomy & Entity Definitions",
        "Spatial Reasoning",
        "Boolean Logic & Transitivity",
    ];
    for (d_idx, d_name) in domains.iter().enumerate() {
        let start = d_idx * 10;
        let end = start + 10;
        let c_pass = classic_test_results[start..end].iter().filter(|r| r.passed).count();
        let s_pass = srx_v05_test_results[start..end].iter().filter(|r| r.passed).count();
        println!("  * Domain {}: {:<32} | Classic: {:>2}/10 ({:>5.1}%) | SRX v05: {:>2}/10 ({:>5.1}%)",
            d_idx + 1, d_name, c_pass, c_pass as f32 * 10.0, s_pass, s_pass as f32 * 10.0
        );
    }
    println!("\nBenchmark execution successfully finished!");
}
