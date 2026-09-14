//! Sprint 4: Multi-Scale Iso-FLOPs Benchmark & Long Context Stress Test (The Memory Wall Challenge).
//!
//! Evaluates:
//! 1. Iso-FLOPs Chinchilla two-stage training (Pretrain + Instruct with 25% replay mix)
//!    for both Classical Transformer (896 params) and SRX v05 Quantum-Algebraic Core (896 params).
//! 2. Pareto-frontier checkpoints (20%, 40%, 60%, 80%, 100% compute) on 60 heterogeneous control tasks.
//! 3. 6 Domains of 10 tasks each:
//!    - Domain 1: Addition Arithmetic (+)
//!    - Domain 2: Subtraction Arithmetic (-) (non-commutative pairs A - B != B - A)
//!    - Domain 3: Multiplication & Division Arithmetic (*, /)
//!    - Domain 4: Taxonomy & Entity Facts
//!    - Domain 5: Spatial Reasoning (где)
//!    - Domain 6: Boolean Logic & Transitivity Chains ("ест", "если...то")
//! 4. Long Context Stress Test (The Memory Wall Challenge):
//!    - Context lengths N ∈ [32, 128, 512, 1 024, 4 096, 16 384, 65 536].
//!    - KV-cache growth: 2 * N * d_model * 4 bytes (2 KB at N=32 to 4.19 MB at N=64K).
//!    - Crossover points: spills L1D (32 KB) at N >= 512; spills L2 (256 KB) at N >= 4K; spills into DRAM at N >= 16K/64K.
//!    - SRX v05 state footprint: STRICTLY 160 BYTES AT ANY N (100% L1D resident, 0 DRAM traffic).
//!    - Memory advantage: 26,214.4x more compact at N = 65,536!
//!    - Real step generation latency measurement on Intel Xeon E5-2650 v2.
//! 5. Telemetry generation:
//!    - `telemetry_classic_chinchilla.txt`
//!    - `telemetry_srx_v05_chinchilla.txt`

use std::fs;
use std::time::Instant;

use srxformer::{
    classic::{
        backward_loss as classic_backward_loss, forward_loss as classic_forward_loss,
        split_into_eos_sequences as classic_split_sequences, AdamW as ClassicAdamW, FastRng,
        InferenceTelemetry, InferenceWorkspace, KvCache, TestCaseResult, TelemetryReport,
        Tokenizer, Transformer, TransformerConfig, TransformerGrad as ClassicGrad,
        TrainTelemetry, TrainWorkspace as ClassicTrainWorkspace, EOS_TOKEN_ID,
    },
    srx_v05::{
        backward_loss as srx_v05_backward_loss, forward_loss as srx_v05_forward_loss,
        split_into_eos_sequences as srx_v05_split_sequences, SrxAdamW as SrxAdamWV05,
        SrxGrad as SrxGradV05, SrxState as SrxStateV05,
        SrxTelemetryReport as SrxTelemetryReportV05,
        SrxTrainWorkspace as SrxTrainWorkspaceV05, SrxTransformer as SrxTransformerV05,
        SrxWorkspace as SrxWorkspaceV05,
    },
};

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CheckpointRecord {
    pct: usize,
    stage: String,
    epoch: usize,
    mflops: f64,
    loss: f32,
    perplexity: f32,
    passed: usize,
    total: usize,
    accuracy: f32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MemoryWallResult {
    n: usize,
    classic_kv_bytes: usize,
    srx_state_bytes: usize,
    advantage_ratio: f64,
    cache_residence_classic: String,
    classic_step_latency_ns: f64,
    classic_step_latency_us: f64,
    srx_step_latency_ns: f64,
    srx_step_latency_us: f64,
    speedup: f64,
}

fn main() {
    let bench_start_time = Instant::now();
    println!("=================================================================================================================");
    println!(" SRXformer Sprint 4: Multi-Scale Iso-FLOPs Benchmark & Long Context Stress Test (The Memory Wall Challenge)     ");
    println!(" Target Hardware: Intel Xeon E5-2650 v2 (Ivy Bridge-EP, AVX FP32, L1D 32KB per core, L2 256KB, L3 20MB)        ");
    println!(" Compute Parity:  Strict Iso-FLOPs Parity on Chinchilla 20:1 Scaling Law (Pretrain + Instruct with 25% Replay)  ");
    println!(" Architecture:    Strict Parameter Parity @ 896 Weights (0.00% delta)                                            ");
    println!(" Control Suite:   60 Heterogeneous Tasks Across 6 Domains (Addition, Subtraction, Mul/Div, Facts, Space, Logic)");
    println!(" Memory Wall:     Lengths N ∈ [32, 128, 512, 1 024, 4 096, 16 384, 65 536] (O(N) KV-Cache vs O(1) 160-Byte State)");
    println!("=================================================================================================================\n");

    // -------------------------------------------------------------------------
    // 1. ARCHITECTURE CONFIGURATIONS & PARAMETER ACCOUNTING
    // -------------------------------------------------------------------------
    let config = TransformerConfig::lang_chinchilla();
    let tokenizer = Tokenizer::chinchilla();

    println!("[1] Architecture & Parameter Accounting (Exact Iso-Parameter Corridor 896 ± 0 weights):");
    println!("    * Vocab Size (V):      {} tokens (data/vocab_chinchilla.txt)", config.vocab_size);
    println!("    * Hidden Dim (d):       {}", config.d_model);
    println!("    * Attention Heads:      {} (head_dim = {})", config.n_heads, config.head_dim());
    println!("    * Decoder Layers:       {}", config.n_layers);
    println!("    * Context Window:       {} tokens", config.max_seq_len);
    println!("    * Feed-Forward Dim:     d_ff = {}", config.d_ff);
    println!("    * Tied LM Head:         true (Token embeddings tied with LM Head)");
    println!("    * Normalization:        RMSNorm (eps = {:.1e})", config.eps);
    println!("    -----------------------------------------------------------------------------------------");

    let classic_model_probe = Transformer::new(config.clone()).expect("Failed to create Classical Transformer");
    let srx_model_probe = SrxTransformerV05::new(config.clone()).expect("Failed to create SRX v05");

    let classic_params = classic_model_probe.param_count();
    let srx_params = srx_model_probe.param_count();

    println!("    * Classical Transformer: {:>3} params (delta = 0, 0.00%) [100% L1D resident]", classic_params);
    println!("    * SRX v05 Quantum Core:  {:>3} params (delta = 0, 0.00%) [100% L1D resident]", srx_params);
    assert_eq!(classic_params, 896);
    assert_eq!(srx_params, 896);
    println!("    * State Footprint @ N=32: Classic = 2,048 B | SRX v05 = 160 B (12.8x more compact)");
    println!("    * State Footprint @ N=64K: Classic = 4,194,304 B | SRX v05 = 160 B (26,214.4x more compact!)\n");

    // -------------------------------------------------------------------------
    // 2. DATASET ACCOUNTING & VERIFICATION
    // -------------------------------------------------------------------------
    println!("[2] Loading Chinchilla Corpora:");
    let pretrain_path = "data/pretrain_chinchilla.txt";
    let pretrain_text = fs::read_to_string(pretrain_path).unwrap_or_else(|_| {
        panic!("Failed to read {}. Run `cargo run --bin prepare_chinchilla_data` first.", pretrain_path)
    });
    let pretrain_tokens = tokenizer.encode(&pretrain_text);
    println!("    * Pretrain tokens: {} (Chinchilla 20:1 optimal scaling target: 17,920)", pretrain_tokens.len());
    assert_eq!(pretrain_tokens.len(), 17920);

    let instruct_path = "data/instruct_chinchilla.txt";
    let instruct_text = fs::read_to_string(instruct_path).unwrap_or_else(|_| {
        panic!("Failed to read {}. Run `cargo run --bin prepare_chinchilla_data` first.", instruct_path)
    });
    let instruct_tokens = tokenizer.encode(&instruct_text);
    println!("    * Instruct tokens: {} (target: 1,800..2,200)", instruct_tokens.len());
    assert_eq!(instruct_tokens.len(), 1901);
    println!("    * Replay Mix Ratio: 25.0% base pretrain facts integrated during Instruct stage\n");

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
        // Domain 2: Subtraction Arithmetic (10 tasks, non-commutative pairs A - B != B - A)
        // =========================================================================
        ("<user> 5 - 2 = <bot>", "3 <eos>", "Subtraction (Pair A: 5 - 2)"),
        ("<user> 5 - 3 = <bot>", "2 <eos>", "Subtraction (Pair A: 5 - 3)"),
        ("<user> 4 - 1 = <bot>", "3 <eos>", "Subtraction (Pair B: 4 - 1)"),
        ("<user> 4 - 3 = <bot>", "1 <eos>", "Subtraction (Pair B: 4 - 3)"),
        ("<user> 9 - 4 = <bot>", "5 <eos>", "Subtraction (Pair C: 9 - 4)"),
        ("<user> 9 - 5 = <bot>", "4 <eos>", "Subtraction (Pair C: 9 - 5)"),
        ("<user> 3 - 1 = <bot>", "2 <eos>", "Subtraction (Pair D: 3 - 1)"),
        ("<user> 3 - 2 = <bot>", "1 <eos>", "Subtraction (Pair D: 3 - 2)"),
        ("4 - 1 =", "3 <eos>", "Subtraction (Base Prompt: 4 - 1)"),
        ("5 - 2 =", "3 <eos>", "Subtraction (Base Prompt: 5 - 2)"),

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
        // Domain 4: Taxonomy & Entity Facts (10 tasks)
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
        // Domain 6: Boolean Logic & Transitivity Chains (10 tasks)
        // =========================================================================
        ("<user> кот это пес <bot>", "нет <eos>", "Boolean Logic (Negation)"),
        ("<user> кот это животное <bot>", "да <eos>", "Boolean Logic (Affirmation)"),
        ("<user> медведь это рыба <bot>", "нет <eos>", "Boolean Logic (Negation)"),
        ("<user> щука это рыба <bot>", "да <eos>", "Boolean Logic (Affirmation)"),
        ("<user> если волк ест заяц то волк хищник <bot>", "да <eos>", "Boolean Logic (Implication)"),
        ("<user> если заяц ест волк <bot>", "нет <eos>", "Boolean Logic (Negative Implication)"),
        ("<user> волк ест заяц <bot>", "да <eos>", "Food Chain Predicate (Affirmation)"),
        ("<user> заяц ест волк <bot>", "нет <eos>", "Food Chain Predicate (Negation)"),
        ("<user> волк ест заяц заяц ест трава <bot>", "да <eos>", "Transitivity Chain (Food Chain)"),
        ("<user> щука ест рыба рыба ест трава <bot>", "да <eos>", "Transitivity Chain (Food Chain)"),
    ];

    println!("[3] Control Suite Verification:");
    println!("    * Total Control Tasks: {} (strictly 10 tasks across 6 domains)", test_cases.len());
    for (i, (prompt, expected, domain)) in test_cases.iter().enumerate() {
        let p_toks = tokenizer.encode(prompt);
        let e_toks = tokenizer.encode(expected);
        assert!(!p_toks.is_empty(), "Prompt {} failed to encode: {}", i, prompt);
        assert!(!e_toks.is_empty(), "Expected {} failed to encode: {}", i, expected);
        if i % 10 == 0 {
            println!("      Domain {}: {:<32} (e.g. \"{}\" -> \"{}\")", i / 10 + 1, domain, prompt, expected);
        }
    }
    println!();

    // -------------------------------------------------------------------------
    // 4. ISO-FLOPS TRAINING BUDGET ACCOUNTING
    // -------------------------------------------------------------------------
    // FLOPs formulas:
    // Full training step = 6 * N_params FLOPs/tok (+ Householder projector overhead for SRX v05 = 864 FLOPs/tok)
    let classic_flops_per_tok = 6 * classic_params as u64; // 5,376 FLOPs/tok
    let srx_flops_per_tok = 6 * srx_params as u64 + 864;   // 6,240 FLOPs/tok

    // Two-stage training configuration:
    // Stage 1: Pretrain on 17,920 tokens
    // Stage 2: Instruct on 1,901 tokens + 25% replay mix from pretrain
    let instruct_seqs = classic_split_sequences(&instruct_tokens, config.max_seq_len);
    let replay_count = ((instruct_seqs.len() as f32 * 0.25) / 0.75).round() as usize;

    let pretrain_toks_epoch = pretrain_tokens.len();
    let instruct_toks_epoch = instruct_tokens.len() + replay_count * 8; // ~2,533 tokens/epoch

    // Let's establish strict Iso-FLOPs budget:
    // Classical: 20 pretrain epochs, 50 instruct epochs
    let classic_pretrain_epochs = 20;
    let classic_instruct_epochs = 50;

    let classic_pretrain_flops = (classic_pretrain_epochs as u64) * (pretrain_toks_epoch as u64) * classic_flops_per_tok;
    let classic_instruct_flops = (classic_instruct_epochs as u64) * (instruct_toks_epoch as u64) * classic_flops_per_tok;
    let total_iso_flops_budget = classic_pretrain_flops + classic_instruct_flops; // ~2,607 MFLOPs (2.61 GFLOPs)

    // For SRX v05: adjust epochs to match the EXACT Iso-FLOPs budget:
    // Pretrain: 17 epochs
    let srx_pretrain_epochs = ((classic_pretrain_flops as f64) / ((pretrain_toks_epoch as f64) * srx_flops_per_tok as f64)).round() as usize; // 17 epochs
    let srx_pretrain_flops = (srx_pretrain_epochs as u64) * (pretrain_toks_epoch as u64) * srx_flops_per_tok;
    let srx_remaining_flops = total_iso_flops_budget.saturating_sub(srx_pretrain_flops);
    let srx_instruct_epochs = ((srx_remaining_flops as f64) / ((instruct_toks_epoch as f64) * srx_flops_per_tok as f64)).round() as usize; // 45 epochs
    let srx_total_flops = srx_pretrain_flops + (srx_instruct_epochs as u64) * (instruct_toks_epoch as u64) * srx_flops_per_tok;

    println!("[4] Iso-FLOPs Compute Budget Accounting:");
    println!("    * Target Total Compute Budget: {:.2} MFLOPs ({:.4} GFLOPs)", total_iso_flops_budget as f64 / 1e6, total_iso_flops_budget as f64 / 1e9);
    println!("    * Classical Transformer Baseline (5,376 FLOPs/tok):");
    println!("      - Stage 1 (Pretrain): {} epochs @ {:.2} MFLOPs = {:.2} MFLOPs",
        classic_pretrain_epochs, (pretrain_toks_epoch as f64 * classic_flops_per_tok as f64) / 1e6, classic_pretrain_flops as f64 / 1e6);
    println!("      - Stage 2 (Instruct): {} epochs @ {:.2} MFLOPs = {:.2} MFLOPs",
        classic_instruct_epochs, (instruct_toks_epoch as f64 * classic_flops_per_tok as f64) / 1e6, classic_instruct_flops as f64 / 1e6);
    println!("      - Total Classical FLOPs: {:.2} MFLOPs", total_iso_flops_budget as f64 / 1e6);
    println!("    * SRX v05 Quantum Core (6,240 FLOPs/tok, +864 FLOPs Householder Projector):");
    println!("      - Stage 1 (Pretrain): {} epochs @ {:.2} MFLOPs = {:.2} MFLOPs",
        srx_pretrain_epochs, (pretrain_toks_epoch as f64 * srx_flops_per_tok as f64) / 1e6, srx_pretrain_flops as f64 / 1e6);
    println!("      - Stage 2 (Instruct): {} epochs @ {:.2} MFLOPs = {:.2} MFLOPs",
        srx_instruct_epochs, (instruct_toks_epoch as f64 * srx_flops_per_tok as f64) / 1e6, (srx_total_flops - srx_pretrain_flops) as f64 / 1e6);
    println!("      - Total SRX v05 FLOPs:   {:.2} MFLOPs (Iso-FLOPs delta = {:.2}%)\n",
        srx_total_flops as f64 / 1e6, ((srx_total_flops as f64 - total_iso_flops_budget as f64) / total_iso_flops_budget as f64) * 100.0);

    // =========================================================================
    // 5. MODEL 1: CLASSICAL TRANSFORMER TWO-STAGE TRAINING
    // =========================================================================
    println!("================================================================================");
    println!(" [5] RUNNING CLASSICAL TRANSFORMER BASELINE (PARITY: 896 PARAMS, SEED 42)");
    println!("================================================================================");
    let mut classic_model = Transformer::new_with_seed(config.clone(), 42)
        .expect("Failed to initialize Classical Transformer");

    let mut classic_train_ws = ClassicTrainWorkspace::new(&config);
    let mut classic_grad = ClassicGrad::new(&config);
    let mut classic_optimizer = ClassicAdamW::new(&config, 0.0);
    let mut classic_rng = FastRng::new(42);

    let mut classic_eval_ws = InferenceWorkspace::new(&config);
    let mut classic_eval_kv = KvCache::new(&config);

    let mut classic_seqs_pretrain = classic_split_sequences(&pretrain_tokens, config.max_seq_len);
    let classic_seqs_instruct = classic_split_sequences(&instruct_tokens, config.max_seq_len);
    let classic_replay_pool = classic_split_sequences(&pretrain_tokens, config.max_seq_len);

    let eval_count = classic_seqs_pretrain.len().min(20);
    let mut classic_initial_loss = 0.0f32;
    for seq in &classic_seqs_pretrain[..eval_count] {
        classic_initial_loss += classic_forward_loss(&mut classic_model, seq, &mut classic_train_ws);
    }
    classic_initial_loss /= eval_count as f32;

    // Checkpoints at 20%, 40%, 60%, 80%, 100% of compute
    let classic_targets = [
        (20, total_iso_flops_budget as f64 * 0.20),
        (40, total_iso_flops_budget as f64 * 0.40),
        (60, total_iso_flops_budget as f64 * 0.60),
        (80, total_iso_flops_budget as f64 * 0.80),
        (100, total_iso_flops_budget as f64 * 1.00),
    ];
    let mut classic_checkpoints = Vec::new();
    let mut classic_loss_history = Vec::new();
    let mut classic_cum_flops = 0u64;
    let mut classic_total_steps = 0usize;
    let mut classic_next_cp = 0;
    let classic_start = Instant::now();

    println!("    * Checkpoint Trajectory (20%, 40%, 60%, 80%, 100% compute):");
    println!("      | Pct | Stage    | Epoch | Compute (MFLOPs) | Loss   | PPL   | Accuracy (60 tests) |");
    println!("      |-----|----------|-------|------------------|--------|-------|---------------------|");

    let evaluate_classic = |model: &Transformer, eval_kv: &mut KvCache, eval_ws: &mut InferenceWorkspace| -> (usize, f32) {
        let mut passed = 0;
        for (prompt, expected, _) in test_cases {
            let p_toks = tokenizer.encode(prompt);
            let gen_ids = model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, eval_kv, eval_ws);
            let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
            if gen_text == expected {
                passed += 1;
            }
        }
        let acc = (passed as f32 / test_cases.len() as f32) * 100.0;
        (passed, acc)
    };

    // Stage 1: Pretrain
    let classic_pretrain_lr = 0.02f32;
    for ep in 1..=classic_pretrain_epochs {
        for i in (1..classic_seqs_pretrain.len()).rev() {
            let j = (classic_rng.next_u64() as usize) % (i + 1);
            classic_seqs_pretrain.swap(i, j);
        }

        let progress = ep as f32 / classic_pretrain_epochs as f32;
        let lr = (classic_pretrain_lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(classic_pretrain_lr * 0.1);

        let mut ep_loss = 0.0f32;
        for seq in &classic_seqs_pretrain {
            classic_grad.zero();
            let loss = classic_forward_loss(&mut classic_model, seq, &mut classic_train_ws);
            ep_loss += loss;
            classic_backward_loss(&mut classic_model, seq, &mut classic_train_ws, &mut classic_grad);
            classic_grad.clip_grad_norm(1.0);
            classic_optimizer.step(&mut classic_model, &classic_grad, lr);
            classic_cum_flops += seq.len() as u64 * classic_flops_per_tok;
            classic_total_steps += 1;
        }
        let avg_loss = ep_loss / classic_seqs_pretrain.len() as f32;
        classic_loss_history.push(avg_loss);

        if classic_next_cp < classic_targets.len() && classic_cum_flops as f64 >= classic_targets[classic_next_cp].1 {
            let pct = classic_targets[classic_next_cp].0;
            let (passed, acc) = evaluate_classic(&classic_model, &mut classic_eval_kv, &mut classic_eval_ws);
            let ppl = avg_loss.exp();
            let mflops = classic_cum_flops as f64 / 1e6;
            println!(
                "      | {:>3}%| Pretrain | {:>5} | {:>16.2} | {:>6.4} | {:>5.2} | {:>2}/60 ({:>5.1}%)      |",
                pct, ep, mflops, avg_loss, ppl, passed, acc
            );
            classic_checkpoints.push(CheckpointRecord {
                pct,
                stage: "Pretrain".to_string(),
                epoch: ep,
                mflops,
                loss: avg_loss,
                perplexity: ppl,
                passed,
                total: test_cases.len(),
                accuracy: acc,
            });
            classic_next_cp += 1;
        }
    }

    // Stage 2: Instruct with 25% Replay Mix
    let classic_instruct_lr = 0.015f32;
    let mut replay_idx = 0;
    for ep in 1..=classic_instruct_epochs {
        let mut ep_seqs = classic_seqs_instruct.clone();
        for _ in 0..replay_count {
            ep_seqs.push(classic_replay_pool[replay_idx % classic_replay_pool.len()].clone());
            replay_idx += 1;
        }
        for i in (1..ep_seqs.len()).rev() {
            let j = (classic_rng.next_u64() as usize) % (i + 1);
            ep_seqs.swap(i, j);
        }

        let progress = ep as f32 / classic_instruct_epochs as f32;
        let lr = (classic_instruct_lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(classic_instruct_lr * 0.1);

        let mut ep_loss = 0.0f32;
        for seq in &ep_seqs {
            classic_grad.zero();
            let loss = classic_forward_loss(&mut classic_model, seq, &mut classic_train_ws);
            ep_loss += loss;
            classic_backward_loss(&mut classic_model, seq, &mut classic_train_ws, &mut classic_grad);
            classic_grad.clip_grad_norm(1.0);
            classic_optimizer.step(&mut classic_model, &classic_grad, lr);
            classic_cum_flops += seq.len() as u64 * classic_flops_per_tok;
            classic_total_steps += 1;
        }
        let avg_loss = ep_loss / ep_seqs.len() as f32;
        classic_loss_history.push(avg_loss);

        if classic_next_cp < classic_targets.len() && (ep == classic_instruct_epochs || classic_cum_flops as f64 >= classic_targets[classic_next_cp].1) {
            let pct = classic_targets[classic_next_cp].0;
            let (passed, acc) = evaluate_classic(&classic_model, &mut classic_eval_kv, &mut classic_eval_ws);
            let ppl = avg_loss.exp();
            let mflops = classic_cum_flops as f64 / 1e6;
            println!(
                "      | {:>3}%| Instruct | {:>5} | {:>16.2} | {:>6.4} | {:>5.2} | {:>2}/60 ({:>5.1}%)      |",
                pct, ep, mflops, avg_loss, ppl, passed, acc
            );
            classic_checkpoints.push(CheckpointRecord {
                pct,
                stage: "Instruct".to_string(),
                epoch: ep,
                mflops,
                loss: avg_loss,
                perplexity: ppl,
                passed,
                total: test_cases.len(),
                accuracy: acc,
            });
            classic_next_cp += 1;
        }
    }

    let classic_elapsed_ms = classic_start.elapsed().as_secs_f64() * 1000.0;
    let classic_final_loss = *classic_loss_history.last().unwrap_or(&classic_initial_loss);

    let classic_train_telemetry = TrainTelemetry {
        num_params: classic_params,
        dataset_tokens: pretrain_tokens.len() + instruct_tokens.len(),
        epochs: classic_pretrain_epochs + classic_instruct_epochs,
        total_training_flops: classic_cum_flops,
        elapsed_ms: classic_elapsed_ms,
        mflops_per_sec: (classic_cum_flops as f64 / 1e6) / (classic_elapsed_ms / 1000.0),
        gflops_per_sec: (classic_cum_flops as f64 / 1e9) / (classic_elapsed_ms / 1000.0),
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
            "      [{}] {:<38}: \"{:<28}\" -> \"{:<22}\"",
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

    const BENCH_STEPS: usize = 50_000;
    classic_eval_kv.reset();
    let b_start = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % config.max_seq_len;
        if pos == 0 { classic_eval_kv.reset(); }
        classic_model.step(i % config.vocab_size, pos, &mut classic_eval_kv, &mut classic_eval_ws);
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
    classic_report.save_to_file("telemetry_classic_chinchilla.txt")
        .expect("Failed to write telemetry_classic_chinchilla.txt");
    println!("    * Classical report saved to: telemetry_classic_chinchilla.txt\n");

    // =========================================================================
    // 6. MODEL 2: SRXFORMER v05 QUANTUM-ALGEBRAIC CORE TWO-STAGE TRAINING
    // =========================================================================
    println!("================================================================================");
    println!(" [6] RUNNING SRXFORMER v05 QUANTUM-ALGEBRAIC CORE (PARITY: 896 PARAMS, SEED 42)");
    println!("================================================================================");
    let mut srx_model = SrxTransformerV05::new_with_seed(config.clone(), 42)
        .expect("Failed to initialize SRX v05");

    let mut srx_train_ws = SrxTrainWorkspaceV05::new(&config);
    let mut srx_grad = SrxGradV05::new(&config);
    let mut srx_optimizer = SrxAdamWV05::new(&config, 0.0);
    let mut srx_rng = FastRng::new(42);

    let mut srx_eval_ws = SrxWorkspaceV05::new(&config);
    let mut srx_eval_state = SrxStateV05::new(&config);

    let mut srx_seqs_pretrain = srx_v05_split_sequences(&pretrain_tokens, config.max_seq_len);
    let srx_seqs_instruct = srx_v05_split_sequences(&instruct_tokens, config.max_seq_len);
    let srx_replay_pool = srx_v05_split_sequences(&pretrain_tokens, config.max_seq_len);

    let mut srx_initial_loss = 0.0f32;
    for seq in &srx_seqs_pretrain[..eval_count] {
        srx_initial_loss += srx_v05_forward_loss(&mut srx_model, seq, &mut srx_train_ws, 1.0);
    }
    srx_initial_loss /= eval_count as f32;

    let srx_targets = [
        (20, total_iso_flops_budget as f64 * 0.20),
        (40, total_iso_flops_budget as f64 * 0.40),
        (60, total_iso_flops_budget as f64 * 0.60),
        (80, total_iso_flops_budget as f64 * 0.80),
        (100, total_iso_flops_budget as f64 * 1.00),
    ];
    let mut srx_checkpoints = Vec::new();
    let mut srx_loss_history = Vec::new();
    let mut srx_cum_flops = 0u64;
    let mut srx_total_steps = 0usize;
    let mut srx_next_cp = 0;
    let srx_start = Instant::now();

    println!("    * Checkpoint Trajectory (20%, 40%, 60%, 80%, 100% compute):");
    println!("      | Pct | Stage    | Epoch | Compute (MFLOPs) | Loss   | PPL   | Accuracy (60 tests) |");
    println!("      |-----|----------|-------|------------------|--------|-------|---------------------|");

    let evaluate_srx = |model: &SrxTransformerV05, eval_state: &mut SrxStateV05, eval_ws: &mut SrxWorkspaceV05| -> (usize, f32) {
        let mut passed = 0;
        for (prompt, expected, _) in test_cases {
            let p_toks = tokenizer.encode(prompt);
            let gen_ids = model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, eval_state, eval_ws);
            let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
            if gen_text == expected {
                passed += 1;
            }
        }
        let acc = (passed as f32 / test_cases.len() as f32) * 100.0;
        (passed, acc)
    };

    // Stage 1: Pretrain
    let srx_pretrain_lr = 0.02f32;
    for ep in 1..=srx_pretrain_epochs {
        for i in (1..srx_seqs_pretrain.len()).rev() {
            let j = (srx_rng.next_u64() as usize) % (i + 1);
            srx_seqs_pretrain.swap(i, j);
        }

        let progress = ep as f32 / srx_pretrain_epochs as f32;
        let eps_srx = 1e-3 + (1.0 - 1e-3) * (1.0 - progress).powi(2);
        let lr = (srx_pretrain_lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(srx_pretrain_lr * 0.1);

        let mut ep_loss = 0.0f32;
        for seq in &srx_seqs_pretrain {
            srx_grad.zero();
            let loss = srx_v05_forward_loss(&mut srx_model, seq, &mut srx_train_ws, eps_srx);
            ep_loss += loss;
            srx_v05_backward_loss(&mut srx_model, seq, &mut srx_train_ws, &mut srx_grad, eps_srx);
            srx_grad.clip_grad_norm(1.0);
            srx_optimizer.step(&mut srx_model, &srx_grad, lr);
            srx_cum_flops += seq.len() as u64 * srx_flops_per_tok;
            srx_total_steps += 1;
        }
        let avg_loss = ep_loss / srx_seqs_pretrain.len() as f32;
        srx_loss_history.push(avg_loss);

        if srx_next_cp < srx_targets.len() && srx_cum_flops as f64 >= srx_targets[srx_next_cp].1 {
            let pct = srx_targets[srx_next_cp].0;
            let (passed, acc) = evaluate_srx(&srx_model, &mut srx_eval_state, &mut srx_eval_ws);
            let ppl = avg_loss.exp();
            let mflops = srx_cum_flops as f64 / 1e6;
            println!(
                "      | {:>3}%| Pretrain | {:>5} | {:>16.2} | {:>6.4} | {:>5.2} | {:>2}/60 ({:>5.1}%)      |",
                pct, ep, mflops, avg_loss, ppl, passed, acc
            );
            srx_checkpoints.push(CheckpointRecord {
                pct,
                stage: "Pretrain".to_string(),
                epoch: ep,
                mflops,
                loss: avg_loss,
                perplexity: ppl,
                passed,
                total: test_cases.len(),
                accuracy: acc,
            });
            srx_next_cp += 1;
        }
    }

    // Stage 2: Instruct with 25% Replay Mix
    let srx_instruct_lr = 0.015f32;
    let mut srx_replay_idx = 0;
    for ep in 1..=srx_instruct_epochs {
        let mut ep_seqs = srx_seqs_instruct.clone();
        for _ in 0..replay_count {
            ep_seqs.push(srx_replay_pool[srx_replay_idx % srx_replay_pool.len()].clone());
            srx_replay_idx += 1;
        }
        for i in (1..ep_seqs.len()).rev() {
            let j = (srx_rng.next_u64() as usize) % (i + 1);
            ep_seqs.swap(i, j);
        }

        let progress = ep as f32 / srx_instruct_epochs as f32;
        let eps_srx = 1e-3 + (1.0 - 1e-3) * (1.0 - progress).powi(2);
        let lr = (srx_instruct_lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())).max(srx_instruct_lr * 0.1);

        let mut ep_loss = 0.0f32;
        for seq in &ep_seqs {
            srx_grad.zero();
            let loss = srx_v05_forward_loss(&mut srx_model, seq, &mut srx_train_ws, eps_srx);
            ep_loss += loss;
            srx_v05_backward_loss(&mut srx_model, seq, &mut srx_train_ws, &mut srx_grad, eps_srx);
            srx_grad.clip_grad_norm(1.0);
            srx_optimizer.step(&mut srx_model, &srx_grad, lr);
            srx_cum_flops += seq.len() as u64 * srx_flops_per_tok;
            srx_total_steps += 1;
        }
        let avg_loss = ep_loss / ep_seqs.len() as f32;
        srx_loss_history.push(avg_loss);

        if srx_next_cp < srx_targets.len() && (ep == srx_instruct_epochs || srx_cum_flops as f64 >= srx_targets[srx_next_cp].1) {
            let pct = srx_targets[srx_next_cp].0;
            let (passed, acc) = evaluate_srx(&srx_model, &mut srx_eval_state, &mut srx_eval_ws);
            let ppl = avg_loss.exp();
            let mflops = srx_cum_flops as f64 / 1e6;
            println!(
                "      | {:>3}%| Instruct | {:>5} | {:>16.2} | {:>6.4} | {:>5.2} | {:>2}/60 ({:>5.1}%)      |",
                pct, ep, mflops, avg_loss, ppl, passed, acc
            );
            srx_checkpoints.push(CheckpointRecord {
                pct,
                stage: "Instruct".to_string(),
                epoch: ep,
                mflops,
                loss: avg_loss,
                perplexity: ppl,
                passed,
                total: test_cases.len(),
                accuracy: acc,
            });
            srx_next_cp += 1;
        }
    }

    let srx_elapsed_ms = srx_start.elapsed().as_secs_f64() * 1000.0;
    let srx_final_loss = *srx_loss_history.last().unwrap_or(&srx_initial_loss);

    let srx_train_telemetry = TrainTelemetry {
        num_params: srx_params,
        dataset_tokens: pretrain_tokens.len() + instruct_tokens.len(),
        epochs: srx_pretrain_epochs + srx_instruct_epochs,
        total_training_flops: srx_cum_flops,
        elapsed_ms: srx_elapsed_ms,
        mflops_per_sec: (srx_cum_flops as f64 / 1e6) / (srx_elapsed_ms / 1000.0),
        gflops_per_sec: (srx_cum_flops as f64 / 1e9) / (srx_elapsed_ms / 1000.0),
        initial_loss: srx_initial_loss,
        final_loss: srx_final_loss,
        initial_perplexity: srx_initial_loss.exp(),
        final_perplexity: srx_final_loss.exp(),
        total_steps: srx_total_steps,
        loss_history: srx_loss_history,
    };

    println!("\n    * SRX v05 Quantum Core Final Evaluation (60 control tasks):");
    let mut srx_test_results = Vec::new();
    let mut srx_passed = 0;
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = srx_model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut srx_eval_state, &mut srx_eval_ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        let passed = gen_text == expected;
        if passed { srx_passed += 1; }
        println!(
            "      [{}] {:<38}: \"{:<28}\" -> \"{:<22}\"",
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

    srx_eval_state.reset();
    let b_start_srx = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % config.max_seq_len;
        if pos == 0 { srx_eval_state.reset(); }
        srx_model.step(i % config.vocab_size, pos, &mut srx_eval_state, &mut srx_eval_ws);
    }
    let b_elapsed_srx = b_start_srx.elapsed();
    let srx_step_ns = b_elapsed_srx.as_nanos() as f64 / BENCH_STEPS as f64;
    let srx_step_us = srx_step_ns / 1000.0;
    let srx_tok_sec = BENCH_STEPS as f64 / b_elapsed_srx.as_secs_f64();
    let srx_inf_flops = 2 * srx_params as u64 + 288;
    let srx_inf_gflops = (srx_inf_flops as f64 * srx_tok_sec) / 1e9;

    let srx_inf_telemetry = InferenceTelemetry {
        flops_per_token: srx_inf_flops,
        bench_steps: BENCH_STEPS,
        step_latency_ns: srx_step_ns,
        step_latency_us: srx_step_us,
        tokens_per_sec: srx_tok_sec,
        gflops_per_sec: srx_inf_gflops,
    };

    let srx_report = SrxTelemetryReportV05::new(
        srx_train_telemetry.clone(),
        srx_inf_telemetry.clone(),
        srx_test_results.clone(),
        srx_eval_state.memory_bytes(),
        config.max_seq_len,
    );
    srx_report.save_to_file("telemetry_srx_v05_chinchilla.txt")
        .expect("Failed to write telemetry_srx_v05_chinchilla.txt");
    println!("    * SRX v05 report saved to: telemetry_srx_v05_chinchilla.txt\n");

    // =========================================================================
    // 7. LONG CONTEXT STRESS TEST: THE MEMORY WALL CHALLENGE
    // =========================================================================
    println!("=================================================================================================================");
    println!(" [7] THE MEMORY WALL CHALLENGE: LONG CONTEXT SCALING & LATENCY MICRO-BENCHMARK                                  ");
    println!(" Context Lengths N ∈ [32, 128, 512, 1 024, 4 096, 16 384, 65 536]                                              ");
    println!(" Intel Xeon E5-2650 v2 Hierarchy: L1D = 32 KB | L2 = 256 KB | L3 = 20 MB (Shared)                                ");
    println!("=================================================================================================================");

    let context_lengths = [32, 128, 512, 1024, 4096, 16384, 65536];
    let mut memory_wall_results = Vec::new();

    println!(" | Length N | Classic KV Cache | SRX v05 State | Advantage | Cache Hierarchy Status     | Classic (us) | SRX (us)  | Speedup |");
    println!(" |----------|------------------|---------------|-----------|----------------------------|--------------|-----------|---------|");

    let mut max_bench_config = config.clone();
    max_bench_config.max_seq_len = 65536;
    let mut classic_bench_model = Transformer::new_with_seed(max_bench_config, 42)
        .expect("Failed to initialize classic bench model");
    classic_bench_model.load_weights_flat(&classic_model.extract_weights())
        .expect("Failed to load weights into classic bench model");

    for &n in &context_lengths {
        // 1. Memory calculations:
        // Classical KV Cache size: 2 * N * d_model * sizeof(f32) = 2 * N * 8 * 4 = 64 * N bytes
        let classic_kv_bytes = 2 * n * config.d_model * std::mem::size_of::<f32>();
        // SRX v05 state: strictly (Thetas [2, 4] + M [2, 4, 4]) * 4 = (8 + 32) * 4 = 160 bytes
        let srx_state_bytes = srx_eval_state.memory_bytes(); // strictly 160
        let advantage_ratio = classic_kv_bytes as f64 / srx_state_bytes as f64;

        let cache_status = if classic_kv_bytes < 32 * 1024 {
            format!("{:.1}% L1D Cache", (classic_kv_bytes as f64 / 32768.0) * 100.0)
        } else if classic_kv_bytes == 32 * 1024 {
            "CROSSOVER -> spills L1D (32KB)".to_string()
        } else if classic_kv_bytes <= 256 * 1024 {
            format!("Spills L1D -> in L2 ({:.1}%)", (classic_kv_bytes as f64 / 262144.0) * 100.0)
        } else if classic_kv_bytes <= 20 * 1024 * 1024 {
            format!("CROSSOVER L2 -> in L3 ({:.2}MB)", classic_kv_bytes as f64 / (1024.0 * 1024.0))
        } else {
            "DRAM Memory Wall".to_string()
        };

        // 2. Measure Classical Transformer generation step latency at context length N:
        let mut n_config = config.clone();
        n_config.max_seq_len = n;
        let mut classic_n_kv = KvCache::new(&n_config);
        let mut classic_n_ws = InferenceWorkspace::new(&n_config);

        // Pre-fill KV-cache up to N - 1 tokens
        for pos in 0..n - 1 {
            let offset = pos * config.d_model;
            for c in 0..config.d_model {
                classic_n_kv.k[offset + c] = 0.05 * ((pos + c) % 7) as f32;
                classic_n_kv.v[offset + c] = 0.05 * ((pos * 3 + c) % 7) as f32;
            }
        }
        classic_n_kv.current_len = n - 1;

        // Warmup
        let token_in = 4;
        classic_bench_model.step(token_in, n - 1, &mut classic_n_kv, &mut classic_n_ws);

        // Iteration count adjusted to avoid excessive wallclock while maintaining precision
        let iters = if n <= 1024 {
            2000
        } else if n <= 4096 {
            500
        } else if n <= 16384 {
            100
        } else {
            30
        };

        let t0 = Instant::now();
        for _ in 0..iters {
            classic_n_kv.current_len = n - 1;
            classic_bench_model.step(token_in, n - 1, &mut classic_n_kv, &mut classic_n_ws);
        }
        let classic_elapsed = t0.elapsed();
        let c_step_ns = classic_elapsed.as_nanos() as f64 / iters as f64;
        let c_step_us = c_step_ns / 1000.0;

        // 3. Measure SRX v05 generation step latency at context length N:
        let mut srx_n_state = SrxStateV05::new(&config);
        let mut srx_n_ws = SrxWorkspaceV05::new(&config);

        // Simulate equivalent recurrent steps
        srx_n_state.current_pos = n - 1;

        // Warmup
        srx_model.step(token_in, n - 1, &mut srx_n_state, &mut srx_n_ws);

        let t0_srx = Instant::now();
        for _ in 0..iters {
            srx_model.step(token_in, n - 1, &mut srx_n_state, &mut srx_n_ws);
        }
        let srx_elapsed = t0_srx.elapsed();
        let s_step_ns = srx_elapsed.as_nanos() as f64 / iters as f64;
        let s_step_us = s_step_ns / 1000.0;

        let speedup = c_step_us / s_step_us;

        let kv_display = if classic_kv_bytes < 1024 * 1024 {
            format!("{:.1} KB", classic_kv_bytes as f64 / 1024.0)
        } else {
            format!("{:.2} MB", classic_kv_bytes as f64 / (1024.0 * 1024.0))
        };

        println!(
            " | {:>8} | {:>16} | {:>10} B | {:>8.1}x | {:<26} | {:>12.2} | {:>9.2} | {:>6.2}x |",
            n, kv_display, srx_state_bytes, advantage_ratio, cache_status, c_step_us, s_step_us, speedup
        );

        memory_wall_results.push(MemoryWallResult {
            n,
            classic_kv_bytes,
            srx_state_bytes,
            advantage_ratio,
            cache_residence_classic: cache_status,
            classic_step_latency_ns: c_step_ns,
            classic_step_latency_us: c_step_us,
            srx_step_latency_ns: s_step_ns,
            srx_step_latency_us: s_step_us,
            speedup,
        });
    }

    println!(" |----------|------------------|---------------|-----------|----------------------------|--------------|-----------|---------|\n");

    // -------------------------------------------------------------------------
    // 8. SUMMARY COMPARISON & SPRINT 4 TELEMETRY SYNTHESIS
    // -------------------------------------------------------------------------
    println!("=================================================================================================================");
    println!(" SPRINT 4 COMPREHENSIVE BENCHMARK SUMMARY: CHINCHILLA ISO-FLOPS & MEMORY WALL                                   ");
    println!("=================================================================================================================");
    println!(" Metric                           | Classical Baseline         | SRX v05 Quantum-Algebraic  | Advantage / Delta  ");
    println!("----------------------------------|----------------------------|----------------------------|--------------------");
    println!(" Trainable Parameters             | {:>26} | {:>26} | {:>18}",
        format!("{} weights", classic_params),
        format!("{} weights", srx_params),
        "0 (0.00% parity)"
    );
    println!(" State Footprint (N=32)           | {:>26} | {:>26} | {:>18}",
        "2,048 bytes (2.0 KB)",
        "160 bytes (0.16 KB)",
        "12.8x smaller"
    );
    println!(" State Footprint (N=512, L1D Wall)| {:>26} | {:>26} | {:>18}",
        "32,768 bytes (32.0 KB)",
        "160 bytes (0.16 KB)",
        "204.8x smaller"
    );
    println!(" State Footprint (N=65,536, DRAM) | {:>26} | {:>26} | {:>18}",
        "4,194,304 bytes (4.19 MB)",
        "160 bytes (0.16 KB)",
        "26,214.4x smaller"
    );
    println!(" Total Compute Budget             | {:>26} | {:>26} | {:>18}",
        format!("{:.2} MFLOPs", classic_train_telemetry.total_training_flops as f64 / 1e6),
        format!("{:.2} MFLOPs", srx_train_telemetry.total_training_flops as f64 / 1e6),
        "Iso-FLOPs Parity"
    );
    println!(" Training Wallclock Time          | {:>26} | {:>26} | {:>18}",
        format!("{:.2} s", classic_elapsed_ms / 1000.0),
        format!("{:.2} s", srx_elapsed_ms / 1000.0),
        format!("{:.2}x", classic_elapsed_ms / srx_elapsed_ms)
    );
    println!(" Final Training Loss              | {:>26} | {:>26} | {:>18}",
        format!("{:.4}", classic_final_loss),
        format!("{:.4}", srx_final_loss),
        format!("{:.4}", classic_final_loss - srx_final_loss)
    );
    println!(" Final Perplexity                 | {:>26} | {:>26} | {:>18}",
        format!("{:.2}", classic_final_loss.exp()),
        format!("{:.2}", srx_final_loss.exp()),
        format!("{:.2}", classic_final_loss.exp() - srx_final_loss.exp())
    );
    println!(" 60-Task Exact Match Accuracy     | {:>26} | {:>26} | {:>18}",
        format!("{}/60 ({:.1}%)", classic_passed, (classic_passed as f32 / 60.0) * 100.0),
        format!("{}/60 ({:.1}%)", srx_passed, (srx_passed as f32 / 60.0) * 100.0),
        format!("+{} passed", srx_passed as i32 - classic_passed as i32)
    );
    println!(" Base Step Latency (N=32)         | {:>26} | {:>26} | {:>18}",
        format!("{:.2} us ({:.0} ns)", memory_wall_results[0].classic_step_latency_us, memory_wall_results[0].classic_step_latency_ns),
        format!("{:.2} us ({:.0} ns)", memory_wall_results[0].srx_step_latency_us, memory_wall_results[0].srx_step_latency_ns),
        format!("{:.2}x faster", memory_wall_results[0].speedup)
    );
    println!(" Deep Step Latency (N=65,536)     | {:>26} | {:>26} | {:>18}",
        format!("{:.2} us ({:.0} ns)", memory_wall_results.last().unwrap().classic_step_latency_us, memory_wall_results.last().unwrap().classic_step_latency_ns),
        format!("{:.2} us ({:.0} ns)", memory_wall_results.last().unwrap().srx_step_latency_us, memory_wall_results.last().unwrap().srx_step_latency_ns),
        format!("{:.2}x faster", memory_wall_results.last().unwrap().speedup)
    );
    println!("=================================================================================================================\n");

    // Domain Breakdown
    println!("Domain Accuracy Breakdown (10 tests per domain):");
    let domains = [
        "Addition Arithmetic (+)",
        "Subtraction Arithmetic (-) (A-B != B-A)",
        "Multiplication & Division (*, /)",
        "Taxonomy & Entity Facts",
        "Spatial Reasoning (где)",
        "Boolean Logic & Transitivity Chains",
    ];
    for (d_idx, d_name) in domains.iter().enumerate() {
        let start = d_idx * 10;
        let end = start + 10;
        let c_pass = classic_test_results[start..end].iter().filter(|r| r.passed).count();
        let s_pass = srx_test_results[start..end].iter().filter(|r| r.passed).count();
        println!("  * Domain {}: {:<42} | Classic: {:>2}/10 ({:>5.1}%) | SRX v05: {:>2}/10 ({:>5.1}%)",
            d_idx + 1, d_name, c_pass, c_pass as f32 * 10.0, s_pass, s_pass as f32 * 10.0
        );
    }

    println!("\nTotal Sprint 4 Benchmark execution completed in {:.2} seconds!", bench_start_time.elapsed().as_secs_f64());
}
