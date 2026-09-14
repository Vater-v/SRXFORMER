use std::fs;
use std::time::Instant;

use srxformer::{
    apply_givens, InferenceTelemetry, SrxState, SrxTelemetryReport, SrxTransformer, SrxWorkspace,
    TestCaseResult, Tokenizer, TransformerConfig, EOS_TOKEN_ID,
};

#[test]
fn test_srx_givens_unitarity_thorough() {
    let d = 8;
    let mut rng = srxformer::FastRng::new(42);
    let thetas: Vec<f32> = (0..d - 1).map(|_| rng.gen_range(-3.1415, 3.1415)).collect();
    let original: Vec<f32> = (0..d).map(|_| rng.gen_normal(0.0, 1.0)).collect();
    let mut v = original.clone();

    // Forward
    apply_givens(&mut v, &thetas, false);
    // Inverse
    apply_givens(&mut v, &thetas, true);

    for i in 0..d {
        assert!(
            (original[i] - v[i]).abs() < 1e-5,
            "Unitarity failed at index {i}: orig={}, rec={}",
            original[i],
            v[i]
        );
    }
}

#[test]
fn test_srx_step_forward_equivalence_thorough() {
    let config = TransformerConfig::lang_512();
    let model = SrxTransformer::new_with_seed(config.clone(), 12345).unwrap();
    let mut ws = SrxWorkspace::new(&config);

    let tokens = [2, 7, 14, 5, 8, 3, 10];
    let forward_logits = model.forward(&tokens, &mut ws).to_vec();

    let mut state = SrxState::new(&config);
    let mut step_logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let l = model.step(tok, pos, &mut state, &mut ws);
        if pos == tokens.len() - 1 {
            step_logits = l.to_vec();
        }
    }

    assert_eq!(forward_logits.len(), step_logits.len());
    for i in 0..forward_logits.len() {
        assert!(
            (forward_logits[i] - step_logits[i]).abs() < 1e-4,
            "Step vs Forward mismatch at {i}: fwd={}, step={}",
            forward_logits[i],
            step_logits[i]
        );
    }
}

#[test]
fn test_srx_unified_corpus_training_and_telemetry() {
    let tok = Tokenizer::new();
    let config = TransformerConfig::lang_512();

    let text = fs::read_to_string("data/unified_corpus.txt")
        .expect("data/unified_corpus.txt must exist");
    let tokens = tok.encode(&text);
    assert_eq!(tokens.len(), 980);

    let mut model = SrxTransformer::new_with_seed(config.clone(), 100).unwrap();
    const EPOCHS: usize = 120;
    const LR: f32 = 0.015;

    let train_telemetry = model.train_dataset(&tokens, EPOCHS, LR);

    println!(
        "SRX Training: init_loss={:.4}, final_loss={:.4}, elapsed={:.2}ms, FLOPs={}",
        train_telemetry.initial_loss,
        train_telemetry.final_loss,
        train_telemetry.elapsed_ms,
        train_telemetry.total_training_flops
    );

    assert!(
        train_telemetry.final_loss < 1.0,
        "SRX loss must converge below 1.0, got {}",
        train_telemetry.final_loss
    );

    // Save weights
    let weight_path = "data/srx_v02_model_weights.bin";
    model.save_weights(weight_path).expect("Failed to save SRX weights");

    // Verify weights roundtrip
    let loaded_model = SrxTransformer::load_from_file(weight_path).expect("Failed to load SRX weights");
    let orig_w = model.extract_weights();
    let loaded_w = loaded_model.extract_weights();
    assert_eq!(orig_w.len(), loaded_w.len());
    for i in 0..orig_w.len() {
        assert_eq!(orig_w[i], loaded_w[i], "Weight mismatch at {i}");
    }

    // Benchmark inference
    let mut ws = SrxWorkspace::new(&config);
    let mut state = SrxState::new(&config);
    const BENCH_STEPS: usize = 10_000;

    let start = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % config.max_seq_len;
        if pos == 0 {
            state.reset();
        }
        loaded_model.step(i % config.vocab_size, pos, &mut state, &mut ws);
    }
    let elapsed = start.elapsed();

    let step_latency_ns = elapsed.as_nanos() as f64 / BENCH_STEPS as f64;
    let step_latency_us = step_latency_ns / 1000.0;
    let tokens_per_sec = BENCH_STEPS as f64 / elapsed.as_secs_f64();
    let flops_per_token = 2 * config.param_count() as u64;
    let gflops_per_sec = (flops_per_token as f64 * tokens_per_sec) / 1e9;

    let inf_telemetry = InferenceTelemetry {
        flops_per_token,
        bench_steps: BENCH_STEPS,
        step_latency_ns,
        step_latency_us,
        tokens_per_sec,
        gflops_per_sec,
    };

    // Quality evaluation on 10 control tasks
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

    let mut test_results = Vec::new();
    for (prompt, expected, category) in test_cases {
        let p_toks = tok.encode(prompt);
        let generated_ids = loaded_model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut state, &mut ws);
        let generated_text = tok.decode(&generated_ids[p_toks.len()..]);
        let passed = generated_text == expected;
        println!(
            "[{}] {:<26}: \"{:<24}\" -> \"{:<20}\" (Expected: \"{}\")",
            if passed { "PASS" } else { "FAIL" },
            category,
            prompt,
            generated_text,
            expected
        );
        test_results.push(TestCaseResult {
            prompt: prompt.to_string(),
            generated: generated_text,
            expected: expected.to_string(),
            category: category.to_string(),
            passed,
        });
    }

    let report = SrxTelemetryReport::new(
        train_telemetry,
        inf_telemetry,
        test_results,
        state.memory_bytes(),
        config.max_seq_len,
    );

    let report_file = "telemetry_srx_v02.txt";
    report.save_to_file(report_file).expect("Failed to write telemetry_srx_v02.txt");

    let read_back = fs::read_to_string(report_file).expect("Failed to read telemetry_srx_v02.txt");
    assert!(read_back.contains("SRXFORMER: SUPER-RESOLVENT XFORMER v02 COMPUTE / LATENCY / QUALITY TELEMETRY REPORT"));
    assert!(read_back.contains("SRX State (Theta + M):  152 bytes"));
}
