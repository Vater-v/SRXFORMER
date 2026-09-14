use std::fs;
use std::time::Instant;

use srxformer::{
    apply_butterfly_4,
    classic::{InferenceTelemetry, TestCaseResult, Tokenizer, TransformerConfig, EOS_TOKEN_ID},
    srx_v03::{SrxState, SrxTelemetryReport, SrxTransformer, SrxWorkspace},
};

#[test]
fn test_srx_v03_butterfly_unitarity_thorough() {
    let mut rng = srxformer::FastRng::new(42);
    let thetas: [f32; 4] = [
        rng.gen_range(-3.1415, 3.1415),
        rng.gen_range(-3.1415, 3.1415),
        rng.gen_range(-3.1415, 3.1415),
        rng.gen_range(-3.1415, 3.1415),
    ];
    let original: [f32; 4] = [
        rng.gen_normal(0.0, 1.0),
        rng.gen_normal(0.0, 1.0),
        rng.gen_normal(0.0, 1.0),
        rng.gen_normal(0.0, 1.0),
    ];
    let mut v = [0.0f32; 4];

    // Forward
    apply_butterfly_4(&original, &thetas, false, &mut v);

    // Check norm preservation
    let norm_orig: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm_orig - norm_v).abs() < 1e-5,
        "Norm preservation failed: orig={}, v={}",
        norm_orig,
        norm_v
    );

    // Inverse
    let mut rec = [0.0f32; 4];
    apply_butterfly_4(&v, &thetas, true, &mut rec);

    for i in 0..4 {
        assert!(
            (original[i] - rec[i]).abs() < 1e-5,
            "Unitarity roundtrip failed at index {i}: orig={}, rec={}",
            original[i],
            rec[i]
        );
    }
}

#[test]
fn test_srx_v03_step_forward_equivalence_thorough() {
    let mut config = TransformerConfig::lang_512();
    config.d_ff = 8;
    let model = SrxTransformer::new_with_seed(config.clone(), 12345).unwrap();
    let mut ws = SrxWorkspace::new(&config);

    let tokens = [1, 5, 2, 8, 3, 10, 4, 7];

    let forward_logits = model.forward(&tokens, &mut ws).to_vec();

    let mut state = SrxState::new(&config);
    let mut step_logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let logits = model.step(tok, pos, &mut state, &mut ws);
        if pos == tokens.len() - 1 {
            step_logits = logits.to_vec();
        }
    }

    assert_eq!(forward_logits.len(), step_logits.len());
    for (i, (&f_val, &s_val)) in forward_logits.iter().zip(step_logits.iter()).enumerate() {
        assert!(
            (f_val - s_val).abs() < 1e-4,
            "Forward vs Step mismatch at logit {i}: forward={f_val}, step={s_val}"
        );
    }
}

#[test]
fn test_srx_v03_unified_corpus_training_and_telemetry() {
    let mut config = TransformerConfig::lang_512();
    config.d_ff = 8;
    let tokenizer = Tokenizer::new();

    let unified_path = "data/unified_corpus.txt";
    let corpus_text = fs::read_to_string(unified_path).expect("Failed to read data/unified_corpus.txt");
    let tokens = tokenizer.encode(&corpus_text);

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

    const SEED: u64 = 42;
    const EPOCHS: usize = 160;
    const LR: f32 = 0.018;

    let mut model = SrxTransformer::new_with_seed(config.clone(), SEED).unwrap();
    let train_telemetry = model.train_dataset(&tokens, EPOCHS, LR);
    println!(
        "SRX v03 Training: init_loss={:.4}, final_loss={:.4}, elapsed={:.2}ms, FLOPs={}",
        train_telemetry.initial_loss,
        train_telemetry.final_loss,
        train_telemetry.elapsed_ms,
        train_telemetry.total_training_flops
    );

    assert!(
        train_telemetry.final_loss < 0.75,
        "Final loss should drop below 0.75, got {}",
        train_telemetry.final_loss
    );

    // Save trained weights
    let weights_path = "data/srx_v03_model_weights.bin";
    model.save_weights(weights_path).expect("Failed to save weights");

    // Load back
    let loaded_model = SrxTransformer::load_from_file(weights_path).expect("Failed to load weights");
    assert_eq!(model.param_count(), loaded_model.param_count());

    let mut ws = SrxWorkspace::new(&config);
    let mut state = SrxState::new(&config);
    let mut test_results = Vec::new();

    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = loaded_model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut state, &mut ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);

        let passed = gen_text == expected;
        println!(
            "[{}] {:<26}: \"{:<24}\" -> \"{:<20}\" (Expected: \"{}\")",
            if passed { "PASS" } else { "FAIL" },
            category,
            prompt,
            gen_text,
            expected
        );
        test_results.push(TestCaseResult {
            prompt: prompt.to_string(),
            generated: gen_text,
            expected: expected.to_string(),
            category: category.to_string(),
            passed,
        });
    }

    // Benchmark step latency (50,000 steps)
    const BENCH_STEPS: usize = 50_000;
    state.reset();
    let b_start = Instant::now();
    for i in 0..BENCH_STEPS {
        let pos = i % config.max_seq_len;
        if pos == 0 {
            state.reset();
        }
        loaded_model.step(i % config.vocab_size, pos, &mut state, &mut ws);
    }
    let b_elapsed = b_start.elapsed();

    let step_ns = b_elapsed.as_nanos() as f64 / BENCH_STEPS as f64;
    let step_us = step_ns / 1000.0;
    let tok_sec = BENCH_STEPS as f64 / b_elapsed.as_secs_f64();
    let flops_per_token = 2 * loaded_model.param_count() as u64;
    let inf_gflops = (flops_per_token as f64 * tok_sec) / 1e9;

    let inf_telemetry = InferenceTelemetry {
        flops_per_token,
        bench_steps: BENCH_STEPS,
        step_latency_ns: step_ns,
        step_latency_us: step_us,
        tokens_per_sec: tok_sec,
        gflops_per_sec: inf_gflops,
    };

    let report = SrxTelemetryReport::new(
        train_telemetry,
        inf_telemetry,
        test_results,
        state.memory_bytes(),
        config.max_seq_len,
    );

    report.save_to_file("telemetry_srx_v03.txt").expect("Failed to write telemetry_srx_v03.txt");
    let read_back = fs::read_to_string("telemetry_srx_v03.txt").unwrap();
    assert!(read_back.contains("SRXFORMER: SUPER-RESOLVENT XFORMER v03 (MONARCH BUTTERFLY + SELECTIVE GATE) COMPUTE / LATENCY / QUALITY TELEMETRY REPORT"));
    println!("Telemetry saved to telemetry_srx_v03.txt successfully");
}
