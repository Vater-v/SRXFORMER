use std::fs;
use std::time::Instant;

use srxformer::{
    classic::{InferenceTelemetry, TestCaseResult, Tokenizer, TransformerConfig, EOS_TOKEN_ID},
    srx_v03::{SrxState, SrxTelemetryReport, SrxTransformer, SrxWorkspace},
};

#[test]
fn test_unified_corpus_v2_integrity() {
    let tok = Tokenizer::new();
    let text = fs::read_to_string("data/unified_corpus_v2.txt")
        .expect("data/unified_corpus_v2.txt must exist");

    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let mut seen = std::collections::HashSet::new();

    for l in &lines {
        assert!(l.ends_with("<eos>"), "Line must end with <eos>: {}", l);
        assert!(seen.insert(*l), "Found duplicate line in unified corpus v2: {}", l);
    }

    let tokens = tok.encode(&text);
    println!(
        "Corpus v2 integrity check: {} unique lines, {} tokens",
        lines.len(),
        tokens.len()
    );

    assert_eq!(lines.len(), 440, "Corpus v2 must contain exactly 440 unique lines");
    assert_eq!(tokens.len(), 2940, "Corpus v2 must contain exactly 2,940 tokens (3x of 980)");
    assert!(
        tokens.len() >= 2900 && tokens.len() <= 3050,
        "Token count must be in range 2900..3050, got {}",
        tokens.len()
    );
}

#[test]
fn test_srx_v03_corpus_v2_training_and_telemetry() {
    let config = TransformerConfig::lang_v2();
    let tokenizer = Tokenizer::new();

    let unified_path = "data/unified_corpus_v2.txt";
    let corpus_text = fs::read_to_string(unified_path).expect("Failed to read data/unified_corpus_v2.txt");
    let tokens = tokenizer.encode(&corpus_text);
    assert_eq!(tokens.len(), 2940);

    // Exact 30 Heterogeneous Control Tasks
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

    assert_eq!(test_cases.len(), 30, "Must contain exactly 30 test cases");

    const SEED: u64 = 42;
    const EPOCHS: usize = 340;
    const LR: f32 = 0.024;

    let mut model = SrxTransformer::new_with_seed(config.clone(), SEED).unwrap();
    let train_telemetry = model.train_dataset(&tokens, EPOCHS, LR);
    println!(
        "SRX v03 Corpus v2 Training: init_loss={:.4}, final_loss={:.4}, elapsed={:.2}ms, FLOPs={}",
        train_telemetry.initial_loss,
        train_telemetry.final_loss,
        train_telemetry.elapsed_ms,
        train_telemetry.total_training_flops
    );

    assert!(
        train_telemetry.final_loss < 0.95,
        "Final loss should drop below 0.95, got {}",
        train_telemetry.final_loss
    );

    // Save weights
    let weights_path = "data/srx_v03_corpus_v2_weights.bin";
    model.save_weights(weights_path).expect("Failed to save SRX v03 corpus v2 weights");

    // Load back and verify bitwise equivalence
    let loaded_model = SrxTransformer::load_from_file(weights_path).expect("Failed to load weights");
    assert_eq!(model.param_count(), loaded_model.param_count());

    let mut ws = SrxWorkspace::new(&config);
    let mut state = SrxState::new(&config);
    let mut test_results = Vec::new();
    let mut passed_count = 0;

    println!("\nEvaluating SRX v03 on 30 Control Tasks (Corpus v2):");
    for (prompt, expected, category) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = loaded_model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut state, &mut ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);

        let passed = gen_text == expected;
        if passed {
            passed_count += 1;
        }
        println!(
            "[{}] {:<34}: \"{:<24}\" -> \"{:<22}\" (Expected: \"{}\")",
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

    println!("\nPassed {}/30 tests ({:.1}%)", passed_count, (passed_count as f64 / 30.0) * 100.0);

    // Inference benchmark (50,000 steps)
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

    let report_path = "telemetry_srx_v03_corpus_v2.txt";
    report.save_to_file(report_path).expect("Failed to write telemetry_srx_v03_corpus_v2.txt");
    println!("Telemetry saved to {} successfully", report_path);
}
