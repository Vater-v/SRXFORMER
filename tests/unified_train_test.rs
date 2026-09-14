use std::fs;
use srxformer::{
    train_dataset, InferenceTelemetry, InferenceWorkspace, KvCache, TestCaseResult,
    TelemetryReport, Tokenizer, Transformer, TransformerConfig, EOS_TOKEN_ID,
};

#[test]
fn test_unified_corpus_training_and_telemetry() {
    let tok = Tokenizer::new();
    let config = TransformerConfig::lang_512();

    // 1. Verify unified corpus exists, has strictly no duplicate lines, ends with <eos>
    let text = fs::read_to_string("data/unified_corpus.txt")
        .expect("data/unified_corpus.txt must exist");
    let lines: Vec<&str> = text.lines().collect();
    let mut seen = std::collections::HashSet::new();
    for l in &lines {
        assert!(l.ends_with("<eos>"), "Line must end with <eos>: {}", l);
        assert!(seen.insert(*l), "Found duplicate line in unified corpus: {}", l);
    }
    assert_eq!(lines.len(), 146, "Unified corpus must contain exactly 146 unique sentences");
    let tokens = tok.encode(&text);
    assert_eq!(tokens.len(), 980, "Unified corpus must contain exactly 980 tokens");

    // 2. Train model directly on unified corpus
    let mut model = Transformer::new_with_seed(config.clone(), 100).unwrap();
    let telemetry = train_dataset(&mut model, &tokens, &config, 120, 0.015);

    println!(
        "Training on unified corpus: init_loss={:.4}, final_loss={:.4}, elapsed={:.2}ms, FLOPs={}",
        telemetry.initial_loss, telemetry.final_loss, telemetry.elapsed_ms, telemetry.total_training_flops
    );

    assert!(telemetry.final_loss < 1.0, "Loss must converge below 1.0, got {}", telemetry.final_loss);

    // 3. Test weights serialization roundtrip with trained weights
    let weight_path = "target/test_unified_weights.bin";
    model.save_weights(weight_path).expect("Failed to save weights");

    let loaded_model = Transformer::load_from_file(weight_path).expect("Failed to load weights");
    let _ = fs::remove_file(weight_path);

    let orig_w = model.extract_weights();
    let loaded_w = loaded_model.extract_weights();
    assert_eq!(orig_w.len(), loaded_w.len());
    for i in 0..orig_w.len() {
        assert_eq!(orig_w[i], loaded_w[i], "Weight mismatch after reload at index {i}");
    }

    // 4. Test generation on core competencies
    let mut ws = InferenceWorkspace::new(&config);
    let mut kv = KvCache::new(&config);

    let test_cases = [
        ("<user> 2 + 3 = <bot>", "5 <eos>", "Addition Arithmetic"),
        ("<user> 4 - 1 = <bot>", "3 <eos>", "Subtraction Arithmetic"),
        ("<user> кто кот <bot>", "кот это животное <eos>", "Entity Fact"),
        ("<user> кто пес <bot>", "пес это друг <eos>", "Entity Fact"),
        ("<user> кот это пес <bot>", "нет <eos>", "Logic Negation"),
        ("<user> кот это животное <bot>", "да <eos>", "Logic Affirmation"),
        ("2 + 3 =", "5 <eos>", "Base Addition"),
        ("4 - 1 =", "3 <eos>", "Base Subtraction"),
        ("кот это пес =", "нет <eos>", "Base Logic"),
    ];

    let mut test_results = Vec::new();
    for (prompt, expected, category) in test_cases {
        let p_toks = tok.encode(prompt);
        let generated_ids = loaded_model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut kv, &mut ws);
        let generated_text = tok.decode(&generated_ids[p_toks.len()..]);
        let passed = generated_text == expected;
        println!(
            "[{}] {:<22}: Prompt='{:<25}' -> Generated='{:<20}' (Expected='{}')",
            if passed { "PASS" } else { "FAIL" }, category, prompt, generated_text, expected
        );
        test_results.push(TestCaseResult {
            prompt: prompt.to_string(),
            generated: generated_text,
            expected: expected.to_string(),
            category: category.to_string(),
            passed,
        });
    }

    // 5. Compute inference telemetry
    let bench_steps = 10_000;
    kv.reset();
    let start = std::time::Instant::now();
    for i in 0..bench_steps {
        let pos = i % config.max_seq_len;
        if pos == 0 {
            kv.reset();
        }
        loaded_model.step(i % config.vocab_size, pos, &mut kv, &mut ws);
    }
    let elapsed = start.elapsed();
    let step_latency_ns = elapsed.as_nanos() as f64 / bench_steps as f64;
    let step_latency_us = step_latency_ns / 1000.0;
    let tokens_per_sec = bench_steps as f64 / elapsed.as_secs_f64();
    let flops_per_token = 2 * config.param_count() as u64;
    let gflops_per_sec = (flops_per_token as f64 * tokens_per_sec) / 1e9;

    let inf_telemetry = InferenceTelemetry {
        flops_per_token,
        bench_steps,
        step_latency_ns,
        step_latency_us,
        tokens_per_sec,
        gflops_per_sec,
    };

    let report = TelemetryReport::new(telemetry, inf_telemetry, test_results);
    let report_path = "target/test_telemetry.txt";
    report.save_to_file(report_path).expect("Failed to save telemetry report");
    let read_back = fs::read_to_string(report_path).expect("Failed to read report");
    assert!(read_back.contains("SRXFORMER: CLASSICAL TRANSFORMER (BASELINE) COMPUTE / LATENCY / QUALITY TELEMETRY REPORT"));
    assert!(read_back.contains("Вложили compute:"));
    assert!(read_back.contains("Ответ за compute:"));
    assert!(read_back.contains("Точность (Exact Match):"));
    let _ = fs::remove_file(report_path);
}
