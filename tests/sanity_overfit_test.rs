use std::time::Instant;
use srxformer::scaled::config::Tier;
use srxformer::scaled::nn::ScaledAdamW;
use srxformer::scaled::qreno_srx::QrenoSrxLM;
use srxformer::qreno::model::{QrenoAdamW, QrenoGrad};

#[inline(always)]
fn truncate_chars(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

#[test]
fn test_sanity_overfit_russian_folklore() {
    println!("\n=======================================================");
    println!("  SANITY CHECK OVERFIT TEST: RUSSIAN FOLKLORE MEMORIZATION");
    println!("  Target: Overfit 5 sentences to Loss < 0.15");
    println!("  Architecture: Tier::Pro (d=32, H=8, 640 B L1D resident)");
    println!("=======================================================\n");

    let sentences = [
        "<user> хочешь сей а хочешь куй все равно получишь <bot> результат <eos>",
        "<user> делу время <bot> потехе час <eos>",
        "<user> без труда не выловишь <bot> рыбку из пруда <eos>",
        "<user> терпенье и труд <bot> все перетрут <eos>",
        "<user> семь раз отмерь <bot> один раз отрежь <eos>",
    ];

    let mut lm = QrenoSrxLM::new(Tier::Pro);
    let mut ws = lm.new_workspace();
    let mut seq_ws = lm.new_sequence_workspace(256);
    let mut opt = ScaledAdamW::new(lm.transformer.param_count(), 0.012, 0.0);
    let mut qreno_grad = QrenoGrad::new(&lm.tokenizer.config);
    let mut qreno_opt = QrenoAdamW::new(&lm.tokenizer.config, 0.012, 0.0);

    let start_time = Instant::now();
    let max_epochs = 400;
    let mut final_loss = 999.0f32;

    for epoch in 1..=max_epochs {
        let mut epoch_loss = 0.0f32;

        let mut max_sentence_loss = 0.0f32;

        for sentence in &sentences {
            let bytes = sentence.as_bytes();
            let seq_loss = lm.train_sequence(bytes, &mut seq_ws, &mut qreno_grad);
            epoch_loss += seq_loss;
            if seq_loss > max_sentence_loss {
                max_sentence_loss = seq_loss;
            }

            let mut layers = lm.transformer.get_layers_mut();
            opt.step_layers(&mut layers, 1.0);
            qreno_opt.step(&mut lm.tokenizer.weights, &qreno_grad);
            qreno_grad.zero();
        }

        final_loss = epoch_loss / (sentences.len() as f32);

        if epoch <= 5 || epoch % 25 == 0 || max_sentence_loss < 0.03 {
            println!("  [Epoch {:3}] Avg Loss = {:.4} (PPL = {:.3}), Max Loss = {:.4}", epoch, final_loss, final_loss.exp(), max_sentence_loss);
        }

        if max_sentence_loss < 0.03 {
            println!("  => Target reached! All sentences Loss < 0.03 (Max = {:.4}) at Epoch {}", max_sentence_loss, epoch);
            break;
        }
    }

    let elapsed = start_time.elapsed().as_secs_f32();
    println!("  Training finished in {:.2}s with final Loss = {:.4}\n", elapsed, final_loss);

    // INFERENCE TEST
    let prompts = [
        ("<user> хочешь сей а хочешь куй все равно получишь <bot>", "результат"),
        ("<user> делу время <bot>", "потехе час"),
        ("<user> без труда не выловишь <bot>", "рыбку из пруда"),
        ("<user> терпенье и труд <bot>", "все перетрут"),
        ("<user> семь раз отмерь <bot>", "один раз отрежь"),
    ];

    println!("  INFERENCE RESULTS (Greedy Decoding T=0):");
    println!("  +--------------------------------------------------------------+---------------------------+---------------------------+");
    println!("  | Prompt                                                       | Expected                  | Generated Continuation    |");
    println!("  +--------------------------------------------------------------+---------------------------+---------------------------+");

    let mut success_count = 0;

    for (prompt, expected) in &prompts {
        let generated_bytes = lm.generate_bytes(prompt, 40, &mut ws);
        let gen_str = String::from_utf8_lossy(&generated_bytes);
        let clean_gen = gen_str.trim().replace("<eos>", "").trim().to_string();

        let passed = clean_gen.contains(expected);
        if passed {
            success_count += 1;
        }

        println!("  | {:60} | {:25} | {:25} | [{}]",
                 truncate_chars(prompt, 60),
                 expected,
                 truncate_chars(&clean_gen, 25),
                 if passed { "PASS" } else { "FAIL" });
        println!("    Raw Generated: '{}'", gen_str);
    }
    println!("  +--------------------------------------------------------------+---------------------------+---------------------------+\n");

    println!("  Sanity Check Score: {} / {} exact matches!", success_count, prompts.len());
}

#[test]
fn test_inspect_clusters() {
    let lm = QrenoSrxLM::new(Tier::Pro);
    let mut ws = srxformer::qreno::SshLatticeWorkspace::new();

    let sentences = [
        "<user> хочешь сей а хочешь куй все равно получишь <bot> результат <eos>",
        "<user> делу время <bot> потехе час <eos>",
        "<user> без труда не выловишь <bot> рыбку из пруда <eos>",
        "<user> терпенье и труд <bot> все перетрут <eos>",
        "<user> семь раз отмерь <bot> один раз отрежь <eos>",
    ];

    println!("\n=== PHYSICAL SSH LATTICE CLUSTERS ===");
    for (i, s) in sentences.iter().enumerate() {
        let clusters = lm.tokenizer.tokenize_physical(s, &mut ws);
        println!("\nSentence {}: '{}'", i + 1, s);
        println!("Total clusters: {}", clusters.len());
        let bytes = s.as_bytes();
        for (c_idx, c) in clusters.iter().enumerate() {
            let slice = &bytes[c.start..c.start + c.len];
            let text = String::from_utf8_lossy(slice);
            println!("  [{:2}] len={:2} bytes='{}'", c_idx, c.len, text);
        }
    }
}

#[test]
fn test_sanity_overfit_classic() {
    use srxformer::scaled::classic::{ScaledClassicTransformer, ScaledClassicKvCache};
    use srxformer::scaled::config::ScaledConfig;

    println!("\n=======================================================");
    println!("  SANITY CHECK OVERFIT TEST: CLASSIC TRANSFORMER");
    println!("=======================================================\n");

    let sentence = "<user> хочешь сей а хочешь куй все равно получишь <bot> результат <eos>";
    let config = ScaledConfig::from_tier(Tier::Pro, 256);
    let mut model = ScaledClassicTransformer::new(config);
    let mut opt = ScaledAdamW::new(model.param_count(), 0.015, 0.0001);

    let bytes = sentence.as_bytes();
    let n_steps = bytes.len() - 1;

    for epoch in 1..=400 {
        let mut cache = ScaledClassicKvCache::new(model.config.d_model, 256);
        let mut epoch_loss = 0.0f32;

        for t in 0..n_steps {
            let in_b = bytes[t] as usize;
            let target_b = bytes[t + 1] as usize;
            let step_loss = model.train_step(in_b, target_b, &mut cache);
            epoch_loss += step_loss;

            let mut layers = model.get_layers_mut();
            opt.step_layers(&mut layers, 1.0);
        }

        let avg_loss = epoch_loss / (n_steps as f32);
        if epoch <= 5 || epoch % 25 == 0 || avg_loss < 0.15 {
            println!("  [Classic Epoch {:3}] Loss = {:.4} (PPL = {:.3})", epoch, avg_loss, avg_loss.exp());
        }
        if avg_loss < 0.15 {
            println!("  => Classic Target reached! Loss = {:.4} at Epoch {}", avg_loss, epoch);
            break;
        }
    }
}
