use std::fs;
use srxformer::{
    classic::{Tokenizer, TransformerConfig, EOS_TOKEN_ID},
    srx_v04::{SrxState, SrxTransformer, SrxWorkspace},
};

#[test]
fn test_srx_v04_corpus_v2_training() {
    let config = TransformerConfig::srx_v04_parity();
    let tokenizer = Tokenizer::new();

    let unified_path = "data/unified_corpus_v2.txt";
    let corpus_text = fs::read_to_string(unified_path).expect("Failed to read corpus");
    let tokens = tokenizer.encode(&corpus_text);

    let test_cases = [
        ("<user> 2 + 3 = <bot>", "5 <eos>"),
        ("<user> 1 + 2 = <bot>", "3 <eos>"),
        ("<user> 3 + 4 = <bot>", "7 <eos>"),
        ("<user> 5 + 3 = <bot>", "8 <eos>"),
        ("<user> 4 + 5 = <bot>", "9 <eos>"),
        ("<user> 4 - 1 = <bot>", "3 <eos>"),
        ("<user> 5 - 2 = <bot>", "3 <eos>"),
        ("<user> 9 - 4 = <bot>", "5 <eos>"),
        ("<user> 8 - 3 = <bot>", "5 <eos>"),
        ("<user> 7 - 2 = <bot>", "5 <eos>"),
        ("<user> 2 * 3 = <bot>", "6 <eos>"),
        ("<user> 2 * 2 = <bot>", "4 <eos>"),
        ("<user> 3 * 3 = <bot>", "9 <eos>"),
        ("<user> кто кот <bot>", "кот это животное <eos>"),
        ("<user> кто пес <bot>", "пес это друг <eos>"),
        ("<user> кто волк <bot>", "волк это зверь <eos>"),
        ("<user> кто лиса <bot>", "лиса это хищник <eos>"),
        ("<user> где волк <bot>", "лес <eos>"),
        ("<user> где рыба <bot>", "река <eos>"),
        ("<user> где кот <bot>", "дом <eos>"),
        ("<user> где птица <bot>", "небо <eos>"),
        ("<user> кот это пес <bot>", "нет <eos>"),
        ("<user> волк это пес <bot>", "нет <eos>"),
        ("<user> кот это животное <bot>", "да <eos>"),
        ("<user> волк это зверь <bot>", "да <eos>"),
        ("2 + 3 =", "5 <eos>"),
        ("4 - 1 =", "3 <eos>"),
        ("2 * 3 =", "6 <eos>"),
        ("волк это зверь =", "да <eos>"),
        ("где рыба =", "река <eos>"),
    ];

    let mut model = SrxTransformer::new_with_seed(config.clone(), 42).unwrap();
    let epochs = 263; // 4,350 MFLOPs budget
    let telemetry = model.train_dataset(&tokens, epochs, 0.019);

    let mut state = SrxState::new(&config);
    let mut ws = SrxWorkspace::new(&config);
    let mut passed = 0;

    for (prompt, expected) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut state, &mut ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        if gen_text == expected {
            passed += 1;
        } else {
            println!("FAIL: '{}' -> got '{}', expected '{}'", prompt, gen_text, expected);
        }
    }
    println!("SRX v04 Parity (898 params, 263 epochs): passed={}/30 ({:.1}%), final_loss={:.4}", passed, (passed as f64 / 30.0) * 100.0, telemetry.final_loss);
    assert!(passed >= 27, "SRX v04 must pass at least 27/30 (90%) control tests, got {}/30", passed);
}

#[test]
fn test_classic_parity_training() {
    use srxformer::classic::{InferenceWorkspace, KvCache, Transformer};

    let config = TransformerConfig::lang_v2_parity();
    let tokenizer = Tokenizer::new();

    let unified_path = "data/unified_corpus_v2.txt";
    let corpus_text = fs::read_to_string(unified_path).expect("Failed to read corpus");
    let tokens = tokenizer.encode(&corpus_text);

    let test_cases = [
        ("<user> 2 + 3 = <bot>", "5 <eos>"),
        ("<user> 1 + 2 = <bot>", "3 <eos>"),
        ("<user> 3 + 4 = <bot>", "7 <eos>"),
        ("<user> 5 + 3 = <bot>", "8 <eos>"),
        ("<user> 4 + 5 = <bot>", "9 <eos>"),
        ("<user> 4 - 1 = <bot>", "3 <eos>"),
        ("<user> 5 - 2 = <bot>", "3 <eos>"),
        ("<user> 9 - 4 = <bot>", "5 <eos>"),
        ("<user> 8 - 3 = <bot>", "5 <eos>"),
        ("<user> 7 - 2 = <bot>", "5 <eos>"),
        ("<user> 2 * 3 = <bot>", "6 <eos>"),
        ("<user> 2 * 2 = <bot>", "4 <eos>"),
        ("<user> 3 * 3 = <bot>", "9 <eos>"),
        ("<user> кто кот <bot>", "кот это животное <eos>"),
        ("<user> кто пес <bot>", "пес это друг <eos>"),
        ("<user> кто волк <bot>", "волк это зверь <eos>"),
        ("<user> кто лиса <bot>", "лиса это хищник <eos>"),
        ("<user> где волк <bot>", "лес <eos>"),
        ("<user> где рыба <bot>", "река <eos>"),
        ("<user> где кот <bot>", "дом <eos>"),
        ("<user> где птица <bot>", "небо <eos>"),
        ("<user> кот это пес <bot>", "нет <eos>"),
        ("<user> волк это пес <bot>", "нет <eos>"),
        ("<user> кот это животное <bot>", "да <eos>"),
        ("<user> волк это зверь <bot>", "да <eos>"),
        ("2 + 3 =", "5 <eos>"),
        ("4 - 1 =", "3 <eos>"),
        ("2 * 3 =", "6 <eos>"),
        ("волк это зверь =", "да <eos>"),
        ("где рыба =", "река <eos>"),
    ];

    let mut model = Transformer::new_with_seed(config.clone(), 42).unwrap();
    let epochs = 275; // 4,350 MFLOPs budget
    let telemetry = model.train_dataset(&tokens, epochs, 0.015);

    let mut ws = InferenceWorkspace::new(&config);
    let mut kv = KvCache::new(&config);
    let mut passed = 0;

    for (prompt, expected) in test_cases {
        let p_toks = tokenizer.encode(prompt);
        let gen_ids = model.generate_until_eos(&p_toks, 8, EOS_TOKEN_ID, &mut kv, &mut ws);
        let gen_text = tokenizer.decode(&gen_ids[p_toks.len()..]);
        if gen_text == expected {
            passed += 1;
        } else {
            println!("Classic FAIL: '{}' -> got '{}', expected '{}'", prompt, gen_text, expected);
        }
    }
    println!("Classic Parity (896 params, 275 epochs): passed={}/30 ({:.1}%), final_loss={:.4}", passed, (passed as f64 / 30.0) * 100.0, telemetry.final_loss);
}
