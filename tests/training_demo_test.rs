use std::fs;
use srxformer::{
    train_instruct, train_pretrain, InferenceWorkspace, KvCache, Tokenizer, Transformer,
    TransformerConfig,
};

#[test]
fn test_train_and_generate_answers() {
    let tok = Tokenizer::new();
    let config = TransformerConfig::lang_512();
    let pretrain_text = fs::read_to_string("data/pretrain.txt").unwrap();
    let pretrain_tokens = tok.encode(&pretrain_text);

    let instruct_text = fs::read_to_string("data/instruct.txt").unwrap();
    let instruct_tokens = tok.encode(&instruct_text);

    let mut model = Transformer::new_with_seed(config.clone(), 100).unwrap();
    let pre_metrics = train_pretrain(&mut model, &pretrain_tokens, &config, 15, 0.015);
    println!("Pretrain metrics: init={:.4}, final={:.4}, elapsed={:.2}ms",
        pre_metrics.initial_loss, pre_metrics.final_loss, pre_metrics.elapsed_ms);

    let inst_metrics = train_instruct(&mut model, &instruct_tokens, &config, 60, 0.015);
    println!("Instruct metrics: init={:.4}, final={:.4}, elapsed={:.2}ms",
        inst_metrics.initial_loss, inst_metrics.final_loss, inst_metrics.elapsed_ms);

    let mut ws = InferenceWorkspace::new(&config);
    let mut kv = KvCache::new(&config);

    println!("\n--- Instruct Mode Verification ---");
    let instruct_prompts = [
        ("<user> 2 + 3 = <bot>", "5 <eos>"),
        ("<user> кто кот <bot>", "кот это животное <eos>"),
        ("<user> кот это пес <bot>", "нет <eos>"),
        ("<user> 4 - 1 = <bot>", "3 <eos>"),
    ];

    for (p, expected) in instruct_prompts {
        let p_toks = tok.encode(p);
        let generated = model.generate_until_eos(&p_toks, 8, srxformer::EOS_TOKEN_ID, &mut kv, &mut ws);
        let completion = tok.decode(&generated[p_toks.len()..]);
        println!("Prompt: '{}' -> Completion: '{}' (Expected: '{}')", p, completion, expected);
        assert_eq!(completion, expected);
    }

    println!("\n--- Catastrophic Forgetting Check (Base Pretrain Facts) ---");
    let base_prompts = [
        ("2 + 3 =", "5 <eos>"),
        ("4 - 1 =", "3 <eos>"),
        ("кот это", "животное <eos>"),
        ("кот это пес =", "нет <eos>"),
    ];

    for (p, expected) in base_prompts {
        let p_toks = tok.encode(p);
        let generated = model.generate_until_eos(&p_toks, 6, srxformer::EOS_TOKEN_ID, &mut kv, &mut ws);
        let completion = tok.decode(&generated[p_toks.len()..]);
        println!("Base Prompt: '{}' -> Completion: '{}' (Expected: '{}')", p, completion, expected);
    }
}
