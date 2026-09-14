use std::fs;
use srxformer::{
    classic::{Tokenizer, TransformerConfig, BOT_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, USER_TOKEN_ID},
    VOCAB_CHINCHILLA,
};

#[test]
fn test_chinchilla_vocab_properties() {
    let tok = Tokenizer::chinchilla();
    assert_eq!(tok.vocab_size(), 65, "Vocab size must be exactly 65 tokens");
    assert_eq!(VOCAB_CHINCHILLA.len(), 65);

    // Mandatory control tokens
    assert_eq!(tok.token_to_id("<pad>"), Some(PAD_TOKEN_ID));
    assert_eq!(tok.token_to_id("<eos>"), Some(EOS_TOKEN_ID));
    assert_eq!(tok.token_to_id("<user>"), Some(USER_TOKEN_ID));
    assert_eq!(tok.token_to_id("<bot>"), Some(BOT_TOKEN_ID));

    // Mandatory digits and arithmetic operators
    for d in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] {
        assert!(tok.token_to_id(d).is_some(), "Digit {} must exist in vocab", d);
    }
    for op in ["+", "-", "*", "/", "="] {
        assert!(tok.token_to_id(op).is_some(), "Operator {} must exist in vocab", op);
    }

    // Mandatory logical connectives & inquiry words
    for word in ["да", "нет", "это", "кто", "где", "что", "ест"] {
        assert!(tok.token_to_id(word).is_some(), "Logical word {} must exist in vocab", word);
    }

    // Foundational science, philosophy, AI and logic tokens
    for token in [
        "не", "как", "почему", "если", "то", "или", "наука", "число", "модель", "разум",
        "знание", "логика",
    ] {
        assert!(
            tok.token_to_id(token).is_some(),
            "Foundational token {} must exist in vocab",
            token
        );
    }
}

#[test]
fn test_chinchilla_model_param_parity() {
    let config = TransformerConfig::chinchilla();
    assert!(config.validate().is_ok(), "Chinchilla config must be valid");
    assert_eq!(
        config.param_count(),
        896,
        "Chinchilla config must have EXACTLY 896 parameters (0.00% parity delta)!"
    );

    // Verify Chinchilla 20:1 token-to-parameter law
    let chinchilla_target_tokens = config.param_count() * 20;
    assert_eq!(
        chinchilla_target_tokens, 17920,
        "Chinchilla 20:1 scaling law requires exactly 17,920 tokens for an 896-parameter model!"
    );
}

#[test]
fn test_pretrain_chinchilla_exact_tokens() {
    let tok = Tokenizer::chinchilla();
    let text = fs::read_to_string("data/pretrain_chinchilla.txt")
        .expect("Failed to read data/pretrain_chinchilla.txt");
    let pretrain_tokens = tok.encode(&text);
    assert_eq!(
        pretrain_tokens.len(),
        17920,
        "Pretrain chinchilla corpus must contain EXACTLY 17,920 tokens (actual: {})",
        pretrain_tokens.len()
    );
}

#[test]
fn test_instruct_chinchilla_token_length() {
    let tok = Tokenizer::chinchilla();
    let text = fs::read_to_string("data/instruct_chinchilla.txt")
        .expect("Failed to read data/instruct_chinchilla.txt");
    let instruct_tokens = tok.encode(&text);
    assert!(
        instruct_tokens.len() >= 1800 && instruct_tokens.len() <= 2500,
        "Instruct chinchilla corpus must contain 1800..2500 tokens (actual: {})",
        instruct_tokens.len()
    );
}

#[test]
fn test_chinchilla_corpora_100_percent_roundtrip_fidelity() {
    let tok = Tokenizer::chinchilla();

    // 1. Verify pretrain_chinchilla.txt roundtrip fidelity
    let pretrain_text = fs::read_to_string("data/pretrain_chinchilla.txt")
        .expect("Failed to read data/pretrain_chinchilla.txt");
    let pretrain_lines: Vec<&str> = pretrain_text.lines().filter(|l| !l.is_empty()).collect();
    assert!(!pretrain_lines.is_empty(), "Pretrain corpus must not be empty");

    for (idx, line) in pretrain_lines.iter().enumerate() {
        assert!(
            line.ends_with("<eos>"),
            "Pretrain line {} does not end with <eos>: {}",
            idx,
            line
        );
        let enc = tok.encode(line);
        let dec = tok.decode(&enc);
        assert_eq!(
            &dec, line,
            "Pretrain 100% roundtrip fidelity failure at line {}: expected '{}', got '{}'",
            idx, line, dec
        );
    }

    // 2. Verify instruct_chinchilla.txt roundtrip fidelity and structure
    let instruct_text = fs::read_to_string("data/instruct_chinchilla.txt")
        .expect("Failed to read data/instruct_chinchilla.txt");
    let instruct_lines: Vec<&str> = instruct_text.lines().filter(|l| !l.is_empty()).collect();
    assert!(!instruct_lines.is_empty(), "Instruct corpus must not be empty");

    for (idx, line) in instruct_lines.iter().enumerate() {
        assert!(
            line.starts_with("<user> "),
            "Instruct line {} must start with '<user> ': {}",
            idx,
            line
        );
        assert!(
            line.contains(" <bot> "),
            "Instruct line {} must contain ' <bot> ': {}",
            idx,
            line
        );
        assert!(
            line.ends_with(" <eos>"),
            "Instruct line {} must end with ' <eos>': {}",
            idx,
            line
        );
        let enc = tok.encode(line);
        let dec = tok.decode(&enc);
        assert_eq!(
            &dec, line,
            "Instruct 100% roundtrip fidelity failure at line {}: expected '{}', got '{}'",
            idx, line, dec
        );
    }

    println!(
        "Verified 100% roundtrip fidelity across {} pretrain lines and {} instruct lines!",
        pretrain_lines.len(),
        instruct_lines.len()
    );
}
