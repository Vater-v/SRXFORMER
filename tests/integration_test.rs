use std::fs;
use srxformer::Tokenizer;

#[test]
fn test_pretrain_corpus_chinchilla_tokens() {
    let tok = Tokenizer::new();
    let text = fs::read_to_string("data/pretrain.txt")
        .expect("Failed to read data/pretrain.txt");
    let tokens = tok.encode(&text);
    assert_eq!(
        tokens.len(),
        10240,
        "Pretrain corpus must contain exactly 10,240 tokens according to Chinchilla 20:1 law!"
    );
}

#[test]
fn test_instruct_corpus_token_length() {
    let tok = Tokenizer::new();
    let text = fs::read_to_string("data/instruct.txt")
        .expect("Failed to read data/instruct.txt");
    let tokens = tok.encode(&text);
    assert!(
        tokens.len() >= 550 && tokens.len() <= 700,
        "Instruct corpus must contain ~600 tokens (actual: {})",
        tokens.len()
    );
}

#[test]
fn test_tokenizer_roundtrip_exhaustive() {
    let tok = Tokenizer::new();
    let phrases = [
        "<user> 2 + 3 = <bot> 5 <eos>",
        "<user> кто кот <bot> кот это животное <eos>",
        "<user> кот это пес <bot> нет <eos>",
        "<user> 4 - 1 = <bot> 3 <eos>",
        "<user> пес это друг <bot> да <eos>",
        "5 - 2 = 3 <eos>",
        "кто пес = пес это друг <eos>",
    ];

    for phrase in phrases {
        let enc = tok.encode(phrase);
        let dec = tok.decode(&enc);
        assert_eq!(phrase, dec);
    }
}
