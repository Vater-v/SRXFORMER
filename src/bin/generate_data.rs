use std::collections::HashSet;
use std::fs;
use srxformer::Tokenizer;

fn generate_pretrain_sentences() -> Vec<String> {
    let mut pool = Vec::new();

    // 1. Arithmetic addition (0..5)
    for a in 0..=5 {
        for b in 0..=5 {
            if a + b <= 5 {
                pool.push(format!("{} + {} = {} <eos>", a, b, a + b));
            }
        }
    }

    // 2. Arithmetic subtraction (0..5)
    for a in 0..=5 {
        for b in 0..=5 {
            if a >= b {
                pool.push(format!("{} - {} = {} <eos>", a, b, a - b));
            }
        }
    }

    // 3. Russian Facts
    pool.push("кот это животное <eos>".to_string());
    pool.push("пес это животное <eos>".to_string());
    pool.push("пес это друг <eos>".to_string());
    pool.push("кот это друг <eos>".to_string());
    pool.push("кто кот = кот это животное <eos>".to_string());
    pool.push("кто пес = пес это животное <eos>".to_string());
    pool.push("кто кот = кот это друг <eos>".to_string());
    pool.push("кто пес = пес это друг <eos>".to_string());
    pool.push("кто кот = животное <eos>".to_string());
    pool.push("кто пес = животное <eos>".to_string());
    pool.push("кто кот = друг <eos>".to_string());
    pool.push("кто пес = друг <eos>".to_string());

    // 4. Logic relations
    pool.push("кот это пес = нет <eos>".to_string());
    pool.push("пес это кот = нет <eos>".to_string());
    pool.push("кот это животное = да <eos>".to_string());
    pool.push("пес это животное = да <eos>".to_string());
    pool.push("пес это друг = да <eos>".to_string());
    pool.push("кот это друг = да <eos>".to_string());
    pool.push("животное это кот = да <eos>".to_string());
    pool.push("животное это пес = да <eos>".to_string());
    pool.push("друг это пес = да <eos>".to_string());
    pool.push("друг это кот = да <eos>".to_string());
    pool.push("кот это 0 = нет <eos>".to_string());
    pool.push("пес это 1 = нет <eos>".to_string());
    pool.push("кот это 2 = нет <eos>".to_string());
    pool.push("пес это 3 = нет <eos>".to_string());
    pool.push("1 это 2 = нет <eos>".to_string());
    pool.push("2 это 3 = нет <eos>".to_string());
    pool.push("3 это 4 = нет <eos>".to_string());
    pool.push("4 это 5 = нет <eos>".to_string());
    pool.push("0 это 0 = да <eos>".to_string());
    pool.push("1 это 1 = да <eos>".to_string());
    pool.push("2 это 2 = да <eos>".to_string());
    pool.push("3 это 3 = да <eos>".to_string());
    pool.push("4 это 4 = да <eos>".to_string());
    pool.push("5 это 5 = да <eos>".to_string());

    pool
}

fn build_pretrain_corpus(tok: &Tokenizer, target_len: usize) -> (String, usize) {
    let pool = generate_pretrain_sentences();
    let mut selected_lines = Vec::new();
    let mut current_len = 0;
    let mut pool_idx = 0;

    // Fill corpus deterministically until close to target
    while current_len + 25 < target_len {
        let line = &pool[pool_idx % pool.len()];
        let len = tok.encode(line).len();
        selected_lines.push(line.clone());
        current_len += len;
        pool_idx += 1;
    }

    // Now fill remaining exact tokens
    let remainder = target_len - current_len;
    let filler_lines = match remainder {
        4 => vec!["кот это животное <eos>".to_string()],
        5 => vec!["кто кот = животное <eos>".to_string()],
        6 => vec!["1 + 1 = 2 <eos>".to_string()],
        7 => vec!["кто кот = кот это животное <eos>".to_string()],
        8 => vec!["кот это животное <eos>".to_string(), "пес это друг <eos>".to_string()],
        9 => vec!["кто кот = животное <eos>".to_string(), "кот это друг <eos>".to_string()],
        10 => vec!["1 + 1 = 2 <eos>".to_string(), "кот это животное <eos>".to_string()],
        11 => vec!["1 + 1 = 2 <eos>".to_string(), "кто кот = животное <eos>".to_string()],
        12 => vec!["1 + 1 = 2 <eos>".to_string(), "2 + 2 = 4 <eos>".to_string()],
        13 => vec!["кто кот = кот это животное <eos>".to_string(), "1 + 1 = 2 <eos>".to_string()],
        14 => vec!["кто кот = кот это животное <eos>".to_string(), "кто пес = пес это друг <eos>".to_string()],
        15 => vec!["1 + 1 = 2 <eos>".to_string(), "кто кот = животное <eos>".to_string(), "кот это друг <eos>".to_string()],
        16 => vec!["1 + 1 = 2 <eos>".to_string(), "2 + 2 = 4 <eos>".to_string(), "кот это животное <eos>".to_string()],
        17 => vec!["1 + 1 = 2 <eos>".to_string(), "2 + 2 = 4 <eos>".to_string(), "кто кот = животное <eos>".to_string()],
        18 => vec!["1 + 1 = 2 <eos>".to_string(), "2 + 2 = 4 <eos>".to_string(), "3 + 2 = 5 <eos>".to_string()],
        19 => vec!["кто кот = кот это животное <eos>".to_string(), "1 + 1 = 2 <eos>".to_string(), "2 + 2 = 4 <eos>".to_string()],
        20 => vec!["1 + 1 = 2 <eos>".to_string(), "2 + 2 = 4 <eos>".to_string(), "кот это животное <eos>".to_string(), "пес это друг <eos>".to_string()],
        21 => vec!["1 + 1 = 2 <eos>".to_string(), "2 + 2 = 4 <eos>".to_string(), "3 + 2 = 5 <eos>".to_string(), "4 - 1 = 3 <eos>".to_string()],
        22 => vec!["1 + 1 = 2 <eos>".to_string(), "2 + 2 = 4 <eos>".to_string(), "3 + 2 = 5 <eos>".to_string(), "кто кот = животное <eos>".to_string()],
        23 => vec!["кто кот = кот это животное <eos>".to_string(), "1 + 1 = 2 <eos>".to_string(), "2 + 2 = 4 <eos>".to_string(), "кот это животное <eos>".to_string()],
        24 => vec!["1 + 1 = 2 <eos>".to_string(), "2 + 2 = 4 <eos>".to_string(), "3 + 2 = 5 <eos>".to_string(), "5 - 1 = 4 <eos>".to_string()],
        25 => vec!["кто кот = кот это животное <eos>".to_string(), "1 + 1 = 2 <eos>".to_string(), "2 + 2 = 4 <eos>".to_string(), "3 + 2 = 5 <eos>".to_string()],
        _ => panic!("Unhandled remainder: {}", remainder),
    };

    for fl in filler_lines {
        let len = tok.encode(&fl).len();
        selected_lines.push(fl);
        current_len += len;
    }

    assert_eq!(current_len, target_len);
    let full_text = selected_lines.join("\n") + "\n";
    let tokenized = tok.encode(&full_text);
    assert_eq!(tokenized.len(), target_len);

    (full_text, tokenized.len())
}

fn build_instruct_corpus(tok: &Tokenizer) -> (String, usize) {
    let mut lines = Vec::new();

    // 1. Addition QA Dialogs (14 dialogs = 112 tokens)
    lines.push("<user> 0 + 1 = <bot> 1 <eos>".to_string());
    lines.push("<user> 1 + 1 = <bot> 2 <eos>".to_string());
    lines.push("<user> 1 + 2 = <bot> 3 <eos>".to_string());
    lines.push("<user> 2 + 1 = <bot> 3 <eos>".to_string());
    lines.push("<user> 2 + 2 = <bot> 4 <eos>".to_string());
    lines.push("<user> 2 + 3 = <bot> 5 <eos>".to_string());
    lines.push("<user> 2 + 3 = <bot> 5 <eos>".to_string());
    lines.push("<user> 3 + 2 = <bot> 5 <eos>".to_string());
    lines.push("<user> 3 + 1 = <bot> 4 <eos>".to_string());
    lines.push("<user> 4 + 1 = <bot> 5 <eos>".to_string());
    lines.push("<user> 1 + 4 = <bot> 5 <eos>".to_string());
    lines.push("<user> 0 + 5 = <bot> 5 <eos>".to_string());
    lines.push("<user> 2 + 0 = <bot> 2 <eos>".to_string());
    lines.push("<user> 3 + 0 = <bot> 3 <eos>".to_string());
    lines.push("<user> 1 + 3 = <bot> 4 <eos>".to_string());

    // 2. Subtraction QA Dialogs (Balanced answers across 0..5)
    lines.push("<user> 5 - 0 = <bot> 5 <eos>".to_string());
    lines.push("<user> 5 - 1 = <bot> 4 <eos>".to_string());
    lines.push("<user> 5 - 2 = <bot> 3 <eos>".to_string());
    lines.push("<user> 5 - 3 = <bot> 2 <eos>".to_string());
    lines.push("<user> 5 - 4 = <bot> 1 <eos>".to_string());
    lines.push("<user> 5 - 5 = <bot> 0 <eos>".to_string());
    lines.push("<user> 4 - 0 = <bot> 4 <eos>".to_string());
    lines.push("<user> 4 - 1 = <bot> 3 <eos>".to_string());
    lines.push("<user> 4 - 1 = <bot> 3 <eos>".to_string());
    lines.push("<user> 4 - 2 = <bot> 2 <eos>".to_string());
    lines.push("<user> 4 - 3 = <bot> 1 <eos>".to_string());
    lines.push("<user> 3 - 0 = <bot> 3 <eos>".to_string());
    lines.push("<user> 3 - 1 = <bot> 2 <eos>".to_string());
    lines.push("<user> 3 - 2 = <bot> 1 <eos>".to_string());
    lines.push("<user> 2 - 1 = <bot> 1 <eos>".to_string());
    lines.push("<user> 1 - 1 = <bot> 0 <eos>".to_string());

    // 3. Fact QA Dialogs (Consistently teaching core facts)
    for _ in 0..6 {
        lines.push("<user> кто кот <bot> кот это животное <eos>".to_string());
        lines.push("<user> кто пес <bot> пес это друг <eos>".to_string());
    }

    // 4. Logic QA Dialogs (Balanced 'нет' and 'да' with clear predicate contrast)
    for _ in 0..3 {
        lines.push("<user> кот это пес <bot> нет <eos>".to_string());
        lines.push("<user> пес это кот <bot> нет <eos>".to_string());
    }
    lines.push("<user> кот это 0 <bot> нет <eos>".to_string());
    lines.push("<user> пес это 1 <bot> нет <eos>".to_string());
    lines.push("<user> 1 это 2 <bot> нет <eos>".to_string());
    lines.push("<user> 2 это 3 <bot> нет <eos>".to_string());
    lines.push("<user> 3 это 4 <bot> нет <eos>".to_string());
    lines.push("<user> 4 это 5 <bot> нет <eos>".to_string());

    for _ in 0..2 {
        lines.push("<user> кот это животное <bot> да <eos>".to_string());
        lines.push("<user> пес это животное <bot> да <eos>".to_string());
        lines.push("<user> кот это друг <bot> да <eos>".to_string());
        lines.push("<user> пес это друг <bot> да <eos>".to_string());
    }
    lines.push("<user> 0 это 0 <bot> да <eos>".to_string());
    lines.push("<user> 1 это 1 <bot> да <eos>".to_string());
    lines.push("<user> 2 это 2 <bot> да <eos>".to_string());
    lines.push("<user> 3 это 3 <bot> да <eos>".to_string());

    // 5. Replay Mix (Base facts to prevent catastrophic forgetting)
    lines.push("1 + 1 = 2 <eos>".to_string());
    lines.push("2 + 3 = 5 <eos>".to_string());
    lines.push("3 + 2 = 5 <eos>".to_string());
    lines.push("4 + 1 = 5 <eos>".to_string());
    lines.push("5 - 1 = 4 <eos>".to_string());
    lines.push("5 - 2 = 3 <eos>".to_string());
    lines.push("4 - 1 = 3 <eos>".to_string());
    lines.push("3 - 2 = 1 <eos>".to_string());
    lines.push("2 - 1 = 1 <eos>".to_string());
    lines.push("кот это животное <eos>".to_string());
    lines.push("пес это животное <eos>".to_string());
    lines.push("пес это друг <eos>".to_string());
    lines.push("кот это друг <eos>".to_string());
    lines.push("кто кот = кот это животное <eos>".to_string());
    lines.push("кто пес = пес это друг <eos>".to_string());
    lines.push("кот это пес = нет <eos>".to_string());
    lines.push("пес это кот = нет <eos>".to_string());
    lines.push("кот это животное = да <eos>".to_string());
    lines.push("пес это животное = да <eos>".to_string());
    lines.push("пес это друг = да <eos>".to_string());
    lines.push("кот это друг = да <eos>".to_string());
    lines.push("2 + 2 = 4 <eos>".to_string());
    lines.push("4 - 2 = 2 <eos>".to_string());
    lines.push("5 - 3 = 2 <eos>".to_string());
    lines.push("3 + 1 = 4 <eos>".to_string());
    lines.push("1 + 2 = 3 <eos>".to_string());
    lines.push("3 - 1 = 2 <eos>".to_string());
    lines.push("5 - 5 = 0 <eos>".to_string());
    lines.push("0 + 0 = 0 <eos>".to_string());
    lines.push("0 + 3 = 3 <eos>".to_string());

    let full_text = lines.join("\n") + "\n";
    let tokenized = tok.encode(&full_text);
    (full_text, tokenized.len())
}

/// Builds the unified non-duplicate corpus containing:
/// 1. 21 addition equations (0..5)
/// 2. 21 subtraction equations (0..5)
/// 3. 12 Russian facts
/// 4. 24 Russian logic relations
/// 5. Dialogue pairs (<user> ... <bot> ... <eos>)
///
/// Strictly 0 duplicates, every line ending with <eos>.
pub fn build_unified_corpus(tok: &Tokenizer) -> (String, usize, usize) {
    let mut lines = Vec::new();
    let mut seen = HashSet::new();

    // 1. All unique addition equations 0..5 (21 equations)
    let mut add_count = 0;
    for a in 0..=5 {
        for b in 0..=5 {
            if a + b <= 5 {
                let line = format!("{} + {} = {} <eos>", a, b, a + b);
                assert!(seen.insert(line.clone()), "Duplicate line: {}", line);
                lines.push(line);
                add_count += 1;
            }
        }
    }
    assert_eq!(add_count, 21);

    // 2. All unique subtraction equations 0..5 (21 equations)
    let mut sub_count = 0;
    for a in 0..=5 {
        for b in 0..=5 {
            if a >= b {
                let line = format!("{} - {} = {} <eos>", a, b, a - b);
                assert!(seen.insert(line.clone()), "Duplicate line: {}", line);
                lines.push(line);
                sub_count += 1;
            }
        }
    }
    assert_eq!(sub_count, 21);

    // 3. Russian Facts (12 unique statements)
    let facts = [
        "кот это животное <eos>",
        "пес это животное <eos>",
        "пес это друг <eos>",
        "кот это друг <eos>",
        "кто кот = кот это животное <eos>",
        "кто пес = пес это животное <eos>",
        "кто кот = кот это друг <eos>",
        "кто пес = пес это друг <eos>",
        "кто кот = животное <eos>",
        "кто пес = животное <eos>",
        "кто кот = друг <eos>",
        "кто пес = друг <eos>",
    ];
    for f in facts {
        let line = f.to_string();
        assert!(seen.insert(line.clone()), "Duplicate line: {}", line);
        lines.push(line);
    }

    // 4. Logic Relations (24 unique statements)
    let logic = [
        "кот это пес = нет <eos>",
        "пес это кот = нет <eos>",
        "кот это животное = да <eos>",
        "пес это животное = да <eos>",
        "пес это друг = да <eos>",
        "кот это друг = да <eos>",
        "животное это кот = да <eos>",
        "животное это пес = да <eos>",
        "друг это пес = да <eos>",
        "друг это кот = да <eos>",
        "кот это 0 = нет <eos>",
        "пес это 1 = нет <eos>",
        "кот это 2 = нет <eos>",
        "пес это 3 = нет <eos>",
        "1 это 2 = нет <eos>",
        "2 это 3 = нет <eos>",
        "3 это 4 = нет <eos>",
        "4 это 5 = нет <eos>",
        "0 это 0 = да <eos>",
        "1 это 1 = да <eos>",
        "2 это 2 = да <eos>",
        "3 это 3 = да <eos>",
        "4 это 4 = да <eos>",
        "5 это 5 = да <eos>",
    ];
    for l in logic {
        let line = l.to_string();
        assert!(seen.insert(line.clone()), "Duplicate line: {}", line);
        lines.push(line);
    }

    // 5. Dialogue Examples (<user> ... <bot> ... <eos>)
    // 5.1 Dialogue addition (21 equations)
    for a in 0..=5 {
        for b in 0..=5 {
            if a + b <= 5 {
                let line = format!("<user> {} + {} = <bot> {} <eos>", a, b, a + b);
                assert!(seen.insert(line.clone()), "Duplicate line: {}", line);
                lines.push(line);
            }
        }
    }

    // 5.2 Dialogue subtraction (21 equations)
    for a in 0..=5 {
        for b in 0..=5 {
            if a >= b {
                let line = format!("<user> {} - {} = <bot> {} <eos>", a, b, a - b);
                assert!(seen.insert(line.clone()), "Duplicate line: {}", line);
                lines.push(line);
            }
        }
    }

    // 5.3 Dialogue facts (2 statements)
    let dialog_facts = [
        "<user> кто кот <bot> кот это животное <eos>",
        "<user> кто пес <bot> пес это друг <eos>",
    ];
    for df in dialog_facts {
        let line = df.to_string();
        assert!(seen.insert(line.clone()), "Duplicate line: {}", line);
        lines.push(line);
    }

    // 5.4 Dialogue logic (24 statements)
    let dialog_logic = [
        "<user> кот это пес <bot> нет <eos>",
        "<user> пес это кот <bot> нет <eos>",
        "<user> кот это животное <bot> да <eos>",
        "<user> пес это животное <bot> да <eos>",
        "<user> пес это друг <bot> да <eos>",
        "<user> кот это друг <bot> да <eos>",
        "<user> животное это кот <bot> да <eos>",
        "<user> животное это пес <bot> да <eos>",
        "<user> друг это пес <bot> да <eos>",
        "<user> друг это кот <bot> да <eos>",
        "<user> кот это 0 <bot> нет <eos>",
        "<user> пес это 1 <bot> нет <eos>",
        "<user> кот это 2 <bot> нет <eos>",
        "<user> пес это 3 <bot> нет <eos>",
        "<user> 1 это 2 <bot> нет <eos>",
        "<user> 2 это 3 <bot> нет <eos>",
        "<user> 3 это 4 <bot> нет <eos>",
        "<user> 4 это 5 <bot> нет <eos>",
        "<user> 0 это 0 <bot> да <eos>",
        "<user> 1 это 1 <bot> да <eos>",
        "<user> 2 это 2 <bot> да <eos>",
        "<user> 3 это 3 <bot> да <eos>",
        "<user> 4 это 4 <bot> да <eos>",
        "<user> 5 это 5 <bot> да <eos>",
    ];
    for dl in dialog_logic {
        let line = dl.to_string();
        assert!(seen.insert(line.clone()), "Duplicate line: {}", line);
        lines.push(line);
    }

    let full_text = lines.join("\n") + "\n";
    let tokenized = tok.encode(&full_text);
    (full_text, lines.len(), tokenized.len())
}

fn main() {
    let tok = Tokenizer::new();
    fs::create_dir_all("data").expect("Failed to create data dir");

    // 1. Unified Corpus: strictly 0 duplicates, every line ends with <eos>
    let (unified_text, num_lines, unified_len) = build_unified_corpus(&tok);
    fs::write("data/unified_corpus.txt", &unified_text)
        .expect("Failed to write data/unified_corpus.txt");
    println!(
        "Generated data/unified_corpus.txt: {} unique lines, {} tokens (STRICTLY 0 DUPLICATES)",
        num_lines, unified_len
    );

    // 2. Pretrain Corpus: Exactly 10,240 tokens (Chinchilla 20:1 ratio for 512 params)
    let (pretrain_text, pretrain_len) = build_pretrain_corpus(&tok, 10240);
    fs::write("data/pretrain.txt", &pretrain_text).expect("Failed to write data/pretrain.txt");
    println!("Generated data/pretrain.txt: {} tokens (TARGET: 10240)", pretrain_len);
    assert_eq!(pretrain_len, 10240);

    // 3. Instruct Corpus: ~600 tokens (e.g. 600..650)
    let (instruct_text, instruct_len) = build_instruct_corpus(&tok);
    fs::write("data/instruct.txt", &instruct_text).expect("Failed to write data/instruct.txt");
    println!("Generated data/instruct.txt: {} tokens (TARGET: ~600)", instruct_len);
    assert!((550..=700).contains(&instruct_len));

    println!("All corpora generation completed successfully!");
}
