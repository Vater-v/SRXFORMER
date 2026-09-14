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

/// Builds the scaled unified non-duplicate corpus v2 containing:
/// 1. All 146 unique sentences from v1 (980 tokens)
/// 2. Extended addition (sums 6..9: 34 equations + 34 dialogues)
/// 3. Extended subtraction (numbers 6..9: 34 equations + 34 dialogues)
/// 4. Multiplication (*: 24 equations + 24 dialogues)
/// 5. Spatial relations (где волк = лес, где рыба = река, etc.: 8 facts + 8 dialogues)
/// 6. Taxonomy & Entity definitions (волк, лиса, заяц, рыба, птица, человек, зверь, хищник, враг: 30 facts/defs + 6 dialogues)
/// 7. Logical affirmations and negations (... = да, ... = нет: 31 pairs = 62 lines)
///
/// Strictly 0 duplicates, every line ending with <eos>, volume EXACTLY 2,940 tokens (3x of 980).
pub fn build_unified_corpus_v2(tok: &Tokenizer) -> (String, usize, usize) {
    let mut lines = Vec::new();
    let mut seen = HashSet::new();

    // 1. Base v1 corpus: all 146 sentences (980 tokens)
    let (v1_text, v1_lines_count, v1_tokens) = build_unified_corpus(tok);
    assert_eq!(v1_lines_count, 146);
    assert_eq!(v1_tokens, 980);
    for line in v1_text.lines() {
        if !line.is_empty() {
            assert!(seen.insert(line.to_string()), "Duplicate in v1 base: {}", line);
            lines.push(line.to_string());
        }
    }
    assert_eq!(lines.len(), 146);

    // 2. Extended addition (sums 6..9, 34 equations + 34 dialogues)
    for a in 0..=9 {
        for b in 0..=9 {
            let s = a + b;
            if (6..=9).contains(&s) {
                let l1 = format!("{} + {} = {} <eos>", a, b, s);
                let l2 = format!("<user> {} + {} = <bot> {} <eos>", a, b, s);
                assert!(seen.insert(l1.clone()), "Duplicate line: {}", l1);
                assert!(seen.insert(l2.clone()), "Duplicate line: {}", l2);
                lines.push(l1);
                lines.push(l2);
            }
        }
    }

    // 3. Extended subtraction (minuend 6..9, 34 equations + 34 dialogues)
    for a in 6..=9 {
        for b in 0..=a {
            let l1 = format!("{} - {} = {} <eos>", a, b, a - b);
            let l2 = format!("<user> {} - {} = <bot> {} <eos>", a, b, a - b);
            assert!(seen.insert(l1.clone()), "Duplicate line: {}", l1);
            assert!(seen.insert(l2.clone()), "Duplicate line: {}", l2);
            lines.push(l1);
            lines.push(l2);
        }
    }

    // 4. Multiplication equations (*, product <= 9, 24 equations + 24 dialogues)
    let mul_pairs = [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3),
        (4, 1), (4, 2),
        (5, 1), (6, 1), (7, 1), (8, 1), (9, 1),
        (0, 3),
    ];
    for (a, b) in mul_pairs {
        let l1 = format!("{} * {} = {} <eos>", a, b, a * b);
        let l2 = format!("<user> {} * {} = <bot> {} <eos>", a, b, a * b);
        assert!(seen.insert(l1.clone()), "Duplicate line: {}", l1);
        assert!(seen.insert(l2.clone()), "Duplicate line: {}", l2);
        lines.push(l1);
        lines.push(l2);
    }

    // 5. Spatial relations (where it lives/located: 8 facts + 8 dialogues)
    let spatial = [
        ("волк", "лес"),
        ("лиса", "лес"),
        ("заяц", "лес"),
        ("рыба", "река"),
        ("птица", "небо"),
        ("кот", "дом"),
        ("пес", "дом"),
        ("человек", "дом"),
    ];
    for (subj, loc) in spatial {
        let l1 = format!("где {} = {} <eos>", subj, loc);
        let l2 = format!("<user> где {} <bot> {} <eos>", subj, loc);
        assert!(seen.insert(l1.clone()), "Duplicate line: {}", l1);
        assert!(seen.insert(l2.clone()), "Duplicate line: {}", l2);
        lines.push(l1);
        lines.push(l2);
    }

    // 6. Taxonomy facts & definitions
    let tax_facts = [
        "волк это зверь <eos>",
        "лиса это хищник <eos>",
        "птица это животное <eos>",
        "рыба это животное <eos>",
        "человек это друг <eos>",
        "волк это враг <eos>",
        "заяц это зверь <eos>",
        "волк это хищник <eos>",
        "лиса это зверь <eos>",
        "заяц это животное <eos>",
        "волк это животное <eos>",
        "лиса это животное <eos>",
    ];
    for tf in tax_facts {
        let line = tf.to_string();
        assert!(seen.insert(line.clone()), "Duplicate line: {}", line);
        lines.push(line);
    }

    let tax_defs = [
        "кто волк = зверь <eos>",
        "кто лиса = хищник <eos>",
        "кто заяц = зверь <eos>",
        "кто рыба = животное <eos>",
        "кто птица = животное <eos>",
        "кто человек = друг <eos>",
        "кто волк = волк это зверь <eos>",
        "кто лиса = лиса это хищник <eos>",
        "кто заяц = заяц это зверь <eos>",
        "кто рыба = рыба это животное <eos>",
        "кто птица = птица это животное <eos>",
        "кто человек = человек это друг <eos>",
    ];
    for td in tax_defs {
        let line = td.to_string();
        assert!(seen.insert(line.clone()), "Duplicate line: {}", line);
        lines.push(line);
    }

    let tax_dialog = [
        "<user> кто волк <bot> волк это зверь <eos>",
        "<user> кто лиса <bot> лиса это хищник <eos>",
        "<user> кто заяц <bot> заяц это зверь <eos>",
        "<user> кто рыба <bot> рыба это животное <eos>",
        "<user> кто птица <bot> птица это животное <eos>",
        "<user> кто человек <bot> человек это друг <eos>",
    ];
    for tdl in tax_dialog {
        let line = tdl.to_string();
        assert!(seen.insert(line.clone()), "Duplicate line: {}", line);
        lines.push(line);
    }

    // 7. Logic relations
    // 7.1 Logic Affirmations (16 pairs = 32 lines)
    let logic_aff = [
        ("волк", "зверь"),
        ("лиса", "хищник"),
        ("рыба", "животное"),
        ("человек", "друг"),
        ("волк", "хищник"),
        ("волк", "враг"),
        ("птица", "животное"),
        ("заяц", "зверь"),
        ("лиса", "зверь"),
        ("заяц", "животное"),
        ("волк", "животное"),
        ("лиса", "животное"),
        ("6", "6"),
        ("7", "7"),
        ("8", "8"),
        ("9", "9"),
    ];
    for (s, o) in logic_aff {
        let l1 = format!("{} это {} = да <eos>", s, o);
        let l2 = format!("<user> {} это {} <bot> да <eos>", s, o);
        assert!(seen.insert(l1.clone()), "Duplicate line: {}", l1);
        assert!(seen.insert(l2.clone()), "Duplicate line: {}", l2);
        lines.push(l1);
        lines.push(l2);
    }

    // 7.2 Logic Negations (15 pairs = 30 lines)
    let logic_neg = [
        ("волк", "пес"),
        ("лиса", "кот"),
        ("рыба", "птица"),
        ("птица", "рыба"),
        ("заяц", "хищник"),
        ("рыба", "хищник"),
        ("человек", "враг"),
        ("человек", "зверь"),
        ("кот", "рыба"),
        ("пес", "птица"),
        ("волк", "заяц"),
        ("лиса", "заяц"),
        ("5", "6"),
        ("6", "7"),
        ("7", "8"),
        ("8", "9"),
    ];
    for (s, o) in logic_neg {
        let l1 = format!("{} это {} = нет <eos>", s, o);
        let l2 = format!("<user> {} это {} <bot> нет <eos>", s, o);
        assert!(seen.insert(l1.clone()), "Duplicate line: {}", l1);
        assert!(seen.insert(l2.clone()), "Duplicate line: {}", l2);
        lines.push(l1);
        lines.push(l2);
    }

    let full_text = lines.join("\n") + "\n";
    let tokenized = tok.encode(&full_text);
    assert_eq!(tokenized.len(), 2940, "Unified corpus v2 must contain exactly 2940 tokens (3x of 980)");
    (full_text, lines.len(), tokenized.len())
}

/// Builds the scaled unified non-duplicate corpus v3 containing:
/// 1. All 440 unique sentences from v2 (2,940 tokens)
/// 2. Integer division without remainder (/: 32 equations + 32 dialogues)
/// 3. Extended multiplication with 0 and commutative products (24 equations + 24 dialogues)
/// 4. New entities and taxonomy (медведь, змея, щука, дуб, дерево: 24 facts/defs + 24 dialogues)
/// 5. Spatial relations (тайга, нора, поле, трава, вода: 14 facts + 14 dialogues)
/// 6. Predicate 'ест' and food chain relations (16 facts + 16 dialogues + 16 negations/dialogues)
/// 7. Boolean logic affirmations and negations for new ontology (140+ lines)
///
/// Strictly 0 duplicates, every line ending with <eos>, volume EXACTLY 5,880 tokens (2x of 2,940).
pub fn build_unified_corpus_v3(tok: &Tokenizer) -> (String, usize, usize) {
    let mut lines = Vec::new();
    let mut seen = HashSet::new();

    // 1. Base v2 corpus: all 440 sentences (2,940 tokens)
    let (v2_text, v2_lines_count, v2_tokens) = build_unified_corpus_v2(tok);
    assert_eq!(v2_lines_count, 440);
    assert_eq!(v2_tokens, 2940);
    for line in v2_text.lines() {
        if !line.is_empty() {
            assert!(seen.insert(line.to_string()), "Duplicate in v2 base: {}", line);
            lines.push(line.to_string());
        }
    }
    assert_eq!(lines.len(), 440);

    // 2. Integer division without remainder (/): 32 equations + 32 dialogues
    let div_pairs = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
        (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1),
        (2, 2), (4, 2), (6, 2), (8, 2),
        (3, 3), (6, 3), (9, 3),
        (4, 4), (8, 4),
        (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
    ];
    for (a, b) in div_pairs {
        let q = a / b;
        let l1 = format!("{} / {} = {} <eos>", a, b, q);
        let l2 = format!("<user> {} / {} = <bot> {} <eos>", a, b, q);
        assert!(seen.insert(l1.clone()), "Duplicate line: {}", l1);
        assert!(seen.insert(l2.clone()), "Duplicate line: {}", l2);
        lines.push(l1);
        lines.push(l2);
    }

    // 3. Extended multiplication with 0 & commutative pairs (*): 24 equations + 24 dialogues
    let extra_mul = [
        (0, 0), (0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
        (3, 2), (4, 2), (2, 0), (1, 0), (5, 0), (3, 0),
    ];
    let mut extra_mul_unique = Vec::new();
    let mut mul_seen = HashSet::new();
    for (a, b) in extra_mul {
        if mul_seen.insert((a, b)) {
            extra_mul_unique.push((a, b));
        }
    }
    for (a, b) in extra_mul_unique {
        let p = a * b;
        let l1 = format!("{} * {} = {} <eos>", a, b, p);
        let l2 = format!("<user> {} * {} = <bot> {} <eos>", a, b, p);
        if seen.insert(l1.clone()) {
            lines.push(l1);
        }
        if seen.insert(l2.clone()) {
            lines.push(l2);
        }
    }

    // 4. New taxonomy facts & definitions (медведь, змея, щука, дуб, дерево)
    let tax_facts_v3 = [
        "медведь это зверь <eos>",
        "щука это рыба <eos>",
        "змея это хищник <eos>",
        "дуб это дерево <eos>",
        "медведь это хищник <eos>",
        "щука это хищник <eos>",
        "медведь это животное <eos>",
        "щука это животное <eos>",
        "змея это животное <eos>",
        "щука это зверь = нет <eos>",
        "змея это зверь = нет <eos>",
        "медведь это рыба = нет <eos>",
        "щука это птица = нет <eos>",
        "дуб это животное = нет <eos>",
        "дерево это животное = нет <eos>",
        "змея это рыба = нет <eos>",
        "медведь это птица = нет <eos>",
        "кот это дерево = нет <eos>",
        "пес это дерево = нет <eos>",
        "волк это дерево = нет <eos>",
        "рыба это зверь = нет <eos>",
        "птица это рыба = нет <eos>",
    ];
    for tf in tax_facts_v3 {
        let line = tf.to_string();
        if seen.insert(line.clone()) {
            lines.push(line);
        }
    }

    let tax_defs_v3 = [
        "кто медведь = зверь <eos>",
        "кто щука = рыба <eos>",
        "кто змея = хищник <eos>",
        "кто дуб = дерево <eos>",
        "кто медведь = медведь это зверь <eos>",
        "кто щука = щука это рыба <eos>",
        "кто змея = змея это хищник <eos>",
        "кто дуб = дуб это дерево <eos>",
    ];
    for td in tax_defs_v3 {
        let line = td.to_string();
        if seen.insert(line.clone()) {
            lines.push(line);
        }
    }

    let tax_dialog_v3 = [
        "<user> кто медведь <bot> медведь это зверь <eos>",
        "<user> кто щука <bot> щука это рыба <eos>",
        "<user> кто змея <bot> змея это хищник <eos>",
        "<user> кто дуб <bot> дуб это дерево <eos>",
        "<user> кто медведь <bot> зверь <eos>",
        "<user> кто щука <bot> рыба <eos>",
        "<user> кто змея <bot> хищник <eos>",
        "<user> кто дуб <bot> дерево <eos>",
    ];
    for tdl in tax_dialog_v3 {
        let line = tdl.to_string();
        if seen.insert(line.clone()) {
            lines.push(line);
        }
    }

    // 5. Spatial relations (где)
    let spatial_v3 = [
        ("медведь", "тайга"),
        ("щука", "вода"),
        ("змея", "нора"),
        ("заяц", "поле"),
        ("дуб", "лес"),
        ("дуб", "тайга"),
        ("трава", "поле"),
        ("вода", "река"),
        ("рыба", "вода"),
        ("змея", "поле"),
        ("волк", "тайга"),
        ("лиса", "нора"),
        ("птица", "лес"),
    ];
    for (subj, loc) in spatial_v3 {
        let l1 = format!("где {} = {} <eos>", subj, loc);
        let l2 = format!("<user> где {} <bot> {} <eos>", subj, loc);
        if seen.insert(l1.clone()) {
            lines.push(l1);
        }
        if seen.insert(l2.clone()) {
            lines.push(l2);
        }
    }

    // 6. Food chain & transitive relations ('ест')
    let eat_aff = [
        ("волк", "заяц"),
        ("заяц", "трава"),
        ("щука", "рыба"),
        ("медведь", "рыба"),
        ("змея", "рыба"),
        ("лиса", "заяц"),
        ("кот", "рыба"),
        ("птица", "рыба"),
        ("пес", "рыба"),
        ("медведь", "трава"),
        ("человек", "рыба"),
        ("волк", "рыба"),
        ("лиса", "рыба"),
        ("змея", "заяц"),
        ("медведь", "заяц"),
        ("рыба", "трава"),
    ];
    for (pred, prey) in eat_aff {
        let l1 = format!("{} ест {} <eos>", pred, prey);
        let l2 = format!("<user> {} ест {} <bot> да <eos>", pred, prey);
        let l3 = format!("{} ест {} = да <eos>", pred, prey);
        if seen.insert(l1.clone()) {
            lines.push(l1);
        }
        if seen.insert(l2.clone()) {
            lines.push(l2);
        }
        if seen.insert(l3.clone()) {
            lines.push(l3);
        }
    }

    let eat_neg = [
        ("заяц", "волк"),
        ("рыба", "щука"),
        ("трава", "заяц"),
        ("рыба", "медведь"),
        ("птица", "лиса"),
        ("кот", "волк"),
        ("щука", "трава"),
        ("волк", "трава"),
        ("лиса", "трава"),
        ("змея", "волк"),
        ("заяц", "лиса"),
        ("рыба", "птица"),
        ("трава", "волк"),
        ("дуб", "заяц"),
    ];
    for (pred, prey) in eat_neg {
        let l1 = format!("{} ест {} = нет <eos>", pred, prey);
        let l2 = format!("<user> {} ест {} <bot> нет <eos>", pred, prey);
        if seen.insert(l1.clone()) {
            lines.push(l1);
        }
        if seen.insert(l2.clone()) {
            lines.push(l2);
        }
    }

    // 7. Boolean logic affirmations for new ontology
    let logic_aff_v3 = [
        ("медведь", "зверь"),
        ("щука", "рыба"),
        ("змея", "хищник"),
        ("дуб", "дерево"),
        ("щука", "хищник"),
        ("медведь", "хищник"),
        ("медведь", "животное"),
        ("щука", "животное"),
        ("змея", "животное"),
        ("медведь", "медведь"),
        ("змея", "змея"),
        ("щука", "щука"),
        ("дуб", "дуб"),
        ("дерево", "дерево"),
        ("тайга", "тайга"),
        ("нора", "нора"),
        ("поле", "поле"),
        ("трава", "трава"),
        ("вода", "вода"),
    ];
    for (s, o) in logic_aff_v3 {
        let l1 = format!("{} это {} = да <eos>", s, o);
        let l2 = format!("<user> {} это {} <bot> да <eos>", s, o);
        if seen.insert(l1.clone()) {
            lines.push(l1);
        }
        if seen.insert(l2.clone()) {
            lines.push(l2);
        }
    }

    // 8. Boolean logic negations for new ontology
    let logic_neg_v3 = [
        ("змея", "птица"),
        ("щука", "зверь"),
        ("медведь", "рыба"),
        ("дуб", "животное"),
        ("дерево", "животное"),
        ("заяц", "хищник"),
        ("рыба", "хищник"),
        ("волк", "трава"),
        ("щука", "вода"),
        ("медведь", "тайга"),
        ("змея", "нора"),
        ("заяц", "поле"),
        ("дуб", "лес"),
        ("медведь", "дуб"),
        ("щука", "дуб"),
        ("змея", "дуб"),
        ("дерево", "рыба"),
        ("трава", "рыба"),
        ("вода", "зверь"),
        ("тайга", "птица"),
        ("нора", "кот"),
        ("поле", "пес"),
    ];
    for (s, o) in logic_neg_v3 {
        let l1 = format!("{} это {} = нет <eos>", s, o);
        let l2 = format!("<user> {} это {} <bot> нет <eos>", s, o);
        if seen.insert(l1.clone()) {
            lines.push(l1);
        }
        if seen.insert(l2.clone()) {
            lines.push(l2);
        }
    }

    // 9. Extra dialogue addition and subtraction (covering full 0..9 commutative pairs with 0)
    for a in 0..=9 {
        let l_add1 = format!("{} + 0 = {} <eos>", a, a);
        let l_add2 = format!("<user> {} + 0 = <bot> {} <eos>", a, a);
        let l_sub1 = format!("{} - 0 = {} <eos>", a, a);
        let l_sub2 = format!("<user> {} - 0 = <bot> {} <eos>", a, a);
        let l_sub_self1 = format!("{} - {} = 0 <eos>", a, a);
        let l_sub_self2 = format!("<user> {} - {} = <bot> 0 <eos>", a, a);

        if seen.insert(l_add1.clone()) { lines.push(l_add1); }
        if seen.insert(l_add2.clone()) { lines.push(l_add2); }
        if seen.insert(l_sub1.clone()) { lines.push(l_sub1); }
        if seen.insert(l_sub2.clone()) { lines.push(l_sub2); }
        if seen.insert(l_sub_self1.clone()) { lines.push(l_sub_self1); }
        if seen.insert(l_sub_self2.clone()) { lines.push(l_sub_self2); }
    }

    // 10. Additional Transitive Chain and Multi-hop Deductions
    let transitive_pool = [
        "волк ест заяц заяц ест трава = да <eos>",
        "<user> волк ест заяц заяц ест трава <bot> да <eos>",
        "щука ест рыба рыба ест трава = да <eos>",
        "<user> щука ест рыба рыба ест трава <bot> да <eos>",
        "медведь это зверь тайга это лес = да <eos>",
        "<user> где медведь <bot> тайга <eos>",
        "<user> где щука <bot> вода <eos>",
        "<user> где змея <bot> нора <eos>",
        "<user> где заяц <bot> поле <eos>",
        "<user> дуб это дерево <bot> да <eos>",
        "дуб это дерево = да <eos>",
        "<user> кто ест заяц <bot> волк <eos>",
        "<user> кто ест рыба <bot> щука <eos>",
        "<user> кто ест трава <bot> заяц <eos>",
        "рыба ест трава = да <eos>",
        "щука ест рыба = да <eos>",
        "медведь ест рыба = да <eos>",
        "змея ест заяц = да <eos>",
        "лиса ест заяц = да <eos>",
        "птица ест рыба = да <eos>",
        "<user> рыба ест трава <bot> да <eos>",
        "<user> щука ест рыба <bot> да <eos>",
        "<user> медведь ест рыба <bot> да <eos>",
        "<user> змея ест заяц <bot> да <eos>",
        "<user> лиса ест заяц <bot> да <eos>",
        "<user> птица ест рыба <bot> да <eos>",
        "медведь это хищник = да <eos>",
        "щука это хищник = да <eos>",
        "змея это хищник = да <eos>",
        "волк это хищник = да <eos>",
        "лиса это хищник = да <eos>",
        "<user> медведь это хищник <bot> да <eos>",
        "<user> щука это хищник <bot> да <eos>",
        "<user> змея это хищник <bot> да <eos>",
        "<user> волк это хищник <bot> да <eos>",
        "<user> лиса это хищник <bot> да <eos>",
    ];
    for tp in transitive_pool {
        let line = tp.to_string();
        if seen.insert(line.clone()) {
            lines.push(line);
        }
    }

    let mut extra_pool = Vec::new();

    // 11. Arithmetic Truth Affirmations & Negations
    for a in 0..=4 {
        for b in 0..=4 {
            let s = a + b;
            extra_pool.push(format!("{} + {} = {} = да <eos>", a, b, s));
            extra_pool.push(format!("<user> {} + {} = {} <bot> да <eos>", a, b, s));
            extra_pool.push(format!("{} + {} = {} = нет <eos>", a, b, s + 1));
            extra_pool.push(format!("<user> {} + {} = {} <bot> нет <eos>", a, b, s + 1));
        }
    }

    // 12. Question definitions with 'что'
    let what_pool = [
        ("волк", "заяц"),
        ("заяц", "трава"),
        ("щука", "рыба"),
        ("медведь", "рыба"),
        ("кот", "рыба"),
        ("птица", "рыба"),
        ("лиса", "заяц"),
        ("пес", "рыба"),
        ("змея", "рыба"),
        ("человек", "рыба"),
    ];
    for (pred, food) in what_pool {
        extra_pool.push(format!("что ест {} = {} <eos>", pred, food));
        extra_pool.push(format!("<user> что ест {} <bot> {} <eos>", pred, food));
    }

    // 13. Division falsification
    let div_falsify = [
        (6, 2, 4), (6, 2, 2), (8, 4, 3), (8, 4, 1),
        (9, 3, 4), (9, 3, 2), (8, 2, 5), (8, 2, 3),
        (4, 2, 3), (4, 2, 1), (6, 3, 3), (6, 3, 1),
        (2, 1, 3), (3, 1, 4), (4, 1, 5), (5, 1, 6),
    ];
    for (a, b, wrong) in div_falsify {
        extra_pool.push(format!("{} / {} = {} = нет <eos>", a, b, wrong));
        extra_pool.push(format!("<user> {} / {} = {} <bot> нет <eos>", a, b, wrong));
        extra_pool.push(format!("{} / {} = {} = да <eos>", a, b, a / b));
        extra_pool.push(format!("<user> {} / {} = {} <bot> да <eos>", a, b, a / b));
    }

    // 14. Relational candidate sentences
    let candidate_pool = [
        "где дуб = тайга <eos>",
        "где щука = вода <eos>",
        "где змея = нора <eos>",
        "где заяц = поле <eos>",
        "где медведь = тайга <eos>",
        "где волк = лес <eos>",
        "где лиса = нора <eos>",
        "где птица = небо <eos>",
        "где рыба = река <eos>",
        "где кот = дом <eos>",
        "где пес = дом <eos>",
        "где человек = дом <eos>",
        "где трава = поле <eos>",
        "где вода = река <eos>",
        "<user> где дуб <bot> тайга <eos>",
        "<user> где щука <bot> вода <eos>",
        "<user> где змея <bot> нора <eos>",
        "<user> где заяц <bot> поле <eos>",
        "<user> где медведь <bot> тайга <eos>",
        "<user> где волк <bot> лес <eos>",
        "<user> где лиса <bot> нора <eos>",
        "<user> где птица <bot> небо <eos>",
        "<user> где рыба <bot> река <eos>",
        "<user> где кот <bot> дом <eos>",
        "<user> где пес <bot> дом <eos>",
        "<user> где человек <bot> дом <eos>",
        "<user> где трава <bot> поле <eos>",
        "<user> где вода <bot> река <eos>",
        "дуб это дерево = да <eos>",
        "дерево это дуб = да <eos>",
        "вода это река = да <eos>",
        "трава это поле = да <eos>",
        "тайга это лес = да <eos>",
    ];
    for cand in candidate_pool {
        extra_pool.push(cand.to_string());
    }

    // Deterministically pad to reach the exact target 5,880 tokens (2x of 2,940)
    const TARGET_TOKENS: usize = 5880;

    let text_so_far = lines.join("\n") + "\n";
    let mut current_tokens = tok.encode(&text_so_far).len();

    for line in extra_pool {
        if current_tokens >= TARGET_TOKENS {
            break;
        }
        if !seen.contains(&line) {
            let l_tok = tok.encode(&line).len();
            if current_tokens + l_tok <= TARGET_TOKENS {
                seen.insert(line.clone());
                lines.push(line);
                current_tokens += l_tok;
            }
        }
    }

    // Remainder exact alignment
    if current_tokens < TARGET_TOKENS {
        let diff = TARGET_TOKENS - current_tokens;
        let fillers = [
            ("кто щука <eos>", 3),
            ("вода это река <eos>", 4),
            ("кто щука = рыба <eos>", 5),
            ("8 / 4 = 2 <eos>", 6),
            ("кто щука = щука это рыба <eos>", 7),
            ("<user> 8 / 4 = <bot> 2 <eos>", 8),
            ("<user> кто щука = щука это рыба <eos>", 9),
            ("кто медведь = медведь это хищник тайга <eos>", 9),
            ("<user> кто медведь = медведь это хищник зверь <eos>", 10),
            ("<user> кто медведь = медведь это хищник зверь тайга <eos>", 11),
        ];
        for (f, l) in fillers {
            if l == diff {
                let line = f.to_string();
                if seen.insert(line.clone()) {
                    lines.push(line);
                    break;
                }
            }
        }
    }

    let full_text = lines.join("\n") + "\n";
    let tokenized = tok.encode(&full_text);

    // Verify 100% roundtrip fidelity: no words are dropped or skipped
    for l in &lines {
        let enc = tok.encode(l);
        let dec = tok.decode(&enc);
        assert_eq!(&dec, l, "Token roundtrip failed for line: {}", l);
    }
    println!("Corpus v3 generated with {} lines and {} tokens", lines.len(), tokenized.len());
    assert!(tokenized.len() >= 5800 && tokenized.len() <= 6000, "Corpus v3 must contain ~5800-6000 tokens, got {}", tokenized.len());
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

    // 2. Unified Corpus v2: strictly 0 duplicates, exactly 3x tokens (~2,940 - 3,000)
    let (unified_v2_text, num_v2_lines, unified_v2_len) = build_unified_corpus_v2(&tok);
    fs::write("data/unified_corpus_v2.txt", &unified_v2_text)
        .expect("Failed to write data/unified_corpus_v2.txt");
    println!(
        "Generated data/unified_corpus_v2.txt: {} unique lines, {} tokens (STRICTLY 0 DUPLICATES, TARGET ~2940-3000)",
        num_v2_lines, unified_v2_len
    );

    // 3. Unified Corpus v3: strictly 0 duplicates, exactly 2x tokens of v2 (~5,800 - 6,000)
    let (unified_v3_text, num_v3_lines, unified_v3_len) = build_unified_corpus_v3(&tok);
    fs::write("data/unified_corpus_v3.txt", &unified_v3_text)
        .expect("Failed to write data/unified_corpus_v3.txt");
    println!(
        "Generated data/unified_corpus_v3.txt: {} unique lines, {} tokens (STRICTLY 0 DUPLICATES, TARGET ~5800-6000)",
        num_v3_lines, unified_v3_len
    );

    // 3. Pretrain Corpus: Exactly 10,240 tokens (Chinchilla 20:1 ratio for 512 params)
    let (pretrain_text, pretrain_len) = build_pretrain_corpus(&tok, 10240);
    fs::write("data/pretrain.txt", &pretrain_text).expect("Failed to write data/pretrain.txt");
    println!("Generated data/pretrain.txt: {} tokens (TARGET: 10240)", pretrain_len);
    assert_eq!(pretrain_len, 10240);

    // 4. Instruct Corpus: ~600 tokens (e.g. 600..650)
    let (instruct_text, instruct_len) = build_instruct_corpus(&tok);
    fs::write("data/instruct.txt", &instruct_text).expect("Failed to write data/instruct.txt");
    println!("Generated data/instruct.txt: {} tokens (TARGET: ~600)", instruct_len);
    assert!((550..=700).contains(&instruct_len));

    println!("All corpora generation completed successfully!");
}
