//! Generator utility for Sprint 1: Chinchilla 20:1 Dataset & Controlled Vocabulary.
//!
//! Generates:
//! 1. `data/vocab_chinchilla.txt`: 65-token controlled vocabulary mapping with IDs.
//! 2. `data/pretrain_chinchilla.txt`: EXACTLY 17,920 tokens (20:1 Chinchilla ratio for 896-parameter model).
//! 3. `data/instruct_chinchilla.txt`: ~2,000 tokens (1,800..2,200 tokens) in `<user> ... <bot> ... <eos>` dialogue format.
//!
//! Enforces:
//! - Pure Rust `std` only (zero external crates).
//! - 100% roundtrip fidelity: `tokenizer.decode(&tokenizer.encode(line)) == line` on every line.
//! - Every line ends with `<eos>`.

use std::collections::HashSet;
use std::fs;
use srxformer::{Tokenizer, VOCAB_CHINCHILLA};

/// Generates the base pool of pretrain sentences covering:
/// 1. Foundations of science, mathematics, epistemology, and AI reasoning (from corpus_0001.json).
/// 2. Deductive logic, if-then implications, disjunctions, and syllogisms.
/// 3. Exhaustive arithmetic (addition, subtraction, multiplication, division 0..9).
/// 4. Arithmetic truth checks and falsifications.
/// 5. Taxonomy, habitats, and ecological food chains.
/// 6. Negations and predicate contrast.
fn generate_pretrain_pool() -> Vec<String> {
    let mut pool = Vec::new();
    let mut seen = HashSet::new();

    let add_line = |line: String, pool: &mut Vec<String>, seen: &mut HashSet<String>| {
        if seen.insert(line.clone()) {
            pool.push(line);
        }
    };

    // -------------------------------------------------------------------------
    // 1. SCIENCE, EPISTEMOLOGY, AI, AND MATHEMATICAL FOUNDATIONS
    // -------------------------------------------------------------------------
    let foundational_facts = [
        "логика это наука <eos>",
        "наука это знание <eos>",
        "модель это знание <eos>",
        "логика это знание <eos>",
        "наука это логика <eos>",
        "модель это наука <eos>",
        "человек это разум <eos>",
        "человек это знание <eos>",
        "человек это друг <eos>",
        "человек это не зверь <eos>",
        "человек это не хищник <eos>",
        "человек это не враг <eos>",
        "разум это знание <eos>",
        "разум это логика <eos>",
        "разум это наука <eos>",
        "знание это наука <eos>",
        "знание это разум <eos>",
        "знание это логика <eos>",
        "модель это разум <eos>",
        "0 это число <eos>",
        "1 это число <eos>",
        "2 это число <eos>",
        "3 это число <eos>",
        "4 это число <eos>",
        "5 это число <eos>",
        "6 это число <eos>",
        "7 это число <eos>",
        "8 это число <eos>",
        "9 это число <eos>",
        "число это 0 или 1 <eos>",
        "число это 2 или 3 <eos>",
        "число это 4 или 5 <eos>",
        "число это 6 или 7 <eos>",
        "число это 8 или 9 <eos>",
        "0 это не 1 <eos>",
        "1 это не 2 <eos>",
        "2 это не 3 <eos>",
        "3 это не 4 <eos>",
        "4 это не 5 <eos>",
        "5 это не 6 <eos>",
        "6 это не 7 <eos>",
        "7 это не 8 <eos>",
        "8 это не 9 <eos>",
    ];
    for f in foundational_facts {
        add_line(f.to_string(), &mut pool, &mut seen);
    }

    // -------------------------------------------------------------------------
    // 2. DEDUCTIVE REASONING & IF-THEN SYLLOGISMS
    // -------------------------------------------------------------------------
    let deductive_syllogisms = [
        "если логика это наука то наука это знание <eos>",
        "если наука это знание то знание это разум <eos>",
        "если модель это знание то модель это наука <eos>",
        "если человек это разум то человек не зверь <eos>",
        "если человек это друг то человек не враг <eos>",
        "если волк ест заяц то волк хищник <eos>",
        "если лиса ест заяц то лиса хищник <eos>",
        "если щука ест рыба то щука хищник <eos>",
        "если медведь ест рыба то медведь хищник <eos>",
        "если змея ест заяц то змея хищник <eos>",
        "если заяц ест трава то заяц не хищник <eos>",
        "если рыба ест трава то рыба не хищник <eos>",
        "если дуб это дерево то дуб не животное <eos>",
        "если дерево это дуб то дерево не рыба <eos>",
        "если медведь это зверь то медведь не дерево <eos>",
        "если щука это рыба то щука не зверь <eos>",
        "если 1 + 1 = 2 то да <eos>",
        "если 2 + 2 = 4 то да <eos>",
        "если 3 + 3 = 6 то да <eos>",
        "если 4 + 4 = 8 то да <eos>",
        "если 2 * 3 = 6 то да <eos>",
        "если 6 / 2 = 3 то да <eos>",
        "если 8 / 4 = 2 то да <eos>",
        "если 9 / 3 = 3 то да <eos>",
        "если 5 - 2 = 3 то да <eos>",
        "если 7 - 4 = 3 то да <eos>",
        "если 8 - 3 = 5 то да <eos>",
        "если 9 - 5 = 4 то да <eos>",
    ];
    for ds in deductive_syllogisms {
        add_line(ds.to_string(), &mut pool, &mut seen);
    }

    // -------------------------------------------------------------------------
    // 3. TAXONOMY & HABITAT RELATIONS
    // -------------------------------------------------------------------------
    let taxonomy_and_habitats = [
        "кот это животное <eos>",
        "пес это животное <eos>",
        "пес это друг <eos>",
        "кот это друг <eos>",
        "волк это зверь <eos>",
        "лиса это зверь <eos>",
        "заяц это зверь <eos>",
        "рыба это животное <eos>",
        "птица это животное <eos>",
        "медведь это зверь <eos>",
        "змея это животное <eos>",
        "щука это рыба <eos>",
        "дуб это дерево <eos>",
        "тайга это лес <eos>",
        "вода это река <eos>",
        "трава это поле <eos>",
        "нора это дом <eos>",
        "лиса это хищник <eos>",
        "волк это хищник <eos>",
        "щука это хищник <eos>",
        "медведь это хищник <eos>",
        "змея это хищник <eos>",
        "птица это хищник <eos>",
        "где волк = лес <eos>",
        "где лиса = нора <eos>",
        "где заяц = поле <eos>",
        "где рыба = река <eos>",
        "где птица = небо <eos>",
        "где медведь = тайга <eos>",
        "где щука = вода <eos>",
        "где змея = нора <eos>",
        "где дуб = лес <eos>",
        "где человек = дом <eos>",
        "где кот = дом <eos>",
        "где пес = дом <eos>",
        "где трава = поле <eos>",
        "где вода = река <eos>",
    ];
    for th in taxonomy_and_habitats {
        add_line(th.to_string(), &mut pool, &mut seen);
    }

    // -------------------------------------------------------------------------
    // 4. FOOD CHAIN & TRANSITIVE RELATIONS
    // -------------------------------------------------------------------------
    let food_chain = [
        "волк ест заяц <eos>",
        "лиса ест заяц <eos>",
        "щука ест рыба <eos>",
        "медведь ест рыба <eos>",
        "кот ест рыба <eos>",
        "птица ест рыба <eos>",
        "змея ест заяц <eos>",
        "заяц ест трава <eos>",
        "рыба ест трава <eos>",
        "медведь ест трава <eos>",
        "человек ест рыба <eos>",
        "волк ест заяц заяц ест трава = да <eos>",
        "щука ест рыба рыба ест трава = да <eos>",
        "медведь ест рыба рыба ест трава = да <eos>",
        "лиса ест заяц заяц ест трава = да <eos>",
        "змея ест рыба рыба ест трава = да <eos>",
    ];
    for fc in food_chain {
        add_line(fc.to_string(), &mut pool, &mut seen);
    }

    // -------------------------------------------------------------------------
    // 5. NEGATIONS AND CONTRADICTION CHECKS
    // -------------------------------------------------------------------------
    let negations = [
        "волк ест трава = нет <eos>",
        "заяц ест волк = нет <eos>",
        "рыба ест щука = нет <eos>",
        "трава ест заяц = нет <eos>",
        "дуб ест рыба = нет <eos>",
        "кот это пес = нет <eos>",
        "пес это кот = нет <eos>",
        "волк это друг = нет <eos>",
        "человек это враг = нет <eos>",
        "дуб это животное = нет <eos>",
        "щука это птица = нет <eos>",
        "змея это зверь = нет <eos>",
        "медведь это рыба = нет <eos>",
        "кот это не пес <eos>",
        "пес это не кот <eos>",
        "волк это не друг <eos>",
        "дуб это не рыба <eos>",
        "дерево это не птица <eos>",
        "трава это не вода <eos>",
        "нора это не река <eos>",
        "кот это 0 = нет <eos>",
        "пес это 1 = нет <eos>",
        "человек это 2 = нет <eos>",
        "дуб это 3 = нет <eos>",
        "рыба это 4 = нет <eos>",
        "волк это 5 = нет <eos>",
    ];
    for neg in negations {
        add_line(neg.to_string(), &mut pool, &mut seen);
    }

    // -------------------------------------------------------------------------
    // 6. EXHAUSTIVE ARITHMETIC ADDITION (0..9)
    // -------------------------------------------------------------------------
    for a in 0..=9 {
        for b in 0..=9 {
            if a + b <= 9 {
                let s = a + b;
                add_line(format!("{} + {} = {} <eos>", a, b, s), &mut pool, &mut seen);
                add_line(format!("{} + {} = {} = да <eos>", a, b, s), &mut pool, &mut seen);
                add_line(format!("{} + {} = {} = нет <eos>", a, b, (s + 1) % 10), &mut pool, &mut seen);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 7. EXHAUSTIVE ARITHMETIC SUBTRACTION (0..9)
    // -------------------------------------------------------------------------
    for a in 0..=9 {
        for b in 0..=a {
            let diff = a - b;
            add_line(format!("{} - {} = {} <eos>", a, b, diff), &mut pool, &mut seen);
            add_line(format!("{} - {} = {} = да <eos>", a, b, diff), &mut pool, &mut seen);
            add_line(format!("{} - {} = {} = нет <eos>", a, b, (diff + 1) % 10), &mut pool, &mut seen);
        }
    }

    // -------------------------------------------------------------------------
    // 8. EXHAUSTIVE ARITHMETIC MULTIPLICATION (Products <= 9)
    // -------------------------------------------------------------------------
    for a in 0..=9 {
        for b in 0..=9 {
            if a * b <= 9 {
                let p = a * b;
                add_line(format!("{} * {} = {} <eos>", a, b, p), &mut pool, &mut seen);
                add_line(format!("{} * {} = {} = да <eos>", a, b, p), &mut pool, &mut seen);
                add_line(format!("{} * {} = {} = нет <eos>", a, b, (p + 1) % 10), &mut pool, &mut seen);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 9. EXHAUSTIVE ARITHMETIC DIVISION (Exact integer division 0..9)
    // -------------------------------------------------------------------------
    for a in 0..=9 {
        for b in 1..=9 {
            if a % b == 0 {
                let q = a / b;
                add_line(format!("{} / {} = {} <eos>", a, b, q), &mut pool, &mut seen);
                add_line(format!("{} / {} = {} = да <eos>", a, b, q), &mut pool, &mut seen);
                add_line(format!("{} / {} = {} = нет <eos>", a, b, (q + 1) % 10), &mut pool, &mut seen);
            }
        }
    }

    pool
}

/// Solves exact remainder token alignment using DFS combination search over valid sentences.
fn solve_remainder(target: usize, tok: &Tokenizer) -> Vec<String> {
    let pool: [&str; 12] = [
        "где волк <eos>",
        "кто кот <eos>",
        "логика это наука <eos>",
        "наука это знание <eos>",
        "0 это не 1 <eos>",
        "человек это разум <eos>",
        "человек это не зверь <eos>",
        "1 + 1 = 2 <eos>",
        "кто кот = кот это животное <eos>",
        "число это 0 или 1 <eos>",
        "если 1 + 1 = 2 то да <eos>",
        "если волк ест заяц то волк хищник <eos>",
    ];

    fn dfs(target: usize, tok: &Tokenizer, pool: &[&str], current: &mut Vec<String>) -> bool {
        if target == 0 {
            return true;
        }
        for &s in pool {
            let len = tok.encode(s).len();
            if len <= target {
                current.push(s.to_string());
                if dfs(target - len, tok, pool, current) {
                    return true;
                }
                current.pop();
            }
        }
        false
    }

    let mut result = Vec::new();
    let ok = dfs(target, tok, &pool, &mut result);
    assert!(ok, "Failed to partition remainder of {} tokens", target);
    result
}

/// Builds pretrain corpus of EXACTLY `target_len` tokens.
fn build_pretrain_corpus(tok: &Tokenizer, target_len: usize) -> String {
    let pool = generate_pretrain_pool();
    let mut selected_lines = Vec::new();
    let mut current_len = 0;
    let mut pool_idx = 0;

    // Fill deterministically while leaving a 30-token margin for exact alignment
    while current_len + 30 < target_len {
        let line = &pool[pool_idx % pool.len()];
        let len = tok.encode(line).len();
        selected_lines.push(line.clone());
        current_len += len;
        pool_idx += 1;
    }

    let remainder = target_len - current_len;
    let fillers = solve_remainder(remainder, tok);
    for fl in fillers {
        let len = tok.encode(&fl).len();
        selected_lines.push(fl);
        current_len += len;
    }

    assert_eq!(current_len, target_len, "Pretrain token count must match target_len");
    let full_text = selected_lines.join("\n") + "\n";
    let encoded = tok.encode(&full_text);
    assert_eq!(
        encoded.len(),
        target_len,
        "Encoded pretrain tokens ({}) != target ({})",
        encoded.len(),
        target_len
    );

    // Verify 100% roundtrip fidelity
    for line in full_text.lines() {
        if !line.is_empty() {
            assert!(line.ends_with("<eos>"), "Line does not end with <eos>: {}", line);
            let enc = tok.encode(line);
            let dec = tok.decode(&enc);
            assert_eq!(&dec, line, "Roundtrip mismatch in pretrain line: {}", line);
        }
    }

    full_text
}

/// Builds the instruct dialogue corpus containing ~2,000 tokens (1,800..2,200).
fn build_instruct_corpus(tok: &Tokenizer) -> String {
    let mut lines = Vec::new();
    let mut seen = HashSet::new();

    let add_dialogue = |line: String, lines: &mut Vec<String>, seen: &mut HashSet<String>| {
        if seen.insert(line.clone()) {
            lines.push(line);
        }
    };

    // -------------------------------------------------------------------------
    // 1. DIALOGUE: SCIENCE, EPISTEMOLOGY & DEEP REASONING (from dialogues/*.json)
    // -------------------------------------------------------------------------
    let reasoning_dialogues = [
        "<user> что это наука <bot> наука это знание <eos>",
        "<user> что это логика <bot> логика это наука <eos>",
        "<user> что это модель <bot> модель это знание <eos>",
        "<user> что это разум <bot> разум это знание <eos>",
        "<user> что это знание <bot> знание это наука <eos>",
        "<user> кто человек <bot> человек это разум <eos>",
        "<user> человек это разум <bot> да <eos>",
        "<user> человек это зверь <bot> нет <eos>",
        "<user> логика это наука <bot> да <eos>",
        "<user> наука это знание <bot> да <eos>",
        "<user> модель это наука <bot> да <eos>",
        "<user> почему человек не зверь <bot> человек это разум <eos>",
        "<user> почему дуб не хищник <bot> дуб это дерево <eos>",
        "<user> почему волк ест заяц <bot> волк это хищник <eos>",
        "<user> почему лиса ест заяц <bot> лиса это хищник <eos>",
        "<user> почему щука ест рыба <bot> щука это хищник <eos>",
        "<user> почему медведь ест рыба <bot> медведь это хищник <eos>",
        "<user> почему змея ест заяц <bot> змея это хищник <eos>",
        "<user> почему заяц ест трава <bot> заяц не хищник <eos>",
        "<user> как модель это знание <bot> да <eos>",
        "<user> как разум это логика <bot> да <eos>",
        "<user> как знание это наука <bot> да <eos>",
        "<user> как 1 + 1 <bot> 2 <eos>",
        "<user> как 2 + 2 <bot> 4 <eos>",
        "<user> как 3 + 3 <bot> 6 <eos>",
        "<user> как 4 + 4 <bot> 8 <eos>",
        "<user> как 6 / 2 <bot> 3 <eos>",
        "<user> как 8 / 4 <bot> 2 <eos>",
        "<user> как 9 / 3 <bot> 3 <eos>",
    ];
    for rd in reasoning_dialogues {
        add_dialogue(rd.to_string(), &mut lines, &mut seen);
    }

    // -------------------------------------------------------------------------
    // 2. DIALOGUE: DEDUCTIVE & TRANSITIVE REASONING
    // -------------------------------------------------------------------------
    let deductive_dialogues = [
        "<user> если волк ест заяц то волк хищник <bot> да <eos>",
        "<user> если заяц ест волк <bot> нет <eos>",
        "<user> если щука ест рыба то щука хищник <bot> да <eos>",
        "<user> если рыба ест щука <bot> нет <eos>",
        "<user> если медведь это зверь то медведь не дерево <bot> да <eos>",
        "<user> если дуб это дерево то дуб не животное <bot> да <eos>",
        "<user> если человек это разум то человек друг <bot> да <eos>",
        "<user> волк ест заяц заяц ест трава <bot> да <eos>",
        "<user> щука ест рыба рыба ест трава <bot> да <eos>",
        "<user> медведь ест рыба рыба ест трава <bot> да <eos>",
        "<user> лиса ест заяц заяц ест трава <bot> да <eos>",
        "<user> волк это хищник или трава <bot> хищник <eos>",
        "<user> дуб это дерево или рыба <bot> дерево <eos>",
        "<user> щука это рыба или зверь <bot> рыба <eos>",
        "<user> человек это разум или враг <bot> разум <eos>",
        "<user> 5 это число или зверь <bot> число <eos>",
        "<user> кот это число <bot> нет <eos>",
        "<user> пес это число <bot> нет <eos>",
        "<user> дуб это число <bot> нет <eos>",
        "<user> 0 это число <bot> да <eos>",
        "<user> 1 это число <bot> да <eos>",
        "<user> 2 это число <bot> да <eos>",
        "<user> 3 это число <bot> да <eos>",
        "<user> 4 это число <bot> да <eos>",
        "<user> 5 это число <bot> да <eos>",
        "<user> 6 это число <bot> да <eos>",
        "<user> 7 это число <bot> да <eos>",
        "<user> 8 это число <bot> да <eos>",
        "<user> 9 это число <bot> да <eos>",
    ];
    for dd in deductive_dialogues {
        add_dialogue(dd.to_string(), &mut lines, &mut seen);
    }

    // -------------------------------------------------------------------------
    // 3. DIALOGUE: SUBTRACTION (minuends 0..9)
    // -------------------------------------------------------------------------
    for a in 0..=9 {
        for b in 0..=a {
            let diff = a - b;
            add_dialogue(format!("<user> {} - {} = <bot> {} <eos>", a, b, diff), &mut lines, &mut seen);
        }
    }

    // -------------------------------------------------------------------------
    // 4. DIALOGUE: DIVISION (exact integer division)
    // -------------------------------------------------------------------------
    for a in 0..=9 {
        for b in 1..=9 {
            if a % b == 0 {
                let q = a / b;
                add_dialogue(format!("<user> {} / {} = <bot> {} <eos>", a, b, q), &mut lines, &mut seen);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 5. DIALOGUE: ADDITION & MULTIPLICATION
    // -------------------------------------------------------------------------
    for a in 0..=5 {
        for b in 0..=5 {
            if a + b <= 9 {
                add_dialogue(format!("<user> {} + {} = <bot> {} <eos>", a, b, a + b), &mut lines, &mut seen);
            }
            if a * b <= 9 {
                add_dialogue(format!("<user> {} * {} = <bot> {} <eos>", a, b, a * b), &mut lines, &mut seen);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 6. DIALOGUE: TAXONOMY, HABITAT & FOOD CHAIN QA
    // -------------------------------------------------------------------------
    let taxonomy_qa = [
        "<user> где волк <bot> лес <eos>",
        "<user> где лиса <bot> нора <eos>",
        "<user> где заяц <bot> поле <eos>",
        "<user> где рыба <bot> река <eos>",
        "<user> где птица <bot> небо <eos>",
        "<user> где медведь <bot> тайга <eos>",
        "<user> где щука <bot> вода <eos>",
        "<user> где змея <bot> нора <eos>",
        "<user> где дуб <bot> лес <eos>",
        "<user> где человек <bot> дом <eos>",
        "<user> где кот <bot> дом <eos>",
        "<user> где пес <bot> дом <eos>",
        "<user> кто волк <bot> волк это зверь <eos>",
        "<user> кто лиса <bot> лиса это хищник <eos>",
        "<user> кто заяц <bot> заяц это зверь <eos>",
        "<user> кто рыба <bot> рыба это животное <eos>",
        "<user> кто птица <bot> птица это животное <eos>",
        "<user> кто медведь <bot> медведь это зверь <eos>",
        "<user> кто щука <bot> щука это рыба <eos>",
        "<user> кто дуб <bot> дуб это дерево <eos>",
        "<user> кто человек <bot> человек это разум <eos>",
        "<user> что ест волк <bot> заяц <eos>",
        "<user> что ест лиса <bot> заяц <eos>",
        "<user> что ест щука <bot> рыба <eos>",
        "<user> что ест медведь <bot> рыба <eos>",
        "<user> что ест заяц <bot> трава <eos>",
        "<user> что ест кот <bot> рыба <eos>",
        "<user> что ест пес <bot> рыба <eos>",
        "<user> что ест птица <bot> рыба <eos>",
        "<user> что ест змея <bot> заяц <eos>",
        "<user> кто ест заяц <bot> волк <eos>",
        "<user> кто ест рыба <bot> щука <eos>",
        "<user> кто ест трава <bot> заяц <eos>",
    ];
    for tq in taxonomy_qa {
        add_dialogue(tq.to_string(), &mut lines, &mut seen);
    }

    let full_text = lines.join("\n") + "\n";
    let tokens = tok.encode(&full_text);

    // Verify constraints
    assert!(
        tokens.len() >= 1800 && tokens.len() <= 2200,
        "Instruct token count must be in range 1800..2200, got {}",
        tokens.len()
    );

    for line in full_text.lines() {
        if !line.is_empty() {
            assert!(line.starts_with("<user> "), "Line does not start with <user>: {}", line);
            assert!(line.contains(" <bot> "), "Line missing <bot>: {}", line);
            assert!(line.ends_with(" <eos>"), "Line does not end with <eos>: {}", line);
            let enc = tok.encode(line);
            let dec = tok.decode(&enc);
            assert_eq!(&dec, line, "Roundtrip mismatch in instruct line: {}", line);
        }
    }

    full_text
}

fn main() {
    let tok = Tokenizer::chinchilla();
    fs::create_dir_all("data").expect("Failed to create data dir");

    // 1. Write data/vocab_chinchilla.txt
    let mut vocab_content = String::new();
    for (i, token) in VOCAB_CHINCHILLA.iter().enumerate() {
        vocab_content.push_str(&format!("{}: {}\n", i, token));
    }
    fs::write("data/vocab_chinchilla.txt", &vocab_content)
        .expect("Failed to write data/vocab_chinchilla.txt");
    println!(
        "[1/3] Generated data/vocab_chinchilla.txt ({} tokens, V = {})",
        VOCAB_CHINCHILLA.len(),
        VOCAB_CHINCHILLA.len()
    );

    // 2. Generate data/pretrain_chinchilla.txt (EXACTLY 17,920 tokens, Chinchilla 20:1)
    const PRETRAIN_TARGET: usize = 17920;
    let pretrain_text = build_pretrain_corpus(&tok, PRETRAIN_TARGET);
    fs::write("data/pretrain_chinchilla.txt", &pretrain_text)
        .expect("Failed to write data/pretrain_chinchilla.txt");
    let pretrain_tokens = tok.encode(&pretrain_text);
    println!(
        "[2/3] Generated data/pretrain_chinchilla.txt (EXACTLY {} tokens, 100% Chinchilla 20:1 Law)",
        pretrain_tokens.len()
    );
    assert_eq!(pretrain_tokens.len(), PRETRAIN_TARGET);

    // 3. Generate data/instruct_chinchilla.txt (~2,000 tokens, range 1,800..2,200)
    let instruct_text = build_instruct_corpus(&tok);
    fs::write("data/instruct_chinchilla.txt", &instruct_text)
        .expect("Failed to write data/instruct_chinchilla.txt");
    let instruct_tokens = tok.encode(&instruct_text);
    println!(
        "[3/3] Generated data/instruct_chinchilla.txt ({} tokens, target 1,800..2,200)",
        instruct_tokens.len()
    );
    assert!((1800..=2200).contains(&instruct_tokens.len()));

    println!("All Chinchilla Sprint 1 datasets generated and verified successfully!");
}
