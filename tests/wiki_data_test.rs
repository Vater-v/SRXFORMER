//! Integration tests for Russian Wikipedia dataset integrity.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[test]
fn test_wiki_datasets_exist_and_non_empty() {
    let train_path = Path::new("data/wiki_train.txt");
    let val_path = Path::new("data/wiki_val.txt");
    let typo_path = Path::new("data/wiki_typo_eval.txt");

    assert!(train_path.exists(), "data/wiki_train.txt must exist");
    assert!(val_path.exists(), "data/wiki_val.txt must exist");
    assert!(typo_path.exists(), "data/wiki_typo_eval.txt must exist");

    let train_file = File::open(train_path).expect("open train file");
    let train_lines: Vec<String> = BufReader::new(train_file).lines().map(|l| l.unwrap()).collect();
    assert_eq!(train_lines.len(), 20000, "Train set must have 20,000 sentences");

    let val_file = File::open(val_path).expect("open val file");
    let val_lines: Vec<String> = BufReader::new(val_file).lines().map(|l| l.unwrap()).collect();
    assert_eq!(val_lines.len(), 2000, "Val set must have 2,000 sentences");

    let typo_file = File::open(typo_path).expect("open typo file");
    let typo_lines: Vec<String> = BufReader::new(typo_file).lines().map(|l| l.unwrap()).collect();
    assert_eq!(typo_lines.len(), 100, "Typo eval set must have 100 pairs");

    // Check line format: every line ends with <eos>
    for line in &train_lines[..100] {
        assert!(line.ends_with("<eos>"), "Each sentence must end with <eos>");
    }
    for line in &val_lines[..100] {
        assert!(line.ends_with("<eos>"), "Each sentence must end with <eos>");
    }
    for line in &typo_lines {
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(parts.len(), 2, "Typo lines must have 2 tab-separated fields");
        assert!(parts[0].ends_with("<eos>"));
        assert!(parts[1].ends_with("<eos>"));
        assert_ne!(parts[0], parts[1], "Corrupted text must differ from original");
    }
}
