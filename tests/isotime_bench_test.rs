use std::path::Path;
use srxformer::scaled::{
    load_corpus_lines, load_typo_pairs, run_isotime_benchmark, IsoTimeConfig, Tier,
};

#[test]
fn test_wiki_corpora_integrity() {
    let train_lines = load_corpus_lines("data/wiki_train.txt");
    assert_eq!(train_lines.len(), 20000, "Train corpus must contain 20,000 sentences");

    let val_lines = load_corpus_lines("data/wiki_val.txt");
    assert_eq!(val_lines.len(), 2000, "Val corpus must contain 2,000 sentences");

    let typo_pairs = load_typo_pairs("data/wiki_typo_eval.txt");
    assert_eq!(typo_pairs.len(), 100, "Typo eval corpus must contain 100 pairs");
}

#[test]
fn test_isotime_mini_benchmark() {
    let config = IsoTimeConfig {
        duration_secs: 2.0, // 2 seconds per model for automated test
        tier: Tier::Micro,
        train_path: "data/wiki_train.txt".to_string(),
        val_path: "data/wiki_val.txt".to_string(),
        typo_path: "data/wiki_typo_eval.txt".to_string(),
        checkpoint_interval_secs: 1.0,
    };

    let (classic_res, srx_res) = run_isotime_benchmark(&config);

    // 1. Check durations
    assert!(classic_res.duration_sec >= 1.8, "Classic must train for ~2s, got {:.2}s", classic_res.duration_sec);
    assert!(srx_res.duration_sec >= 1.8, "SRX must train for ~2s, got {:.2}s", srx_res.duration_sec);

    // 2. Check tokens processed
    assert!(classic_res.tokens_processed > 0, "Classic must process tokens");
    assert!(srx_res.tokens_processed > 0, "SRX must process tokens");

    // 3. Check loss & perplexity sanity
    assert!(classic_res.final_train_loss > 0.0 && classic_res.final_train_loss.is_finite());
    assert!(srx_res.final_train_loss > 0.0 && srx_res.final_train_loss.is_finite());
    assert!(classic_res.val_loss > 0.0 && classic_res.val_ppl >= 1.0);
    assert!(srx_res.val_loss > 0.0 && srx_res.val_ppl >= 1.0);

    // 4. Check Typo Robustness
    assert_eq!(classic_res.typo_eval.num_pairs, 100);
    assert_eq!(srx_res.typo_eval.num_pairs, 100);
    assert!(
        srx_res.typo_eval.mean_cosine_similarity >= 0.85,
        "SRX Q-RENO typo cosine similarity must be >= 0.85, got {:.4}",
        srx_res.typo_eval.mean_cosine_similarity
    );

    // 5. Check Memory Wall Challenge points
    assert_eq!(classic_res.memory_wall.len(), 7);
    assert_eq!(srx_res.memory_wall.len(), 7);

    // 6. Check Telemetry Files generated
    assert!(Path::new("telemetry_isotime_classic.txt").exists());
    assert!(Path::new("telemetry_isotime_srx_qreno.txt").exists());
}
