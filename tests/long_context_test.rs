use srxformer::scaled::config::Tier;
use srxformer::scaled::generator::{sample_token, FastRng, SamplingConfig};
use srxformer::scaled::qreno_srx::QrenoSrxLM;

#[test]
fn test_long_context_1024_tokens_srx_stability() {
    let lm = QrenoSrxLM::new(Tier::Micro);
    let mut state = lm.init_state();
    let mut ws = lm.new_workspace();
    let mut logits = vec![0.0f32; 256];

    // State size check: strictly constant O(1)
    let state_bytes = state.memory_bytes();
    assert_eq!(state_bytes, lm.config.n_heads * 80, "SRX state must be exactly H * 80 bytes");

    // Process a synthetic 1024-step sequence
    for step in 0..1024 {
        let b = ((step * 37 + 13) % 256) as u8;
        lm.step_cluster(&[b], &mut state, &mut logits, &mut ws);
    }

    // Verify state memory remained strictly constant
    assert_eq!(state.memory_bytes(), state_bytes, "Memory must never grow beyond O(1) across 1024 steps");

    // Verify logits are non-trivial and finite
    assert_eq!(logits.len(), 256);
    for &l in &logits {
        assert!(l.is_finite(), "Logits must remain numerically stable and finite after 1024 steps");
    }
}

#[test]
fn test_stochastic_sampling_non_trivial_diversity() {
    let mut logits = vec![0.0f32; 256];
    // Dominated by byte 'o' (111 in ascii)
    logits[111] = 5.0;
    // Alternative candidates
    logits[97] = 4.8; // 'a'
    logits[101] = 4.6; // 'e'

    let mut rng = FastRng::new(42);
    let cfg = SamplingConfig {
        temperature: 0.7,
        top_k: 3,
        repetition_penalty: 1.2,
        penalty_window: 4,
    };

    let mut counts = std::collections::HashMap::new();
    let mut recent = Vec::new();

    for _ in 0..100 {
        let mut l_copy = logits.clone();
        let tok = sample_token(&mut l_copy, &recent, &cfg, &mut rng);
        *counts.entry(tok).or_insert(0) += 1;
        recent.push(tok);
    }

    // Check that we didn't collapse exclusively into token 111
    assert!(counts.len() >= 2, "Stochastic sampling must prevent single-token mode collapse, got {:?}", counts);
    assert!(*counts.get(&111).unwrap_or(&0) < 95, "Dominated token must not consume 95%+ of generations");
}
