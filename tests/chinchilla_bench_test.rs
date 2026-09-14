//! Tests for Sprint 4: Chinchilla Iso-FLOPs Benchmark and Memory Wall Challenge.

use srxformer::{
    classic::{KvCache, Tokenizer, Transformer, TransformerConfig},
    srx_v05::{SrxState, SrxTransformer},
};

#[test]
fn test_chinchilla_memory_wall_scaling() {
    let config = TransformerConfig::lang_chinchilla();
    let srx_state = SrxState::new(&config);

    // SRX v05 state is strictly 160 bytes at any sequence length
    assert_eq!(srx_state.memory_bytes(), 160);

    let test_lengths = [32, 128, 512, 1024, 4096, 16384, 65536];
    let expected_kv_bytes = [
        2048,      // 2 KB @ N=32
        8192,      // 8 KB @ N=128
        32768,     // 32 KB @ N=512 (spills L1D)
        65536,     // 65.5 KB @ N=1K
        262144,    // 262 KB @ N=4K (spills L2)
        1048576,   // 1.05 MB @ N=16K
        4194304,   // 4.19 MB @ N=64K (spills to DRAM)
    ];

    for (&n, &expected_bytes) in test_lengths.iter().zip(expected_kv_bytes.iter()) {
        let mut n_config = config.clone();
        n_config.max_seq_len = n;
        let kv = KvCache::new(&n_config);

        let kv_bytes = (kv.k.len() + kv.v.len()) * std::mem::size_of::<f32>();
        assert_eq!(kv_bytes, expected_bytes);

        let ratio = kv_bytes as f64 / srx_state.memory_bytes() as f64;
        if n == 65536 {
            assert!((ratio - 26214.4).abs() < 1e-3);
        }
    }
}

#[test]
fn test_chinchilla_isoflops_formula_parity() {
    let config = TransformerConfig::lang_chinchilla();
    let classic = Transformer::new(config.clone()).unwrap();
    let srx = SrxTransformer::new(config.clone()).unwrap();

    assert_eq!(classic.param_count(), 896);
    assert_eq!(srx.param_count(), 896);

    let classic_fwd = 2 * classic.param_count() as u64;
    let classic_bwd = 4 * classic.param_count() as u64;
    let classic_step = 6 * classic.param_count() as u64;

    assert_eq!(classic_fwd, 1792);
    assert_eq!(classic_bwd, 3584);
    assert_eq!(classic_step, 5376);

    let srx_fwd = 2 * srx.param_count() as u64 + 288;
    let srx_bwd = 4 * srx.param_count() as u64 + 576;
    let srx_step = 6 * srx.param_count() as u64 + 864;

    assert_eq!(srx_fwd, 2080);
    assert_eq!(srx_bwd, 4160);
    assert_eq!(srx_step, 6240);
}

#[test]
fn test_chinchilla_60_tasks_encodability() {
    let tok = Tokenizer::chinchilla();
    let prompts = [
        "<user> 2 + 3 = <bot>",
        "<user> 5 - 2 = <bot>",
        "<user> 5 - 3 = <bot>",
        "<user> 2 * 3 = <bot>",
        "<user> 6 / 2 = <bot>",
        "<user> кто кот <bot>",
        "<user> где волк <bot>",
        "<user> кот это пес <bot>",
        "<user> если волк ест заяц то волк хищник <bot>",
        "<user> волк ест заяц заяц ест трава <bot>",
    ];

    for p in prompts {
        let enc = tok.encode(p);
        assert!(!enc.is_empty());
        let dec = tok.decode(&enc);
        assert_eq!(dec, p);
    }
}
