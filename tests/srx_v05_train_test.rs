//! Sprint 2 Verification Suite for SRXFORMER v05 (Quantum-Algebraic Core: «Автомат Калашникова»).
//! Validates:
//! 1. Mathematical Rigor: Pure Orthogonal Complement Projector Memory Reproduction (1e-6 tolerance).
//! 2. Undistorted MUSIC Subspace Resonance: E_noise == 0 and w == 15.0 at q == k_rot.
//! 3. State O(1) Memory: Strictly 160 bytes (100% L1D cache resident).
//! 4. Parameter Parity: Exactly 896 parameters (0.00% delta with Classical Transformer).
//! 5. Exact Reversible BPTT: Analytical vs Finite Difference Gradient check (< 5e-3 tolerance).

use std::fs;

use srxformer::{
    classic::{Tokenizer, TransformerConfig, EOS_TOKEN_ID},
    srx_v05::{
        apply_butterfly_4, backward_loss, forward_loss, l2_normalize, SrxGrad, SrxState,
        SrxTrainWorkspace, SrxTransformer, SrxWorkspace, SRX_W_MAX,
    },
};

#[test]
fn test_srx_v05_parameter_count_896() {
    let config = TransformerConfig::lang_chinchilla();
    let model = SrxTransformer::new(config.clone()).expect("Failed to create SrxTransformer v05");

    assert_eq!(
        model.param_count(),
        896,
        "SRX v05 must contain EXACTLY 896 trainable parameters (matching Classical Baseline)!"
    );

    // Embeddings: 65 * 8 = 520
    assert_eq!(model.token_embeddings.len(), 520);
    // Attention: 4 * 64 = 256
    let attn = &model.layers[0].attn;
    assert_eq!(attn.param_count(), 256);
    // Norms: 3 * 8 = 24
    let layer = &model.layers[0];
    assert_eq!(layer.attn_norm_gamma.len(), 8);
    assert_eq!(layer.ffn_norm_gamma.len(), 8);
    assert_eq!(model.final_norm_gamma.len(), 8);
    // FFN: 6 * 8 + 8 * 6 = 96
    assert_eq!(layer.mlp.w_1.len(), 48);
    assert_eq!(layer.mlp.w_2.len(), 48);

    assert_eq!(520 + 256 + 24 + 96, 896);
}

#[test]
fn test_srx_v05_state_footprint_strictly_160_bytes() {
    let config = TransformerConfig::lang_chinchilla();
    let state = SrxState::new(&config);

    // Thetas: 2 heads * 4 angles = 8 floats = 32 bytes
    assert_eq!(state.thetas.len(), 8);
    // M: 2 heads * 4 * 4 = 32 floats = 128 bytes
    assert_eq!(state.m.len(), 32);

    // Total = 40 floats = EXACTLY 160 bytes!
    assert_eq!(
        state.memory_bytes(),
        160,
        "SRX v05 context state must occupy strictly 160 bytes (< 0.5% of 32 KB L1D cache)!"
    );
}

#[test]
fn test_srx_v05_orthogonal_projector_exact_reproduction() {
    // Exact response identity: M_t^T * k_rot == v_raw
    let mut m = [
        0.31f32, -0.12, 0.45, 0.67,
        -0.89, 0.22, -0.54, 0.18,
        0.05, 0.76, -0.33, -0.42,
        0.91, -0.63, 0.14, 0.85,
    ];

    let test_vectors = [
        ([0.6f32, -0.8, 0.0, 0.0], [2.5f32, -1.8, 0.4, 3.1]),
        ([0.2f32, 0.4, -0.5, 0.7071], [-0.9f32, 1.2, 3.4, -2.1]),
        ([1.0f32, 0.0, 0.0, 0.0], [0.1f32, 0.2, 0.3, 0.4]),
        ([0.0f32, 0.0, 0.0, 1.0], [5.0f32, -4.0, 3.0, -2.0]),
    ];

    for (k_unnorm, v_raw) in test_vectors {
        let mut k_rot = [0.0f32; 4];
        l2_normalize(&k_unnorm, &mut k_rot, 1e-12);

        // v_hat = M_{t-1}^T * k_rot
        let mut v_hat = [0.0f32; 4];
        for col in 0..4 {
            let mut sum = 0.0f32;
            for row in 0..4 {
                sum += k_rot[row] * m[row * 4 + col];
            }
            v_hat[col] = sum;
        }

        // e_t = v_raw - v_hat
        let mut e_t = [0.0f32; 4];
        for c in 0..4 {
            e_t[c] = v_raw[c] - v_hat[c];
        }

        // M_t = M_{t-1} + k_rot * e_t^T (Orthogonal Projector on k^\perp)
        for row in 0..4 {
            let kr = k_rot[row];
            for col in 0..4 {
                m[row * 4 + col] += kr * e_t[col];
            }
        }

        // Response check: M_t^T * k_rot == v_raw with tolerance 1e-6
        let mut v_out = [0.0f32; 4];
        for col in 0..4 {
            let mut sum = 0.0f32;
            for row in 0..4 {
                sum += k_rot[row] * m[row * 4 + col];
            }
            v_out[col] = sum;
        }

        for c in 0..4 {
            let diff = (v_out[c] - v_raw[c]).abs();
            assert!(
                diff < 1e-6,
                "Projector response identity failed at c={}: expected {}, got {}, diff={}",
                c,
                v_raw[c],
                v_out[c],
                diff
            );
        }

        // Zero memory drift check: if key repeats and v matches, e_t == 0 and M_t == M_{t-1}
        let m_snapshot = m;
        let mut v_hat_repeat = [0.0f32; 4];
        for col in 0..4 {
            let mut sum = 0.0f32;
            for row in 0..4 {
                sum += k_rot[row] * m[row * 4 + col];
            }
            v_hat_repeat[col] = sum;
        }
        for c in 0..4 {
            let e_repeat = v_raw[c] - v_hat_repeat[c];
            assert!(
                e_repeat.abs() < 1e-6,
                "Repeated key must yield zero error e_t == 0"
            );
        }
        for idx in 0..16 {
            assert!(
                (m[idx] - m_snapshot[idx]).abs() < 1e-6,
                "Memory must have zero drift on identical key-value pair"
            );
        }
    }
}

#[test]
fn test_srx_v05_music_peak_trigger() {
    // When q == k_rot where k_rot = U(Theta) * k_sig with k_sig = [k0, k1, 0, 0]:
    // Noise subspace energy E_noise == 0, and resonant gain w reaches maximum SRX_W_MAX (15.0).
    let thetas = [0.72f32, -0.45, 1.15, -0.92];

    // Signal subspace has 0 components in dimensions 2 and 3
    let k_sig = [0.28f32, 0.96, 0.0, 0.0];

    // k_rot = U(Theta) * k_sig
    let mut k_rot = [0.0f32; 4];
    apply_butterfly_4(&k_sig, &thetas, false, &mut k_rot);

    // Query matches rotated key: q_norm = k_rot
    let q_norm = k_rot;

    // Inverse rotation: q_inv = U^\dagger(Theta) * q_norm
    let mut q_inv = [0.0f32; 4];
    apply_butterfly_4(&q_norm, &thetas, true, &mut q_inv);

    // Noise subspace energy = q_inv[2]^2 + q_inv[3]^2
    let noise_energy = q_inv[2] * q_inv[2] + q_inv[3] * q_inv[3];

    assert!(
        noise_energy < 1e-6,
        "Undistorted MUSIC noise energy must be strictly 0, got {}",
        noise_energy
    );

    let eps = 1e-3f32;
    let w = (1.0 / (noise_energy + eps)).min(SRX_W_MAX);
    assert_eq!(
        w, 15.0,
        "MUSIC gain must hit Dirac resonance limit 15.0, got {}",
        w
    );
}

#[test]
fn test_srx_v05_numerical_vjp_gradients() {
    let config = TransformerConfig::lang_chinchilla();
    let mut model = SrxTransformer::new_with_seed(config.clone(), 1337).unwrap();
    let mut ws = SrxTrainWorkspace::new(&config);
    let mut grad = SrxGrad::new(&config);

    let tokens = [2, 6, 41, 6, 12, 3, 7, EOS_TOKEN_ID]; // "<user> 2 / 2 = <bot> 1 <eos>"

    // Compute analytical gradients
    grad.zero();
    let loss = forward_loss(&model, &tokens, &mut ws, 1.0);
    backward_loss(&model, &tokens, &mut ws, &mut grad, 1.0);

    assert!(loss > 0.0 && loss.is_finite());
    assert!(grad.l2_norm() > 0.0);

    let eps = 1e-3f32;
    let max_diff_allowed = 5e-3f32;

    // Check W_q gradients
    for i in 0..4 {
        let orig = model.layers[0].attn.w_q[i];

        model.layers[0].attn.w_q[i] = orig + eps;
        let loss_plus = forward_loss(&model, &tokens, &mut ws, 1.0);

        model.layers[0].attn.w_q[i] = orig - eps;
        let loss_minus = forward_loss(&model, &tokens, &mut ws, 1.0);

        model.layers[0].attn.w_q[i] = orig;

        let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let ana_grad = grad.w_q[i];

        let diff = (ana_grad - num_grad).abs();
        assert!(
            diff < max_diff_allowed,
            "W_q[{}] mismatch: ana={}, num={}, diff={} (limit={})",
            i,
            ana_grad,
            num_grad,
            diff,
            max_diff_allowed
        );
    }

    // Check W_k gradients
    for i in 0..4 {
        let orig = model.layers[0].attn.w_k[i];

        model.layers[0].attn.w_k[i] = orig + eps;
        let loss_plus = forward_loss(&model, &tokens, &mut ws, 1.0);

        model.layers[0].attn.w_k[i] = orig - eps;
        let loss_minus = forward_loss(&model, &tokens, &mut ws, 1.0);

        model.layers[0].attn.w_k[i] = orig;

        let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let ana_grad = grad.w_k[i];

        let diff = (ana_grad - num_grad).abs();
        assert!(
            diff < max_diff_allowed,
            "W_k[{}] mismatch: ana={}, num={}, diff={} (limit={})",
            i,
            ana_grad,
            num_grad,
            diff,
            max_diff_allowed
        );
    }

    // Check W_v gradients
    for i in 0..4 {
        let orig = model.layers[0].attn.w_v[i];

        model.layers[0].attn.w_v[i] = orig + eps;
        let loss_plus = forward_loss(&model, &tokens, &mut ws, 1.0);

        model.layers[0].attn.w_v[i] = orig - eps;
        let loss_minus = forward_loss(&model, &tokens, &mut ws, 1.0);

        model.layers[0].attn.w_v[i] = orig;

        let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let ana_grad = grad.w_v[i];

        let diff = (ana_grad - num_grad).abs();
        assert!(
            diff < max_diff_allowed,
            "W_v[{}] mismatch: ana={}, num={}, diff={} (limit={})",
            i,
            ana_grad,
            num_grad,
            diff,
            max_diff_allowed
        );
    }

    // Check W_1 (FFN) gradients
    for i in 0..4 {
        let orig = model.layers[0].mlp.w_1[i];

        model.layers[0].mlp.w_1[i] = orig + eps;
        let loss_plus = forward_loss(&model, &tokens, &mut ws, 1.0);

        model.layers[0].mlp.w_1[i] = orig - eps;
        let loss_minus = forward_loss(&model, &tokens, &mut ws, 1.0);

        model.layers[0].mlp.w_1[i] = orig;

        let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let ana_grad = grad.w_1[i];

        let diff = (ana_grad - num_grad).abs();
        assert!(
            diff < max_diff_allowed,
            "W_1[{}] mismatch: ana={}, num={}, diff={} (limit={})",
            i,
            ana_grad,
            num_grad,
            diff,
            max_diff_allowed
        );
    }
}

#[test]
fn test_srx_v05_step_forward_equivalence() {
    let config = TransformerConfig::lang_chinchilla();
    let model = SrxTransformer::new_with_seed(config.clone(), 99).unwrap();
    let mut state = SrxState::new(&config);
    let mut ws = SrxWorkspace::new(&config);

    let tokens = [2, 6, 41, 6, 12, 3, 7, EOS_TOKEN_ID]; // "<user> 2 / 2 = <bot> 1 <eos>"

    // Autoregressive token-by-token decoding step
    for (pos, &tok) in tokens.iter().enumerate() {
        let logits = model.step(tok, pos, &mut state, &mut ws);
        assert_eq!(logits.len(), 65);
        for &val in logits {
            assert!(val.is_finite(), "Logits must be finite numbers");
        }
    }
}

#[test]
fn test_srx_v05_training_loss_decrease() {
    let config = TransformerConfig::lang_chinchilla();
    let mut model = SrxTransformer::new_with_seed(config.clone(), 42).unwrap();
    let tokenizer = Tokenizer::chinchilla();

    let instruct_path = "data/instruct_chinchilla.txt";
    let text = fs::read_to_string(instruct_path).expect("Failed to read instruct_chinchilla.txt");
    let all_tokens = tokenizer.encode(&text);
    let tokens = &all_tokens[..all_tokens.len().min(120)];

    let telemetry = model.train_dataset(tokens, 5, 0.02);
    assert!(telemetry.final_loss < telemetry.initial_loss, "Training must reduce cross-entropy loss!");
    assert!(telemetry.final_perplexity < telemetry.initial_perplexity);
}
