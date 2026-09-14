use srxformer::scaled::*;

#[test]
fn test_scaling_calculator_tiers() {
    // 1) Scaling calculator parameter and memory accounting
    let vocab_size = 256;

    let micro = ScalingCalculator::compute_config(Tier::Micro, vocab_size);
    assert_eq!(micro.n_heads, 2);
    assert_eq!(micro.d_model, 8);
    assert_eq!(micro.d_ff, 12);
    assert_eq!(micro.srx_state_bytes(), 160);
    assert_eq!(ScalingCalculator::cache_tier(micro.srx_state_bytes()), "L1D Cache (<= 32 KB, Resident)");

    let standard = ScalingCalculator::compute_config(Tier::Standard, vocab_size);
    assert_eq!(standard.n_heads, 4);
    assert_eq!(standard.d_model, 16);
    assert_eq!(standard.d_ff, 24);
    assert_eq!(standard.srx_state_bytes(), 320);

    let pro = ScalingCalculator::compute_config(Tier::Pro, vocab_size);
    assert_eq!(pro.n_heads, 8);
    assert_eq!(pro.d_model, 32);
    assert_eq!(pro.d_ff, 48);
    assert_eq!(pro.srx_state_bytes(), 640);

    let ultra = ScalingCalculator::compute_config(Tier::Ultra, vocab_size);
    assert_eq!(ultra.n_heads, 16);
    assert_eq!(ultra.d_model, 64);
    assert_eq!(ultra.d_ff, 96);
    assert_eq!(ultra.srx_state_bytes(), 1280);

    // KV Cache Memory Wall crossover check
    let kv_small = ScalingCalculator::classic_kv_bytes(pro.d_model, 32);
    let kv_wall = ScalingCalculator::classic_kv_bytes(pro.d_model, 65536);
    assert_eq!(kv_small, 2 * 32 * 32 * 4); // 8,192 B (8 KB)
    assert_eq!(kv_wall, 2 * 65536 * 32 * 4); // 16,777,216 B (16 MB)
    assert_eq!(ScalingCalculator::cache_tier(kv_small), "L1D Cache (<= 32 KB, Resident)");
    assert_eq!(ScalingCalculator::cache_tier(kv_wall), "L3 Cache (256 KB - 20 MB, Shared Bus)");
}

#[test]
fn test_scaled_srx_attention_forward_step_equivalence() {
    // 2) Test ScaledSrxAttention equivalence on H=4 (d=16) and H=8 (d=32)
    for &n_heads in &[4, 8] {
        let attn = ScaledSrxAttention::new(n_heads);
        let config = ScaledConfig::from_tier(
            if n_heads == 4 { Tier::Standard } else { Tier::Pro },
            256,
        );
        let mut ws1 = ScaledWorkspace::new(&config);
        let mut ws2 = ScaledWorkspace::new(&config);

        let t_steps = 6;
        let d_model = config.d_model;
        let mut seq_x = vec![0.0f32; t_steps * d_model];

        for i in 0..seq_x.len() {
            seq_x[i] = ((i as f32) * 0.37).sin() * 0.5;
        }

        // 1. Full sequence forward
        let mut seq_y = vec![0.0f32; t_steps * d_model];
        attn.forward(&seq_x, &mut seq_y, &mut ws1);

        // 2. Sequential step calls
        let mut step_y = vec![0.0f32; t_steps * d_model];
        let mut state = ScaledSrxState::new(n_heads);

        for t in 0..t_steps {
            let in_slice = &seq_x[t * d_model..(t + 1) * d_model];
            let out_slice = &mut step_y[t * d_model..(t + 1) * d_model];
            attn.step(in_slice, &mut state, out_slice, &mut ws2);
        }

        // Compare outputs
        let mut max_diff = 0.0f32;
        for i in 0..seq_y.len() {
            let diff = (seq_y[i] - step_y[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        assert!(
            max_diff < 1e-5,
            "Forward vs step mismatch for H={n_heads}: max_diff={max_diff}"
        );
    }
}

#[test]
fn test_scaled_srx_state_memory_bytes() {
    // 3) State footprint must be strictly H * 80 bytes
    for n_heads in [2, 4, 8, 16, 32] {
        let state = ScaledSrxState::new(n_heads);
        assert_eq!(state.memory_bytes(), n_heads * 80);
    }
}

#[test]
fn test_scaled_nn_gradient_check() {
    // 4) Numerical vs analytical gradient check on modular layers
    let in_dim = 16;
    let out_dim = 16;
    let mut linear = ScaledLinear::new(in_dim, out_dim);
    let mut norm = ScaledRMSNorm::new(in_dim, 1e-5);

    let x = vec![0.3f32; in_dim];
    let dy = vec![0.5f32; out_dim];

    // Linear backward
    let mut dx = vec![0.0f32; in_dim];
    linear.backward(&x, &dy, &mut dx);

    // Numerical gradient check on weight[2, 3]
    let row = 2;
    let col = 3;
    let idx = row * in_dim + col;
    let eps = 1e-3f32;

    linear.weight[idx] += eps;
    let mut y_plus = vec![0.0f32; out_dim];
    linear.forward(&x, &mut y_plus);
    let loss_plus: f32 = y_plus.iter().zip(&dy).map(|(y, g)| y * g).sum();

    linear.weight[idx] -= 2.0 * eps;
    let mut y_minus = vec![0.0f32; out_dim];
    linear.forward(&x, &mut y_minus);
    let loss_minus: f32 = y_minus.iter().zip(&dy).map(|(y, g)| y * g).sum();
    linear.weight[idx] += eps;

    let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    let ana_grad = linear.grad[idx];

    let diff = (num_grad - ana_grad).abs();
    assert!(
        diff < 1e-3,
        "Linear gradient mismatch: num={num_grad}, ana={ana_grad}, diff={diff}"
    );

    // RMSNorm backward check
    let mut dx_norm = vec![0.0f32; in_dim];
    norm.backward(&x, &dy, &mut dx_norm);
    assert_eq!(norm.grad.len(), in_dim);
}

#[test]
fn test_qreno_srx_lm_end_to_end() {
    // 5) End-to-end integration: Q-RENO frontend + Scaled SRX core
    let model = QrenoSrxLM::new(Tier::Standard);
    let mut state = model.init_state();
    let mut ws = model.new_workspace();

    assert_eq!(state.memory_bytes(), 4 * 80); // 320 bytes for Tier::Standard (H=4)

    let phrase = "Привет квантовый мир!";
    let logits_seq = model.forward_text(phrase, &mut state, &mut ws);

    assert!(!logits_seq.is_empty(), "Must produce logit sequence");
    assert_eq!(logits_seq[0].len(), 256, "Logits must cover 256 byte codes");

    // Autoregressive generation
    let generated = model.generate_bytes("алгебра", 8, &mut ws);
    assert!(!generated.is_empty(), "Must generate continuation bytes");
}
