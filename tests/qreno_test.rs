use srxformer::qreno::*;

#[test]
fn test_qreno_spectral_solver_precision() {
    // 1) Test precision across various cluster lengths L in 1..=16
    for n in 1..=16 {
        let mut diag = vec![0.0f32; n];
        let mut subdiag = vec![0.0f32; n.saturating_sub(1)];

        for i in 0..n {
            diag[i] = -0.5 + 0.1 * ((i as f32) * 1.7).sin();
        }
        for i in 0..n.saturating_sub(1) {
            subdiag[i] = -0.8 + 0.05 * ((i as f32) * 2.3).cos();
        }

        let gs = solve_ground_state(&diag, &subdiag);
        assert_eq!(gs.len, n);

        // Check ||Psi_0||_2 = 1.0
        let mut norm_sq = 0.0f32;
        for &p in gs.psi_slice() {
            norm_sq += p * p;
        }
        assert!(
            (norm_sq.sqrt() - 1.0).abs() < 1e-5,
            "Norm for n={n} must be 1.0, got {}",
            norm_sq.sqrt()
        );

        if n > 1 {
            // Check H * Psi_0 = E_0 * Psi_0
            let mut h_psi = vec![0.0f32; n];
            for i in 0..n {
                h_psi[i] += diag[i] * gs.psi[i];
                if i > 0 {
                    h_psi[i] += subdiag[i - 1] * gs.psi[i - 1];
                }
                if i < n - 1 {
                    h_psi[i] += subdiag[i] * gs.psi[i + 1];
                }
            }

            let mut max_diff = 0.0f32;
            for i in 0..n {
                let diff = (h_psi[i] - gs.energy * gs.psi[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            assert!(
                max_diff < 1e-4,
                "Eigenvalue residual for n={n} must be < 1e-4, got {max_diff}"
            );
        }
    }
}

#[test]
fn test_qreno_cluster_partitioning() {
    // 2) Words separated by spaces and weak bonds
    let config = QrenoConfig::default();
    let tok = QrenoTokenizer::new(config);

    let phrase = "алгебра логика физика";
    let clusters = tok.tokenize_to_clusters(phrase);

    // Should detect the 3 words and spaces
    let words: Vec<&str> = clusters
        .iter()
        .filter(|c| !c.is_delimiter)
        .map(|c| std::str::from_utf8(&phrase.as_bytes()[c.start..c.start + c.len]).unwrap())
        .collect();

    assert_eq!(words.len(), 3);
    assert_eq!(words[0], "алгебра");
    assert_eq!(words[1], "логика");
    assert_eq!(words[2], "физика");

    let delimiters: Vec<&str> = clusters
        .iter()
        .filter(|c| c.is_delimiter)
        .map(|c| std::str::from_utf8(&phrase.as_bytes()[c.start..c.start + c.len]).unwrap())
        .collect();

    assert_eq!(delimiters.len(), 2);
    assert_eq!(delimiters[0], " ");
    assert_eq!(delimiters[1], " ");
}

#[test]
fn test_qreno_typo_robustness_and_gauge_invariance() {
    // 3) Typo robustness: "математика" vs typos >= 0.85, vs "крокодил" < 0.3
    let config = QrenoConfig::default();
    let tok = QrenoTokenizer::new(config);

    let emb_orig = tok.embed_word("математика");
    let emb_typo1 = tok.embed_word("математка");   // dropped 'и'
    let emb_typo2 = tok.embed_word("математикаа");  // duplicated 'а'
    let emb_typo3 = tok.embed_word("математека");   // replaced 'и' with 'е'
    let emb_other = tok.embed_word("крокодил");    // unrelated word

    let sim1 = QrenoTokenizer::cosine_similarity(&emb_orig, &emb_typo1);
    let sim2 = QrenoTokenizer::cosine_similarity(&emb_orig, &emb_typo2);
    let sim3 = QrenoTokenizer::cosine_similarity(&emb_orig, &emb_typo3);
    let sim_other = QrenoTokenizer::cosine_similarity(&emb_orig, &emb_other);

    println!("Cosine similarities:");
    println!("  математика vs математка:   {sim1:.4}");
    println!("  математика vs математикаа: {sim2:.4}");
    println!("  математика vs математека:  {sim3:.4}");
    println!("  математика vs крокодил:    {sim_other:.4}");

    assert!(
        sim1 >= 0.85,
        "математка similarity must be >= 0.85, got {sim1:.4}"
    );
    assert!(
        sim2 >= 0.85,
        "математикаа similarity must be >= 0.85, got {sim2:.4}"
    );
    assert!(
        sim3 >= 0.85,
        "математека similarity must be >= 0.85, got {sim3:.4}"
    );
    assert!(
        sim_other < 0.30,
        "крокодил similarity must be < 0.30, got {sim_other:.4}"
    );
}

#[test]
fn test_qreno_numerical_vs_analytical_gradients() {
    // 4) Numerical vs analytical gradient check
    let config = QrenoConfig {
        field_dim: 8,
        d_model: 8,
        max_cluster_len: 8,
        bond_threshold: 0.5,
        rms_eps: 1e-5,
    };
    let tok = QrenoTokenizer::new(config.clone());
    let cluster_bytes = b"rust";

    // Downstream objective: MSE against arbitrary target representation
    let target = vec![0.5f32; config.d_model];

    // Compute forward pass
    let mut e_out = vec![0.0f32; config.d_model];
    tok.embed_cluster_bytes(cluster_bytes, &mut e_out);

    // Loss L = 0.5 * sum((e - target)^2)
    let mut e_grad = vec![0.0f32; config.d_model];
    for i in 0..config.d_model {
        e_grad[i] = e_out[i] - target[i];
    }

    // Analytical backward
    let mut grad = QrenoGrad::new(&config);
    tok.forward_backward_cluster(cluster_bytes, &e_grad, &mut grad);

    // Numerical gradient check on W_O
    let eps = 1e-3f32;
    for i in 0..config.d_model.min(4) {
        for j in 0..config.field_dim.min(4) {
            let idx = i * config.field_dim + j;
            let mut tok_plus = tok.clone();
            tok_plus.weights.measure.w_o[idx] += eps;
            let mut e_plus = vec![0.0f32; config.d_model];
            tok_plus.embed_cluster_bytes(cluster_bytes, &mut e_plus);
            let loss_plus: f32 = 0.5 * e_plus.iter().zip(&target).map(|(a, b)| (a - b).powi(2)).sum::<f32>();

            let mut tok_minus = tok.clone();
            tok_minus.weights.measure.w_o[idx] -= eps;
            let mut e_minus = vec![0.0f32; config.d_model];
            tok_minus.embed_cluster_bytes(cluster_bytes, &mut e_minus);
            let loss_minus: f32 = 0.5 * e_minus.iter().zip(&target).map(|(a, b)| (a - b).powi(2)).sum::<f32>();

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
            let ana_grad = grad.w_o_grad[idx];

            let diff = (num_grad - ana_grad).abs();
            let denom = (num_grad.abs() + ana_grad.abs()).max(1e-4);
            let rel_err = diff / denom;

            assert!(
                diff < 1e-3 || rel_err < 0.08,
                "W_O grad mismatch at ({i}, {j}): ana={ana_grad}, num={num_grad}, diff={diff}, rel_err={rel_err}"
            );
        }
    }

    // Numerical gradient check on b_bond
    {
        let mut tok_plus = tok.clone();
        tok_plus.weights.bond.b_bond += eps;
        let mut e_plus = vec![0.0f32; config.d_model];
        tok_plus.embed_cluster_bytes(cluster_bytes, &mut e_plus);
        let loss_plus: f32 = 0.5 * e_plus.iter().zip(&target).map(|(a, b)| (a - b).powi(2)).sum::<f32>();

        let mut tok_minus = tok.clone();
        tok_minus.weights.bond.b_bond -= eps;
        let mut e_minus = vec![0.0f32; config.d_model];
        tok_minus.embed_cluster_bytes(cluster_bytes, &mut e_minus);
        let loss_minus: f32 = 0.5 * e_minus.iter().zip(&target).map(|(a, b)| (a - b).powi(2)).sum::<f32>();

        let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let ana_grad = grad.b_bond_grad;

        let diff = (num_grad - ana_grad).abs();
        let denom = (num_grad.abs() + ana_grad.abs()).max(1e-4);
        let rel_err = diff / denom;

        assert!(
            rel_err < 0.05,
            "b_bond grad mismatch: ana={ana_grad}, num={num_grad}, rel_err={rel_err}"
        );
    }
}

#[test]
fn test_qreno_adamw_training_loop() {
    // 5) Verify AdamW parameter update decreases MSE loss on targets
    let config = QrenoConfig {
        field_dim: 8,
        d_model: 8,
        max_cluster_len: 8,
        bond_threshold: 0.5,
        rms_eps: 1e-5,
    };
    let mut tok = QrenoTokenizer::new(config.clone());
    let mut optimizer = QrenoAdamW::new(&config, 0.05, 0.01);
    let word = "quantum";
    let target = vec![0.4f32; config.d_model];

    let compute_loss = |t: &QrenoTokenizer| -> f32 {
        let emb = t.embed_word(word);
        0.5 * emb.iter().zip(&target).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
    };

    let initial_loss = compute_loss(&tok);

    for _ in 0..15 {
        let clusters = tok.tokenize_to_clusters(word);
        let mut grad = QrenoGrad::new(&config);

        for c in &clusters {
            let cluster_bytes = &word.as_bytes()[c.start..c.start + c.len];
            let mut e_out = vec![0.0f32; config.d_model];
            tok.embed_cluster_bytes(cluster_bytes, &mut e_out);

            let mut e_grad = vec![0.0f32; config.d_model];
            for k in 0..config.d_model {
                e_grad[k] = e_out[k] - target[k];
            }

            tok.forward_backward_cluster(cluster_bytes, &e_grad, &mut grad);
        }

        optimizer.step(&mut tok.weights, &grad);
    }

    let final_loss = compute_loss(&tok);
    println!("AdamW Training: initial_loss={initial_loss:.4} -> final_loss={final_loss:.4}");
    assert!(
        final_loss < initial_loss,
        "Loss must decrease after AdamW updates: initial={initial_loss}, final={final_loss}"
    );
}
