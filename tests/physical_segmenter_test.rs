use srxformer::qreno::{
    charge_kl_divergence, morse_bond_energy, physical_segment, QrenoField, SshLatticeParams,
    SshLatticeWorkspace,
};

#[test]
fn test_morse_bond_physics_continuity() {
    let d_e = 2.5f32;
    let a = 1.8f32;
    let r_0 = 0.25f32;

    // Minimum distance / compressed lattice: repulsive wall
    let e_repulsive = morse_bond_energy(0.05, d_e, a, r_0);
    // Equilibrium distance: maximum binding energy
    let e_eq = morse_bond_energy(r_0, d_e, a, r_0);
    // Dissociated distance: approaching zero
    let e_dissoc = morse_bond_energy(r_0 + 3.0, d_e, a, r_0);

    assert!(e_eq > e_repulsive, "Equilibrium must have higher bound energy than compressed wall");
    assert!(e_eq > e_dissoc, "Equilibrium must have higher bound energy than dissociated limit");
    assert!((e_eq - d_e).abs() < 1e-5);
    assert!(e_dissoc < 0.1, "At r >> r0 energy must asymptotically approach 0");
}

#[test]
fn test_charge_kl_divergence_properties() {
    let c1 = vec![0.5f32; 8];
    let c2 = vec![0.5f32; 8];

    let kl_ident = charge_kl_divergence(&c1, &c2);
    assert!(kl_ident < 1e-5, "KL divergence of identical charges must be ~0: {}", kl_ident);

    let mut c3 = vec![0.1f32; 8];
    c3[0] = 2.0;
    let kl_diff = charge_kl_divergence(&c1, &c3);
    assert!(kl_diff > 0.1, "KL divergence of distinct charges must be positive: {}", kl_diff);
}

#[test]
fn test_physical_segmentation_russian_folklore() {
    let field = QrenoField::new(8);
    let params = SshLatticeParams::default();
    let mut ws = SshLatticeWorkspace::new();

    let text = "хочешь сей а хочешь куй";
    let clusters = physical_segment(text.as_bytes(), &field, &params, &mut ws);

    assert!(!clusters.is_empty());
    // Verify complete coverage without gaps
    let mut total_bytes = 0;
    for (idx, c) in clusters.iter().enumerate() {
        assert_eq!(c.start, total_bytes, "Cluster {} has discontinuous start offset", idx);
        total_bytes += c.len;
    }
    assert_eq!(total_bytes, text.as_bytes().len(), "Total cluster bytes must match text length exactly");
}

#[test]
fn test_physical_segmentation_typo_resilience() {
    let field = QrenoField::new(8);
    let params = SshLatticeParams::default();
    let mut ws = SshLatticeWorkspace::new();

    // Original vs Typo
    let text1 = "математика";
    let text2 = "математка";

    let c1 = physical_segment(text1.as_bytes(), &field, &params, &mut ws);
    let c2 = physical_segment(text2.as_bytes(), &field, &params, &mut ws);

    // Both should remain unified words or clean morphemes, not fragmented into single letters
    assert!(c1.len() <= 2, "Original word should not over-segment: {:?}", c1);
    assert!(c2.len() <= 2, "Typo word should not over-segment: {:?}", c2);
}
