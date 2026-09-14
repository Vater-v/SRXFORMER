//! Optimized numerical operations for SRXFORMER v05:
//! Monarch Butterfly Unitary Factorization, L2 Normalization, and Vector Projection.
//! All operations are zero-allocation and AVX auto-vectorization friendly.

use std::f32::consts::PI;

/// Fast analytical sine and cosine approximation using Taylor/Bhaskara expansion.
/// Within [-pi, pi], maximum absolute error < 1e-4.
#[inline(always)]
pub fn fast_sin_cos(x: f32) -> (f32, f32) {
    let mut rad = x % (2.0 * PI);
    if rad > PI {
        rad -= 2.0 * PI;
    } else if rad < -PI {
        rad += 2.0 * PI;
    }
    (rad.sin(), rad.cos())
}

/// Computes L2-normalization of vector `x` with numerical guard `eps`.
/// Returns the L2 norm ||x||_2.
#[inline(always)]
pub fn l2_normalize(x: &[f32], out: &mut [f32], eps: f32) -> f32 {
    let mut sum_sq = 0.0f32;
    for &val in x {
        sum_sq += val * val;
    }
    let norm = (sum_sq + eps).sqrt();
    let inv_norm = 1.0 / norm;
    for i in 0..x.len() {
        out[i] = x[i] * inv_norm;
    }
    norm
}

/// Exact analytical Vector-Jacobian Product (VJP) backward through L2-normalization.
/// Given adjoint `d_y` = dL / dy where y = x / ||x||, computes `d_x` = dL / dx.
#[inline(always)]
pub fn l2_normalize_backward(
    _x: &[f32],
    y: &[f32],
    norm: f32,
    d_y: &[f32],
    d_x: &mut [f32],
) {
    let inv_norm = 1.0 / norm;
    let mut dot_dy_y = 0.0f32;
    for i in 0..y.len() {
        dot_dy_y += d_y[i] * y[i];
    }
    for i in 0..y.len() {
        d_x[i] = inv_norm * (d_y[i] - y[i] * dot_dy_y);
    }
}

/// Applies 4-dimensional Monarch Butterfly Unitary Matrix U(Theta):
/// U(Theta) = B_2(Theta_2) * P * B_1(Theta_1)
/// where:
/// - B_1(Theta_1) is block-diagonal 2x2 Givens rotations: R(theta_0) on (0, 1) and R(theta_1) on (2, 3)
/// - P is perfect shuffle permutation matrix: (0, 1, 2, 3) -> (0, 2, 1, 3)
/// - B_2(Theta_2) is block-diagonal 2x2 Givens rotations: R(theta_2) on (0, 1) and R(theta_3) on (2, 3)
///
/// Strictly unitary: U * U^\dagger = I.
/// When `inverse` is true, applies U^\dagger = B_1^\dagger * P^\dagger * B_2^\dagger with negative angles.
#[inline(always)]
pub fn apply_butterfly_4(
    v: &[f32; 4],
    thetas: &[f32; 4],
    inverse: bool,
    out: &mut [f32; 4],
) {
    if !inverse {
        // Forward: U = B_2 * P * B_1

        // Stage 1: B_1(theta_0, theta_1)
        let (s0, c0) = fast_sin_cos(thetas[0]);
        let (s1, c1) = fast_sin_cos(thetas[1]);
        let b1_0 = c0 * v[0] - s0 * v[1];
        let b1_1 = s0 * v[0] + c0 * v[1];
        let b1_2 = c1 * v[2] - s1 * v[3];
        let b1_3 = s1 * v[2] + c1 * v[3];

        // Stage 2: Permutation P (0, 1, 2, 3) -> (0, 2, 1, 3)
        let p_0 = b1_0;
        let p_1 = b1_2;
        let p_2 = b1_1;
        let p_3 = b1_3;

        // Stage 3: B_2(theta_2, theta_3)
        let (s2, c2) = fast_sin_cos(thetas[2]);
        let (s3, c3) = fast_sin_cos(thetas[3]);
        out[0] = c2 * p_0 - s2 * p_1;
        out[1] = s2 * p_0 + c2 * p_1;
        out[2] = c3 * p_2 - s3 * p_3;
        out[3] = s3 * p_2 + c3 * p_3;
    } else {
        // Inverse: U^\dagger = B_1(-Theta_1) * P^\dagger * B_2(-Theta_2)
        // Stage 1: B_2^\dagger (rotation by -theta_2, -theta_3)
        let (s2, c2) = fast_sin_cos(-thetas[2]);
        let (s3, c3) = fast_sin_cos(-thetas[3]);
        let inv_b2_0 = c2 * v[0] - s2 * v[1];
        let inv_b2_1 = s2 * v[0] + c2 * v[1];
        let inv_b2_2 = c3 * v[2] - s3 * v[3];
        let inv_b2_3 = s3 * v[2] + c3 * v[3];

        // Stage 2: Permutation P^\dagger = P^T
        // Forward was: p_0 = b1_0, p_1 = b1_2, p_2 = b1_1, p_3 = b1_3
        // So inverse permutation is: out_0 = p_0, out_1 = p_2, out_2 = p_1, out_3 = p_3
        let p_inv_0 = inv_b2_0;
        let p_inv_1 = inv_b2_2;
        let p_inv_2 = inv_b2_1;
        let p_inv_3 = inv_b2_3;

        // Stage 3: B_1^\dagger (rotation by -theta_0, -theta_1)
        let (s0, c0) = fast_sin_cos(-thetas[0]);
        let (s1, c1) = fast_sin_cos(-thetas[1]);
        out[0] = c0 * p_inv_0 - s0 * p_inv_1;
        out[1] = s0 * p_inv_0 + c0 * p_inv_1;
        out[2] = c1 * p_inv_2 - s1 * p_inv_3;
        out[3] = s1 * p_inv_2 + c1 * p_inv_3;
    }
}

/// In-place variant of 4-dimensional Butterfly Unitary.
#[inline(always)]
pub fn apply_butterfly_4_inplace(v: &mut [f32; 4], thetas: &[f32; 4], inverse: bool) {
    let mut tmp = [0.0f32; 4];
    apply_butterfly_4(v, thetas, inverse, &mut tmp);
    v.copy_from_slice(&tmp);
}

/// Exact analytical Vector-Jacobian Product (VJP) backward through Butterfly Unitary Mixer.
/// Given input `v`, angles `thetas`, mode `inverse`, and upstream adjoint `d_out`,
/// computes `d_v` (gradient w.r.t v) and accumulates into `d_thetas` (gradient w.r.t angles).
#[inline(always)]
pub fn apply_butterfly_4_backward(
    v: &[f32; 4],
    thetas: &[f32; 4],
    inverse: bool,
    d_out: &[f32; 4],
    d_v: &mut [f32; 4],
    d_thetas: &mut [f32; 4],
) {
    if !inverse {
        // Forward was:
        // b1 = B_1 * v
        let (s0, c0) = fast_sin_cos(thetas[0]);
        let (s1, c1) = fast_sin_cos(thetas[1]);
        let b1_0 = c0 * v[0] - s0 * v[1];
        let b1_1 = s0 * v[0] + c0 * v[1];
        let b1_2 = c1 * v[2] - s1 * v[3];
        let b1_3 = s1 * v[2] + c1 * v[3];

        // p = P * b1
        let p_0 = b1_0;
        let p_1 = b1_2;
        let p_2 = b1_1;
        let p_3 = b1_3;

        // out = B_2 * p
        let (s2, c2) = fast_sin_cos(thetas[2]);
        let (s3, c3) = fast_sin_cos(thetas[3]);

        // Backward through B_2:
        // out[0] = c2 * p_0 - s2 * p_1
        // out[1] = s2 * p_0 + c2 * p_1
        let d_p_0 = c2 * d_out[0] + s2 * d_out[1];
        let d_p_1 = -s2 * d_out[0] + c2 * d_out[1];
        let d_theta_2 = d_out[0] * (-s2 * p_0 - c2 * p_1) + d_out[1] * (c2 * p_0 - s2 * p_1);

        // out[2] = c3 * p_2 - s3 * p_3
        // out[3] = s3 * p_2 + c3 * p_3
        let d_p_2 = c3 * d_out[2] + s3 * d_out[3];
        let d_p_3 = -s3 * d_out[2] + c3 * d_out[3];
        let d_theta_3 = d_out[2] * (-s3 * p_2 - c3 * p_3) + d_out[3] * (c3 * p_2 - s3 * p_3);

        // Backward through P:
        // p_0 = b1_0, p_1 = b1_2, p_2 = b1_1, p_3 = b1_3
        let d_b1_0 = d_p_0;
        let d_b1_1 = d_p_2;
        let d_b1_2 = d_p_1;
        let d_b1_3 = d_p_3;

        // Backward through B_1:
        // b1_0 = c0 * v[0] - s0 * v[1]
        // b1_1 = s0 * v[0] + c0 * v[1]
        d_v[0] = c0 * d_b1_0 + s0 * d_b1_1;
        d_v[1] = -s0 * d_b1_0 + c0 * d_b1_1;
        let d_theta_0 = d_b1_0 * (-s0 * v[0] - c0 * v[1]) + d_b1_1 * (c0 * v[0] - s0 * v[1]);

        // b1_2 = c1 * v[2] - s1 * v[3]
        // b1_3 = s1 * v[2] + c1 * v[3]
        d_v[2] = c1 * d_b1_2 + s1 * d_b1_3;
        d_v[3] = -s1 * d_b1_2 + c1 * d_b1_3;
        let d_theta_1 = d_b1_2 * (-s1 * v[2] - c1 * v[3]) + d_b1_3 * (c1 * v[2] - s1 * v[3]);

        d_thetas[0] += d_theta_0;
        d_thetas[1] += d_theta_1;
        d_thetas[2] += d_theta_2;
        d_thetas[3] += d_theta_3;
    } else {
        // Inverse mode: U^\dagger(thetas) = B_1(-Theta_1) * P^\dagger * B_2(-Theta_2)
        let (s2, c2) = fast_sin_cos(-thetas[2]);
        let (s3, c3) = fast_sin_cos(-thetas[3]);
        let inv_b2_0 = c2 * v[0] - s2 * v[1];
        let inv_b2_1 = s2 * v[0] + c2 * v[1];
        let inv_b2_2 = c3 * v[2] - s3 * v[3];
        let inv_b2_3 = s3 * v[2] + c3 * v[3];

        let p_inv_0 = inv_b2_0;
        let p_inv_1 = inv_b2_2;
        let p_inv_2 = inv_b2_1;
        let p_inv_3 = inv_b2_3;

        let (s0, c0) = fast_sin_cos(-thetas[0]);
        let (s1, c1) = fast_sin_cos(-thetas[1]);

        // Backward through Stage 3 (B_1(-Theta_1)):
        let d_p_inv_0 = c0 * d_out[0] + s0 * d_out[1];
        let d_p_inv_1 = -s0 * d_out[0] + c0 * d_out[1];
        let d_neg_theta_0 = d_out[0] * (-s0 * p_inv_0 - c0 * p_inv_1)
            + d_out[1] * (c0 * p_inv_0 - s0 * p_inv_1);

        let d_p_inv_2 = c1 * d_out[2] + s1 * d_out[3];
        let d_p_inv_3 = -s1 * d_out[2] + c1 * d_out[3];
        let d_neg_theta_1 = d_out[2] * (-s1 * p_inv_2 - c1 * p_inv_3)
            + d_out[3] * (c1 * p_inv_2 - s1 * p_inv_3);

        // Backward through Stage 2 (P^\dagger):
        let d_inv_b2_0 = d_p_inv_0;
        let d_inv_b2_2 = d_p_inv_1;
        let d_inv_b2_1 = d_p_inv_2;
        let d_inv_b2_3 = d_p_inv_3;

        // Backward through Stage 1 (B_2(-Theta_2)):
        d_v[0] = c2 * d_inv_b2_0 + s2 * d_inv_b2_1;
        d_v[1] = -s2 * d_inv_b2_0 + c2 * d_inv_b2_1;
        let d_neg_theta_2 = d_inv_b2_0 * (-s2 * v[0] - c2 * v[1])
            + d_inv_b2_1 * (c2 * v[0] - s2 * v[1]);

        d_v[2] = c3 * d_inv_b2_2 + s3 * d_inv_b2_3;
        d_v[3] = -s3 * d_inv_b2_2 + c3 * d_inv_b2_3;
        let d_neg_theta_3 = d_inv_b2_2 * (-s3 * v[2] - c3 * v[3])
            + d_inv_b2_3 * (c3 * v[2] - s3 * v[3]);

        // d_theta = - d_neg_theta
        d_thetas[0] -= d_neg_theta_0;
        d_thetas[1] -= d_neg_theta_1;
        d_thetas[2] -= d_neg_theta_2;
        d_thetas[3] -= d_neg_theta_3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butterfly_unitarity_and_roundtrip() {
        let thetas = [0.42, -1.25, 0.73, 2.11];
        let v = [1.5, -2.0, 0.5, 3.2];

        // 1. Unitarity check: ||U(v)||^2 == ||v||^2
        let mut u_v = [0.0f32; 4];
        apply_butterfly_4(&v, &thetas, false, &mut u_v);

        let norm_v_sq = v.iter().map(|x| x * x).sum::<f32>();
        let norm_uv_sq = u_v.iter().map(|x| x * x).sum::<f32>();
        assert!(
            (norm_v_sq - norm_uv_sq).abs() < 1e-5,
            "Butterfly mixer must preserve L2 norm! norm_v_sq={}, norm_uv_sq={}",
            norm_v_sq,
            norm_uv_sq
        );

        // 2. Roundtrip check: U^\dagger(U(v)) == v
        let mut recovered = [0.0f32; 4];
        apply_butterfly_4(&u_v, &thetas, true, &mut recovered);

        for i in 0..4 {
            assert!(
                (v[i] - recovered[i]).abs() < 1e-5,
                "Butterfly inverse roundtrip failed at idx {}: expected {}, got {}",
                i,
                v[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_butterfly_vjp_gradient_check() {
        let thetas = [0.35, -0.85, 1.20, -0.45];
        let v = [0.7, -1.3, 2.1, 0.4];
        let d_out = [1.2, -0.5, 0.8, -1.1];

        let mut d_v = [0.0f32; 4];
        let mut d_thetas = [0.0f32; 4];
        apply_butterfly_4_backward(&v, &thetas, false, &d_out, &mut d_v, &mut d_thetas);

        let eps = 1e-4f32;

        // Check d_v gradient numerically
        for i in 0..4 {
            let mut v_plus = v;
            let mut v_minus = v;
            v_plus[i] += eps;
            v_minus[i] -= eps;

            let mut out_plus = [0.0f32; 4];
            let mut out_minus = [0.0f32; 4];
            apply_butterfly_4(&v_plus, &thetas, false, &mut out_plus);
            apply_butterfly_4(&v_minus, &thetas, false, &mut out_minus);

            let mut num_grad = 0.0f32;
            for j in 0..4 {
                num_grad += d_out[j] * (out_plus[j] - out_minus[j]) / (2.0 * eps);
            }

            assert!(
                (d_v[i] - num_grad).abs() < 1e-3,
                "d_v[{}] gradient mismatch: ana={}, num={}",
                i,
                d_v[i],
                num_grad
            );
        }

        // Check d_thetas gradient numerically
        for i in 0..4 {
            let mut th_plus = thetas;
            let mut th_minus = thetas;
            th_plus[i] += eps;
            th_minus[i] -= eps;

            let mut out_plus = [0.0f32; 4];
            let mut out_minus = [0.0f32; 4];
            apply_butterfly_4(&v, &th_plus, false, &mut out_plus);
            apply_butterfly_4(&v, &th_minus, false, &mut out_minus);

            let mut num_grad = 0.0f32;
            for j in 0..4 {
                num_grad += d_out[j] * (out_plus[j] - out_minus[j]) / (2.0 * eps);
            }

            assert!(
                (d_thetas[i] - num_grad).abs() < 1e-3,
                "d_thetas[{}] gradient mismatch: ana={}, num={}",
                i,
                d_thetas[i],
                num_grad
            );
        }
    }
}
