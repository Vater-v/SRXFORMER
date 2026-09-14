//! Low-level mathematical operations for SRXFORMER:
//! Givens rotation chains (forward & analytical VJP backward),
//! L2 vector normalization (forward & analytical VJP backward).

/// Applies a chain of (d - 1) Givens rotations to vector `x` in-place.
///
/// If `inverse` is false:
///   U(Theta) = G_0(theta_0) G_1(theta_1) ... G_{d-2}(theta_{d-2})
///   Rotating coordinate pair (i, i+1) by angle theta_i for i = 0 .. d-2.
///
/// If `inverse` is true:
///   U^\dagger(Theta) = G_{d-2}(-theta_{d-2}) ... G_0(-theta_0)
///   Rotating coordinate pair (i, i+1) by -theta_i for i = d-2 down to 0.
#[inline]
pub fn apply_givens(x: &mut [f32], thetas: &[f32], inverse: bool) {
    let d = x.len();
    if d <= 1 {
        return;
    }
    assert_eq!(thetas.len(), d - 1, "thetas length must be d - 1");

    if !inverse {
        for i in 0..d - 1 {
            let theta = thetas[i];
            let c = theta.cos();
            let s = theta.sin();
            let xi = x[i];
            let xip = x[i + 1];
            x[i] = c * xi - s * xip;
            x[i + 1] = s * xi + c * xip;
        }
    } else {
        for i in (0..d - 1).rev() {
            let theta = thetas[i];
            let c = theta.cos();
            let s = -theta.sin(); // sin(-theta) = -sin(theta)
            let xi = x[i];
            let xip = x[i + 1];
            x[i] = c * xi - s * xip;
            x[i + 1] = s * xi + c * xip;
        }
    }
}

/// Forward Givens rotation recording all intermediate states for exact VJP backpropagation.
/// `intermediates` buffer must have length `d * d` (storing d vectors of length d: step 0 to step d-1).
#[inline]
pub fn apply_givens_forward_with_intermediates(
    x: &[f32],
    thetas: &[f32],
    inverse: bool,
    intermediates: &mut [f32],
) {
    let d = x.len();
    assert_eq!(thetas.len(), d - 1);
    assert_eq!(intermediates.len(), d * d);

    // Step 0: copy initial input
    intermediates[..d].copy_from_slice(x);

    if !inverse {
        for step in 0..d - 1 {
            let prev = step * d;
            let next = (step + 1) * d;
            for c in 0..d {
                intermediates[next + c] = intermediates[prev + c];
            }

            let theta = thetas[step];
            let c = theta.cos();
            let s = theta.sin();
            let xi = intermediates[prev + step];
            let xip = intermediates[prev + step + 1];
            intermediates[next + step] = c * xi - s * xip;
            intermediates[next + step + 1] = s * xi + c * xip;
        }
    } else {
        for (step_idx, i) in (0..d - 1).rev().enumerate() {
            let prev = step_idx * d;
            let next = (step_idx + 1) * d;
            for c in 0..d {
                intermediates[next + c] = intermediates[prev + c];
            }

            let theta = thetas[i];
            let c = theta.cos();
            let s = -theta.sin();
            let xi = intermediates[prev + i];
            let xip = intermediates[prev + i + 1];
            intermediates[next + i] = c * xi - s * xip;
            intermediates[next + i + 1] = s * xi + c * xip;
        }
    }
}

/// Analytical Vector-Jacobian Product (VJP) backward pass through Givens rotations.
/// Computes adjoints with respect to input `d_x` and accumulated into `d_thetas`.
#[inline]
pub fn apply_givens_backward(
    intermediates: &[f32],
    thetas: &[f32],
    inverse: bool,
    d_out: &[f32],
    d_x: &mut [f32],
    d_thetas: &mut [f32],
) {
    let d = d_out.len();
    assert_eq!(thetas.len(), d - 1);
    assert_eq!(d_x.len(), d);
    assert_eq!(d_thetas.len(), d - 1);

    // We start from the output adjoint at step d - 1
    d_x.copy_from_slice(d_out);

    if !inverse {
        for step in (0..d - 1).rev() {
            let prev = step * d;
            let theta = thetas[step];
            let c = theta.cos();
            let s = theta.sin();

            let d_xi_next = d_x[step];
            let d_xip_next = d_x[step + 1];

            let xi_prev = intermediates[prev + step];
            let xip_prev = intermediates[prev + step + 1];

            // Gradient w.r.t input pair
            let d_xi_prev = c * d_xi_next + s * d_xip_next;
            let d_xip_prev = -s * d_xi_next + c * d_xip_next;

            // Gradient w.r.t theta_step:
            // d(xi_next)/d(theta) = -s * xi - c * xip
            // d(xip_next)/d(theta) = c * xi - s * xip
            let d_theta = d_xi_next * (-s * xi_prev - c * xip_prev)
                + d_xip_next * (c * xi_prev - s * xip_prev);

            d_x[step] = d_xi_prev;
            d_x[step + 1] = d_xip_prev;
            d_thetas[step] += d_theta;
        }
    } else {
        for step_idx in (0..d - 1).rev() {
            let i = (d - 2) - step_idx;
            let prev = step_idx * d;
            let theta = thetas[i];
            let c = theta.cos();
            let s = -theta.sin(); // sin(-theta)

            let d_xi_next = d_x[i];
            let d_xip_next = d_x[i + 1];

            let xi_prev = intermediates[prev + i];
            let xip_prev = intermediates[prev + i + 1];

            let d_xi_prev = c * d_xi_next + s * d_xip_next;
            let d_xip_prev = -s * d_xi_next + c * d_xip_next;

            // Derivative w.r.t (-theta):
            let d_phi = d_xi_next * (-s * xi_prev - c * xip_prev)
                + d_xip_next * (c * xi_prev - s * xip_prev);
            // phi = -theta => d_theta = -d_phi
            let d_theta = -d_phi;

            d_x[i] = d_xi_prev;
            d_x[i + 1] = d_xip_prev;
            d_thetas[i] += d_theta;
        }
    }
}

/// L2 normalization of vector `x`.
/// Returns the Euclidean norm.
#[inline]
pub fn l2_normalize(x: &[f32], out: &mut [f32], eps: f32) -> f32 {
    let mut sum_sq = 0.0f32;
    for &val in x {
        sum_sq += val * val;
    }
    let norm = (sum_sq + eps).sqrt();
    let inv_norm = 1.0 / norm;
    for (o, &val) in out.iter_mut().zip(x.iter()) {
        *o = val * inv_norm;
    }
    norm
}

/// Analytical VJP backward for L2 normalization.
/// If u = x / norm, then:
/// d_x = (1 / norm) * (d_u - (d_u \cdot u) * u)
#[inline]
pub fn l2_normalize_backward(
    _x: &[f32],
    u: &[f32],
    norm: f32,
    d_u: &[f32],
    d_x: &mut [f32],
) {
    let mut dot_du_u = 0.0f32;
    for (&du, &unit) in d_u.iter().zip(u.iter()) {
        dot_du_u += du * unit;
    }

    let inv_norm = 1.0 / norm;
    for (dx, (&du, &unit)) in d_x.iter_mut().zip(d_u.iter().zip(u.iter())) {
        *dx = inv_norm * (du - dot_du_u * unit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_givens_unitarity() {
        let thetas = [0.35f32, -0.82f32, 1.21f32];
        let original = [1.0f32, -2.0f32, 0.5f32, 3.2f32];
        let mut v = original;

        // Apply forward rotation U
        apply_givens(&mut v, &thetas, false);

        // Check norm preservation
        let orig_norm: f32 = original.iter().map(|x| x * x).sum();
        let rot_norm: f32 = v.iter().map(|x| x * x).sum();
        assert!(
            (orig_norm - rot_norm).abs() < 1e-5,
            "Norm must be preserved by unitary Givens rotations: orig={}, rot={}",
            orig_norm,
            rot_norm
        );

        // Apply inverse rotation U^\dagger
        apply_givens(&mut v, &thetas, true);

        // Must recover original vector within machine precision
        for (i, (&orig, &recovered)) in original.iter().zip(v.iter()).enumerate() {
            assert!(
                (orig - recovered).abs() < 1e-5,
                "Mismatch at coord {}: orig={}, recovered={}",
                i,
                orig,
                recovered
            );
        }
    }

    #[test]
    fn test_givens_vjp_gradient_check() {
        let thetas = [0.4f32, -0.6f32, 0.8f32];
        let x = [0.5f32, -1.2f32, 0.9f32, 2.1f32];
        let d = x.len();

        let mut intermediates = vec![0.0f32; d * d];
        apply_givens_forward_with_intermediates(&x, &thetas, false, &mut intermediates);

        let d_out = [1.0f32, -0.5f32, 0.3f32, -0.8f32];

        let mut d_x = vec![0.0f32; d];
        let mut d_thetas = vec![0.0f32; thetas.len()];
        apply_givens_backward(&intermediates, &thetas, false, &d_out, &mut d_x, &mut d_thetas);

        // Numerical verification for theta gradients
        let eps = 1e-4f32;
        for i in 0..thetas.len() {
            let mut thetas_plus = thetas;
            let mut thetas_minus = thetas;
            thetas_plus[i] += eps;
            thetas_minus[i] -= eps;

            let mut x_plus = x;
            let mut x_minus = x;
            apply_givens(&mut x_plus, &thetas_plus, false);
            apply_givens(&mut x_minus, &thetas_minus, false);

            let mut dot_plus = 0.0f32;
            let mut dot_minus = 0.0f32;
            for j in 0..d {
                dot_plus += d_out[j] * x_plus[j];
                dot_minus += d_out[j] * x_minus[j];
            }

            let num_grad = (dot_plus - dot_minus) / (2.0 * eps);
            assert!(
                (d_thetas[i] - num_grad).abs() < 2e-3,
                "Theta {} grad mismatch: analytical={}, numerical={}",
                i,
                d_thetas[i],
                num_grad
            );
        }

        // Numerical verification for x gradients
        for i in 0..d {
            let mut x_plus = x;
            let mut x_minus = x;
            x_plus[i] += eps;
            x_minus[i] -= eps;

            apply_givens(&mut x_plus, &thetas, false);
            apply_givens(&mut x_minus, &thetas, false);

            let mut dot_plus = 0.0f32;
            let mut dot_minus = 0.0f32;
            for j in 0..d {
                dot_plus += d_out[j] * x_plus[j];
                dot_minus += d_out[j] * x_minus[j];
            }

            let num_grad = (dot_plus - dot_minus) / (2.0 * eps);
            assert!(
                (d_x[i] - num_grad).abs() < 1e-3,
                "x {} grad mismatch: analytical={}, numerical={}",
                i,
                d_x[i],
                num_grad
            );
        }
    }

    #[test]
    fn test_givens_inverse_vjp_gradient_check() {
        let thetas = [0.25f32, -0.45f32, 0.75f32];
        let x = [1.5f32, -0.8f32, 0.3f32, 1.1f32];
        let d = x.len();

        let mut intermediates = vec![0.0f32; d * d];
        apply_givens_forward_with_intermediates(&x, &thetas, true, &mut intermediates);

        let d_out = [0.7f32, -1.1f32, 0.4f32, -0.2f32];

        let mut d_x = vec![0.0f32; d];
        let mut d_thetas = vec![0.0f32; thetas.len()];
        apply_givens_backward(&intermediates, &thetas, true, &d_out, &mut d_x, &mut d_thetas);

        let eps = 1e-4f32;
        for i in 0..thetas.len() {
            let mut thetas_plus = thetas;
            let mut thetas_minus = thetas;
            thetas_plus[i] += eps;
            thetas_minus[i] -= eps;

            let mut x_plus = x;
            let mut x_minus = x;
            apply_givens(&mut x_plus, &thetas_plus, true);
            apply_givens(&mut x_minus, &thetas_minus, true);

            let mut dot_plus = 0.0f32;
            let mut dot_minus = 0.0f32;
            for j in 0..d {
                dot_plus += d_out[j] * x_plus[j];
                dot_minus += d_out[j] * x_minus[j];
            }

            let num_grad = (dot_plus - dot_minus) / (2.0 * eps);
            assert!(
                (d_thetas[i] - num_grad).abs() < 2e-3,
                "Inverse Theta {} grad mismatch: analytical={}, numerical={}",
                i,
                d_thetas[i],
                num_grad
            );
        }
    }

    #[test]
    fn test_l2_normalize_vjp() {
        let x = [1.2f32, -0.7f32, 2.3f32, -1.5f32];
        let d = x.len();
        let eps_norm = 1e-6f32;

        let mut u = vec![0.0f32; d];
        let norm = l2_normalize(&x, &mut u, eps_norm);

        let d_u = [0.5f32, 1.2f32, -0.8f32, 0.3f32];
        let mut d_x = vec![0.0f32; d];
        l2_normalize_backward(&x, &u, norm, &d_u, &mut d_x);

        let eps = 1e-4f32;
        for i in 0..d {
            let mut x_plus = x;
            let mut x_minus = x;
            x_plus[i] += eps;
            x_minus[i] -= eps;

            let mut u_plus = vec![0.0f32; d];
            let mut u_minus = vec![0.0f32; d];
            l2_normalize(&x_plus, &mut u_plus, eps_norm);
            l2_normalize(&x_minus, &mut u_minus, eps_norm);

            let mut dot_plus = 0.0f32;
            let mut dot_minus = 0.0f32;
            for j in 0..d {
                dot_plus += d_u[j] * u_plus[j];
                dot_minus += d_u[j] * u_minus[j];
            }

            let num_grad = (dot_plus - dot_minus) / (2.0 * eps);
            assert!(
                (d_x[i] - num_grad).abs() < 1e-3,
                "L2 norm grad mismatch at {}: analytical={}, numerical={}",
                i,
                d_x[i],
                num_grad
            );
        }
    }
}
