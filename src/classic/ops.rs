/// Math and linear algebra primitives optimized for Ivy Bridge cache hierarchy and AVX.
/// All loops operate on contiguous memory slices with stride 1 to enable auto-vectorization.

/// Computes inner product of two contiguous f32 slices.
/// Uses 4 independent accumulators and 32-float loop unrolling to saturate
/// Ivy Bridge execution ports (3-cycle latency on FP ADD, 1 throughput per cycle).
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot_product: slice length mismatch");

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks_a = a.chunks_exact(32);
    let chunks_b = b.chunks_exact(32);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        for i in 0..8 {
            sum0 += ca[i] * cb[i];
            sum1 += ca[i + 8] * cb[i + 8];
            sum2 += ca[i + 16] * cb[i + 16];
            sum3 += ca[i + 24] * cb[i + 24];
        }
    }

    let chunks_a8 = remainder_a.chunks_exact(8);
    let chunks_b8 = remainder_b.chunks_exact(8);
    let rem8_a = chunks_a8.remainder();
    let rem8_b = chunks_b8.remainder();

    for (ca, cb) in chunks_a8.zip(chunks_b8) {
        for i in 0..8 {
            sum0 += ca[i] * cb[i];
        }
    }

    for (&x, &y) in rem8_a.iter().zip(rem8_b) {
        sum1 += x * y;
    }

    (sum0 + sum1) + (sum2 + sum3)
}

/// Elementwise addition dst[i] += src[i]
#[inline]
pub fn vector_add(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len(), "vector_add: slice length mismatch");
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

/// Elementwise scaled addition dst[i] += alpha * src[i] (AXPY)
#[inline]
pub fn vector_add_scaled(dst: &mut [f32], src: &[f32], alpha: f32) {
    debug_assert_eq!(dst.len(), src.len(), "vector_add_scaled: slice length mismatch");
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += alpha * s;
    }
}

/// Vector-matrix multiplication: out = x * W^T (+ bias)
/// where weights W has shape [out_dim, in_dim] in row-major layout.
/// Each output element is the dot product of x with row i of W.
#[inline]
pub fn matvec(
    out: &mut [f32],
    weights: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out_dim: usize,
    in_dim: usize,
) {
    debug_assert_eq!(out.len(), out_dim);
    debug_assert_eq!(weights.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);

    if let Some(b) = bias {
        debug_assert_eq!(b.len(), out_dim);
        for (i, (out_elem, &b_val)) in out.iter_mut().zip(b.iter()).enumerate() {
            let row = &weights[i * in_dim..(i + 1) * in_dim];
            *out_elem = dot_product(row, x) + b_val;
        }
    } else {
        for (i, out_elem) in out.iter_mut().enumerate() {
            let row = &weights[i * in_dim..(i + 1) * in_dim];
            *out_elem = dot_product(row, x);
        }
    }
}

/// Matrix-matrix multiplication: C = A * B^T (+ bias)
/// A has shape [m, k], B has shape [n, k] (weights transposed), C has shape [m, n].
#[inline]
pub fn matmul(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(c.len(), m * n);
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        matvec(c_row, b, a_row, bias, n, k);
    }
}

/// In-place numerically stable Softmax:
/// 1. Subtract maximum value for float stability
/// 2. Exp and sum
/// 3. Normalize by sum
#[inline]
pub fn softmax(slice: &mut [f32]) {
    if slice.is_empty() {
        return;
    }

    let mut max_val = f32::NEG_INFINITY;
    for &x in slice.iter() {
        if x > max_val {
            max_val = x;
        }
    }

    // If all values are -inf (e.g. completely masked), set uniform zero
    if max_val == f32::NEG_INFINITY {
        slice.fill(0.0);
        return;
    }

    let mut sum = 0.0f32;
    for x in slice.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }

    let inv_sum = 1.0 / sum;
    for x in slice.iter_mut() {
        *x *= inv_sum;
    }
}

/// Layer Normalization:
/// out_i = ((x_i - mean) / sqrt(var + eps)) * gamma_i + beta_i
pub fn layernorm(
    out: &mut [f32],
    x: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    eps: f32,
) {
    let d = x.len();
    debug_assert_eq!(out.len(), d);
    debug_assert_eq!(gamma.len(), d);

    let mut sum = 0.0f32;
    for &val in x {
        sum += val;
    }
    let mean = sum / d as f32;

    let mut var_sum = 0.0f32;
    for &val in x {
        let diff = val - mean;
        var_sum += diff * diff;
    }
    let var = var_sum / d as f32;
    let inv_std = 1.0 / (var + eps).sqrt();

    if let Some(b) = beta {
        debug_assert_eq!(b.len(), d);
        for i in 0..d {
            out[i] = (x[i] - mean) * inv_std * gamma[i] + b[i];
        }
    } else {
        for i in 0..d {
            out[i] = (x[i] - mean) * inv_std * gamma[i];
        }
    }
}

/// Root Mean Square Normalization (RMSNorm):
/// out_i = (x_i / sqrt(mean(x^2) + eps)) * gamma_i
pub fn rmsnorm(out: &mut [f32], x: &[f32], gamma: &[f32], eps: f32) {
    let d = x.len();
    debug_assert_eq!(out.len(), d);
    debug_assert_eq!(gamma.len(), d);

    let mut sum_sq = 0.0f32;
    for &val in x {
        sum_sq += val * val;
    }
    let mean_sq = sum_sq / d as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    for i in 0..d {
        out[i] = x[i] * inv_rms * gamma[i];
    }
}

/// In-place Gaussian Error Linear Unit (GELU) with tanh approximation:
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
pub fn gelu(slice: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;

    for x in slice.iter_mut() {
        let val = *x;
        let inner = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
        *x = 0.5 * val * (1.0 + inner.tanh());
    }
}

/// In-place Rectified Linear Unit (ReLU):
/// ReLU(x) = max(0, x)
#[inline]
pub fn relu(slice: &mut [f32]) {
    for x in slice.iter_mut() {
        *x = x.max(0.0);
    }
}

/// Computes sinusoidal positional encoding vector for a single position:
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
pub fn sinusoidal_pe_vector(pos: usize, d_model: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), d_model);
    let half_dim = d_model / 2;
    for i in 0..half_dim {
        let div_term = (-((2 * i) as f32) * (10000.0f32.ln() / d_model as f32)).exp();
        let angle = pos as f32 * div_term;
        out[2 * i] = angle.sin();
        out[2 * i + 1] = angle.cos();
    }
    // If d_model is odd (unlikely in transformers), fill remainder with sin
    if d_model % 2 != 0 {
        let i = half_dim;
        let div_term = (-((2 * i) as f32) * (10000.0f32.ln() / d_model as f32)).exp();
        out[d_model - 1] = (pos as f32 * div_term).sin();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        // 5 + 12 + 21 + 32 = 70
        let result = dot_product(&a, &b);
        assert!((result - 70.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_unrolled_length() {
        let len = 75; // More than 32 + 32 + 8, exercises all loop tiers
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|_| 1.0f32).collect();
        // Sum from 0 to 74 = 74 * 75 / 2 = 2775
        let result = dot_product(&a, &b);
        assert!((result - 2775.0).abs() < 1e-4);
    }

    #[test]
    fn test_matvec() {
        // W is 2x3:
        // [1.0, 2.0, 3.0]
        // [4.0, 5.0, 6.0]
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [0.5, 1.0, 2.0];
        let bias = [0.1, -0.5];
        let mut out = [0.0; 2];
        matvec(&mut out, &w, &x, Some(&bias), 2, 3);
        // row 0: 0.5*1 + 1*2 + 2*3 + 0.1 = 0.5 + 2 + 6 + 0.1 = 8.6
        // row 1: 0.5*4 + 1*5 + 2*6 - 0.5 = 2 + 5 + 12 - 0.5 = 18.5
        assert!((out[0] - 8.6).abs() < 1e-5);
        assert!((out[1] - 18.5).abs() < 1e-5);
    }

    #[test]
    fn test_matmul() {
        // A is 2x2, B is 2x2
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A * B^T
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0; 4];
        matmul(&mut c, &a, &b, None, 2, 2, 2);
        // c[0, 0] = dot([1, 2], [5, 6]) = 17
        // c[0, 1] = dot([1, 2], [7, 8]) = 23
        // c[1, 0] = dot([3, 4], [5, 6]) = 39
        // c[1, 1] = dot([3, 4], [7, 8]) = 53
        assert!((c[0] - 17.0).abs() < 1e-5);
        assert!((c[1] - 23.0).abs() < 1e-5);
        assert!((c[2] - 39.0).abs() < 1e-5);
        assert!((c[3] - 53.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_properties() {
        let mut v = [1.0, 2.0, 3.0, 4.0, 5.0];
        softmax(&mut v);
        // Sum must be 1.0
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Must be monotonically increasing
        for i in 1..v.len() {
            assert!(v[i] > v[i - 1]);
        }
    }

    #[test]
    fn test_softmax_shift_invariance() {
        let mut v1 = [10.0, 20.0, 30.0];
        let mut v2 = [1010.0, 1020.0, 1030.0];
        softmax(&mut v1);
        softmax(&mut v2);
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_layernorm_normalization() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = [1.0; 8];
        let beta = [0.0; 8];
        let mut out = [0.0; 8];
        layernorm(&mut out, &x, &gamma, Some(&beta), 1e-5);

        let mean: f32 = out.iter().sum::<f32>() / 8.0;
        let var: f32 = out.iter().map(|&v| v * v).sum::<f32>() / 8.0;
        assert!(mean.abs() < 1e-5, "Mean must be ~0, got {mean}");
        assert!((var - 1.0).abs() < 1e-3, "Variance must be ~1, got {var}");
    }

    #[test]
    fn test_rmsnorm_normalization() {
        let x = [2.0, -2.0, 2.0, -2.0];
        let gamma = [1.0; 4];
        let mut out = [0.0; 4];
        rmsnorm(&mut out, &x, &gamma, 1e-5);

        let mean_sq: f32 = out.iter().map(|&v| v * v).sum::<f32>() / 4.0;
        assert!((mean_sq - 1.0).abs() < 1e-3, "RMS must be ~1, got {mean_sq}");
    }

    #[test]
    fn test_gelu_and_relu() {
        let mut g = [0.0, -2.0, 2.0];
        gelu(&mut g);
        assert!((g[0] - 0.0).abs() < 1e-6);
        assert!(g[1] < 0.0 && g[1] > -0.1); // Small negative dip
        assert!((g[2] - 1.9545).abs() < 1e-3);

        let mut r = [-3.0, 0.0, 4.5];
        relu(&mut r);
        assert_eq!(r[0], 0.0);
        assert_eq!(r[1], 0.0);
        assert_eq!(r[2], 4.5);
    }
}
