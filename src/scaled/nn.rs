//! # PyTorch-like Ergonomic Neural Network Primitives
//!
//! Provides composable traits and modular layers:
//! - `Module`: composable forward, backward, parameter inspection, and zero_grad.
//! - `AutoregressiveModel`: stateful token/vector step and autoregressive generation.
//! - `Optimizer`: first-order and second-order parameter updates.
//! - Core layers: `ScaledRMSNorm`, `ScaledLinear`, `ScaledFFN`.

/// Base trait for modular neural network layers and blocks.
pub trait Module {
    /// Dimension of input vector or features.
    fn in_dim(&self) -> usize;
    /// Dimension of output vector or features.
    fn out_dim(&self) -> usize;
    /// Total number of trainable scalar parameters.
    fn param_count(&self) -> usize;
    /// Zeroes out all accumulated parameter gradients.
    fn zero_grad(&mut self);
}

/// Trait for autoregressive models supporting $O(1)$ recurrent step execution.
pub trait AutoregressiveModel {
    /// Context state type maintained across tokens.
    type State;

    /// Initializes a fresh blank context state.
    fn init_state(&self) -> Self::State;

    /// Generates logits for a single input step and advances the recurrent state.
    fn step(&self, input_vector: &[f32], state: &mut Self::State, logits_out: &mut [f32]);
}

/// Root Mean Square Layer Normalization (RMSNorm).
#[derive(Debug, Clone, PartialEq)]
pub struct ScaledRMSNorm {
    pub dim: usize,
    pub weight: Vec<f32>,
    pub grad: Vec<f32>,
    pub eps: f32,
}

impl ScaledRMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            dim,
            weight: vec![1.0f32; dim],
            grad: vec![0.0f32; dim],
            eps,
        }
    }

    /// Evaluates RMSNorm on input slice: $y = \frac{x}{\text{rms}(x)} \odot \gamma$.
    #[inline(always)]
    pub fn forward(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(y.len(), self.dim);

        let mut sum_sq = 0.0f32;
        for i in 0..self.dim {
            sum_sq += x[i] * x[i];
        }
        let rms = (sum_sq / (self.dim as f32) + self.eps).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..self.dim {
            y[i] = x[i] * inv_rms * self.weight[i];
        }
    }

    /// Computes analytical backward gradients through RMSNorm and parameter scale $\gamma$.
    #[inline(always)]
    pub fn backward(&mut self, x: &[f32], dy: &[f32], dx: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(dy.len(), self.dim);
        debug_assert_eq!(dx.len(), self.dim);

        let mut sum_sq = 0.0f32;
        for i in 0..self.dim {
            sum_sq += x[i] * x[i];
        }
        let rms = (sum_sq / (self.dim as f32) + self.eps).sqrt();
        let inv_rms = 1.0 / rms;
        let d_f32 = self.dim as f32;

        let mut dot_w_dy_x = 0.0f32;
        for i in 0..self.dim {
            self.grad[i] += dy[i] * (x[i] * inv_rms);
            dot_w_dy_x += (self.weight[i] * dy[i]) * x[i];
        }

        let factor = dot_w_dy_x / (d_f32 * rms * rms);
        for i in 0..self.dim {
            dx[i] = inv_rms * (self.weight[i] * dy[i] - x[i] * factor);
        }
    }
}

impl Module for ScaledRMSNorm {
    fn in_dim(&self) -> usize { self.dim }
    fn out_dim(&self) -> usize { self.dim }
    fn param_count(&self) -> usize { self.dim }
    fn zero_grad(&mut self) { self.grad.fill(0.0); }
}

/// Linear projection layer without bias ($y = W x$).
#[derive(Debug, Clone, PartialEq)]
pub struct ScaledLinear {
    pub in_features: usize,
    pub out_features: usize,
    pub weight: Vec<f32>, // row-major: [out_features, in_features]
    pub grad: Vec<f32>,
}

impl ScaledLinear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let n = in_features * out_features;
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let mut weight = vec![0.0f32; n];
        for i in 0..n {
            let angle = (std::f32::consts::PI * (i + 1) as f32 * 1.6180339).sin();
            weight[i] = scale * angle;
        }

        Self {
            in_features,
            out_features,
            weight,
            grad: vec![0.0f32; n],
        }
    }

    #[inline(always)]
    pub fn forward(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_features);
        debug_assert_eq!(y.len(), self.out_features);

        for row in 0..self.out_features {
            let offset = row * self.in_features;
            let mut dot = 0.0f32;
            for col in 0..self.in_features {
                dot += self.weight[offset + col] * x[col];
            }
            y[row] = dot;
        }
    }

    #[inline(always)]
    pub fn backward(&mut self, x: &[f32], dy: &[f32], dx: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_features);
        debug_assert_eq!(dy.len(), self.out_features);
        debug_assert_eq!(dx.len(), self.in_features);

        dx.fill(0.0);
        for row in 0..self.out_features {
            let offset = row * self.in_features;
            let dy_r = dy[row];
            for col in 0..self.in_features {
                self.grad[offset + col] += dy_r * x[col];
                dx[col] += dy_r * self.weight[offset + col];
            }
        }
    }
}

impl Module for ScaledLinear {
    fn in_dim(&self) -> usize { self.in_features }
    fn out_dim(&self) -> usize { self.out_features }
    fn param_count(&self) -> usize { self.weight.len() }
    fn zero_grad(&mut self) { self.grad.fill(0.0); }
}

/// Two-layer Feed-Forward Network with ReLU activation: $y = W_2 \cdot \text{ReLU}(W_1 x)$.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaledFFN {
    pub w1: ScaledLinear,
    pub w2: ScaledLinear,
}

impl ScaledFFN {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            w1: ScaledLinear::new(d_model, d_ff),
            w2: ScaledLinear::new(d_ff, d_model),
        }
    }

    #[inline(always)]
    pub fn forward(&self, x: &[f32], hidden_act: &mut [f32], y: &mut [f32]) {
        self.w1.forward(x, hidden_act);
        for val in hidden_act.iter_mut() {
            if *val < 0.0 {
                *val = 0.0; // ReLU
            }
        }
        self.w2.forward(hidden_act, y);
    }

    #[inline(always)]
    pub fn backward(
        &mut self,
        x: &[f32],
        hidden_act: &[f32],
        dy: &[f32],
        d_hidden: &mut [f32],
        dx: &mut [f32],
    ) {
        self.w2.backward(hidden_act, dy, d_hidden);
        for i in 0..d_hidden.len() {
            if hidden_act[i] <= 0.0 {
                d_hidden[i] = 0.0; // ReLU backward
            }
        }
        self.w1.backward(x, d_hidden, dx);
    }
}

impl Module for ScaledFFN {
    fn in_dim(&self) -> usize { self.w1.in_features }
    fn out_dim(&self) -> usize { self.w2.out_features }
    fn param_count(&self) -> usize { self.w1.param_count() + self.w2.param_count() }
    fn zero_grad(&mut self) {
        self.w1.zero_grad();
        self.w2.zero_grad();
    }
}

/// AdamW optimizer maintaining first and second moments for flat parameter vectors.
#[derive(Debug, Clone)]
pub struct ScaledAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: usize,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl ScaledAdamW {
    pub fn new(param_count: usize, lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step: 0,
            m: vec![0.0f32; param_count],
            v: vec![0.0f32; param_count],
        }
    }

    /// Performs one AdamW optimization update step on a slice of weights and gradients.
    #[inline(always)]
    pub fn step(&mut self, weights: &mut [f32], grads: &[f32]) {
        debug_assert_eq!(weights.len(), grads.len());
        debug_assert_eq!(weights.len(), self.m.len());

        self.step += 1;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;
        let lr = self.lr;

        let bias_correction1 = 1.0 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step as i32);
        let step_size = lr * (bias_correction2.sqrt() / bias_correction1);

        for i in 0..weights.len() {
            let g = grads[i];
            let p = &mut weights[i];
            *p -= lr * wd * *p;
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;
            let denom = self.v[i].sqrt() + eps;
            *p -= step_size * (self.m[i] / denom);
        }
    }

    /// Performs one AdamW step across multiple parameter-gradient slice pairs in place,
    /// scaling gradients by `grad_scale` and clipping to `[-5.0, 5.0]`, then zeroing gradients.
    #[inline(always)]
    pub fn step_layers(&mut self, layers: &mut [(&mut [f32], &mut [f32])], grad_scale: f32) {
        self.step += 1;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;
        let lr = self.lr;

        let bias_correction1 = 1.0 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step as i32);
        let step_size = lr * (bias_correction2.sqrt() / bias_correction1);

        let mut offset = 0;
        for (w, g) in layers.iter_mut() {
            let n = w.len();
            for i in 0..n {
                let mut grad_val = g[i] * grad_scale;
                if grad_val > 5.0 {
                    grad_val = 5.0;
                } else if grad_val < -5.0 {
                    grad_val = -5.0;
                }

                let p = &mut w[i];
                *p -= lr * wd * *p;
                self.m[offset + i] = beta1 * self.m[offset + i] + (1.0 - beta1) * grad_val;
                self.v[offset + i] = beta2 * self.v[offset + i] + (1.0 - beta2) * grad_val * grad_val;
                let denom = self.v[offset + i].sqrt() + eps;
                *p -= step_size * (self.m[offset + i] / denom);
                g[i] = 0.0;
            }
            offset += n;
        }
    }
}
