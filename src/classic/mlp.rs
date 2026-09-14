use super::cache::MlpWorkspace;
use super::config::{ActivationType, TransformerConfig};
use super::ops::{gelu, matmul, matvec, relu};
use super::rng::FastRng;

/// Feed-Forward / MLP sublayer for Transformer:
/// Linear(d_model -> d_ff) -> Activation -> Linear(d_ff -> d_model)
#[derive(Debug, Clone)]
pub struct FeedForward {
    pub w_1: Vec<f32>,            // [d_ff, d_model]
    pub b_1: Option<Vec<f32>>,    // [d_ff]
    pub w_2: Vec<f32>,            // [d_model, d_ff]
    pub b_2: Option<Vec<f32>>,    // [d_model]
    pub activation: ActivationType,
    pub d_model: usize,
    pub d_ff: usize,
}

impl FeedForward {
    /// Creates and initializes MLP layer weights with Xavier normal initialization.
    pub fn new(config: &TransformerConfig, rng: &mut FastRng) -> Self {
        let d_model = config.d_model;
        let d_ff = config.d_ff;

        // Xavier standard deviations:
        // W1: sqrt(2 / (d_model + d_ff))
        let std_1 = (2.0 / (d_model + d_ff) as f32).sqrt();
        // W2: sqrt(2 / (d_ff + d_model))
        let std_2 = (2.0 / (d_ff + d_model) as f32).sqrt();

        let w_1 = (0..d_ff * d_model)
            .map(|_| rng.gen_normal(0.0, std_1))
            .collect();
        let w_2 = (0..d_model * d_ff)
            .map(|_| rng.gen_normal(0.0, std_2))
            .collect();

        let b_1 = if config.use_bias {
            Some(vec![0.0; d_ff])
        } else {
            None
        };
        let b_2 = if config.use_bias {
            Some(vec![0.0; d_model])
        } else {
            None
        };

        Self {
            w_1,
            b_1,
            w_2,
            b_2,
            activation: config.activation,
            d_model,
            d_ff,
        }
    }

    /// Sequence forward pass for MLP.
    /// `input`: slice of shape [seq_len, d_model]
    /// Writes output into `out` of shape [seq_len, d_model].
    pub fn forward(
        &self,
        input: &[f32],
        seq_len: usize,
        workspace: &mut MlpWorkspace,
        out: &mut [f32],
    ) {
        let d_model = self.d_model;
        let d_ff = self.d_ff;

        // 1. Linear in: input * W1^T (+ b1) -> mlp_hidden [seq_len, d_ff]
        let hidden = &mut workspace.hidden[..seq_len * d_ff];
        matmul(hidden, input, &self.w_1, self.b_1.as_deref(), seq_len, d_model, d_ff);

        // 2. Activation function
        match self.activation {
            ActivationType::Gelu => gelu(hidden),
            ActivationType::Relu => relu(hidden),
        }

        // 3. Linear out: mlp_hidden * W2^T (+ b2) -> out [seq_len, d_model]
        matmul(out, hidden, &self.w_2, self.b_2.as_deref(), seq_len, d_ff, d_model);
    }

    /// Single-token autoregressive step for MLP.
    /// `input`: slice of shape [d_model]
    /// Writes output into `out` of shape [d_model].
    pub fn step(
        &self,
        input: &[f32],
        workspace: &mut MlpWorkspace,
        out: &mut [f32],
    ) {
        let d_model = self.d_model;
        let d_ff = self.d_ff;

        // 1. Linear in: input * W1^T (+ b1) -> step_mlp_hidden [d_ff]
        matvec(
            &mut workspace.step_hidden,
            &self.w_1,
            input,
            self.b_1.as_deref(),
            d_ff,
            d_model,
        );

        // 2. Activation
        match self.activation {
            ActivationType::Gelu => gelu(&mut workspace.step_hidden),
            ActivationType::Relu => relu(&mut workspace.step_hidden),
        }

        // 3. Linear out: step_mlp_hidden * W2^T (+ b2) -> out [d_model]
        matvec(
            out,
            &self.w_2,
            &workspace.step_hidden,
            self.b_2.as_deref(),
            d_model,
            d_ff,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_forward_step_equivalence() {
        let config = TransformerConfig::default_512();
        let mut rng = FastRng::new(54321);
        let mlp = FeedForward::new(&config, &mut rng);
        let mut ws = MlpWorkspace::new(&config);

        let seq_len = 4;
        let mut input = vec![0.0; seq_len * config.d_model];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i * 13 + 7) % 29) as f32 * 0.1 - 1.0;
        }

        let mut forward_out = vec![0.0; seq_len * config.d_model];
        mlp.forward(&input, seq_len, &mut ws, &mut forward_out);

        let mut step_out = vec![0.0; config.d_model];
        for pos in 0..seq_len {
            let tok_in = &input[pos * config.d_model..(pos + 1) * config.d_model];
            mlp.step(tok_in, &mut ws, &mut step_out);

            let fwd_slice = &forward_out[pos * config.d_model..(pos + 1) * config.d_model];
            for c in 0..config.d_model {
                let diff = (fwd_slice[c] - step_out[c]).abs();
                assert!(
                    diff < 1e-6,
                    "Mismatch at pos {pos}, dim {c}: fwd={}, step={}",
                    fwd_slice[c],
                    step_out[c]
                );
            }
        }
    }
}
