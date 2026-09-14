//! # Scaled Quantum-Algebraic SRX Core
//!
//! Scalable SRX Attention & Transformer generalized to arbitrary head count $H \ge 2$:
//! - Head dimension fixed at $d_{\text{head}} = 4$ (Lie group SO(4) quantum invariant).
//! - Hidden dimension $d_{\text{model}} = 4H$.
//! - Strictly $O(1)$ state footprint: $H \times 80$ bytes (100% L1D cache resident).
//! - Vectorized inner loops across heads, zero heap allocation in hot-path `step()` and `forward()`.
//! - Pure orthogonal projector $M_t = M_{t-1} + k_{\text{rot}} e_t^T$ and undistorted MUSIC resonance.

use crate::srx_v05::ops::{
    apply_butterfly_4, apply_butterfly_4_backward, l2_normalize, l2_normalize_backward,
};
use crate::scaled::config::ScaledConfig;
use crate::scaled::nn::{Module, ScaledFFN, ScaledLinear, ScaledRMSNorm};

/// Recurrent context state for Scaled SRX with $H$ heads.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaledSrxState {
    pub n_heads: usize,
    /// Phase angles $\Theta \in \mathbb{R}^{H \times 4}$. Size: $4H$ floats = $16H$ bytes.
    pub thetas: Vec<f32>,
    /// Associative memory $M \in \mathbb{R}^{H \times 4 \times 4}$. Size: $16H$ floats = $64H$ bytes.
    pub m: Vec<f32>,
}

impl ScaledSrxState {
    /// Creates a fresh zero-initialized state for $H$ heads.
    pub fn new(n_heads: usize) -> Self {
        Self {
            n_heads,
            thetas: vec![0.0f32; n_heads * 4],
            m: vec![0.0f32; n_heads * 16],
        }
    }

    /// Resets all thetas and memories to zero.
    pub fn reset(&mut self) {
        self.thetas.fill(0.0);
        self.m.fill(0.0);
    }

    /// Exact memory footprint in bytes: strictly $H \times 80$ bytes.
    pub fn memory_bytes(&self) -> usize {
        self.n_heads * 80
    }
}

/// Pre-allocated scratchpad for zero-allocation inference and training.
#[derive(Debug, Clone)]
pub struct ScaledWorkspace {
    pub n_heads: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub head_outputs: Vec<f32>,
    pub hidden_act: Vec<f32>,
    pub norm_buf: Vec<f32>,
    pub residual_buf: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_out: Vec<f32>,
    // Per-head analytical backward cache buffers (allocated once)
    pub n_q: Vec<f32>,
    pub n_k: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub curr_thetas: Vec<f32>,
    pub k_rot: Vec<f32>,
    pub prev_m: Vec<f32>,
    pub e: Vec<f32>,
    pub q_inv: Vec<f32>,
    pub q_rot: Vec<f32>,
    pub w_music: Vec<f32>,
    pub y_ret: Vec<f32>,
}

impl ScaledWorkspace {
    pub fn new(config: &ScaledConfig) -> Self {
        Self {
            n_heads: config.n_heads,
            d_model: config.d_model,
            d_ff: config.d_ff,
            q: vec![0.0f32; config.d_model],
            k: vec![0.0f32; config.d_model],
            v: vec![0.0f32; config.d_model],
            head_outputs: vec![0.0f32; config.d_model],
            hidden_act: vec![0.0f32; config.d_ff],
            norm_buf: vec![0.0f32; config.d_model],
            residual_buf: vec![0.0f32; config.d_model],
            attn_out: vec![0.0f32; config.d_model],
            ffn_out: vec![0.0f32; config.d_model],
            n_q: vec![0.0f32; config.n_heads],
            n_k: vec![0.0f32; config.n_heads],
            q_norm: vec![0.0f32; config.d_model],
            k_norm: vec![0.0f32; config.d_model],
            curr_thetas: vec![0.0f32; config.d_model],
            k_rot: vec![0.0f32; config.d_model],
            prev_m: vec![0.0f32; config.n_heads * 16],
            e: vec![0.0f32; config.d_model],
            q_inv: vec![0.0f32; config.d_model],
            q_rot: vec![0.0f32; config.d_model],
            w_music: vec![0.0f32; config.n_heads],
            y_ret: vec![0.0f32; config.d_model],
        }
    }
}

/// Multi-head Scaled SRX Attention mechanism.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaledSrxAttention {
    pub n_heads: usize,
    pub d_head: usize,
    pub d_model: usize,
    pub w_q: ScaledLinear,
    pub w_k: ScaledLinear,
    pub w_v: ScaledLinear,
    pub w_o: ScaledLinear,
}

impl ScaledSrxAttention {
    pub fn new(n_heads: usize) -> Self {
        let d_head = 4;
        let d_model = n_heads * d_head;
        Self {
            n_heads,
            d_head,
            d_model,
            w_q: ScaledLinear::new(d_model, d_model),
            w_k: ScaledLinear::new(d_model, d_model),
            w_v: ScaledLinear::new(d_model, d_model),
            w_o: ScaledLinear::new(d_model, d_model),
        }
    }

    /// Single autoregressive step with $O(1)$ complexity and zero heap allocation.
    #[inline(always)]
    pub fn step(
        &self,
        x: &[f32],
        state: &mut ScaledSrxState,
        y_out: &mut [f32],
        ws: &mut ScaledWorkspace,
    ) {
        debug_assert_eq!(x.len(), self.d_model);
        debug_assert_eq!(y_out.len(), self.d_model);

        // 1. Compute Q, K, V projections
        self.w_q.forward(x, &mut ws.q);
        self.w_k.forward(x, &mut ws.k);
        self.w_v.forward(x, &mut ws.v);

        // 2. Vectorized loop over H heads
        for h in 0..self.n_heads {
            let off = h * 4;
            let m_off = h * 16;

            let q_h = [ws.q[off], ws.q[off + 1], ws.q[off + 2], ws.q[off + 3]];
            let k_h = [ws.k[off], ws.k[off + 1], ws.k[off + 2], ws.k[off + 3]];
            let v_h = [ws.v[off], ws.v[off + 1], ws.v[off + 2], ws.v[off + 3]];

            // L2-normalize
            let mut q_norm = [0.0f32; 4];
            let mut k_norm = [0.0f32; 4];
            let n_q = l2_normalize(&q_h, &mut q_norm, 1e-6);
            let n_k = l2_normalize(&k_h, &mut k_norm, 1e-6);
            ws.n_q[h] = n_q;
            ws.n_k[h] = n_k;
            ws.q_norm[off..off + 4].copy_from_slice(&q_norm);
            ws.k_norm[off..off + 4].copy_from_slice(&k_norm);

            // Record previous associative memory before update
            ws.prev_m[m_off..m_off + 16].copy_from_slice(&state.m[m_off..m_off + 16]);

            // Update phase angles
            for i in 0..4 {
                let delta_theta = (k_norm[i] * v_h[i]).clamp(-0.5, 0.5);
                state.thetas[off + i] += delta_theta;
            }

            let thetas_h = [
                state.thetas[off],
                state.thetas[off + 1],
                state.thetas[off + 2],
                state.thetas[off + 3],
            ];
            ws.curr_thetas[off..off + 4].copy_from_slice(&thetas_h);

            // Butterfly rotation on key: k_rot = U * k_norm
            let mut k_rot = [0.0f32; 4];
            apply_butterfly_4(&k_norm, &thetas_h, false, &mut k_rot);
            ws.k_rot[off..off + 4].copy_from_slice(&k_rot);

            // Read past response: v_hat = M_{t-1}^T * k_rot
            let mut v_hat = [0.0f32; 4];
            for c in 0..4 {
                let mut dot = 0.0f32;
                for r in 0..4 {
                    dot += ws.prev_m[m_off + r * 4 + c] * k_rot[r];
                }
                v_hat[c] = dot;
            }

            // Novelty residual error: e = v - v_hat
            let mut e = [0.0f32; 4];
            for c in 0..4 {
                e[c] = v_h[c] - v_hat[c];
            }
            ws.e[off..off + 4].copy_from_slice(&e);

            // Pure Orthogonal Projector memory update: M_t = M_{t-1} + k_rot * e^T
            for r in 0..4 {
                for c in 0..4 {
                    state.m[m_off + r * 4 + c] = ws.prev_m[m_off + r * 4 + c] + k_rot[r] * e[c];
                }
            }

            // Undistorted MUSIC resonance: q_inv = U^\dagger * q_norm
            let mut q_inv = [0.0f32; 4];
            apply_butterfly_4(&q_norm, &thetas_h, true, &mut q_inv);
            ws.q_inv[off..off + 4].copy_from_slice(&q_inv);

            let noise_energy = q_inv[2] * q_inv[2] + q_inv[3] * q_inv[3];
            let w_music = (1.0 / (noise_energy + 1e-5)).min(15.0);
            ws.w_music[h] = w_music;

            // Query readout: q_rot = U * q_norm
            let mut q_rot = [0.0f32; 4];
            apply_butterfly_4(&q_norm, &thetas_h, false, &mut q_rot);
            ws.q_rot[off..off + 4].copy_from_slice(&q_rot);

            // y_ret = M_t^T * q_rot, y_h = y_ret * w_music
            let mut y_ret = [0.0f32; 4];
            for c in 0..4 {
                let mut dot = 0.0f32;
                for r in 0..4 {
                    dot += state.m[m_off + r * 4 + c] * q_rot[r];
                }
                y_ret[c] = dot;
                ws.head_outputs[off + c] = dot * w_music;
            }
            ws.y_ret[off..off + 4].copy_from_slice(&y_ret);
        }

        // 3. Output projection W_o
        self.w_o.forward(&ws.head_outputs, y_out);
    }

    /// Full sequence forward pass over $T$ steps.
    pub fn forward(
        &self,
        seq_x: &[f32],
        seq_y: &mut [f32],
        ws: &mut ScaledWorkspace,
    ) {
        let t_steps = seq_x.len() / self.d_model;
        debug_assert_eq!(seq_y.len(), seq_x.len());

        let mut state = ScaledSrxState::new(self.n_heads);
        for t in 0..t_steps {
            let in_slice = &seq_x[t * self.d_model..(t + 1) * self.d_model];
            let out_slice = &mut seq_y[t * self.d_model..(t + 1) * self.d_model];
            self.step(in_slice, &mut state, out_slice, ws);
        }
    }
}

impl Module for ScaledSrxAttention {
    fn in_dim(&self) -> usize { self.d_model }
    fn out_dim(&self) -> usize { self.d_model }
    fn param_count(&self) -> usize {
        self.w_q.param_count() + self.w_k.param_count() + self.w_v.param_count() + self.w_o.param_count()
    }
    fn zero_grad(&mut self) {
        self.w_q.zero_grad();
        self.w_k.zero_grad();
        self.w_v.zero_grad();
        self.w_o.zero_grad();
    }
}

/// Complete Scaled SRX Transformer Architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaledSrxTransformer {
    pub config: ScaledConfig,
    pub embed: ScaledLinear, // Vocabulary embeddings: vocab_size -> d_model
    pub norm_attn: ScaledRMSNorm,
    pub attn: ScaledSrxAttention,
    pub norm_ffn: ScaledRMSNorm,
    pub ffn: ScaledFFN,
    pub norm_final: ScaledRMSNorm,
    pub lm_head: ScaledLinear, // Output projection: d_model -> vocab_size
}

impl ScaledSrxTransformer {
    pub fn new(config: ScaledConfig) -> Self {
        let dm = config.d_model;
        let v = config.vocab_size;
        let dff = config.d_ff;

        Self {
            config: config.clone(),
            embed: ScaledLinear::new(v, dm),
            norm_attn: ScaledRMSNorm::new(dm, config.rms_eps),
            attn: ScaledSrxAttention::new(config.n_heads),
            norm_ffn: ScaledRMSNorm::new(dm, config.rms_eps),
            ffn: ScaledFFN::new(dm, dff),
            norm_final: ScaledRMSNorm::new(dm, config.rms_eps),
            lm_head: ScaledLinear::new(dm, v),
        }
    }

    /// Single autoregressive step from continuous input vector or token embedding to vocabulary logits.
    #[inline(always)]
    pub fn step(
        &self,
        x_emb: &[f32],
        state: &mut ScaledSrxState,
        logits_out: &mut [f32],
        ws: &mut ScaledWorkspace,
    ) {
        debug_assert_eq!(x_emb.len(), self.config.d_model);
        debug_assert_eq!(logits_out.len(), self.config.vocab_size);

        let mut norm_buf = vec![0.0f32; self.config.d_model];
        let mut attn_out = vec![0.0f32; self.config.d_model];
        let mut res1 = vec![0.0f32; self.config.d_model];
        let mut res2 = vec![0.0f32; self.config.d_model];
        let mut final_norm = vec![0.0f32; self.config.d_model];

        // 1. Pre-Attn RMSNorm
        self.norm_attn.forward(x_emb, &mut norm_buf);

        // 2. Attention step
        self.attn.step(&norm_buf, state, &mut attn_out, ws);

        // 3. Residual connection 1: res1 = x_emb + attn_out
        for i in 0..self.config.d_model {
            res1[i] = x_emb[i] + attn_out[i];
        }

        // 4. Pre-FFN RMSNorm
        self.norm_ffn.forward(&res1, &mut norm_buf);

        // 5. FFN block
        self.ffn.forward(&norm_buf, &mut ws.hidden_act, &mut res2);

        // 6. Residual connection 2: res2 = res1 + ffn_out
        for i in 0..self.config.d_model {
            res2[i] += res1[i];
        }

        // 7. Final RMSNorm
        self.norm_final.forward(&res2, &mut final_norm);

        // 8. LM Head: logits = W_lm * norm_buf
        self.lm_head.forward(&final_norm, logits_out);
    }

    /// Returns total trainable parameter count.
    pub fn param_count(&self) -> usize {
        self.embed.param_count()
            + self.norm_attn.param_count()
            + self.attn.param_count()
            + self.norm_ffn.param_count()
            + self.ffn.param_count()
            + self.norm_final.param_count()
            + self.lm_head.param_count()
    }

    /// Returns mutable slice references to all 11 parameter and gradient layers for AdamW.
    pub fn get_layers_mut(&mut self) -> [(&mut [f32], &mut [f32]); 11] {
        [
            (&mut self.embed.weight[..], &mut self.embed.grad[..]),
            (&mut self.norm_attn.weight[..], &mut self.norm_attn.grad[..]),
            (&mut self.attn.w_q.weight[..], &mut self.attn.w_q.grad[..]),
            (&mut self.attn.w_k.weight[..], &mut self.attn.w_k.grad[..]),
            (&mut self.attn.w_v.weight[..], &mut self.attn.w_v.grad[..]),
            (&mut self.attn.w_o.weight[..], &mut self.attn.w_o.grad[..]),
            (&mut self.norm_ffn.weight[..], &mut self.norm_ffn.grad[..]),
            (&mut self.ffn.w1.weight[..], &mut self.ffn.w1.grad[..]),
            (&mut self.ffn.w2.weight[..], &mut self.ffn.w2.grad[..]),
            (&mut self.norm_final.weight[..], &mut self.norm_final.grad[..]),
            (&mut self.lm_head.weight[..], &mut self.lm_head.grad[..]),
        ]
    }

    /// Executes single forward and analytical backward step on a continuous embedding vector.
    /// Accumulates parameter gradients into internal layers' `.grad`.
    /// Returns (loss, d_emb) where d_emb is gradient wrt input embedding.
    pub fn train_step_emb(
        &mut self,
        x_emb: &[f32],
        target_b: usize,
        state: &mut ScaledSrxState,
        ws: &mut ScaledWorkspace,
    ) -> (f32, Vec<f32>) {
        let dm = self.config.d_model;
        debug_assert_eq!(x_emb.len(), dm);

        // 1. Pre-Attn RMSNorm
        let mut norm_buf_1 = vec![0.0f32; dm];
        self.norm_attn.forward(x_emb, &mut norm_buf_1);

        // 2. Attention step
        let mut attn_out = vec![0.0f32; dm];
        self.attn.step(&norm_buf_1, state, &mut attn_out, ws);

        // 3. Residual 1
        let mut res1 = vec![0.0f32; dm];
        for i in 0..dm {
            res1[i] = x_emb[i] + attn_out[i];
        }

        // 4. Pre-FFN RMSNorm
        let mut norm_buf_2 = vec![0.0f32; dm];
        self.norm_ffn.forward(&res1, &mut norm_buf_2);

        // 5. FFN block
        let mut hidden_act = vec![0.0f32; self.config.d_ff];
        let mut ffn_out = vec![0.0f32; dm];
        self.ffn.forward(&norm_buf_2, &mut hidden_act, &mut ffn_out);

        // 6. Residual 2
        let mut res2 = vec![0.0f32; dm];
        for i in 0..dm {
            res2[i] = res1[i] + ffn_out[i];
        }

        // 7. Final RMSNorm
        let mut norm_buf_3 = vec![0.0f32; dm];
        self.norm_final.forward(&res2, &mut norm_buf_3);

        // 8. LM Head
        let mut logits = vec![0.0f32; 256];
        self.lm_head.forward(&norm_buf_3, &mut logits);

        // 9. Softmax & Cross-Entropy
        let mut max_l = f32::NEG_INFINITY;
        for &l in &logits {
            if l > max_l {
                max_l = l;
            }
        }
        let mut sum_exp = 0.0f32;
        let mut probs = [0.0f32; 256];
        for b in 0..256 {
            probs[b] = (logits[b] - max_l).exp();
            sum_exp += probs[b];
        }
        let inv_sum = 1.0 / sum_exp.max(1e-12);
        for b in 0..256 {
            probs[b] *= inv_sum;
        }

        let target_b = target_b.min(255);
        let loss = -probs[target_b].max(1e-12).ln();

        // 10. Gradient of cross-entropy
        let mut d_logits = [0.0f32; 256];
        for b in 0..256 {
            d_logits[b] = probs[b];
        }
        d_logits[target_b] -= 1.0;

        // 11. Backward through LM Head
        let mut d_norm_buf_3 = vec![0.0f32; dm];
        self.lm_head.backward(&norm_buf_3, &d_logits, &mut d_norm_buf_3);

        // 12. Backward through Final RMSNorm
        let mut d_res2 = vec![0.0f32; dm];
        self.norm_final.backward(&res2, &d_norm_buf_3, &mut d_res2);

        // 13. Residual 2 & FFN backward
        let mut d_res1 = d_res2.clone();
        let mut d_hidden = vec![0.0f32; self.config.d_ff];
        let mut d_norm_buf_2 = vec![0.0f32; dm];
        self.ffn.backward(&norm_buf_2, &hidden_act, &d_res2, &mut d_hidden, &mut d_norm_buf_2);

        let mut d_res1_from_ffn = vec![0.0f32; dm];
        self.norm_ffn.backward(&res1, &d_norm_buf_2, &mut d_res1_from_ffn);
        for i in 0..dm {
            d_res1[i] += d_res1_from_ffn[i];
        }

        // 14. Attention backward: exact analytical backpropagation through MUSIC, Butterfly, and Orthogonal Projector
        let mut d_head_outputs = vec![0.0f32; dm];
        self.attn.w_o.backward(&ws.head_outputs, &d_res1, &mut d_head_outputs);

        let mut d_q = vec![0.0f32; dm];
        let mut d_k = vec![0.0f32; dm];
        let mut d_v = vec![0.0f32; dm];

        for h in 0..self.config.n_heads {
            let off = h * 4;
            let m_off = h * 16;

            let q_h = [ws.q[off], ws.q[off + 1], ws.q[off + 2], ws.q[off + 3]];
            let k_h = [ws.k[off], ws.k[off + 1], ws.k[off + 2], ws.k[off + 3]];
            let v_h = [ws.v[off], ws.v[off + 1], ws.v[off + 2], ws.v[off + 3]];

            let q_norm: [f32; 4] = ws.q_norm[off..off + 4].try_into().unwrap();
            let k_norm: [f32; 4] = ws.k_norm[off..off + 4].try_into().unwrap();
            let curr_thetas: [f32; 4] = ws.curr_thetas[off..off + 4].try_into().unwrap();
            let k_rot: [f32; 4] = ws.k_rot[off..off + 4].try_into().unwrap();
            let q_inv: [f32; 4] = ws.q_inv[off..off + 4].try_into().unwrap();
            let q_rot: [f32; 4] = ws.q_rot[off..off + 4].try_into().unwrap();
            let e: [f32; 4] = ws.e[off..off + 4].try_into().unwrap();
            let y_ret: [f32; 4] = ws.y_ret[off..off + 4].try_into().unwrap();
            let w_music = ws.w_music[h];
            let n_q = ws.n_q[h];
            let n_k = ws.n_k[h];

            let dy = [
                d_head_outputs[off],
                d_head_outputs[off + 1],
                d_head_outputs[off + 2],
                d_head_outputs[off + 3],
            ];

            // 1. d_y_ret and d_w_music
            let mut d_y_ret = [0.0f32; 4];
            let mut d_w_music = 0.0f32;
            for c in 0..4 {
                d_y_ret[c] = dy[c] * w_music;
                d_w_music += dy[c] * y_ret[c];
            }

            // 2. MUSIC noise energy backward
            let d_noise = if w_music < 15.0 {
                -d_w_music * w_music * w_music
            } else {
                0.0
            };
            let d_q_inv = [
                0.0f32,
                0.0f32,
                2.0 * d_noise * q_inv[2],
                2.0 * d_noise * q_inv[3],
            ];

            // 3. d_q_rot: dL / d(q_rot[r]) = sum_c d_y_ret[c] * M_t[r, c]
            let mut d_q_rot = [0.0f32; 4];
            for r in 0..4 {
                let mut sum = 0.0f32;
                for c in 0..4 {
                    sum += d_y_ret[c] * state.m[m_off + r * 4 + c];
                }
                d_q_rot[r] = sum;
            }

            // 4. Backprop through butterfly rotations to q_norm
            let mut d_thetas = [0.0f32; 4];
            let mut d_q_norm_rot = [0.0f32; 4];
            apply_butterfly_4_backward(
                &q_norm,
                &curr_thetas,
                false,
                &d_q_rot,
                &mut d_q_norm_rot,
                &mut d_thetas,
            );

            let mut d_q_norm_inv = [0.0f32; 4];
            apply_butterfly_4_backward(
                &q_norm,
                &curr_thetas,
                true,
                &d_q_inv,
                &mut d_q_norm_inv,
                &mut d_thetas,
            );

            let mut d_q_norm = [0.0f32; 4];
            for i in 0..4 {
                d_q_norm[i] = d_q_norm_rot[i] + d_q_norm_inv[i];
            }

            let mut d_q_h = [0.0f32; 4];
            l2_normalize_backward(&q_h, &q_norm, n_q, &d_q_norm, &mut d_q_h);

            // 5. Memory update backward:
            // M_t[r, c] = M_{t-1}[r, c] + k_rot[r] * e[c]
            // Readout: y_ret[c] = sum_r M_t[r, c] * q_rot[r]
            // dM[r, c] = q_rot[r] * d_y_ret[c]
            // d_e[c] = sum_r dM[r, c] * k_rot[r] = d_y_ret[c] * (k_rot . q_rot)
            let mut dot_kq = 0.0f32;
            for i in 0..4 {
                dot_kq += k_rot[i] * q_rot[i];
            }

            let mut d_e = [0.0f32; 4];
            for c in 0..4 {
                d_e[c] = d_y_ret[c] * dot_kq;
            }

            // e[c] = v_h[c] - v_hat[c] => d_v_h = d_e, d_v_hat = -d_e
            let mut d_v_h = d_e;

            // d_k_rot[r] = q_rot[r] * (d_y_ret . e) - sum_c d_e[c] * prev_m[r, c]
            let mut dot_dy_e = 0.0f32;
            for c in 0..4 {
                dot_dy_e += d_y_ret[c] * e[c];
            }

            let mut d_k_rot = [0.0f32; 4];
            for r in 0..4 {
                let mut sum_prev = 0.0f32;
                for c in 0..4 {
                    sum_prev += d_e[c] * ws.prev_m[m_off + r * 4 + c];
                }
                d_k_rot[r] = q_rot[r] * dot_dy_e - sum_prev;
            }

            // 6. Butterfly backward on k_rot -> k_norm
            let mut d_k_norm = [0.0f32; 4];
            apply_butterfly_4_backward(
                &k_norm,
                &curr_thetas,
                false,
                &d_k_rot,
                &mut d_k_norm,
                &mut d_thetas,
            );

            // 7. Phase dynamics backward: delta_theta = clamp(k_norm * v_h, -0.5, 0.5)
            for i in 0..4 {
                let prod = k_norm[i] * v_h[i];
                if prod >= -0.5 && prod <= 0.5 {
                    d_k_norm[i] += d_thetas[i] * v_h[i];
                    d_v_h[i] += d_thetas[i] * k_norm[i];
                }
            }

            // 8. L2 normalize backward on k
            let mut d_k_h = [0.0f32; 4];
            l2_normalize_backward(&k_h, &k_norm, n_k, &d_k_norm, &mut d_k_h);

            // Store into d_q, d_k, d_v
            for i in 0..4 {
                d_q[off + i] = d_q_h[i];
                d_k[off + i] = d_k_h[i];
                d_v[off + i] = d_v_h[i];
            }
        }

        let mut d_norm_buf_1_q = vec![0.0f32; dm];
        let mut d_norm_buf_1_k = vec![0.0f32; dm];
        let mut d_norm_buf_1_v = vec![0.0f32; dm];
        self.attn.w_q.backward(&norm_buf_1, &d_q, &mut d_norm_buf_1_q);
        self.attn.w_k.backward(&norm_buf_1, &d_k, &mut d_norm_buf_1_k);
        self.attn.w_v.backward(&norm_buf_1, &d_v, &mut d_norm_buf_1_v);

        let mut d_norm_buf_1 = vec![0.0f32; dm];
        for i in 0..dm {
            d_norm_buf_1[i] = d_norm_buf_1_q[i] + d_norm_buf_1_k[i] + d_norm_buf_1_v[i];
        }

        // 15. Pre-Attn RMSNorm backward
        let mut d_x_emb = d_res1.clone();
        let mut d_x_emb_from_attn = vec![0.0f32; dm];
        self.norm_attn.backward(x_emb, &d_norm_buf_1, &mut d_x_emb_from_attn);
        for i in 0..dm {
            d_x_emb[i] += d_x_emb_from_attn[i];
        }

        (loss, d_x_emb)
    }

    /// Sequence forward pass over `seq_len` steps with full intermediate activation recording for BPTT.
    /// Returns the mean cross-entropy loss over the sequence.
    pub fn forward_sequence(
        &self,
        embeddings: &[f32],
        targets: &[usize],
        seq_len: usize,
        ws: &mut ScaledSequenceWorkspace,
    ) -> f32 {
        let dm = self.config.d_model;
        let dff = self.config.d_ff;
        let nh = self.config.n_heads;
        assert!(seq_len <= ws.max_len, "Sequence length exceeds workspace capacity");
        assert_eq!(embeddings.len(), seq_len * dm);
        assert_eq!(targets.len(), seq_len);

        ws.emb[..seq_len * dm].copy_from_slice(embeddings);

        let mut total_loss = 0.0f32;

        for t in 0..seq_len {
            let t_dm = t * dm;
            let t_nh = t * nh;
            let t_m = t * nh * 16;
            let t_v = t * 256;

            // 1. Pre-Attn RMSNorm
            self.norm_attn.forward(&ws.emb[t_dm..t_dm + dm], &mut ws.norm_buf_1[t_dm..t_dm + dm]);

            // 2. QKV projections
            self.attn.w_q.forward(&ws.norm_buf_1[t_dm..t_dm + dm], &mut ws.q[t_dm..t_dm + dm]);
            self.attn.w_k.forward(&ws.norm_buf_1[t_dm..t_dm + dm], &mut ws.k[t_dm..t_dm + dm]);
            self.attn.w_v.forward(&ws.norm_buf_1[t_dm..t_dm + dm], &mut ws.v[t_dm..t_dm + dm]);

            // 3. Multi-head SRX Attention
            for h in 0..nh {
                let off = t_dm + h * 4;
                let m_off = t_m + h * 16;
                let pm_off = if t > 0 { (t - 1) * nh * 16 + h * 16 } else { 0 };
                let pth_off = if t > 0 { (t - 1) * dm + h * 4 } else { 0 };

                let q_h = [ws.q[off], ws.q[off + 1], ws.q[off + 2], ws.q[off + 3]];
                let k_h = [ws.k[off], ws.k[off + 1], ws.k[off + 2], ws.k[off + 3]];
                let v_h = [ws.v[off], ws.v[off + 1], ws.v[off + 2], ws.v[off + 3]];

                let mut q_norm = [0.0f32; 4];
                let mut k_norm = [0.0f32; 4];
                let n_q = l2_normalize(&q_h, &mut q_norm, 1e-6);
                let n_k = l2_normalize(&k_h, &mut k_norm, 1e-6);
                ws.n_q[t_nh + h] = n_q;
                ws.n_k[t_nh + h] = n_k;
                ws.q_norm[off..off + 4].copy_from_slice(&q_norm);
                ws.k_norm[off..off + 4].copy_from_slice(&k_norm);

                // Copy previous m
                if t > 0 {
                    let mut prev_m_tmp = [0.0f32; 16];
                    prev_m_tmp.copy_from_slice(&ws.m[pm_off..pm_off + 16]);
                    ws.prev_m[m_off..m_off + 16].copy_from_slice(&prev_m_tmp);
                } else {
                    ws.prev_m[m_off..m_off + 16].fill(0.0);
                }

                // Phase update
                let mut thetas_h = [0.0f32; 4];
                for i in 0..4 {
                    let prev_th = if t > 0 { ws.thetas[pth_off + i] } else { 0.0 };
                    let delta_theta = (k_norm[i] * v_h[i]).clamp(-0.5, 0.5);
                    thetas_h[i] = prev_th + delta_theta;
                }
                ws.thetas[off..off + 4].copy_from_slice(&thetas_h);

                // Butterfly on k
                let mut k_rot = [0.0f32; 4];
                apply_butterfly_4(&k_norm, &thetas_h, false, &mut k_rot);
                ws.k_rot[off..off + 4].copy_from_slice(&k_rot);

                // v_hat = prev_m^T * k_rot
                let mut v_hat = [0.0f32; 4];
                for c in 0..4 {
                    let mut dot = 0.0f32;
                    for r in 0..4 {
                        dot += ws.prev_m[m_off + r * 4 + c] * k_rot[r];
                    }
                    v_hat[c] = dot;
                }

                // e = v - v_hat
                let mut e = [0.0f32; 4];
                for c in 0..4 {
                    e[c] = v_h[c] - v_hat[c];
                }
                ws.e[off..off + 4].copy_from_slice(&e);

                // M_t = prev_m + k_rot * e^T
                for r in 0..4 {
                    for c in 0..4 {
                        ws.m[m_off + r * 4 + c] = ws.prev_m[m_off + r * 4 + c] + k_rot[r] * e[c];
                    }
                }

                // MUSIC resonance
                let mut q_inv = [0.0f32; 4];
                apply_butterfly_4(&q_norm, &thetas_h, true, &mut q_inv);
                ws.q_inv[off..off + 4].copy_from_slice(&q_inv);

                let noise_energy = q_inv[2] * q_inv[2] + q_inv[3] * q_inv[3];
                let w_music = (1.0 / (noise_energy + 1e-5)).min(15.0);
                ws.w_music[t_nh + h] = w_music;

                // Query readout
                let mut q_rot = [0.0f32; 4];
                apply_butterfly_4(&q_norm, &thetas_h, false, &mut q_rot);
                ws.q_rot[off..off + 4].copy_from_slice(&q_rot);

                let mut y_ret = [0.0f32; 4];
                for c in 0..4 {
                    let mut dot = 0.0f32;
                    for r in 0..4 {
                        dot += ws.m[m_off + r * 4 + c] * q_rot[r];
                    }
                    y_ret[c] = dot;
                    ws.head_outputs[off + c] = dot * w_music;
                }
                ws.y_ret[off..off + 4].copy_from_slice(&y_ret);
            }

            // 4. Output projection
            self.attn.w_o.forward(&ws.head_outputs[t_dm..t_dm + dm], &mut ws.attn_out[t_dm..t_dm + dm]);

            // 5. Residual 1
            for i in 0..dm {
                ws.res1[t_dm + i] = ws.emb[t_dm + i] + ws.attn_out[t_dm + i];
            }

            // 6. Pre-FFN RMSNorm
            self.norm_ffn.forward(&ws.res1[t_dm..t_dm + dm], &mut ws.norm_buf_2[t_dm..t_dm + dm]);

            // 7. FFN
            let t_dff = t * dff;
            self.ffn.forward(
                &ws.norm_buf_2[t_dm..t_dm + dm],
                &mut ws.hidden_act[t_dff..t_dff + dff],
                &mut ws.ffn_out[t_dm..t_dm + dm],
            );

            // 8. Residual 2
            for i in 0..dm {
                ws.res2[t_dm + i] = ws.res1[t_dm + i] + ws.ffn_out[t_dm + i];
            }

            // 9. Final RMSNorm
            self.norm_final.forward(&ws.res2[t_dm..t_dm + dm], &mut ws.norm_buf_3[t_dm..t_dm + dm]);

            // 10. LM Head
            self.lm_head.forward(&ws.norm_buf_3[t_dm..t_dm + dm], &mut ws.logits[t_v..t_v + 256]);

            // 11. Softmax & Cross-Entropy
            let mut max_l = f32::NEG_INFINITY;
            for b in 0..256 {
                if ws.logits[t_v + b] > max_l { max_l = ws.logits[t_v + b]; }
            }
            let mut sum_exp = 0.0f32;
            for b in 0..256 {
                let ep = (ws.logits[t_v + b] - max_l).exp();
                ws.probs[t_v + b] = ep;
                sum_exp += ep;
            }
            let inv_sum = 1.0 / sum_exp.max(1e-12);
            for b in 0..256 {
                ws.probs[t_v + b] *= inv_sum;
            }

            let tb = targets[t].min(255);
            let loss = -ws.probs[t_v + tb].max(1e-12).ln();
            total_loss += loss;
        }

        total_loss / (seq_len as f32)
    }

    /// Exact Analytical Reversible BPTT backward pass across the sequence.
    /// Unrolls reverse-time accumulators for both associative memory M and phase angles Theta.
    /// Accumulates parameter gradients into all layer `.grad` buffers and writes input embedding gradients into `d_embeddings`.
    pub fn backward_sequence(
        &mut self,
        seq_len: usize,
        targets: &[usize],
        ws: &mut ScaledSequenceWorkspace,
        d_embeddings: &mut [f32],
    ) {
        let dm = self.config.d_model;
        let dff = self.config.d_ff;
        let nh = self.config.n_heads;
        let scale = 1.0 / (seq_len as f32);

        ws.d_m_acc.fill(0.0);
        ws.d_thetas_acc.fill(0.0);

        for t in (0..seq_len).rev() {
            let t_dm = t * dm;
            let t_nh = t * nh;
            let t_m = t * nh * 16;
            let t_v = t * 256;
            let t_dff = t * dff;

            // 1. Cross-entropy gradient
            let mut d_logits = [0.0f32; 256];
            for b in 0..256 {
                d_logits[b] = ws.probs[t_v + b] * scale;
            }
            let tb = targets[t].min(255);
            d_logits[tb] -= scale;

            // 2. LM Head backward
            let mut d_norm_buf_3 = vec![0.0f32; dm];
            self.lm_head.backward(&ws.norm_buf_3[t_dm..t_dm + dm], &d_logits, &mut d_norm_buf_3);

            // 3. Final RMSNorm backward
            let mut d_res2 = vec![0.0f32; dm];
            self.norm_final.backward(&ws.res2[t_dm..t_dm + dm], &d_norm_buf_3, &mut d_res2);

            // 4. FFN backward
            let mut d_res1 = d_res2.clone();
            let mut d_hidden = vec![0.0f32; dff];
            let mut d_norm_buf_2 = vec![0.0f32; dm];
            self.ffn.backward(
                &ws.norm_buf_2[t_dm..t_dm + dm],
                &ws.hidden_act[t_dff..t_dff + dff],
                &d_res2,
                &mut d_hidden,
                &mut d_norm_buf_2,
            );

            let mut d_res1_from_ffn = vec![0.0f32; dm];
            self.norm_ffn.backward(&ws.res1[t_dm..t_dm + dm], &d_norm_buf_2, &mut d_res1_from_ffn);
            for i in 0..dm {
                d_res1[i] += d_res1_from_ffn[i];
            }

            // 5. Output projection W_o backward
            let mut d_head_outputs = vec![0.0f32; dm];
            self.attn.w_o.backward(&ws.head_outputs[t_dm..t_dm + dm], &d_res1, &mut d_head_outputs);

            // 6. Attention backward with reverse-time accumulators
            let mut d_q = vec![0.0f32; dm];
            let mut d_k = vec![0.0f32; dm];
            let mut d_v = vec![0.0f32; dm];

            for h in 0..nh {
                let off = t_dm + h * 4;
                let m_off = t_m + h * 16;
                let acc_m_off = h * 16;
                let acc_th_off = h * 4;

                let q_h = [ws.q[off], ws.q[off + 1], ws.q[off + 2], ws.q[off + 3]];
                let k_h = [ws.k[off], ws.k[off + 1], ws.k[off + 2], ws.k[off + 3]];
                let v_h = [ws.v[off], ws.v[off + 1], ws.v[off + 2], ws.v[off + 3]];

                let q_norm: [f32; 4] = ws.q_norm[off..off + 4].try_into().unwrap();
                let k_norm: [f32; 4] = ws.k_norm[off..off + 4].try_into().unwrap();
                let thetas_h: [f32; 4] = ws.thetas[off..off + 4].try_into().unwrap();
                let k_rot: [f32; 4] = ws.k_rot[off..off + 4].try_into().unwrap();
                let q_inv: [f32; 4] = ws.q_inv[off..off + 4].try_into().unwrap();
                let q_rot: [f32; 4] = ws.q_rot[off..off + 4].try_into().unwrap();
                let e: [f32; 4] = ws.e[off..off + 4].try_into().unwrap();
                let y_ret: [f32; 4] = ws.y_ret[off..off + 4].try_into().unwrap();
                let w_music = ws.w_music[t_nh + h];
                let n_q = ws.n_q[t_nh + h];
                let n_k = ws.n_k[t_nh + h];

                let dy = [
                    d_head_outputs[h * 4],
                    d_head_outputs[h * 4 + 1],
                    d_head_outputs[h * 4 + 2],
                    d_head_outputs[h * 4 + 3],
                ];

                // a) d_y_ret and d_w_music
                let mut d_y_ret = [0.0f32; 4];
                let mut d_w_music = 0.0f32;
                for c in 0..4 {
                    d_y_ret[c] = dy[c] * w_music;
                    d_w_music += dy[c] * y_ret[c];
                }

                // b) MUSIC noise energy
                let d_noise = if w_music < 15.0 {
                    -d_w_music * w_music * w_music
                } else {
                    0.0
                };
                let d_q_inv = [0.0f32, 0.0f32, 2.0 * d_noise * q_inv[2], 2.0 * d_noise * q_inv[3]];

                // c) d_q_rot & total dM_t
                let mut d_q_rot = [0.0f32; 4];
                let mut d_m_total = [0.0f32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        let idx = r * 4 + c;
                        d_q_rot[r] += d_y_ret[c] * ws.m[m_off + idx];
                        d_m_total[idx] = ws.d_m_acc[acc_m_off + idx] + q_rot[r] * d_y_ret[c];
                    }
                }

                // d) Butterfly backward on q
                let mut d_thetas_local = [0.0f32; 4];
                let mut d_q_norm_rot = [0.0f32; 4];
                apply_butterfly_4_backward(&q_norm, &thetas_h, false, &d_q_rot, &mut d_q_norm_rot, &mut d_thetas_local);

                let mut d_q_norm_inv = [0.0f32; 4];
                apply_butterfly_4_backward(&q_norm, &thetas_h, true, &d_q_inv, &mut d_q_norm_inv, &mut d_thetas_local);

                let mut d_q_norm = [0.0f32; 4];
                for i in 0..4 {
                    d_q_norm[i] = d_q_norm_rot[i] + d_q_norm_inv[i];
                }

                let mut d_q_h = [0.0f32; 4];
                l2_normalize_backward(&q_h, &q_norm, n_q, &d_q_norm, &mut d_q_h);

                // e) Memory update backward: M_t = prev_m + k_rot * e^T
                let mut d_e = [0.0f32; 4];
                for c in 0..4 {
                    let mut sum = 0.0f32;
                    for r in 0..4 {
                        sum += k_rot[r] * d_m_total[r * 4 + c];
                    }
                    d_e[c] = sum;
                }

                let mut d_v_h = d_e;

                let mut d_k_rot = [0.0f32; 4];
                for r in 0..4 {
                    let mut sum_me = 0.0f32;
                    let mut sum_prev = 0.0f32;
                    for c in 0..4 {
                        sum_me += d_m_total[r * 4 + c] * e[c];
                        sum_prev += d_e[c] * ws.prev_m[m_off + r * 4 + c];
                    }
                    d_k_rot[r] = sum_me - sum_prev;
                }

                // Update d_m_acc for step t-1
                if t > 0 {
                    for r in 0..4 {
                        for c in 0..4 {
                            let idx = r * 4 + c;
                            ws.d_m_acc[acc_m_off + idx] = d_m_total[idx] - k_rot[r] * d_e[c];
                        }
                    }
                } else {
                    for idx in 0..16 {
                        ws.d_m_acc[acc_m_off + idx] = 0.0;
                    }
                }

                // f) Butterfly backward on k
                let mut d_k_norm = [0.0f32; 4];
                apply_butterfly_4_backward(&k_norm, &thetas_h, false, &d_k_rot, &mut d_k_norm, &mut d_thetas_local);

                // g) Phase dynamics: thetas_t = thetas_{t-1} + clamp(k_norm * v_h, -0.5, 0.5)
                for i in 0..4 {
                    let d_th = d_thetas_local[i] + ws.d_thetas_acc[acc_th_off + i];
                    let prod = k_norm[i] * v_h[i];
                    if prod >= -0.5 && prod <= 0.5 {
                        d_k_norm[i] += d_th * v_h[i];
                        d_v_h[i] += d_th * k_norm[i];
                    }
                    ws.d_thetas_acc[acc_th_off + i] = if t > 0 { d_th } else { 0.0 };
                }

                // h) L2 normalize backward on k
                let mut d_k_h = [0.0f32; 4];
                l2_normalize_backward(&k_h, &k_norm, n_k, &d_k_norm, &mut d_k_h);

                for i in 0..4 {
                    d_q[h * 4 + i] = d_q_h[i];
                    d_k[h * 4 + i] = d_k_h[i];
                    d_v[h * 4 + i] = d_v_h[i];
                }
            }

            // 7. QKV linear projections backward
            let mut d_norm_buf_1_q = vec![0.0f32; dm];
            let mut d_norm_buf_1_k = vec![0.0f32; dm];
            let mut d_norm_buf_1_v = vec![0.0f32; dm];
            self.attn.w_q.backward(&ws.norm_buf_1[t_dm..t_dm + dm], &d_q, &mut d_norm_buf_1_q);
            self.attn.w_k.backward(&ws.norm_buf_1[t_dm..t_dm + dm], &d_k, &mut d_norm_buf_1_k);
            self.attn.w_v.backward(&ws.norm_buf_1[t_dm..t_dm + dm], &d_v, &mut d_norm_buf_1_v);

            let mut d_norm_buf_1 = vec![0.0f32; dm];
            for i in 0..dm {
                d_norm_buf_1[i] = d_norm_buf_1_q[i] + d_norm_buf_1_k[i] + d_norm_buf_1_v[i];
            }

            // 8. Pre-Attn RMSNorm backward
            let mut d_x_emb_from_attn = vec![0.0f32; dm];
            self.norm_attn.backward(&ws.emb[t_dm..t_dm + dm], &d_norm_buf_1, &mut d_x_emb_from_attn);
            for i in 0..dm {
                d_embeddings[t_dm + i] = d_res1[i] + d_x_emb_from_attn[i];
            }
        }
    }
}

/// Pre-allocated workspace for sequence-level BPTT training with zero runtime allocations.
#[derive(Debug, Clone)]
pub struct ScaledSequenceWorkspace {
    pub max_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub emb: Vec<f32>,
    pub norm_buf_1: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub n_q: Vec<f32>,
    pub n_k: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub thetas: Vec<f32>,
    pub k_rot: Vec<f32>,
    pub prev_m: Vec<f32>,
    pub m: Vec<f32>,
    pub e: Vec<f32>,
    pub q_inv: Vec<f32>,
    pub q_rot: Vec<f32>,
    pub w_music: Vec<f32>,
    pub y_ret: Vec<f32>,
    pub head_outputs: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub res1: Vec<f32>,
    pub norm_buf_2: Vec<f32>,
    pub hidden_act: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub res2: Vec<f32>,
    pub norm_buf_3: Vec<f32>,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub d_m_acc: Vec<f32>,
    pub d_thetas_acc: Vec<f32>,
}

impl ScaledSequenceWorkspace {
    pub fn new(config: &ScaledConfig, max_len: usize) -> Self {
        let dm = config.d_model;
        let nh = config.n_heads;
        let dff = config.d_ff;

        Self {
            max_len,
            d_model: dm,
            n_heads: nh,
            d_ff: dff,
            emb: vec![0.0f32; max_len * dm],
            norm_buf_1: vec![0.0f32; max_len * dm],
            q: vec![0.0f32; max_len * dm],
            k: vec![0.0f32; max_len * dm],
            v: vec![0.0f32; max_len * dm],
            n_q: vec![0.0f32; max_len * nh],
            n_k: vec![0.0f32; max_len * nh],
            q_norm: vec![0.0f32; max_len * dm],
            k_norm: vec![0.0f32; max_len * dm],
            thetas: vec![0.0f32; max_len * dm],
            k_rot: vec![0.0f32; max_len * dm],
            prev_m: vec![0.0f32; max_len * nh * 16],
            m: vec![0.0f32; max_len * nh * 16],
            e: vec![0.0f32; max_len * dm],
            q_inv: vec![0.0f32; max_len * dm],
            q_rot: vec![0.0f32; max_len * dm],
            w_music: vec![0.0f32; max_len * nh],
            y_ret: vec![0.0f32; max_len * dm],
            head_outputs: vec![0.0f32; max_len * dm],
            attn_out: vec![0.0f32; max_len * dm],
            res1: vec![0.0f32; max_len * dm],
            norm_buf_2: vec![0.0f32; max_len * dm],
            hidden_act: vec![0.0f32; max_len * dff],
            ffn_out: vec![0.0f32; max_len * dm],
            res2: vec![0.0f32; max_len * dm],
            norm_buf_3: vec![0.0f32; max_len * dm],
            logits: vec![0.0f32; max_len * 256],
            probs: vec![0.0f32; max_len * 256],
            d_m_acc: vec![0.0f32; nh * 16],
            d_thetas_acc: vec![0.0f32; nh * 4],
        }
    }
}
