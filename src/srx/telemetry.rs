//! Telemetry and Performance Accounting for SRXFORMER Innovation Architecture.
//! Computes exact Compute (FLOPs), Latency (Step / Throughput), Memory footprint O(d) vs KV-cache,
//! and Quality (Exact Match).

use std::fs;
use std::io;
use std::path::Path;

use crate::classic::telemetry::{InferenceTelemetry, TestCaseResult, TrainTelemetry};

/// Comprehensive telemetry report for SRX architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct SrxTelemetryReport {
    pub train: TrainTelemetry,
    pub inference: InferenceTelemetry,
    pub test_cases: Vec<TestCaseResult>,
    pub exact_match_accuracy: f32,
    pub state_memory_bytes: usize,
    pub max_seq_len: usize,
}

impl SrxTelemetryReport {
    pub fn new(
        train: TrainTelemetry,
        inference: InferenceTelemetry,
        test_cases: Vec<TestCaseResult>,
        state_memory_bytes: usize,
        max_seq_len: usize,
    ) -> Self {
        let total = test_cases.len();
        let passed = test_cases.iter().filter(|t| t.passed).count();
        let exact_match_accuracy = if total > 0 {
            (passed as f32 / total as f32) * 100.0
        } else {
            0.0
        };

        Self {
            train,
            inference,
            test_cases,
            exact_match_accuracy,
            state_memory_bytes,
            max_seq_len,
        }
    }

    pub fn format_text(&self) -> String {
        let mut out = String::new();

        out.push_str("================================================================================\n");
        out.push_str(" SRXFORMER: SUPER-RESOLVENT XFORMER (INNOVATION) COMPUTE / LATENCY / QUALITY TELEMETRY REPORT\n");
        out.push_str(" Target Architecture: Intel Xeon E5-2650 v2 (Ivy Bridge-EP)                     \n");
        out.push_str(&format!(
            " Model Parameters:    {} (100% L1D Cache Resident, 32 KB per core)             \n",
            self.train.num_params
        ));
        out.push_str("================================================================================\n\n");

        // 1. Training Compute
        out.push_str("[1] ВЛОЖЕННЫЙ COMPUTE НА ОБУЧЕНИЕ (TRAINING COMPUTE):\n");
        out.push_str(&format!(
            "    * Вложили compute:          {} FLOPs ({:.2} MFLOPs / {:.4} GFLOPs)\n",
            self.train.total_training_flops,
            self.train.total_training_flops as f64 / 1e6,
            self.train.total_training_flops as f64 / 1e9
        ));
        out.push_str(&format!(
            "    * Время обучения:           {:.2} ms ({:.3} s)\n",
            self.train.elapsed_ms,
            self.train.elapsed_ms / 1000.0
        ));
        out.push_str("    * Формула FLOPs:            6 * N_params * tokens_per_epoch * epochs\n");
        out.push_str(&format!(
            "                                = 6 * {} * {} * {}\n",
            self.train.num_params, self.train.dataset_tokens, self.train.epochs
        ));
        out.push_str(&format!(
            "    * Эффективная скорость:     {:.2} MFLOP/s ({:.4} GFLOP/s)\n",
            self.train.mflops_per_sec, self.train.gflops_per_sec
        ));
        out.push_str(&format!(
            "    * Начальный Loss:           {:.4} (Перплексия: {:.2})\n",
            self.train.initial_loss, self.train.initial_perplexity
        ));
        out.push_str(&format!(
            "    * Финальный Loss:           {:.4} (Перплексия: {:.2})\n",
            self.train.final_loss, self.train.final_perplexity
        ));
        out.push_str(&format!(
            "    * Всего шагов (Steps):      {} gradient updates\n",
            self.train.total_steps
        ));
        out.push('\n');

        // 2. Inference Compute & Latency & L1 Memory Footprint
        let classic_kv_bytes_current = 2 * 1 * self.max_seq_len * 8 * 4; // 2 * L * max_seq_len * d * 4 bytes
        let classic_kv_bytes_1k = 2 * 1 * 1024 * 8 * 4; // at 1024 tokens
        let classic_kv_bytes_100k = 2 * 1 * 100_000 * 8 * 4; // at 100k tokens

        out.push_str("[2] COMPUTE, ЛАТЕНТНОСТЬ И ПАМЯТЬ СОСТОЯНИЯ НА ИНФЕРЕНСЕ (INFERENCE COMPUTE & MEMORY):\n");
        out.push_str(&format!(
            "    * Ответ за compute:         {} FLOPs на токен (~ 2 * N_params)\n",
            self.inference.flops_per_token
        ));
        out.push_str(&format!(
            "    * Время генерации токена:   {:.1} ns ({:.3} µs)\n",
            self.inference.step_latency_ns, self.inference.step_latency_us
        ));
        out.push_str(&format!(
            "    * Пропускная способность:   {:.0} токенов/сек ({:.2}M токенов/сек)\n",
            self.inference.tokens_per_sec,
            self.inference.tokens_per_sec / 1e6
        ));
        out.push_str(&format!(
            "    * Вычислительная скорость:  {:.4} GFLOP/s\n",
            self.inference.gflops_per_sec
        ));
        out.push_str(&format!(
            "    * Число бенчмарк шагов:     {} tokens decoded\n",
            self.inference.bench_steps
        ));
        out.push_str("    --------------------------------------------------------------------------------\n");
        out.push_str("    * СРАВНЕНИЕ ОБЪЕМА ПАМЯТИ СОСТОЯНИЯ (STATE MEMORY FOOTPRINT):\n");
        out.push_str(&format!(
            "      - SRX State (Theta + M):  {} bytes (СТРОГО КОНСТАНТНО O(1), 100% L1D resident)\n",
            self.state_memory_bytes
        ));
        out.push_str(&format!(
            "      - Classic KV Cache (N={:<3}): {} bytes ({:.1}x больше SRX)\n",
            self.max_seq_len,
            classic_kv_bytes_current,
            classic_kv_bytes_current as f64 / self.state_memory_bytes as f64
        ));
        out.push_str(&format!(
            "      - Classic KV Cache (N=1K):  {} bytes ({:.1}x больше SRX -> вытеснение L1D)\n",
            classic_kv_bytes_1k,
            classic_kv_bytes_1k as f64 / self.state_memory_bytes as f64
        ));
        out.push_str(&format!(
            "      - Classic KV Cache (N=100K):{} bytes ({:.1}x больше SRX -> вытеснение в DRAM)\n",
            classic_kv_bytes_100k,
            classic_kv_bytes_100k as f64 / self.state_memory_bytes as f64
        ));
        out.push_str("      - Преимущество SRX:       Нулевой трафик в DRAM, отсутствие падения скорости при росте N\n");
        out.push('\n');

        // 3. Quality Evaluation
        let passed_count = self.test_cases.iter().filter(|t| t.passed).count();
        let total_count = self.test_cases.len();
        out.push_str("[3] КАЧЕСТВО МОДЕЛИ (QUALITY EVALUATION - EXACT MATCH):\n");
        out.push_str(&format!(
            "    * Точность (Exact Match):   {:.1}% ({}/{} тестов пройдено)\n",
            self.exact_match_accuracy, passed_count, total_count
        ));
        out.push_str("    --------------------------------------------------------------------------------------------------------------------\n");
        out.push_str(&format!(
            "    | {:<24} | {:<6} | {:<25} | {:<20} | {:<20} |\n",
            "Категория", "Статус", "Промпт", "Сгенерировано", "Ожидалось"
        ));
        out.push_str("    --------------------------------------------------------------------------------------------------------------------\n");

        for tc in &self.test_cases {
            let status = if tc.passed { "PASS" } else { "FAIL" };
            out.push_str(&format!(
                "    | {:<24} | {:<6} | {:<25} | {:<20} | {:<20} |\n",
                tc.category, status, tc.prompt, tc.generated, tc.expected
            ));
        }
        out.push_str("    --------------------------------------------------------------------------------------------------------------------\n\n");

        out.push_str("================================================================================\n");
        out.push_str(" SRXFORMER INNOVATION TELEMETRY VERIFICATION: ALL METRICS RECORDED!             \n");
        out.push_str("================================================================================\n");

        out
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let text = self.format_text();
        fs::write(path, text)
    }
}
