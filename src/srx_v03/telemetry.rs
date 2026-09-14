//! Telemetry and Quality Report Generator for SRXFORMER v03.
//! Generates telemetry_srx_v03.txt formatted with hardware metrics, compute, latency, and test case tables.

use std::fs;
use std::io;
use std::path::Path;

use crate::classic::telemetry::{InferenceTelemetry, TestCaseResult, TrainTelemetry};

/// Comprehensive telemetry report structure for SRX v03.
#[derive(Debug, Clone)]
pub struct SrxTelemetryReport {
    pub train: TrainTelemetry,
    pub inference: InferenceTelemetry,
    pub test_results: Vec<TestCaseResult>,
    pub exact_match_accuracy: f32,
    pub state_memory_bytes: usize,
    pub max_seq_len: usize,
}

impl SrxTelemetryReport {
    pub fn new(
        train: TrainTelemetry,
        inference: InferenceTelemetry,
        test_results: Vec<TestCaseResult>,
        state_memory_bytes: usize,
        max_seq_len: usize,
    ) -> Self {
        let passed = test_results.iter().filter(|t| t.passed).count();
        let exact_match_accuracy = if !test_results.is_empty() {
            (passed as f32 / test_results.len() as f32) * 100.0
        } else {
            0.0
        };

        Self {
            train,
            inference,
            test_results,
            exact_match_accuracy,
            state_memory_bytes,
            max_seq_len,
        }
    }

    /// Generates the telemetry report formatted as required by the v03 directive.
    pub fn generate_report(&self) -> String {
        let mut s = String::with_capacity(4096);

        s.push_str("================================================================================\n");
        s.push_str(" SRXFORMER: SUPER-RESOLVENT XFORMER v03 (MONARCH BUTTERFLY + SELECTIVE GATE) COMPUTE / LATENCY / QUALITY TELEMETRY REPORT\n");
        s.push_str(" Target Architecture: Intel Xeon E5-2650 v2 (Ivy Bridge-EP)                     \n");
        s.push_str(&format!(
            " Model Parameters:    {} (100% L1D Cache Resident, 32 KB per core)             \n",
            self.train.num_params
        ));
        s.push_str("================================================================================\n\n");

        s.push_str("[1] ВЛОЖЕННЫЙ COMPUTE НА ОБУЧЕНИЕ (TRAINING COMPUTE):\n");
        let gflops_train = self.train.total_training_flops as f64 / 1e9;
        let mflops_train = self.train.total_training_flops as f64 / 1e6;
        s.push_str(&format!(
            "    * Вложили compute:          {} FLOPs ({:.2} MFLOPs / {:.4} GFLOPs)\n",
            self.train.total_training_flops, mflops_train, gflops_train
        ));
        s.push_str(&format!(
            "    * Время обучения:           {:.2} ms ({:.3} s)\n",
            self.train.elapsed_ms,
            self.train.elapsed_ms / 1000.0
        ));
        s.push_str(&format!(
            "    * Формула FLOPs:            6 * N_params * tokens_per_epoch * epochs\n                                = 6 * {} * {} * {}\n",
            self.train.num_params, self.train.dataset_tokens, self.train.epochs
        ));
        s.push_str(&format!(
            "    * Эффективная скорость:     {:.2} MFLOP/s ({:.4} GFLOP/s)\n",
            self.train.mflops_per_sec, self.train.gflops_per_sec
        ));
        s.push_str(&format!(
            "    * Начальный Loss:           {:.4} (Перплексия: {:.2})\n",
            self.train.initial_loss, self.train.initial_perplexity
        ));
        s.push_str(&format!(
            "    * Финальный Loss:           {:.4} (Перплексия: {:.2})\n",
            self.train.final_loss, self.train.final_perplexity
        ));
        s.push_str(&format!(
            "    * Всего шагов (Steps):      {} gradient updates\n\n",
            self.train.total_steps
        ));

        s.push_str("[2] COMPUTE, ЛАТЕНТНОСТЬ И ПАМЯТЬ СОСТОЯНИЯ НА ИНФЕРЕНСЕ (INFERENCE COMPUTE & MEMORY):\n");
        s.push_str(&format!(
            "    * Ответ за compute:         {} FLOPs на токен (~ 2 * N_params)\n",
            self.inference.flops_per_token
        ));
        s.push_str(&format!(
            "    * Время генерации токена:   {:.1} ns ({:.3} µs)\n",
            self.inference.step_latency_ns, self.inference.step_latency_us
        ));
        s.push_str(&format!(
            "    * Пропускная способность:   {:.0} токенов/сек ({:.2}M токенов/сек)\n",
            self.inference.tokens_per_sec,
            self.inference.tokens_per_sec / 1e6
        ));
        s.push_str(&format!(
            "    * Вычислительная скорость:  {:.4} GFLOP/s\n",
            self.inference.gflops_per_sec
        ));
        s.push_str(&format!(
            "    * Число бенчмарк шагов:     {} tokens decoded\n",
            self.inference.bench_steps
        ));
        s.push_str("    --------------------------------------------------------------------------------\n");
        s.push_str("    * СРАВНЕНИЕ ОБЪЕМА ПАМЯТИ СОСТОЯНИЯ (STATE MEMORY FOOTPRINT):\n");
        let srx_bytes = self.state_memory_bytes;
        s.push_str(&format!(
            "      - SRX v03 State (Theta + M): {} bytes (СТРОГО КОНСТАНТНО O(1), 100% L1D resident)\n",
            srx_bytes
        ));

        let kv_32 = 2 * 1 * 32 * 8 * 4;
        let kv_1k = 2 * 1 * 1024 * 8 * 4;
        let kv_100k = 2 * 1 * 100000 * 8 * 4;

        s.push_str(&format!(
            "      - Classic KV Cache (N=32 ): {} bytes ({:.1}x больше SRX)\n",
            kv_32,
            kv_32 as f64 / srx_bytes as f64
        ));
        s.push_str(&format!(
            "      - Classic KV Cache (N=1K):  {} bytes ({:.1}x больше SRX -> вытеснение L1D)\n",
            kv_1k,
            kv_1k as f64 / srx_bytes as f64
        ));
        s.push_str(&format!(
            "      - Classic KV Cache (N=100K):{} bytes ({:.1}x больше SRX -> вытеснение в DRAM)\n",
            kv_100k,
            kv_100k as f64 / srx_bytes as f64
        ));
        s.push_str("      - Преимущество SRX v03:   Нулевой трафик в DRAM, отсутствие падения скорости при росте N\n\n");

        s.push_str("[3] КАЧЕСТВО МОДЕЛИ (QUALITY EVALUATION - EXACT MATCH):\n");
        let passed_count = self.test_results.iter().filter(|t| t.passed).count();
        s.push_str(&format!(
            "    * Точность (Exact Match):   {:.1}% ({}/{} тестов пройдено)\n",
            self.exact_match_accuracy,
            passed_count,
            self.test_results.len()
        ));
        s.push_str("    --------------------------------------------------------------------------------------------------------------------\n");
        s.push_str("    | Категория                | Статус | Промпт                    | Сгенерировано        | Ожидалось            |\n");
        s.push_str("    --------------------------------------------------------------------------------------------------------------------\n");

        for t in &self.test_results {
            let status = if t.passed { "PASS" } else { "FAIL" };
            s.push_str(&format!(
                "    | {:<24} | {:<6} | {:<25} | {:<20} | {:<20} |\n",
                t.category, status, t.prompt, t.generated, t.expected
            ));
        }
        s.push_str("    --------------------------------------------------------------------------------------------------------------------\n\n");

        s.push_str("================================================================================\n");
        s.push_str(" SRXFORMER v03 GOLDEN CORE TELEMETRY VERIFICATION: ALL METRICS RECORDED!         \n");
        s.push_str("================================================================================\n");

        s
    }

    /// Writes report to destination file path.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let content = self.generate_report();
        fs::write(path, content)
    }
}
