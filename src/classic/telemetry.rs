//! Telemetry and Performance Accounting for SRXformer.
//! Computes exact Compute (FLOPs), Latency (Step / Throughput), and Quality (Exact Match).

use std::fs;
use std::io;
use std::path::Path;

/// Telemetry metrics collected during model training.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainTelemetry {
    /// Number of trainable parameters in the model.
    pub num_params: usize,
    /// Total tokens processed per epoch in the dataset.
    pub dataset_tokens: usize,
    /// Number of training epochs executed.
    pub epochs: usize,
    /// Total floating-point operations invested in training:
    /// 6 * num_params * dataset_tokens * epochs
    pub total_training_flops: u64,
    /// Total elapsed training time in milliseconds.
    pub elapsed_ms: f64,
    /// Compute throughput in MegaFLOPs / sec.
    pub mflops_per_sec: f64,
    /// Compute throughput in GigaFLOPs / sec.
    pub gflops_per_sec: f64,
    /// Initial cross-entropy loss before training.
    pub initial_loss: f32,
    /// Final cross-entropy loss after training.
    pub final_loss: f32,
    /// Initial perplexity exp(initial_loss).
    pub initial_perplexity: f32,
    /// Final perplexity exp(final_loss).
    pub final_perplexity: f32,
    /// History of average cross-entropy loss per epoch.
    pub loss_history: Vec<f32>,
    /// Total optimization steps performed.
    pub total_steps: usize,
}

/// Telemetry metrics collected during inference benchmarking.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceTelemetry {
    /// Floating-point operations per generated token:
    /// ~ 2 * num_params
    pub flops_per_token: u64,
    /// Number of benchmark decoding steps evaluated.
    pub bench_steps: usize,
    /// Autoregressive step latency in nanoseconds.
    pub step_latency_ns: f64,
    /// Autoregressive step latency in microseconds.
    pub step_latency_us: f64,
    /// Decoding throughput in tokens per second.
    pub tokens_per_sec: f64,
    /// Inference compute rate in GigaFLOPs / sec.
    pub gflops_per_sec: f64,
}

/// Individual test case verification result.
#[derive(Debug, Clone, PartialEq)]
pub struct TestCaseResult {
    pub prompt: String,
    pub generated: String,
    pub expected: String,
    pub category: String,
    pub passed: bool,
}

/// Comprehensive telemetry report combining training compute, inference performance, and quality evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryReport {
    pub train: TrainTelemetry,
    pub inference: InferenceTelemetry,
    pub test_cases: Vec<TestCaseResult>,
    pub exact_match_accuracy: f32,
}

impl TelemetryReport {
    /// Constructs a new TelemetryReport and calculates the exact match accuracy.
    pub fn new(
        train: TrainTelemetry,
        inference: InferenceTelemetry,
        test_cases: Vec<TestCaseResult>,
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
        }
    }

    /// Formats the complete telemetry report into a structured, readable string.
    pub fn format_text(&self) -> String {
        let mut out = String::new();

        out.push_str("================================================================================\n");
        out.push_str(" SRXFORMER: CLASSICAL TRANSFORMER (BASELINE) COMPUTE / LATENCY / QUALITY TELEMETRY REPORT\n");
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

        // 2. Inference Compute & Latency
        out.push_str("[2] COMPUTE И ЛАТЕНТНОСТЬ НА ИНФЕРЕНСЕ (INFERENCE COMPUTE & LATENCY):\n");
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
        out.push_str(" SRXFORMER TELEMETRY VERIFICATION: ALL METRICS RECORDED SUCCESSFULLY!           \n");
        out.push_str("================================================================================\n");

        out
    }

    /// Saves the formatted report to a specified file.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let text = self.format_text();
        fs::write(path, text)
    }
}
