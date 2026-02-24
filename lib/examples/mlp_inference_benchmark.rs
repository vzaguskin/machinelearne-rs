//! MLP Inference Benchmark Example
//!
//! Compares inference performance across different backends:
//! - Native CPU inference
//! - Native WGPU inference (when available)
//! - ONNX Runtime inference (when available)
//!
//! # Usage
//!
//! ```bash
//! cargo run --example mlp_inference_benchmark
//! cargo run --example mlp_inference_benchmark --features wgpu
//! cargo run --example mlp_inference_benchmark --features onnx
//! ```

use machinelearne_rs::backend::{CpuBackend, Tensor2D};
use machinelearne_rs::dataset::memory::InMemoryDataset;
use machinelearne_rs::loss::MSELoss;
use machinelearne_rs::model::{Activation, InferenceModel, MLP};
use machinelearne_rs::optimizer::SGD;
use machinelearne_rs::regularizers::NoRegularizer;
use machinelearne_rs::trainer::Trainer;
use std::time::Instant;

/// Configuration for a benchmark test.
struct BenchmarkConfig {
    name: String,
    layer_sizes: Vec<usize>,
    activations: Vec<Activation>,
}

impl BenchmarkConfig {
    fn new(name: &str, layer_sizes: Vec<usize>, activations: Vec<Activation>) -> Self {
        Self {
            name: name.to_string(),
            layer_sizes,
            activations,
        }
    }

    fn n_features(&self) -> usize {
        *self.layer_sizes.first().unwrap_or(&0)
    }

    fn n_params(&self) -> usize {
        let mut total = 0;
        for window in self.layer_sizes.windows(2) {
            total += window[0] * window[1] + window[1]; // weights + bias
        }
        total
    }
}

/// Results from a benchmark run.
#[derive(Debug)]
struct BenchmarkResult {
    name: String,
    architecture: String,
    batch_size: usize,
    n_samples: usize,
    total_time_ms: f64,
    throughput_samples_per_sec: f64,
    latency_ms_per_sample: f64,
}

impl BenchmarkResult {
    fn print(&self) {
        println!(
            "{:<20} | batch={:<5} | samples={:<6} | time={:>8.2}ms | throughput={:>8.0} samples/s | latency={:>6.3}ms",
            self.name,
            self.batch_size,
            self.n_samples,
            self.total_time_ms,
            self.throughput_samples_per_sec,
            self.latency_ms_per_sample
        );
    }

    fn header() {
        println!(
            "{:<20} | {:<12} | {:<14} | {:<15} | {:<23} | {:<15}",
            "Backend",
            "Batch Size",
            "Samples",
            "Time (ms)",
            "Throughput (samples/s)",
            "Latency (ms)"
        );
        println!("{}", "-".repeat(110));
    }
}

/// Run CPU inference benchmark.
fn benchmark_cpu_inference(
    model: &machinelearne_rs::model::MLPModel<CpuBackend, machinelearne_rs::model::Fitted>,
    data: &Tensor2D<CpuBackend>,
    batch_size: usize,
    n_iterations: usize,
) -> BenchmarkResult {
    let (n_samples, _) = data.shape();

    // Warmup
    for _ in 0..3 {
        let _ = model.predict_batch(data);
    }

    let start = Instant::now();
    for _ in 0..n_iterations {
        let _ = model.predict_batch(data);
    }
    let elapsed = start.elapsed();

    let total_samples = n_samples * n_iterations;
    let total_time_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput = total_samples as f64 / (total_time_ms / 1000.0);
    let latency = total_time_ms / total_samples as f64;

    BenchmarkResult {
        name: "CPU".to_string(),
        architecture: "".to_string(),
        batch_size,
        n_samples: total_samples,
        total_time_ms,
        throughput_samples_per_sec: throughput,
        latency_ms_per_sample: latency,
    }
}

/// Generate synthetic data for benchmarking.
fn generate_synthetic_data(
    n_samples: usize,
    n_features: usize,
) -> (Tensor2D<CpuBackend>, InMemoryDataset) {
    let mut features = Vec::with_capacity(n_samples);
    let mut targets = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let sample: Vec<f32> = (0..n_features)
            .map(|j| ((i * n_features + j) % 100) as f32 / 100.0)
            .collect();
        // Simple target: sum of features (doesn't matter for benchmarking)
        let target = sample.iter().sum();
        features.push(sample);
        targets.push(target);
    }

    let x = Tensor2D::<CpuBackend>::new(
        features.iter().flatten().copied().collect(),
        n_samples,
        n_features,
    );
    let dataset = InMemoryDataset::new(features, targets).unwrap();

    (x, dataset)
}

fn main() {
    println!("MLP Inference Benchmark");
    println!("=======================\n");

    // Define test configurations
    let configs = vec![
        BenchmarkConfig::new(
            "Small (8->16->8->1)",
            vec![8, 16, 8, 1],
            vec![Activation::ReLU, Activation::ReLU, Activation::Identity],
        ),
        BenchmarkConfig::new(
            "Medium (8->64->32->1)",
            vec![8, 64, 32, 1],
            vec![Activation::ReLU, Activation::Tanh, Activation::Identity],
        ),
        BenchmarkConfig::new(
            "Large (8->128->64->1)",
            vec![8, 128, 64, 1],
            vec![Activation::ReLU, Activation::ReLU, Activation::Identity],
        ),
        BenchmarkConfig::new(
            "Deep (8->32->32->32->1)",
            vec![8, 32, 32, 32, 1],
            vec![
                Activation::ReLU,
                Activation::ReLU,
                Activation::ReLU,
                Activation::Identity,
            ],
        ),
    ];

    let batch_sizes = vec![1, 16, 64, 256, 1024];
    let n_iterations = 100; // Number of iterations per benchmark

    println!("Configurations:");
    for config in &configs {
        println!("  {} - {} parameters", config.name, config.n_params());
    }
    println!();

    for config in &configs {
        println!("\n========================================");
        println!("Architecture: {}", config.name);
        println!("Parameters: {}", config.n_params());
        println!("========================================\n");

        // Create and train model briefly (just to get fitted weights)
        let model = MLP::<CpuBackend>::new(&config.layer_sizes, &config.activations);

        // Generate training data for a quick fit
        let (_, train_dataset) = generate_synthetic_data(1000, config.n_features());

        // Train briefly
        let trainer = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
            .batch_size(32)
            .max_epochs(10)
            .verbose(false)
            .build();
        let fitted = trainer.fit(model, &train_dataset).unwrap();

        BenchmarkResult::header();

        for batch_size in &batch_sizes {
            // Generate benchmark data
            let (data, _) = generate_synthetic_data(*batch_size, config.n_features());

            // CPU benchmark
            let cpu_result = benchmark_cpu_inference(&fitted, &data, *batch_size, n_iterations);
            let mut result = cpu_result;
            result.architecture = config.name.clone();
            result.print();
        }

        // WGPU benchmark (if available)
        #[cfg(feature = "wgpu")]
        {
            use machinelearne_rs::backend::wgpu_backend::WgpuDevice;

            println!("\n--- WGPU Backend ---");
            let device = WgpuDevice::global();

            // Convert model to WGPU
            // Note: This would require implementing WGPU backend support for MLP
            // For now, we'll just note that WGPU support is available
            println!("WGPU backend available but MLP WGPU inference not yet optimized.");
            println!("See mlp_cpu_wgpu_comparison example for training benchmarks.");
        }

        // ONNX Runtime benchmark (if available)
        #[cfg(feature = "onnx")]
        {
            println!("\n--- ONNX Runtime Backend ---");
            println!("ONNX export available. See export_mlp_onnx example.");
            println!("Run onnx_inference example for ONNX Runtime benchmarks.");
        }
    }

    println!("\n========================================");
    println!("Benchmark Summary");
    println!("========================================\n");
    println!("This benchmark measures native CPU inference performance.");
    println!("\nFor additional backends:");
    println!("  - WGPU: cargo run --example mlp_inference_benchmark --features wgpu");
    println!("  - ONNX: cargo run --example mlp_inference_benchmark --features onnx");
    println!("\nKey observations:");
    println!("  1. Larger batch sizes generally improve throughput (samples/sec)");
    println!("  2. Latency per sample decreases with larger batches (better parallelism)");
    println!("  3. Model size (parameters) affects both throughput and latency");
    println!("  4. Deep networks may have different characteristics than wide networks");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_n_params() {
        let config = BenchmarkConfig::new(
            "test",
            vec![8, 16, 8],
            vec![Activation::ReLU, Activation::Identity],
        );
        // 8*16 + 16 + 16*8 + 8 = 128 + 16 + 128 + 8 = 280
        assert_eq!(config.n_params(), 280);
    }

    #[test]
    fn test_generate_synthetic_data() {
        let (x, dataset) = generate_synthetic_data(100, 8);
        assert_eq!(x.shape(), (100, 8));
        assert_eq!(dataset.len(), Some(100));
    }
}
