//! MLP Inference Benchmark Example
//!
//! Compares inference performance and correctness across different backends:
//! - Native CPU inference (always available)
//! - Native WGPU inference (when compiled with --features wgpu)
//! - ONNX Runtime CPU inference (when compiled with --features onnx-inference)
//! - ONNX Runtime GPU inference (when compiled with --features onnx-cuda)
//!
//! # Usage
//!
//! ```bash
//! # CPU only
//! cargo run --example mlp_inference_benchmark
//!
//! # With WGPU support
//! cargo run --example mlp_inference_benchmark --features wgpu
//!
//! # With ONNX CPU inference
//! cargo run --example mlp_inference_benchmark --features onnx-inference
//!
//! # With ONNX GPU inference (requires CUDA)
//! cargo run --example mlp_inference_benchmark --features onnx-cuda
//!
//! # All backends
//! cargo run --example mlp_inference_benchmark --features wgpu,onnx-inference
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
#[derive(Debug, Clone)]
struct BenchmarkResult {
    backend: String,
    architecture: String,
    batch_size: usize,
    n_samples: usize,
    n_iterations: usize,
    total_time_ms: f64,
    throughput_samples_per_sec: f64,
    latency_ms_per_sample: f64,
    /// Sample predictions for comparison
    sample_predictions: Vec<f64>,
}

impl BenchmarkResult {
    fn print(&self) {
        println!(
            "{:<15} | batch={:<5} | samples={:<6} | iters={:<4} | time={:>8.2}ms | throughput={:>8.0} samples/s | latency={:>6.3}ms",
            self.backend,
            self.batch_size,
            self.n_samples,
            self.n_iterations,
            self.total_time_ms,
            self.throughput_samples_per_sec,
            self.latency_ms_per_sample
        );
    }

    fn header() {
        println!(
            "{:<15} | {:<12} | {:<14} | {:<10} | {:<15} | {:<23} | {:<15}",
            "Backend",
            "Batch Size",
            "Samples",
            "Iters",
            "Time (ms)",
            "Throughput (samples/s)",
            "Latency (ms)"
        );
        println!("{}", "-".repeat(115));
    }
}

/// Compare predictions between two backends
fn compare_predictions(name1: &str, name2: &str, preds1: &[f64], preds2: &[f64], tolerance: f64) {
    if preds1.len() != preds2.len() {
        println!(
            "  {} vs {}: Different prediction counts ({} vs {}) - SKIPPED",
            name1,
            name2,
            preds1.len(),
            preds2.len()
        );
        return;
    }

    let mut max_diff = 0.0f64;
    let mut sum_diff = 0.0f64;
    let mut mismatches = 0;

    for (i, (a, b)) in preds1.iter().zip(preds2.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;
        if diff > tolerance {
            mismatches += 1;
            if mismatches <= 3 {
                println!(
                    "    Mismatch at [{}]: {} vs {} (diff: {:.6})",
                    i, a, b, diff
                );
            }
        }
    }

    let avg_diff = sum_diff / preds1.len() as f64;

    if mismatches == 0 {
        println!(
            "  {} vs {}: MATCH (max_diff: {:.6}, avg_diff: {:.6})",
            name1, name2, max_diff, avg_diff
        );
    } else {
        println!(
            "  {} vs {}: {} mismatches out of {} (max_diff: {:.6}, avg_diff: {:.6})",
            name1,
            name2,
            mismatches,
            preds1.len(),
            max_diff,
            avg_diff
        );
    }
}

/// Generate synthetic data for benchmarking.
fn generate_synthetic_data(
    n_samples: usize,
    n_features: usize,
) -> (Tensor2D<CpuBackend>, InMemoryDataset, Vec<Vec<f32>>) {
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
    let dataset = InMemoryDataset::new(features.clone(), targets).unwrap();

    (x, dataset, features)
}

/// Run CPU inference benchmark.
fn benchmark_cpu_inference(
    model: &machinelearne_rs::model::MLPModel<CpuBackend, machinelearne_rs::model::Fitted>,
    data: &Tensor2D<CpuBackend>,
    batch_size: usize,
    n_iterations: usize,
) -> BenchmarkResult {
    let (n_samples, _n_features) = data.shape();

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

    // Get sample predictions for comparison
    let predictions = model.predict_batch(data);
    let sample_predictions = predictions.ravel().to_vec();

    BenchmarkResult {
        backend: "CPU".to_string(),
        architecture: "".to_string(),
        batch_size,
        n_samples: total_samples,
        n_iterations,
        total_time_ms,
        throughput_samples_per_sec: throughput,
        latency_ms_per_sample: latency,
        sample_predictions,
    }
}

/// Run WGPU inference benchmark.
#[cfg(feature = "wgpu")]
fn benchmark_wgpu_inference(
    model: &machinelearne_rs::model::MLPModel<CpuBackend, machinelearne_rs::model::Fitted>,
    data: &Tensor2D<CpuBackend>,
    batch_size: usize,
    n_iterations: usize,
) -> BenchmarkResult {
    use machinelearne_rs::backend::WgpuBackend;
    use machinelearne_rs::model::InferenceModel;

    let (n_samples, n_features) = data.shape();
    let data_vec = data.ravel().to_vec();

    // Create WGPU model with same parameters
    let wgpu_model: machinelearne_rs::model::MLPModel<
        WgpuBackend,
        machinelearne_rs::model::Fitted,
    > = {
        // Get layer sizes and activations from the CPU model
        let layer_sizes = model.layer_sizes().to_vec();
        let activations = model.activations().to_vec();

        // Create new model and convert to fitted
        let mlp = MLP::<WgpuBackend>::new(&layer_sizes, &activations);
        mlp.into_fitted()
    };

    // Create WGPU tensor
    let wgpu_data = Tensor2D::<WgpuBackend>::new(data_vec, n_samples, n_features);

    // Warmup
    for _ in 0..3 {
        let _ = wgpu_model.predict_batch(&wgpu_data);
    }

    let start = Instant::now();
    for _ in 0..n_iterations {
        let _ = wgpu_model.predict_batch(&wgpu_data);
    }
    let elapsed = start.elapsed();

    let total_samples = n_samples * n_iterations;
    let total_time_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput = total_samples as f64 / (total_time_ms / 1000.0);
    let latency = total_time_ms / total_samples as f64;

    // Get sample predictions
    let predictions = wgpu_model.predict_batch(&wgpu_data);
    let sample_predictions = predictions.ravel().to_vec();

    BenchmarkResult {
        backend: "WGPU".to_string(),
        architecture: "".to_string(),
        batch_size,
        n_samples: total_samples,
        n_iterations,
        total_time_ms,
        throughput_samples_per_sec: throughput,
        latency_ms_per_sample: latency,
        sample_predictions,
    }
}

/// Run ONNX CPU inference benchmark.
#[cfg(feature = "onnx-inference")]
fn benchmark_onnx_cpu_inference(
    model: &machinelearne_rs::model::MLPModel<CpuBackend, machinelearne_rs::model::Fitted>,
    data: &Tensor2D<CpuBackend>,
    batch_size: usize,
    n_iterations: usize,
) -> BenchmarkResult {
    use machinelearne_rs::onnx::{OnnxExportable, OnnxInferenceSession};

    let (n_samples, n_features) = data.shape();
    let data_vec = data.ravel().to_vec();

    // Export model to ONNX
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let onnx_path = temp_file.path().to_path_buf();
    model
        .save_onnx(&onnx_path, Some("benchmark_model"))
        .unwrap();

    // Load ONNX session
    let session = OnnxInferenceSession::load(&onnx_path).unwrap();

    // Convert data to ndarray
    let input_array = ndarray::Array2::from_shape_vec((n_samples, n_features), data_vec).unwrap();

    // Warmup
    for _ in 0..3 {
        let _ = session.predict(&input_array);
    }

    let start = Instant::now();
    for _ in 0..n_iterations {
        let _ = session.predict(&input_array);
    }
    let elapsed = start.elapsed();

    let total_samples = n_samples * n_iterations;
    let total_time_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput = total_samples as f64 / (total_time_ms / 1000.0);
    let latency = total_time_ms / total_samples as f64;

    // Get sample predictions
    let predictions = session.predict(&input_array).unwrap();
    let sample_predictions: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();

    // Cleanup
    std::fs::remove_file(&onnx_path).ok();

    BenchmarkResult {
        backend: "ONNX-CPU".to_string(),
        architecture: "".to_string(),
        batch_size,
        n_samples: total_samples,
        n_iterations,
        total_time_ms,
        throughput_samples_per_sec: throughput,
        latency_ms_per_sample: latency,
        sample_predictions,
    }
}

/// Run ONNX GPU inference benchmark.
#[cfg(feature = "onnx-cuda")]
fn benchmark_onnx_gpu_inference(
    model: &machinelearne_rs::model::MLPModel<CpuBackend, machinelearne_rs::model::Fitted>,
    data: &Tensor2D<CpuBackend>,
    batch_size: usize,
    n_iterations: usize,
) -> Option<BenchmarkResult> {
    use machinelearne_rs::onnx::{OnnxExportable, OnnxInferenceSession};

    let (n_samples, n_features) = data.shape();
    let data_vec = data.ravel().to_vec();

    // Export model to ONNX
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let onnx_path = temp_file.path().to_path_buf();
    model
        .save_onnx(&onnx_path, Some("benchmark_model"))
        .unwrap();

    // Try to load ONNX session with CUDA
    let session = match OnnxInferenceSession::load_gpu(&onnx_path, 0) {
        Ok(s) => s,
        Err(e) => {
            println!("  CUDA not available: {}", e);
            std::fs::remove_file(&onnx_path).ok();
            return None;
        }
    };

    // Convert data to ndarray
    let input_array = ndarray::Array2::from_shape_vec((n_samples, n_features), data_vec).unwrap();

    // Warmup
    for _ in 0..3 {
        let _ = session.predict(&input_array);
    }

    let start = Instant::now();
    for _ in 0..n_iterations {
        let _ = session.predict(&input_array);
    }
    let elapsed = start.elapsed();

    let total_samples = n_samples * n_iterations;
    let total_time_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput = total_samples as f64 / (total_time_ms / 1000.0);
    let latency = total_time_ms / total_samples as f64;

    // Get sample predictions
    let predictions = session.predict(&input_array).unwrap();
    let sample_predictions: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();

    // Cleanup
    std::fs::remove_file(&onnx_path).ok();

    Some(BenchmarkResult {
        backend: "ONNX-GPU".to_string(),
        architecture: "".to_string(),
        batch_size,
        n_samples: total_samples,
        n_iterations,
        total_time_ms,
        throughput_samples_per_sec: throughput,
        latency_ms_per_sample: latency,
        sample_predictions,
    })
}

fn main() {
    println!("MLP Inference Benchmark");
    println!("=======================\n");

    // Display available backends
    println!("Available Backends:");
    println!("  [x] CPU (native Rust)");
    #[cfg(feature = "wgpu")]
    println!("  [x] WGPU (GPU via wgpu)");
    #[cfg(not(feature = "wgpu"))]
    println!("  [ ] WGPU (not compiled - use --features wgpu)");

    #[cfg(feature = "onnx-inference")]
    println!("  [x] ONNX-CPU (ONNX Runtime)");
    #[cfg(not(feature = "onnx-inference"))]
    println!("  [ ] ONNX-CPU (not compiled - use --features onnx-inference)");

    #[cfg(feature = "onnx-cuda")]
    println!("  [x] ONNX-GPU (ONNX Runtime with CUDA)");
    #[cfg(not(feature = "onnx-cuda"))]
    println!("  [ ] ONNX-GPU (not compiled - use --features onnx-cuda)");
    println!();

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
            "Large (16->128->64->1)",
            vec![16, 128, 64, 1],
            vec![Activation::ReLU, Activation::ReLU, Activation::Identity],
        ),
    ];

    let batch_sizes = vec![1, 32, 256, 1024];
    let n_iterations = 100;

    println!("Configurations:");
    for config in &configs {
        println!("  {} - {} parameters", config.name, config.n_params());
    }
    println!();

    // Store all results for comparison
    let mut all_results: Vec<(String, usize, Vec<BenchmarkResult>)> = Vec::new();

    for config in &configs {
        println!("\n{}", "=".repeat(80));
        println!("Architecture: {}", config.name);
        println!("Parameters: {}", config.n_params());
        println!("{}", "=".repeat(80));

        // Create and train model briefly (just to get fitted weights)
        let model = MLP::<CpuBackend>::new(&config.layer_sizes, &config.activations);

        // Generate training data for a quick fit
        let (_, train_dataset, _) = generate_synthetic_data(1000, config.n_features());

        // Train briefly
        let trainer = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
            .batch_size(32)
            .max_epochs(10)
            .verbose(false)
            .build();
        let fitted = trainer.fit(model, &train_dataset).unwrap();

        println!("\n--- Benchmark Results ---\n");
        BenchmarkResult::header();

        let mut config_results: Vec<BenchmarkResult> = Vec::new();

        for &batch_size in &batch_sizes {
            // Generate benchmark data
            let (data, _, _) = generate_synthetic_data(batch_size, config.n_features());

            // CPU benchmark
            let cpu_result = benchmark_cpu_inference(&fitted, &data, batch_size, n_iterations);
            let mut result = cpu_result.clone();
            result.architecture = config.name.clone();
            result.print();
            config_results.push(cpu_result);

            // WGPU benchmark (if available)
            #[cfg(feature = "wgpu")]
            {
                let wgpu_result =
                    benchmark_wgpu_inference(&fitted, &data, batch_size, n_iterations);
                let mut result = wgpu_result.clone();
                result.architecture = config.name.clone();
                result.print();
                config_results.push(wgpu_result);
            }

            // ONNX CPU benchmark (if available)
            #[cfg(feature = "onnx-inference")]
            {
                let onnx_result =
                    benchmark_onnx_cpu_inference(&fitted, &data, batch_size, n_iterations);
                let mut result = onnx_result.clone();
                result.architecture = config.name.clone();
                result.print();
                config_results.push(onnx_result);
            }

            // ONNX GPU benchmark (if available)
            #[cfg(feature = "onnx-cuda")]
            {
                if let Some(onnx_gpu_result) =
                    benchmark_onnx_gpu_inference(&fitted, &data, batch_size, n_iterations)
                {
                    let mut result = onnx_gpu_result.clone();
                    result.architecture = config.name.clone();
                    result.print();
                    config_results.push(onnx_gpu_result);
                }
            }

            println!();
        }

        all_results.push((config.name.clone(), config.n_params(), config_results));
    }

    // ========================================
    // Comparison Summary
    // ========================================
    println!("\n{}", "=".repeat(80));
    println!("Prediction Comparison (Correctness Check)");
    println!("{}", "=".repeat(80));

    for (config_name, _, results) in &all_results {
        println!("\n{}:", config_name);

        // Find CPU result as baseline (use batch_size=256 for comparison)
        let cpu_result = results
            .iter()
            .find(|r| r.backend == "CPU" && r.batch_size == 256);

        if let Some(cpu) = cpu_result {
            let cpu_preds = &cpu.sample_predictions;

            for result in results {
                if result.backend != "CPU" && result.batch_size == 256 {
                    compare_predictions(
                        "CPU",
                        &result.backend,
                        cpu_preds,
                        &result.sample_predictions,
                        1e-4,
                    );
                }
            }
        }
    }

    // ========================================
    // Performance Summary
    // ========================================
    println!("\n{}", "=".repeat(80));
    println!("Performance Summary (batch_size=256)");
    println!("{}", "=".repeat(80));
    println!();
    println!(
        "{:<20} | {:<12} | {:<15} | {:<15}",
        "Architecture", "Backend", "Throughput", "Latency"
    );
    println!("{}", "-".repeat(70));

    for (config_name, _, results) in &all_results {
        for result in results {
            if result.batch_size == 256 {
                println!(
                    "{:<20} | {:<12} | {:>10.0} samples/s | {:>10.3} ms",
                    config_name,
                    result.backend,
                    result.throughput_samples_per_sec,
                    result.latency_ms_per_sample
                );
            }
        }
    }

    // Speedup comparison (if multiple backends available)
    #[cfg(any(feature = "wgpu", feature = "onnx-inference"))]
    {
        println!("\n{}", "=".repeat(80));
        println!("Speedup vs CPU (batch_size=256)");
        println!("{}", "=".repeat(80));

        for (config_name, _, results) in &all_results {
            let cpu_result = results
                .iter()
                .find(|r| r.backend == "CPU" && r.batch_size == 256);

            if let Some(cpu) = cpu_result {
                println!("\n{}:", config_name);
                for result in results {
                    if result.backend != "CPU" && result.batch_size == 256 {
                        let speedup =
                            result.throughput_samples_per_sec / cpu.throughput_samples_per_sec;
                        println!(
                            "  {}: {:.2}x {}",
                            result.backend,
                            speedup,
                            if speedup > 1.0 { "faster" } else { "slower" }
                        );
                    }
                }
            }
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("Benchmark Complete");
    println!("{}", "=".repeat(80));
    println!("\nTo enable additional backends:");
    println!("  - WGPU:       cargo run --example mlp_inference_benchmark --features wgpu");
    println!(
        "  - ONNX-CPU:   cargo run --example mlp_inference_benchmark --features onnx-inference"
    );
    println!("  - ONNX-GPU:   cargo run --example mlp_inference_benchmark --features onnx-cuda");
    println!(
        "  - All:        cargo run --example mlp_inference_benchmark --features wgpu,onnx-inference"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use machinelearne_rs::dataset::Dataset;
    use machinelearne_rs::model::TrainableModel;

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
        let (x, dataset, features) = generate_synthetic_data(100, 8);
        assert_eq!(x.shape(), (100, 8));
        assert_eq!(dataset.len(), Some(100));
        assert_eq!(features.len(), 100);
    }

    #[test]
    fn test_compare_predictions_match() {
        let preds1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let preds2: Vec<f64> = vec![1.00001, 2.00001, 3.00001, 4.00001, 5.00001];
        // Should not panic - differences are within tolerance
        compare_predictions("A", "B", &preds1, &preds2, 0.001);
    }

    #[test]
    fn test_benchmark_cpu_inference() {
        let model = MLP::<CpuBackend>::new(&[8, 16, 1], &[Activation::ReLU, Activation::Identity]);
        let fitted = model.into_fitted();
        let (data, _, _) = generate_synthetic_data(32, 8);

        let result = benchmark_cpu_inference(&fitted, &data, 32, 10);

        assert_eq!(result.backend, "CPU");
        assert_eq!(result.batch_size, 32);
        assert!(result.total_time_ms > 0.0);
        assert!(result.throughput_samples_per_sec > 0.0);
    }
}
