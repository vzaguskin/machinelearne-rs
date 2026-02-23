//! ONNX Inference Benchmark
//!
//! Compares native Rust inference vs ONNX Runtime inference performance.
//!
//! ## Usage
//!
//! ```bash
//! # Run with onnx export only (no ONNX Runtime)
//! cargo bench --bench onnx_inference --features onnx
//!
//! # Run with ONNX Runtime (requires onnx-inference feature)
//! cargo bench --bench onnx_inference --features onnx-inference
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use machinelearne_rs::{
    backend::{CpuBackend, Tensor2D},
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    model::linear::{Fitted, LinearModel, LinearRegressor},
    model::InferenceModel,
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
};

/// Create a fitted linear regression model with the given number of features.
fn create_model(n_features: usize, n_samples: usize) -> LinearModel<CpuBackend, Fitted> {
    // Generate synthetic data
    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut row = vec![0.0; n_features];
        for j in 0..n_features {
            row[j] = ((i * n_features + j) as f32 % 100.0) / 10.0;
        }
        // y = sum of features
        let y_val: f32 = row.iter().sum();
        x.push(row);
        y.push(y_val);
    }

    let dataset = InMemoryDataset::new(x, y).unwrap();

    let model = LinearRegressor::new(n_features);
    let loss = MSELoss;
    let opt = SGD::new(0.1);
    let reg = NoRegularizer;
    let trainer = Trainer::builder(loss, opt, reg)
        .batch_size(32)
        .max_epochs(50)
        .build();

    trainer.fit(model, &dataset).unwrap()
}

/// Create test input data.
fn create_test_data(n_samples: usize, n_features: usize) -> Tensor2D<CpuBackend> {
    let mut data = Vec::with_capacity(n_samples * n_features);
    for i in 0..n_samples {
        for j in 0..n_features {
            data.push(((i * n_features + j) as f32 % 50.0) / 10.0);
        }
    }
    Tensor2D::new(data, n_samples, n_features)
}

fn bench_native_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("native_inference");

    // Test various batch sizes to show scaling
    let batch_sizes = [100, 1000, 10000];
    let n_features = 10;

    let model = create_model(n_features, 5000);

    for batch_size in batch_sizes.iter() {
        let test_data = create_test_data(*batch_size, n_features);

        group.throughput(Throughput::Elements((*batch_size * n_features) as u64));
        group.bench_with_input(BenchmarkId::new("batch", batch_size), batch_size, |b, _| {
            b.iter(|| {
                let _ = model.predict_batch(&test_data);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "onnx")]
fn bench_onnx_export(c: &mut Criterion) {
    use machinelearne_rs::onnx::OnnxExportable;

    let mut group = c.benchmark_group("onnx_export");

    for n_features in [2, 10, 50, 100].iter() {
        let model = create_model(*n_features, 1000);

        group.bench_with_input(
            BenchmarkId::new("export", n_features),
            n_features,
            |b, _| {
                b.iter(|| {
                    let _ = model.to_onnx_default();
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "onnx-inference")]
fn bench_onnx_inference(c: &mut Criterion) {
    use machinelearne_rs::onnx::{OnnxExportable, OnnxInferenceSession};
    use ndarray::Array2;

    let mut group = c.benchmark_group("onnx_inference");

    // Test various batch sizes to show scaling
    let batch_sizes = [100, 1000, 10000];
    let n_features = 10;

    let model = create_model(n_features, 5000);

    // Export model to temp file
    let temp_path = std::env::temp_dir().join("bench_model.onnx");
    model.save_onnx(&temp_path).unwrap();

    // Load ONNX model
    let session = OnnxInferenceSession::load(&temp_path).unwrap();

    for batch_size in batch_sizes.iter() {
        // Create test data as ndarray
        let test_data: Array2<f32> = Array2::from_shape_vec(
            (*batch_size, n_features),
            (0..*batch_size * n_features)
                .map(|i| (i as f32 % 50.0) / 10.0)
                .collect(),
        )
        .unwrap();

        group.throughput(Throughput::Elements((*batch_size * n_features) as u64));
        group.bench_with_input(BenchmarkId::new("batch", batch_size), batch_size, |b, _| {
            b.iter(|| {
                let _ = session.predict(&test_data);
            });
        });
    }

    // Cleanup
    std::fs::remove_file(&temp_path).ok();

    group.finish();
}

#[cfg(feature = "onnx-inference")]
fn bench_comparison(c: &mut Criterion) {
    use machinelearne_rs::onnx::{OnnxExportable, OnnxInferenceSession};
    use ndarray::Array2;

    let mut group = c.benchmark_group("inference_comparison");

    let n_features = 10;
    let batch_size = 10000; // Larger batch to see real differences

    let model = create_model(n_features, 5000);
    let test_data = create_test_data(batch_size, n_features);

    // Export and load ONNX model
    let temp_path = std::env::temp_dir().join("bench_comparison.onnx");
    model.save_onnx(&temp_path).unwrap();
    let session = OnnxInferenceSession::load(&temp_path).unwrap();

    // Create ndarray test data
    let test_data_nd: Array2<f32> = Array2::from_shape_vec(
        (batch_size, n_features),
        (0..batch_size * n_features)
            .map(|i| (i as f32 % 50.0) / 10.0)
            .collect(),
    )
    .unwrap();

    group.throughput(Throughput::Elements((batch_size * n_features) as u64));

    group.bench_function("native_rust", |b| {
        b.iter(|| {
            let _ = model.predict_batch(&test_data);
        });
    });

    group.bench_function("onnx_runtime_cpu", |b| {
        b.iter(|| {
            let _ = session.predict(&test_data_nd);
        });
    });

    // Cleanup
    std::fs::remove_file(&temp_path).ok();

    group.finish();
}

/// CUDA inference benchmark - only runs if CUDA is available
#[cfg(feature = "onnx-cuda")]
fn bench_onnx_cuda_inference(c: &mut Criterion) {
    use machinelearne_rs::onnx::{OnnxExportable, OnnxInferenceSession};
    use ndarray::Array2;

    let mut group = c.benchmark_group("onnx_cuda_inference");

    // Test various batch sizes to show scaling
    let batch_sizes = [100, 1000, 10000];
    let n_features = 10;

    let model = create_model(n_features, 5000);

    // Export model to temp file
    let temp_path = std::env::temp_dir().join("bench_cuda_model.onnx");
    model.save_onnx(&temp_path).unwrap();

    // Try to load with CUDA - skip if unavailable
    match OnnxInferenceSession::load_gpu(&temp_path, 0) {
        Ok(session) => {
            for batch_size in batch_sizes.iter() {
                // Create test data as ndarray
                let test_data: Array2<f32> = Array2::from_shape_vec(
                    (*batch_size, n_features),
                    (0..*batch_size * n_features)
                        .map(|i| (i as f32 % 50.0) / 10.0)
                        .collect(),
                )
                .unwrap();

                group.throughput(Throughput::Elements((*batch_size * n_features) as u64));
                group.bench_with_input(
                    BenchmarkId::new("batch", batch_size),
                    batch_size,
                    |b, _| {
                        b.iter(|| {
                            let _ = session.predict(&test_data);
                        });
                    },
                );
            }
        }
        Err(e) => {
            eprintln!("CUDA not available for benchmark: {}", e);
        }
    }

    // Cleanup
    std::fs::remove_file(&temp_path).ok();

    group.finish();
}

/// Compare CPU vs CUDA performance
#[cfg(feature = "onnx-cuda")]
fn bench_cpu_cuda_comparison(c: &mut Criterion) {
    use machinelearne_rs::onnx::{OnnxExportable, OnnxInferenceSession};
    use ndarray::Array2;

    let mut group = c.benchmark_group("cpu_cuda_comparison");

    let n_features = 10;
    let batch_size = 10000; // Larger batch to see GPU benefits

    let model = create_model(n_features, 5000);

    // Export model to temp file
    let temp_path = std::env::temp_dir().join("bench_cpu_cuda_comparison.onnx");
    model.save_onnx(&temp_path).unwrap();

    // Load CPU session
    let cpu_session = OnnxInferenceSession::load(&temp_path).unwrap();

    // Try CUDA session
    match OnnxInferenceSession::load_gpu(&temp_path, 0) {
        Ok(cuda_session) => {
            // Create test data
            let test_data: Array2<f32> = Array2::from_shape_vec(
                (batch_size, n_features),
                (0..batch_size * n_features)
                    .map(|i| (i as f32 % 50.0) / 10.0)
                    .collect(),
            )
            .unwrap();

            group.throughput(Throughput::Elements((batch_size * n_features) as u64));

            group.bench_function("onnx_cpu", |b| {
                b.iter(|| {
                    let _ = cpu_session.predict(&test_data);
                });
            });

            group.bench_function("onnx_cuda", |b| {
                b.iter(|| {
                    let _ = cuda_session.predict(&test_data);
                });
            });
        }
        Err(e) => {
            eprintln!("CUDA not available for comparison: {}", e);
        }
    }

    // Cleanup
    std::fs::remove_file(&temp_path).ok();

    group.finish();
}

// Register benchmark groups
#[cfg(not(feature = "onnx"))]
criterion_group!(benches, bench_native_inference);

#[cfg(all(feature = "onnx", not(feature = "onnx-inference")))]
criterion_group!(benches, bench_native_inference, bench_onnx_export);

#[cfg(all(feature = "onnx-inference", not(feature = "onnx-cuda")))]
criterion_group!(
    benches,
    bench_native_inference,
    bench_onnx_export,
    bench_onnx_inference,
    bench_comparison
);

#[cfg(feature = "onnx-cuda")]
criterion_group!(
    benches,
    bench_native_inference,
    bench_onnx_export,
    bench_onnx_inference,
    bench_comparison,
    bench_onnx_cuda_inference,
    bench_cpu_cuda_comparison
);

criterion_main!(benches);
