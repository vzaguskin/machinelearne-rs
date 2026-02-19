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

    for n_features in [2, 10, 50, 100].iter() {
        let model = create_model(*n_features, 1000);
        let test_data = create_test_data(100, *n_features);

        group.throughput(Throughput::Elements(*n_features as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_100", n_features),
            n_features,
            |b, _| {
                b.iter(|| {
                    let _ = model.predict_batch(&test_data);
                });
            },
        );
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

    for n_features in [2, 10, 50, 100].iter() {
        let model = create_model(*n_features, 1000);

        // Export model to temp file
        let temp_path = std::env::temp_dir().join(format!("bench_model_{}.onnx", n_features));
        model.save_onnx(&temp_path).unwrap();

        // Load ONNX model
        let session = OnnxInferenceSession::load(&temp_path).unwrap();

        // Create test data as ndarray
        let test_data: Array2<f32> = Array2::from_shape_vec(
            (100, *n_features),
            (0..100 * n_features)
                .map(|i| (i as f32 % 50.0) / 10.0)
                .collect(),
        )
        .unwrap();

        group.throughput(Throughput::Elements(*n_features as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_100", n_features),
            n_features,
            |b, _| {
                b.iter(|| {
                    let _ = session.predict(&test_data);
                });
            },
        );

        // Cleanup
        std::fs::remove_file(&temp_path).ok();
    }

    group.finish();
}

#[cfg(feature = "onnx-inference")]
fn bench_comparison(c: &mut Criterion) {
    use machinelearne_rs::onnx::{OnnxExportable, OnnxInferenceSession};
    use ndarray::Array2;

    let mut group = c.benchmark_group("inference_comparison");

    let n_features = 10;
    let model = create_model(n_features, 1000);
    let test_data = create_test_data(100, n_features);

    // Export and load ONNX model
    let temp_path = std::env::temp_dir().join("bench_comparison.onnx");
    model.save_onnx(&temp_path).unwrap();
    let session = OnnxInferenceSession::load(&temp_path).unwrap();

    // Create ndarray test data
    let test_data_nd: Array2<f32> = Array2::from_shape_vec(
        (100, n_features),
        (0..100 * n_features)
            .map(|i| (i as f32 % 50.0) / 10.0)
            .collect(),
    )
    .unwrap();

    group.bench_function("native_rust", |b| {
        b.iter(|| {
            let _ = model.predict_batch(&test_data);
        });
    });

    group.bench_function("onnx_runtime", |b| {
        b.iter(|| {
            let _ = session.predict(&test_data_nd);
        });
    });

    // Cleanup
    std::fs::remove_file(&temp_path).ok();

    group.finish();
}

// Register benchmark groups
#[cfg(not(feature = "onnx"))]
criterion_group!(benches, bench_native_inference);

#[cfg(all(feature = "onnx", not(feature = "onnx-inference")))]
criterion_group!(benches, bench_native_inference, bench_onnx_export);

#[cfg(feature = "onnx-inference")]
criterion_group!(
    benches,
    bench_native_inference,
    bench_onnx_export,
    bench_onnx_inference,
    bench_comparison
);

criterion_main!(benches);
