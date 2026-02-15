use benchmarks::metrics::Metrics;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_mse(c: &mut Criterion) {
    // Test different array sizes
    for size in [100, 1000, 10000, 100000].iter() {
        c.bench_with_input(BenchmarkId::new("mse", size), size, |b, &n| {
            let y_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let y_pred: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 0.5).collect();

            b.iter(|| {
                let mse = Metrics::mse(black_box(&y_true), black_box(&y_pred));
                black_box(mse);
            });
        });
    }
}

fn bench_mae(c: &mut Criterion) {
    // Test different array sizes
    for size in [100, 1000, 10000, 100000].iter() {
        c.bench_with_input(BenchmarkId::new("mae", size), size, |b, &n| {
            let y_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let y_pred: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 0.5).collect();

            b.iter(|| {
                let mae = Metrics::mae(black_box(&y_true), black_box(&y_pred));
                black_box(mae);
            });
        });
    }
}

fn bench_r_squared(c: &mut Criterion) {
    // Test different array sizes
    for size in [100, 1000, 10000, 100000].iter() {
        c.bench_with_input(BenchmarkId::new("r_squared", size), size, |b, &n| {
            let y_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let y_pred: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 0.5).collect();

            b.iter(|| {
                let r2 = Metrics::r_squared(black_box(&y_true), black_box(&y_pred));
                black_box(r2);
            });
        });
    }
}

fn bench_rmse(c: &mut Criterion) {
    // Test different array sizes
    for size in [100, 1000, 10000, 100000].iter() {
        c.bench_with_input(BenchmarkId::new("rmse", size), size, |b, &n| {
            let y_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let y_pred: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 0.5).collect();

            b.iter(|| {
                let rmse = Metrics::rmse(black_box(&y_true), black_box(&y_pred));
                black_box(rmse);
            });
        });
    }
}

fn bench_calculate_all(c: &mut Criterion) {
    // Test different array sizes
    for size in [100, 1000, 10000, 100000].iter() {
        c.bench_with_input(BenchmarkId::new("calculate_all", size), size, |b, &n| {
            let y_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let y_pred: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 0.5).collect();

            b.iter(|| {
                let metrics = Metrics::calculate_all(black_box(&y_true), black_box(&y_pred));
                black_box(metrics);
            });
        });
    }
}

fn bench_mse_perfect_prediction(c: &mut Criterion) {
    // Benchmark MSE with perfect predictions (should be fast)
    for size in [100, 1000, 10000].iter() {
        c.bench_with_input(BenchmarkId::new("mse_perfect", size), size, |b, &n| {
            let y_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let y_pred = y_true.clone(); // Perfect prediction

            b.iter(|| {
                let mse = Metrics::mse(black_box(&y_true), black_box(&y_pred));
                black_box(mse);
            });
        });
    }
}

fn bench_r_squared_perfect_prediction(c: &mut Criterion) {
    // Benchmark R² with perfect predictions (should be ~1.0)
    for size in [100, 1000, 10000].iter() {
        c.bench_with_input(
            BenchmarkId::new("r_squared_perfect", size),
            size,
            |b, &n| {
                let y_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
                let y_pred = y_true.clone(); // Perfect prediction

                b.iter(|| {
                    let r2 = Metrics::r_squared(black_box(&y_true), black_box(&y_pred));
                    black_box(r2);
                });
            },
        );
    }
}

fn bench_metrics_comparison(c: &mut Criterion) {
    // Compare all metrics on the same data
    let sizes = [1000, 10000, 100000];

    for size in sizes.iter() {
        let y_true: Vec<f32> = (0..*size).map(|i| i as f32 * 0.1).collect();
        let y_pred: Vec<f32> = (0..*size).map(|i| i as f32 * 0.1 + 0.5).collect();

        let mut group = c.benchmark_group(format!("metrics_comparison_{}", size));

        group.bench_function("mse", |b| {
            b.iter(|| {
                let mse = Metrics::mse(black_box(&y_true), black_box(&y_pred));
                black_box(mse);
            });
        });

        group.bench_function("mae", |b| {
            b.iter(|| {
                let mae = Metrics::mae(black_box(&y_true), black_box(&y_pred));
                black_box(mae);
            });
        });

        group.bench_function("r_squared", |b| {
            b.iter(|| {
                let r2 = Metrics::r_squared(black_box(&y_true), black_box(&y_pred));
                black_box(r2);
            });
        });

        group.bench_function("rmse", |b| {
            b.iter(|| {
                let rmse = Metrics::rmse(black_box(&y_true), black_box(&y_pred));
                black_box(rmse);
            });
        });

        group.finish();
    }
}

fn bench_metrics_large_arrays(c: &mut Criterion) {
    // Benchmark with very large arrays
    c.bench_function("mse_large_1M", |b| {
        let n = 1_000_000;
        let y_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let y_pred: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 0.5).collect();

        b.iter(|| {
            let mse = Metrics::mse(black_box(&y_true), black_box(&y_pred));
            black_box(mse);
        });
    });

    c.bench_function("mae_large_1M", |b| {
        let n = 1_000_000;
        let y_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let y_pred: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 0.5).collect();

        b.iter(|| {
            let mae = Metrics::mae(black_box(&y_true), black_box(&y_pred));
            black_box(mae);
        });
    });

    c.bench_function("r_squared_large_1M", |b| {
        let n = 1_000_000;
        let y_true: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let y_pred: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 0.5).collect();

        b.iter(|| {
            let r2 = Metrics::r_squared(black_box(&y_true), black_box(&y_pred));
            black_box(r2);
        });
    });
}

criterion_group!(
    benches,
    bench_mse,
    bench_mae,
    bench_r_squared,
    bench_rmse,
    bench_calculate_all,
    bench_mse_perfect_prediction,
    bench_r_squared_perfect_prediction,
    bench_metrics_comparison,
    bench_metrics_large_arrays
);
criterion_main!(benches);
