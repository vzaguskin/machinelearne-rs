use benchmarks::data::CaliforniaHousingDataset;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use machinelearne_rs::{
    backend::Tensor2D,
    dataset::InMemoryDataset,
    loss::MSELoss,
    model::{linear::LinearRegression, InferenceModel},
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
    CpuBackend,
};

/// Helper function to convert Vec<Vec<f32>> to Tensor2D<CpuBackend>
fn vec_to_tensor2d(data: &[Vec<f32>]) -> Tensor2D<CpuBackend> {
    let flat: Vec<f32> = data.iter().flatten().copied().collect();
    Tensor2D::new(flat, data.len(), data.first().map(|v| v.len()).unwrap_or(0))
}

/// Train a model once for prediction benchmarks
fn train_model_for_prediction(
) -> machinelearne_rs::model::linear::LinearModel<CpuBackend, machinelearne_rs::model::state::Fitted>
{
    let dataset = CaliforniaHousingDataset::load("datasets/california_housing.csv")
        .expect("Failed to load dataset");
    let (train_dataset, _test_dataset) = dataset.split(0.8);
    let in_memory = train_dataset
        .to_in_memory_dataset()
        .expect("Failed to create dataset");

    let model = LinearRegression::<CpuBackend>::new(8);
    let optimizer = SGD::new(0.001);
    let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
        .batch_size(32)
        .max_epochs(50)
        .build();

    trainer.fit(model, &in_memory).expect("Failed to fit model")
}

fn bench_predict_single(c: &mut Criterion) {
    let model = train_model_for_prediction();

    c.bench_function("predict_single", |b| {
        let input: Vec<f32> = vec![5.0, 30.0, 6.0, 1.0, 500.0, 3.0, 37.8, -122.2];
        let input_tensor = machinelearne_rs::backend::Tensor1D::new(input);
        b.iter(|| {
            let pred = model.predict(black_box(&input_tensor));
            black_box(pred);
        });
    });
}

fn bench_predict_batch(c: &mut Criterion) {
    let model = train_model_for_prediction();

    // Test different batch sizes
    for batch_size in [10, 100, 1000, 10000].iter() {
        c.bench_with_input(
            BenchmarkId::new("predict_batch", batch_size),
            batch_size,
            |b, &bs| {
                // Create test data
                let test_x: Vec<Vec<f32>> = (0..bs)
                    .map(|_| vec![5.0, 30.0, 6.0, 1.0, 500.0, 3.0, 37.8, -122.2])
                    .collect();

                b.iter(|| {
                    let predictions = model.predict_batch(black_box(&vec_to_tensor2d(&test_x)));
                    black_box(predictions);
                });
            },
        );
    }
}

fn bench_predict_throughput(c: &mut Criterion) {
    let model = train_model_for_prediction();

    // Test different batch sizes for throughput
    for batch_size in [100, 1000, 10000].iter() {
        c.bench_with_input(
            BenchmarkId::new("predict_throughput", batch_size),
            batch_size,
            |b, &bs| {
                // Create test data
                let test_x: Vec<Vec<f32>> = (0..bs)
                    .map(|_| vec![5.0, 30.0, 6.0, 1.0, 500.0, 3.0, 37.8, -122.2])
                    .collect();

                b.iter(|| {
                    let predictions = model.predict_batch(black_box(&vec_to_tensor2d(&test_x)));
                    // Throughput is calculated as batch_size / time
                    black_box((predictions.len(), predictions));
                });
            },
        );
    }
}

fn bench_predict_warmup(c: &mut Criterion) {
    let model = train_model_for_prediction();

    c.bench_function("predict_warmup", |b| {
        let input: Vec<f32> = vec![5.0, 30.0, 6.0, 1.0, 500.0, 3.0, 37.8, -122.2];
        let input_tensor = machinelearne_rs::backend::Tensor1D::new(input);

        // Warmup
        for _ in 0..5 {
            let _ = model.predict(&input_tensor);
        }

        b.iter(|| {
            let pred = model.predict(black_box(&input_tensor));
            black_box(pred);
        });
    });
}

fn bench_predict_latencies(c: &mut Criterion) {
    let model = train_model_for_prediction();

    // Measure latency percentiles for single prediction
    let mut group = c.benchmark_group("predict_latency");
    group.sample_size(10000);

    group.bench_function("p50", |b| {
        let input: Vec<f32> = vec![5.0, 30.0, 6.0, 1.0, 500.0, 3.0, 37.8, -122.2];
        let input_tensor = machinelearne_rs::backend::Tensor1D::new(input);
        b.iter(|| {
            let pred = model.predict(black_box(&input_tensor));
            black_box(pred);
        });
    });

    group.finish();
}

fn bench_predict_batch_warmup(c: &mut Criterion) {
    let model = train_model_for_prediction();

    for batch_size in [100, 1000].iter() {
        c.bench_with_input(
            BenchmarkId::new("predict_batch_warmup", batch_size),
            batch_size,
            |b, &bs| {
                let test_x: Vec<Vec<f32>> = (0..bs)
                    .map(|_| vec![5.0, 30.0, 6.0, 1.0, 500.0, 3.0, 37.8, -122.2])
                    .collect();

                // Warmup
                for _ in 0..5 {
                    let _ = model.predict_batch(&vec_to_tensor2d(&test_x));
                }

                b.iter(|| {
                    let predictions = model.predict_batch(black_box(&vec_to_tensor2d(&test_x)));
                    black_box(predictions);
                });
            },
        );
    }
}

fn bench_predict_different_features(c: &mut Criterion) {
    // Train models with different feature counts
    let dataset = CaliforniaHousingDataset::load("datasets/california_housing.csv")
        .expect("Failed to load dataset");
    let (train_dataset, _) = dataset.split(0.8);
    let train_memory = train_dataset
        .to_in_memory_dataset()
        .expect("Failed to create dataset");

    // Train models with 1, 2, 4, and 8 features
    let models = vec![
        (1, {
            let subset = dataset.select_features(&[0]);
            let (train, _) = subset.split(0.8);
            let in_memory = train.to_in_memory_dataset().unwrap();
            let model = LinearRegression::<CpuBackend>::new(1);
            let optimizer = SGD::new(0.01);
            let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
                .batch_size(32)
                .max_epochs(50)
                .build();
            trainer.fit(model, &in_memory).expect("Failed to fit model")
        }),
        (2, {
            let subset = dataset.select_features(&[0, 1]);
            let (train, _) = subset.split(0.8);
            let in_memory = train.to_in_memory_dataset().unwrap();
            let model = LinearRegression::<CpuBackend>::new(2);
            let optimizer = SGD::new(0.001);
            let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
                .batch_size(32)
                .max_epochs(50)
                .build();
            trainer.fit(model, &in_memory).expect("Failed to fit model")
        }),
        (4, {
            let subset = dataset.select_features(&[0, 1, 2, 3]);
            let (train, _) = subset.split(0.8);
            let in_memory = train.to_in_memory_dataset().unwrap();
            let model = LinearRegression::<CpuBackend>::new(4);
            let optimizer = SGD::new(0.001);
            let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
                .batch_size(32)
                .max_epochs(50)
                .build();
            trainer.fit(model, &in_memory).expect("Failed to fit model")
        }),
        (8, {
            let model = LinearRegression::<CpuBackend>::new(8);
            let optimizer = SGD::new(0.001);
            let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
                .batch_size(32)
                .max_epochs(50)
                .build();
            trainer
                .fit(model, &train_memory)
                .expect("Failed to fit model")
        }),
    ];

    for (n_features, model) in models {
        c.bench_with_input(
            BenchmarkId::new("predict_by_features", n_features),
            &n_features,
            |b, n_feat| {
                let input: Vec<f32> = (0..*n_feat).map(|i| i as f32 * 1.0).collect();
                let input_tensor = machinelearne_rs::backend::Tensor1D::new(input);
                b.iter(|| {
                    let pred = model.predict(black_box(&input_tensor));
                    black_box(pred);
                });
            },
        );
    }
}

criterion_group!(
    benches,
    bench_predict_single,
    bench_predict_batch,
    bench_predict_throughput,
    bench_predict_warmup,
    bench_predict_latencies,
    bench_predict_batch_warmup,
    bench_predict_different_features
);
criterion_main!(benches);
