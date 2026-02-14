use benchmarks::data::CaliforniaHousingDataset;
use benchmarks::metrics::Metrics;
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

fn bench_train_2_features(c: &mut Criterion) {
    // Load dataset and select first 2 features
    let dataset = CaliforniaHousingDataset::load("datasets/california_housing.csv")
        .expect("Failed to load dataset");
    let subset = dataset.select_features(&[0, 1]); // MedInc and HouseAge
    let (train_dataset, _test_dataset) = subset.split(0.8);
    let in_memory = train_dataset
        .to_in_memory_dataset()
        .expect("Failed to create dataset");

    // Test different batch sizes
    for batch_size in [16, 32, 64, 128].iter() {
        c.bench_with_input(
            BenchmarkId::new("train_2_features", batch_size),
            batch_size,
            |b, &bs| {
                let optimizer = SGD::new(0.001);
                let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
                    .batch_size(bs)
                    .max_epochs(10)
                    .build();

                b.iter(|| {
                    let model_copy = LinearRegression::<CpuBackend>::new(2);
                    let fitted = trainer
                        .fit(model_copy, &in_memory)
                        .expect("Failed to fit model");

                    // Verify the model works
                    let test_x: Vec<f32> = vec![5.0, 30.0];
                    let test_tensor = machinelearne_rs::backend::Tensor1D::new(test_x);
                    let _pred = fitted.predict(black_box(&test_tensor));

                    fitted
                });
            },
        );
    }
}

fn bench_train_2_features_learning_rate(c: &mut Criterion) {
    // Load dataset and select first 2 features
    let dataset = CaliforniaHousingDataset::load("datasets/california_housing.csv")
        .expect("Failed to load dataset");
    let subset = dataset.select_features(&[0, 1]); // MedInc and HouseAge
    let (train_dataset, _test_dataset) = subset.split(0.8);
    let in_memory = train_dataset
        .to_in_memory_dataset()
        .expect("Failed to create dataset");

    // Test different learning rates
    for lr in [0.001, 0.01, 0.1].iter() {
        c.bench_with_input(
            BenchmarkId::new("train_2_features_lr", lr),
            lr,
            |b, &learning_rate| {
                let optimizer = SGD::new(learning_rate);
                let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
                    .batch_size(32)
                    .max_epochs(10)
                    .build();

                b.iter(|| {
                    let model_copy = LinearRegression::<CpuBackend>::new(2);
                    let _ = trainer
                        .fit(model_copy, &in_memory)
                        .expect("Failed to fit model");
                });
            },
        );
    }
}

fn bench_train_2_features_epochs(c: &mut Criterion) {
    // Load dataset and select first 2 features
    let dataset = CaliforniaHousingDataset::load("datasets/california_housing.csv")
        .expect("Failed to load dataset");
    let subset = dataset.select_features(&[0, 1]); // MedInc and HouseAge
    let (train_dataset, _test_dataset) = subset.split(0.8);
    let in_memory = train_dataset
        .to_in_memory_dataset()
        .expect("Failed to create dataset");

    // Test different epoch counts
    for epochs in [10, 50, 100].iter() {
        c.bench_with_input(
            BenchmarkId::new("train_2_features_epochs", epochs),
            epochs,
            |b, &max_epochs| {
                let optimizer = SGD::new(0.01);
                let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
                    .batch_size(32)
                    .max_epochs(max_epochs)
                    .build();

                b.iter(|| {
                    let model_copy = LinearRegression::<CpuBackend>::new(2);
                    let _ = trainer
                        .fit(model_copy, &in_memory)
                        .expect("Failed to fit model");
                });
            },
        );
    }
}

fn bench_train_2_features_with_metrics(c: &mut Criterion) {
    // Load dataset and select first 2 features
    let dataset = CaliforniaHousingDataset::load("datasets/california_housing.csv")
        .expect("Failed to load dataset");
    let subset = dataset.select_features(&[0, 1]); // MedInc and HouseAge
    let (train_dataset, test_dataset) = subset.split(0.8);
    let train_memory = train_dataset
        .to_in_memory_dataset()
        .expect("Failed to create dataset");

    // Test the full pipeline and measure metrics
    c.bench_function("train_2_features_with_metrics", |b| {
        b.iter(|| {
            let model = LinearRegression::<CpuBackend>::new(2);
            let optimizer = SGD::new(0.001);
            let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
                .batch_size(32)
                .max_epochs(50)
                .build();

            let fitted = trainer
                .fit(model, &train_memory)
                .expect("Failed to fit model");

            // Evaluate on test set
            let test_x = test_dataset.features().to_vec();
            let test_y = test_dataset.target().to_vec();
            let pred_tensor = fitted.predict_batch(black_box(&vec_to_tensor2d(&test_x)));

            let predictions: Vec<f32> =
                pred_tensor.to_vec().into_iter().map(|v| v as f32).collect();

            let mse = Metrics::mse(&test_y, &predictions);
            let mae = Metrics::mae(&test_y, &predictions);
            let r2 = Metrics::r_squared(&test_y, &predictions);

            black_box((mse, mae, r2));
        });
    });
}

criterion_group!(
    benches,
    bench_train_2_features,
    bench_train_2_features_learning_rate,
    bench_train_2_features_epochs,
    bench_train_2_features_with_metrics
);
criterion_main!(benches);
