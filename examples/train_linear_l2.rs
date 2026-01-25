// examples/train_linear.rs или в тестах
use machinelearne_rs::{
    dataset::memory::InMemoryDataset, loss::MSELoss, model::linear::LinearRegressor,
    model::InferenceModel, optimizer::SGD, regularizers::L2, trainer::Trainer, CpuBackend,
    Tensor1D,
};

fn main() {
    let model = LinearRegressor::new(2); // 2 фичи
    let loss = MSELoss;
    let opt = SGD::new(0.01);
    let reg = L2::new(0.1);
    let trainer = Trainer::builder(loss, opt, reg)
        .batch_size(20)
        .max_epochs(500)
        .build();
    let x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    let y = vec![3.0, 5.0, 7.0];

    let dataset = InMemoryDataset::new(x, y).unwrap();

    let fitted_model = trainer.fit(model, &dataset).unwrap();
    println!(
        "Prediction: {:?}",
        fitted_model.predict(&Tensor1D::<CpuBackend>::new((&[4.0, 5.0]).to_vec()))
    );
}
