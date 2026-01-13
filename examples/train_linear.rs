// examples/train_linear.rs или в тестах
use machinelearne_rs::{CpuBackend, 
    Tensor1D, 
    dataset::memory::InMemoryDataset, 
    loss::MSELoss, 
    model::linear::LinearRegressor, optimizer::SGD, regularizers::NoRegularizer, trainer::Trainer, model::InferenceModel};

fn main() {

    let model = LinearRegressor::new(2); // 2 фичи
    let loss = MSELoss;
    let opt = SGD::new(0.1);
    let reg = NoRegularizer;
    let trainer = Trainer::builder(loss, opt, reg)
        .batch_size(20)
        .max_epochs(500)
        .build();
        let x = vec![
    vec![1.0, 1.0],   // y ≈ 1*1 + 2*1 = 3
    vec![2.0, 1.0],   // y ≈ 2 + 2 = 4
    vec![1.0, 2.0],   // y ≈ 1 + 4 = 5
    vec![2.0, 2.0],   // y ≈ 2 + 4 = 6
    vec![3.0, 3.0],   // ← выброс по y!
];
    let y = vec![3.0, 4.0, 5.0, 6.0, 30.0];
    let dataset = InMemoryDataset::new(x, y).unwrap();

    let fitted_model = trainer.fit(model, &dataset).unwrap();
    let inp = Tensor1D::<CpuBackend>::new((&[4.0, 5.0]).to_vec());
    let pred = fitted_model.predict(&inp);
    println!("Prediction: {:?}", pred);
}