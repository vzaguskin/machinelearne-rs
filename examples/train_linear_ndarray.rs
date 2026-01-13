// examples/train_linear.rs или в тестах

use machinelearne_rs::{ 
    Tensor1D, 
    dataset::memory::InMemoryDataset, 
    loss::MSELoss, 
    model::{linear::LinearRegression, InferenceModel}, 
    optimizer::SGD, 
    regularizers::NoRegularizer, 
    trainer::Trainer};
#[cfg(feature = "ndarray")]
fn main() {

    use machinelearne_rs::backend::NdarrayBackend;
    let model = LinearRegression::<NdarrayBackend>::new(2); // 2 фичи
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
    let inp = Tensor1D::<NdarrayBackend>::new((&[4.0, 5.0]).to_vec());
    let pred = fitted_model.predict(&inp);
    println!("Prediction: {:?}", pred);
}

#[cfg(not(feature = "ndarray"))]
fn main() {
    println!("This example requires the `ndarray` feature. Run with:");
    println!("cargo run --example train_linear_ndarray --features ndarray");
    std::process::exit(1);
}