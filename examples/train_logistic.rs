// examples/train_logistic.rs
use machinelearne_rs::{CpuBackend, 
    Tensor1D, 
    dataset::memory::InMemoryDataset, 
    loss::BCEWithLogitsLoss, 
    model::linear::LinearRegressor, optimizer::SGD, regularizers::NoRegularizer, trainer::Trainer, model::InferenceModel, backend::scalar::Scalar};

fn main() {
    // Бинарные данные: y = 1 если x1 + x2 > 3, иначе 0
    let x = vec![
        vec![1.0, 1.0], // sum=2 → y=0
        vec![1.0, 2.0], // sum=3 → y=0 (граничный случай)
        vec![2.0, 2.0], // sum=4 → y=1
        vec![3.0, 1.0], // sum=4 → y=1
        vec![0.5, 0.5], // sum=1 → y=0
        // Добавим выброс: точка с y=1, хотя сумма мала
        vec![1.0, 1.0], // sum=2 → но пометим как y=1 (шум/ошибка)
    ];
    let y = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    let model = LinearRegressor::new(2); // 2 признака → 1 логит
    let loss = BCEWithLogitsLoss;
    let opt = SGD::new(0.5); // может потребоваться больший LR для BCE
    let reg = NoRegularizer;

    let trainer = Trainer::builder(loss, opt, reg)
        .batch_size(10)
        .max_epochs(1000)
        .build();

   let dataset = InMemoryDataset::new(x, y).unwrap();

    let fitted_model = trainer.fit(model, &dataset).unwrap();
    let logit = fitted_model.predict(&Tensor1D::<CpuBackend>::new((&[2.5, 2.0]).to_vec()));
    let one = Scalar::<CpuBackend>::new(1.);
    let minus_one = Scalar::<CpuBackend>::new(-1.);
    let prob = one / (one + ( logit * minus_one));
    let prob = prob.exp();
    println!("Logit: {:?}, Probability: {:?}", logit, prob);
}