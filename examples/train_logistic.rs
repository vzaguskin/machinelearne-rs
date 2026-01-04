// examples/train_logistic.rs
use machinelearne_rs::{
    loss::BCEWithLogitsLoss,
    model::linear::LinearRegressor, // будем использовать как линейный классификатор (выдаёт логит)
    optimizer::SGD,
    trainer::Trainer,
    regularizers::NoRegularizer,
};

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

    let fitted_model = trainer.fit(model, &x, &y).unwrap();

    // Протестируем на новой точке
    let logit = fitted_model.predict(&[2.5, 2.0]); // sum=4.5 → ожидаем y≈1
    let prob = 1.0 / (1.0 + (-logit).exp());
    println!("Logit: {:.3}, Probability: {:.3}", logit, prob);
}