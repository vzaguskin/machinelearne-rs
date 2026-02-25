//! MLP training with callbacks example.
//!
//! This example demonstrates:
//! - Using multiple callbacks together
//! - LoggingCallback for training progress
//! - Learning rate scheduling during training
//! - Callback ordering and interaction

use machinelearne_rs::{
    backend::CpuBackend,
    callbacks::LoggingCallback,
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    model::{Activation, InferenceModel, MLP},
    optimizer::SGD,
    regularizers::NoRegularizer,
    schedulers::ExponentialLR,
    trainer::Trainer,
};

fn main() {
    println!("=== MLP Training with Callbacks ===\n");

    // XOR dataset
    let x = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y = vec![0.0, 1.0, 1.0, 0.0];

    let dataset = InMemoryDataset::new(x.clone(), y.clone()).unwrap();

    println!("Dataset: XOR problem (4 samples)");
    println!("Model: 2 -> 8 (Tanh) -> 1 (Sigmoid)\n");

    let model = MLP::<CpuBackend>::new(&[2, 8, 1], &[Activation::Tanh, Activation::Sigmoid]);

    // Create callbacks
    // Logging callback - outputs training progress to console
    let logging_callback = LoggingCallback::console_only();

    // Create learning rate scheduler
    let lr_scheduler = ExponentialLR::new(0.5, 0.9995); // Decay by 0.9995 each epoch

    println!("Callbacks configured:");
    println!("  1. LoggingCallback (console output)");
    println!("  2. ExponentialLR scheduler (gamma=0.9995)\n");

    // Build trainer with callbacks
    let trainer = Trainer::builder(MSELoss, SGD::new(0.5), NoRegularizer)
        .batch_size(4) // Full batch for XOR
        .max_epochs(5000)
        .verbose(false) // Callbacks handle logging
        .with_callback(Box::new(logging_callback))
        .with_lr_scheduler(Box::new(lr_scheduler))
        .with_initial_lr(0.5)
        .build();

    println!("Training for up to 5000 epochs...\n");

    let fitted = match trainer.fit(model, &dataset) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Training failed: {}", e);
            return;
        }
    };

    println!("\n=== Training Complete ===\n");

    // Test predictions
    println!("Predictions:");
    let mut correct = 0;
    for (inputs, expected) in x.iter().zip(y.iter()) {
        let input_1d = machinelearne_rs::backend::Tensor1D::<CpuBackend>::new(
            inputs.iter().map(|&v| v as f32).collect(),
        );
        let pred = fitted.predict(&input_1d);
        let pred_val = pred.to_vec()[0];
        let rounded = if pred_val > 0.5 { 1.0 } else { 0.0 };
        let is_correct = (rounded - expected).abs() < 0.01;
        if is_correct {
            correct += 1;
        }
        println!(
            "  XOR({:?}) = {:.4} (expected {:.1}) {}",
            inputs,
            pred_val,
            expected,
            if is_correct { "OK" } else { "X" }
        );
    }

    println!(
        "\nAccuracy: {}/{} ({:.0}%)",
        correct,
        x.len(),
        100.0 * correct as f64 / x.len() as f64
    );
}
