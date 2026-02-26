//! Learning rate scheduling example.
//!
//! This example demonstrates:
//! - Using StepLR scheduler to decay learning rate
//! - Training with learning rate callbacks
//! - Comparing training with and without scheduling

use machinelearne_rs::{
    backend::CpuBackend,
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    model::{Activation, InferenceModel, MLP},
    optimizer::SGD,
    regularizers::NoRegularizer,
    schedulers::StepLR,
    trainer::Trainer,
};

fn main() {
    println!("=== Learning Rate Scheduling Example ===\n");

    // Create a simple regression dataset
    let x: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 / 100.0]).collect();
    let y: Vec<f32> = x.iter().map(|v| 2.0 * v[0] + 1.0).collect();

    let dataset = InMemoryDataset::new(x.clone(), y.clone()).unwrap();

    // Split into train and validation (simple 80/20 split)
    let split_point = (x.len() * 80) / 100;
    let train_x: Vec<Vec<f32>> = x.iter().take(split_point).cloned().collect();
    let train_y: Vec<f32> = y.iter().take(split_point).cloned().collect();
    let val_x: Vec<Vec<f32>> = x.iter().skip(split_point).cloned().collect();
    let val_y: Vec<f32> = y.iter().skip(split_point).cloned().collect();

    let train_dataset = InMemoryDataset::new(train_x, train_y).unwrap();
    let _val_dataset = InMemoryDataset::new(val_x, val_y).unwrap();

    // --- Train WITH learning rate scheduling ---
    println!("Training WITH StepLR scheduling:");
    println!("  - Initial LR: 0.1");
    println!("  - Step size: 100 epochs (decay every 100 epochs)");
    println!("  - Gamma: 0.5 (halve the LR each step)\n");

    let model1 = MLP::<CpuBackend>::new(&[1, 4, 1], &[Activation::ReLU, Activation::Identity]);

    let scheduler = StepLR::new(0.1, 100, 0.5);

    let trainer1 = Trainer::builder(MSELoss, SGD::new(0.1), NoRegularizer)
        .batch_size(16)
        .max_epochs(500)
        .verbose(false)
        .with_lr_scheduler(Box::new(scheduler))
        .with_initial_lr(0.1)
        .build();

    let _fitted1 = trainer1.fit(model1, &train_dataset).unwrap();

    println!("Training with LR scheduling completed!\n");

    // --- Train WITHOUT learning rate scheduling ---
    println!("Training WITHOUT scheduling (constant LR):");
    println!("  - Constant LR: 0.01\n");

    let model2 = MLP::<CpuBackend>::new(&[1, 4, 1], &[Activation::ReLU, Activation::Identity]);

    let trainer2 = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
        .batch_size(16)
        .max_epochs(500)
        .verbose(false)
        .build();

    let _fitted2 = trainer2.fit(model2, &train_dataset).unwrap();

    println!("Training without scheduling completed!\n");

    println!("=== Comparison ===");
    println!("With LR scheduling:");
    println!("  - Starts with higher LR (0.1) for fast initial learning");
    println!("  - Decays LR over time for fine-tuning");
    println!("  - Can escape local minima early, then converge precisely");
    println!();
    println!("Without scheduling:");
    println!("  - Fixed LR requires manual tuning");
    println!("  - Too high: may oscillate near minimum");
    println!("  - Too low: slow convergence");
}
