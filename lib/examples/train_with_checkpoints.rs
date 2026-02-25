//! Checkpoint example: saving and resuming training.
//!
//! This example demonstrates:
//! - Using start_epoch to resume training
//! - Manual checkpoint management using checkpoint utilities
//! - Training state restoration

use std::path::Path;

use machinelearne_rs::{
    backend::CpuBackend,
    checkpoint,
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    model::{Activation, InferenceModel, MLP},
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
};

fn main() {
    println!("=== Checkpoint Example ===\n");

    let checkpoint_dir = "checkpoints/demo";

    // Create a simple regression dataset
    let x: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 / 100.0]).collect();
    let y: Vec<f32> = x.iter().map(|v| 2.0 * v[0] + 1.0).collect();

    let dataset = InMemoryDataset::new(x, y).unwrap();

    // Check if we have a checkpoint to resume from
    if Path::new(checkpoint_dir).exists() {
        if let Ok(checkpoint) = checkpoint::find_latest_checkpoint(checkpoint_dir) {
            println!("Found checkpoint from epoch {}", checkpoint.metadata.epoch);
            println!("Previous loss: {:.6}", checkpoint.metadata.loss);
            println!("Resuming training...\n");

            // Create a new model (in a real app, you would restore parameters from checkpoint bytes)
            // Example: let model = MyModel::from_checkpoint_bytes(&checkpoint.path)?;
            let model =
                MLP::<CpuBackend>::new(&[1, 4, 1], &[Activation::ReLU, Activation::Identity]);

            let trainer = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
                .batch_size(16)
                .max_epochs(500)
                .start_epoch(checkpoint.metadata.epoch + 1) // Resume from next epoch
                .verbose(true)
                .build();

            let _fitted = trainer.fit(model, &dataset).unwrap();
            println!("Training completed from checkpoint!");
            return;
        }
    }

    println!("No checkpoint found. Starting fresh training...\n");

    let model = MLP::<CpuBackend>::new(&[1, 4, 1], &[Activation::ReLU, Activation::Identity]);

    let trainer = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
        .batch_size(16)
        .max_epochs(100) // First run: only 100 epochs
        .verbose(true)
        .build();

    let fitted = trainer.fit(model, &dataset).unwrap();

    // In a real app, you would save the fitted model here
    // fitted.save_to_file("checkpoints/demo/model.bin")?;

    println!("\n=== Training Complete ===");
    println!("Model trained for 100 epochs.");
    println!();
    println!("To demonstrate checkpoint resumption:");
    println!("1. Create the checkpoint directory: mkdir -p checkpoints/demo");
    println!("2. Save model metadata manually or using CheckpointCallback");
    println!("3. Run this example again to resume training");
    println!();
    println!("The trainer will detect the checkpoint and resume from epoch 101.");

    // Demo: Show what the fitted model predicts
    let test_input = machinelearne_rs::backend::Tensor1D::<CpuBackend>::new(vec![0.5]);
    let pred = fitted.predict(&test_input);
    println!(
        "\nSample prediction for x=0.5: {:.4} (expected: 2.0)",
        pred.to_vec()[0]
    );
}
