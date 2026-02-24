//! MLP training example: learning the XOR function.
//!
//! This example demonstrates:
//! - Creating an MLP model with configurable architecture
//! - Training with the existing Trainer infrastructure
//! - Multi-layer neural network learning non-linear patterns

use machinelearne_rs::{
    backend::CpuBackend,
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    model::{Activation, InferenceModel, MLP},
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
};

fn main() {
    println!("=== MLP Training: XOR Problem ===\n");

    // XOR dataset: [x1, x2] -> y
    let x = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y = vec![0.0, 1.0, 1.0, 0.0];

    let dataset = InMemoryDataset::new(x.clone(), y.clone()).unwrap();

    // Create MLP: 2 inputs -> 8 hidden (Tanh) -> 1 output (Sigmoid)
    // Using Tanh for hidden layer works well for XOR
    // Using Sigmoid for output to constrain to [0, 1]
    let model = MLP::<CpuBackend>::new(&[2, 8, 1], &[Activation::Tanh, Activation::Sigmoid]);

    println!("Architecture: 2 -> 8 (Tanh) -> 1 (Sigmoid)");
    println!("Training for 10000 epochs with learning rate 0.5...\n");

    // Build trainer
    let trainer = Trainer::builder(MSELoss, SGD::new(0.5), NoRegularizer)
        .batch_size(1) // Process one sample at a time for proper gradient computation
        .max_epochs(10000)
        .verbose(false) // Set to true to see epoch-by-epoch loss
        .build();

    // Train the model
    let fitted = match trainer.fit(model, &dataset) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Training failed: {}", e);
            return;
        }
    };

    println!("\n=== Results ===\n");

    // Test predictions
    for (inputs, expected) in x.iter().zip(y.iter()) {
        let input_1d = machinelearne_rs::backend::Tensor1D::<CpuBackend>::new(
            inputs.iter().map(|&v| v as f32).collect(),
        );
        let pred = fitted.predict(&input_1d);
        let pred_vec = pred.to_vec();
        let pred_val = pred_vec[0];
        let rounded = if pred_val > 0.5 { 1.0 } else { 0.0 };
        let correct = (rounded - expected).abs() < 0.01;
        println!(
            "XOR({:?}) = {:.4} (expected {:.1}) {}",
            inputs,
            pred_val,
            expected,
            if correct { "✓" } else { "✗" }
        );
    }

    // Calculate accuracy
    let mut correct = 0;
    for (inputs, expected) in x.iter().zip(y.iter()) {
        let input_1d = machinelearne_rs::backend::Tensor1D::<CpuBackend>::new(
            inputs.iter().map(|&v| v as f32).collect(),
        );
        let pred = fitted.predict(&input_1d);
        let pred_vec = pred.to_vec();
        let pred_val = pred_vec[0];
        let rounded = if pred_val > 0.5 { 1.0 } else { 0.0 };
        if (rounded - expected).abs() < 0.01 {
            correct += 1;
        }
    }

    println!(
        "\nAccuracy: {}/{} ({:.0}%)",
        correct,
        x.len(),
        100.0 * correct as f64 / x.len() as f64
    );
}
