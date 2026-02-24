//! ONNX export example for MLP models.
//!
//! This example demonstrates:
//! - Training an MLP on XOR problem
//! - Exporting the trained model to ONNX format
//! - Verifying the exported model file

use machinelearne_rs::{
    backend::CpuBackend,
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    model::{Activation, InferenceModel, MLP},
    onnx::OnnxExportable,
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
};

fn main() {
    println!("=== MLP ONNX Export Example ===\n");

    // XOR dataset
    let x = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y = vec![0.0, 1.0, 1.0, 0.0];

    let dataset = InMemoryDataset::new(x.clone(), y.clone()).unwrap();

    // Create and train MLP
    println!("Training MLP on XOR problem...");
    let model = MLP::<CpuBackend>::new(&[2, 8, 1], &[Activation::Tanh, Activation::Sigmoid]);

    let trainer = Trainer::builder(MSELoss, SGD::new(0.5), NoRegularizer)
        .batch_size(1)
        .max_epochs(5000)
        .verbose(false)
        .build();

    let fitted = trainer.fit(model, &dataset).unwrap();
    println!("Training complete!\n");

    // Test predictions
    println!("Trained model predictions:");
    for (inputs, expected) in x.iter().zip(y.iter()) {
        let input_1d = machinelearne_rs::backend::Tensor1D::<CpuBackend>::new(
            inputs.iter().map(|&v| v as f32).collect(),
        );
        let pred = fitted.predict(&input_1d);
        let pred_val = pred.to_vec()[0] as f32;
        let rounded = if pred_val > 0.5 { 1.0 } else { 0.0 };
        let correct = (rounded - expected).abs() < 0.01;
        println!(
            "  XOR({:?}) = {:.4} (expected {:.1}) {}",
            inputs,
            pred_val,
            expected,
            if correct { "✓" } else { "✗" }
        );
    }

    // Export to ONNX
    println!("\n--- Exporting to ONNX ---");
    let output_path = std::env::temp_dir().join("xor_mlp.onnx");

    // Method 1: Using save_onnx with model name
    fitted
        .save_onnx(&output_path, Some("xor_mlp_model"))
        .unwrap();
    println!("Model saved to: {:?}", output_path);

    // Verify the file exists and has content
    let metadata = std::fs::metadata(&output_path).unwrap();
    println!("File size: {} bytes", metadata.len());

    // Read and verify ONNX structure
    let bytes = std::fs::read(&output_path).unwrap();

    // Check for ONNX magic bytes and structure
    // ONNX files are protobuf, so we look for our model name
    let bytes_str = String::from_utf8_lossy(&bytes);
    println!("\nONNX model structure verification:");
    println!(
        "  Contains producer name 'machinelearne-rs': {}",
        bytes_str.contains("machinelearne-rs")
    );
    println!(
        "  Contains model name 'xor_mlp_model': {}",
        bytes_str.contains("xor_mlp_model")
    );
    println!("  Contains 'input' tensor: {}", bytes_str.contains("input"));
    println!(
        "  Contains 'output' tensor: {}",
        bytes_str.contains("output")
    );
    println!("  Contains Gemm nodes: {}", bytes_str.contains("Gemm"));
    println!("  Contains Tanh activation: {}", bytes_str.contains("Tanh"));
    println!(
        "  Contains Sigmoid activation: {}",
        bytes_str.contains("Sigmoid")
    );

    // Method 2: Using to_onnx_default for quick export
    println!("\n--- Alternative Export Methods ---");
    let onnx_bytes = fitted.to_onnx_default().unwrap();
    println!("to_onnx_default() produced {} bytes", onnx_bytes.len());

    // Method 3: Custom export with specific name
    let custom_bytes = fitted.to_onnx("custom_xor_model").unwrap();
    println!(
        "to_onnx(\"custom_xor_model\") produced {} bytes",
        custom_bytes.len()
    );

    // Clean up
    std::fs::remove_file(&output_path).ok();

    println!("\n=== Export Complete ===");
    println!("The ONNX model can now be loaded in:");
    println!("  - ONNX Runtime (Python, C++, etc.)");
    println!("  - TensorRT");
    println!("  - OpenVINO");
    println!("  - Any ONNX-compatible inference engine");
}
