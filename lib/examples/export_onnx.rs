//! ONNX Export Example
//!
//! This example demonstrates how to train a model and export it to ONNX format
//! for portable deployment and optimized inference.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example export_onnx --features onnx
//! ```
//!
//! The exported model can be loaded in Python with ONNX Runtime:
//!
//! ```python
//! import onnxruntime as ort
//! import numpy as np
//!
//! session = ort.InferenceSession("linear_model.onnx")
//! input_data = np.array([[1.0, 2.0]], dtype=np.float32)
//! output = session.run(None, {"input": input_data})
//! print(output)
//! ```

#[cfg(feature = "onnx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use machinelearne_rs::{
        dataset::memory::InMemoryDataset,
        loss::MSELoss,
        model::{linear::LinearRegressor, InferenceModel},
        onnx::OnnxExportable,
        optimizer::SGD,
        regularizers::NoRegularizer,
        trainer::Trainer,
        CpuBackend, Tensor1D,
    };

    println!("=== ONNX Export Example ===\n");

    // 1. Create training data
    // y = 2*x1 + 3*x2 + noise
    println!("1. Creating training data...");
    let x = vec![
        vec![1.0, 1.0],
        vec![2.0, 1.0],
        vec![1.0, 2.0],
        vec![2.0, 2.0],
        vec![3.0, 1.0],
        vec![1.0, 3.0],
        vec![3.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 3.0],
        vec![4.0, 2.0],
    ];
    let y = vec![5.0, 7.0, 8.0, 10.0, 9.0, 11.0, 12.0, 13.0, 15.0, 14.0];
    println!("   Created {} samples with {} features", x.len(), 2);

    // 2. Create dataset
    let dataset = InMemoryDataset::new(x, y)?;

    // 3. Train the model
    println!("\n2. Training linear regression model...");
    let model = LinearRegressor::new(2);
    let loss = MSELoss;
    let opt = SGD::new(0.1);
    let reg = NoRegularizer;
    let trainer = Trainer::builder(loss, opt, reg)
        .batch_size(4)
        .max_epochs(200)
        .build();

    let fitted_model = trainer.fit(model, &dataset)?;
    println!("   Training complete!");

    // 4. Test the trained model
    println!("\n3. Testing trained model...");
    let test_input = Tensor1D::<CpuBackend>::new(vec![2.0, 3.0]);
    let prediction = fitted_model.predict(&test_input);
    println!("   Test input: [2.0, 3.0]");
    println!("   Expected: ~13.0 (2*2 + 3*3)");
    println!("   Prediction: {:.4}", prediction.to_f64());

    // 5. Export the model to ONNX
    println!("\n4. Exporting model to ONNX format...");
    let output_path = "linear_model.onnx";
    fitted_model.save_onnx(output_path)?;
    println!("   Model saved to: {}", output_path);

    // Verify the file was created
    let metadata = std::fs::metadata(output_path)?;
    println!("   File size: {} bytes", metadata.len());

    // 6. Show ONNX model size
    let onnx_bytes = fitted_model.to_onnx_default()?;
    println!("   ONNX model size: {} bytes", onnx_bytes.len());

    println!("\n=== Export Complete ===");
    println!("\nThe exported model can now be loaded in any ONNX Runtime supported language:");
    println!("  - Python: import onnxruntime as ort");
    println!("  - C++: onnxruntime C++ API");
    println!("  - JavaScript: onnxruntime-web");
    println!("  - Java: onnxruntime Java API");

    // Clean up
    std::fs::remove_file(output_path).ok();

    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("This example requires the 'onnx' feature.");
    eprintln!("Run with: cargo run --example export_onnx --features onnx");
}
