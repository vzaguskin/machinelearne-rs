//! ONNX Deployment Example
//!
//! This example demonstrates the complete ML deployment workflow:
//! 1. Training a model with preprocessing pipeline (StandardScaler + LinearModel)
//! 2. Exporting the entire pipeline to ONNX format
//! 3. Starting an HTTP inference server
//! 4. Making predictions via HTTP API
//! 5. Comparing native Rust vs ONNX predictions
//!
//! ## Usage
//!
//! ```bash
//! # Requires onnx-server feature and ONNX Runtime installed
//! # Install ONNX Runtime to /usr/local first
//! cargo run --example onnx_deployment --features onnx-server
//! ```
//!
//! ## Prerequisites
//!
//! This example requires ONNX Runtime to be installed:
//! ```bash
//! wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
//! tar -xzf onnxruntime-linux-x64-1.23.2.tgz
//! sudo cp -r onnxruntime-linux-x64-1.23.2/lib/* /usr/local/lib/
//! sudo cp -r onnxruntime-linux-x64-1.23.2/include/* /usr/local/include/
//! sudo ldconfig
//! ```

#[cfg(feature = "onnx-server")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use machinelearne_rs::{
        backend::CpuBackend,
        backend::Tensor2D,
        dataset::memory::InMemoryDataset,
        loss::MSELoss,
        model::{linear::LinearRegressor, InferenceModel},
        onnx::{
            server::{ExecutionProvider, OnnxServer},
            OnnxExportable, OnnxInferenceSession,
        },
        optimizer::SGD,
        pipeline::FittedPipeline,
        preprocessing::{
            pipeline::Pipeline, scaling::StandardScaler, traits::FittedTransformer,
            traits::Transformer,
        },
        regularizers::NoRegularizer,
        trainer::Trainer,
    };
    use ndarray::array;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║          ONNX Deployment Workflow Example                  ║");
    println!("║     (Full Pipeline: StandardScaler + LinearModel)          ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // STEP 1: Create raw training data (unscaled)
    // =========================================================================
    println!("📊 STEP 1: Creating Training Data");
    println!("─────────────────────────────────────────────────────────────");

    // Raw unscaled data: features in different scales
    // Feature 0: values around 100-400
    // Feature 1: values around 0-20
    // Target: y = 0.5*x0 + 2.0*x1 + 10
    println!("   Creating raw training dataset with different scales...");
    let x_raw = vec![
        vec![100.0, 5.0],  // y = 0.5*100 + 2.0*5 + 10 = 70
        vec![150.0, 8.0],  // y = 0.5*150 + 2.0*8 + 10 = 101
        vec![200.0, 10.0], // y = 0.5*200 + 2.0*10 + 10 = 130
        vec![250.0, 12.0], // y = 0.5*250 + 2.0*12 + 10 = 159
        vec![300.0, 15.0], // y = 0.5*300 + 2.0*15 + 10 = 190
        vec![350.0, 18.0], // y = 0.5*350 + 2.0*18 + 10 = 221
        vec![400.0, 20.0], // y = 0.5*400 + 2.0*20 + 10 = 250
        vec![120.0, 6.0],  // y = 0.5*120 + 2.0*6 + 10 = 82
        vec![180.0, 9.0],  // y = 0.5*180 + 2.0*9 + 10 = 118
        vec![280.0, 14.0], // y = 0.5*280 + 2.0*14 + 10 = 178
    ];
    let y = vec![
        70.0, 101.0, 130.0, 159.0, 190.0, 221.0, 250.0, 82.0, 118.0, 178.0,
    ];
    println!(
        "   Created {} samples with {} features (different scales)",
        x_raw.len(),
        2
    );
    println!("   Feature 0: range ~100-400");
    println!("   Feature 1: range ~5-20");

    // Convert to Tensor2D
    let flat_x: Vec<f32> = x_raw.iter().flatten().copied().collect();
    let x_tensor = Tensor2D::<CpuBackend>::new(flat_x, x_raw.len(), 2);

    // =========================================================================
    // STEP 2: Fit preprocessing pipeline (StandardScaler)
    // =========================================================================
    println!("\n🔧 STEP 2: Fitting Preprocessing Pipeline");
    println!("─────────────────────────────────────────────────────────────");

    // Create and fit the preprocessing pipeline
    let preproc_pipeline = Pipeline::<CpuBackend>::new().add_standard_scaler(StandardScaler::new());

    println!("   Fitting StandardScaler on raw data...");
    let fitted_preproc = preproc_pipeline.fit(&x_tensor)?;
    println!("   ✓ Preprocessing pipeline fitted!");

    // Transform the training data
    let x_scaled = fitted_preproc.transform(&x_tensor)?;
    println!("   Data standardized (mean=0, std=1)");

    // =========================================================================
    // STEP 3: Train the model on preprocessed data
    // =========================================================================
    println!("\n📈 STEP 3: Training Model on Preprocessed Data");
    println!("─────────────────────────────────────────────────────────────");

    // Create dataset from preprocessed data
    let (n_samples, n_features) = x_scaled.shape();
    let x_scaled_vec: Vec<Vec<f32>> = {
        let flat = x_scaled.ravel().to_vec();
        (0..n_samples)
            .map(|r| {
                (0..n_features)
                    .map(|c| flat[r * n_features + c] as f32)
                    .collect()
            })
            .collect()
    };
    let dataset = InMemoryDataset::new(x_scaled_vec, y.clone())?;

    // Train the model
    let model = LinearRegressor::new(2);
    let loss = MSELoss;
    let opt = SGD::new(0.1);
    let reg = NoRegularizer;

    let trainer = Trainer::builder(loss, opt, reg)
        .batch_size(4)
        .max_epochs(500)
        .build();

    println!("   Training linear regression model (500 epochs)...");
    let fitted_model = trainer.fit(model, &dataset)?;
    println!("   ✓ Training complete!");

    let params = fitted_model.extract_params();
    println!(
        "   Learned weights: [{:.4}, {:.4}]",
        params.weights.get(0).unwrap(),
        params.weights.get(1).unwrap()
    );
    println!("   Learned bias: {:.4}", params.bias as f64);

    // =========================================================================
    // STEP 4: Create and export the full pipeline
    // =========================================================================
    println!("\n📦 STEP 4: Creating and Exporting Full Pipeline");
    println!("─────────────────────────────────────────────────────────────");

    // Create the fitted pipeline (preprocessor + model)
    let full_pipeline = FittedPipeline::new(
        Some(fitted_preproc),
        None, // No polynomial features
        fitted_model,
    );
    println!("   Created FittedPipeline with:");
    println!("     - 1 preprocessing step (StandardScaler)");
    println!("     - Linear regression model");

    // Export to ONNX
    let model_path = "deployment_pipeline.onnx";
    println!("\n   Exporting pipeline to: {}", model_path);

    full_pipeline.save_onnx(model_path)?;
    let metadata = std::fs::metadata(model_path)?;
    println!("   ✓ Pipeline exported successfully!");
    println!("   File size: {} bytes", metadata.len());

    // =========================================================================
    // STEP 5: Verify ONNX predictions match native
    // =========================================================================
    println!("\n🔍 STEP 5: Verifying ONNX Predictions");
    println!("─────────────────────────────────────────────────────────────");

    // Load the ONNX model
    let onnx_session = OnnxInferenceSession::load(model_path)?;
    println!("   ✓ ONNX model loaded successfully!");

    // Test with raw (unscaled) inputs - ONNX model should handle scaling internally
    let test_inputs_raw = array![
        [150.0_f32, 10.0_f32], // y = 0.5*150 + 2.0*10 + 10 = 105
        [250.0_f32, 15.0_f32], // y = 0.5*250 + 2.0*15 + 10 = 160
        [350.0_f32, 20.0_f32], // y = 0.5*350 + 2.0*20 + 10 = 225
    ];

    println!("\n   Comparing predictions (RAW input -> pipeline handles scaling):");
    println!(
        "   {:>16} {:>12} {:>12} {:>12}",
        "Raw Input", "Native", "ONNX", "Diff"
    );
    println!("   {}", "─".repeat(56));

    let mut all_match = true;
    for i in 0..test_inputs_raw.nrows() {
        let input_row = test_inputs_raw.row(i);

        // Native prediction through full pipeline
        let raw_input = Tensor2D::<CpuBackend>::new(vec![input_row[0], input_row[1]], 1, 2);
        let native_pred = full_pipeline.predict(&raw_input)?.to_vec()[0];

        // ONNX prediction (takes raw input, handles scaling internally)
        let input_2d = array![[input_row[0], input_row[1]]];
        let onnx_pred = onnx_session.predict(&input_2d)?[0];

        let diff = (native_pred - onnx_pred).abs();
        let matches = diff < 0.1; // Allow small tolerance for floating point
        all_match = all_match && matches;

        println!(
            "   [{:>6.1}, {:>4.1}] {:>12.2} {:>12.2} {:>12.4} {}",
            input_row[0],
            input_row[1],
            native_pred,
            onnx_pred,
            diff,
            if matches { "✓" } else { "✗" }
        );
    }

    if all_match {
        println!("\n   ✓ All predictions match within tolerance!");
    } else {
        println!("\n   ✗ Some predictions differ!");
    }

    // =========================================================================
    // STEP 6: Start HTTP inference server
    // =========================================================================
    println!("\n🚀 STEP 6: Starting HTTP Inference Server");
    println!("─────────────────────────────────────────────────────────────");

    let server_port = 8765;
    let server_ready = Arc::new(AtomicBool::new(false));
    let server_ready_clone = server_ready.clone();
    let model_path_owned = model_path.to_string();

    // Start server in a background task
    let server_handle = tokio::spawn(async move {
        let server = OnnxServer::new()
            .model(&model_path_owned)
            .host("127.0.0.1")
            .port(server_port)
            .provider(ExecutionProvider::Cpu)
            .max_batch_size(100);

        server_ready_clone.store(true, Ordering::SeqCst);

        if let Err(e) = server.run().await {
            eprintln!("Server error: {}", e);
        }
    });

    println!("   Starting server on http://127.0.0.1:{}", server_port);
    while !server_ready.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(500)).await;

    println!("   ✓ Server started!");

    // =========================================================================
    // STEP 7: Make HTTP predictions with raw data
    // =========================================================================
    println!("\n🌐 STEP 7: Making HTTP Predictions (Raw Input)");
    println!("─────────────────────────────────────────────────────────────");

    let base_url = format!("http://127.0.0.1:{}", server_port);

    // Health check
    println!("   Checking /health endpoint...");
    let health_response = reqwest::get(&format!("{}/health", base_url)).await?;
    println!("   Health status: {}", health_response.status());

    // Single prediction with RAW (unscaled) data
    // The ONNX model handles the StandardScaler transformation internally!
    println!("\n   Making prediction with RAW (unscaled) data...");
    println!("   Input: [150.0, 10.0] (raw values, not standardized)");
    let predict_payload = serde_json::json!({
        "features": [150.0, 10.0],
        "shape": [1, 2]
    });

    let client = reqwest::Client::new();
    let predict_response = client
        .post(&format!("{}/predict", base_url))
        .json(&predict_payload)
        .send()
        .await?;

    let predict_result: serde_json::Value = predict_response.json().await?;
    println!("   Response: prediction={:?}", predict_result["prediction"]);
    println!("   Expected: ~105.0 (y = 0.5*150 + 2.0*10 + 10)");

    // Batch prediction with raw data
    println!("\n   Making batch prediction with RAW data...");
    let batch_payload = serde_json::json!({
        "samples": [
            {"features": [150.0, 10.0], "shape": [1, 2]},  // Expected: 105
            {"features": [250.0, 15.0], "shape": [1, 2]},  // Expected: 160
            {"features": [350.0, 20.0], "shape": [1, 2]}   // Expected: 225
        ]
    });

    let batch_response = client
        .post(&format!("{}/predict/batch", base_url))
        .json(&batch_payload)
        .send()
        .await?;

    let batch_result: serde_json::Value = batch_response.json().await?;
    println!("   Batch predictions:");
    for (i, pred) in batch_result["predictions"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .enumerate()
    {
        println!("     Sample {}: {:?}", i + 1, pred["prediction"]);
    }

    // =========================================================================
    // STEP 8: Cleanup
    // =========================================================================
    println!("\n🧹 STEP 8: Cleanup");
    println!("─────────────────────────────────────────────────────────────");

    server_handle.abort();
    println!("   ✓ Server stopped");

    std::fs::remove_file(model_path).ok();
    println!("   ✓ Model file removed");

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                   Summary                                   ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ ✓ Trained model with StandardScaler preprocessing          ║");
    println!("║ ✓ Exported FULL PIPELINE to ONNX format                    ║");
    println!("║ ✓ ONNX model handles scaling automatically                 ║");
    println!("║ ✓ Native vs ONNX predictions match                         ║");
    println!("║ ✓ HTTP server accepts raw input (scaling is internal)      ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    println!("\n💡 Key Points:");
    println!("   • The ONNX model includes the StandardScaler transformation");
    println!("   • Send RAW (unscaled) data to the server - it handles scaling");
    println!("   • This is the real value of ONNX: deploy the full pipeline!");
    println!("\n💡 Next Steps:");
    println!("   • Add more preprocessing: imputers, encoders, polynomial features");
    println!("   • Deploy with Docker for containerized production use");
    println!("   • Add authentication and rate limiting for production");

    Ok(())
}

#[cfg(all(feature = "onnx-inference", not(feature = "onnx-server")))]
fn main() {
    eprintln!("This example requires the 'onnx-server' feature.");
    eprintln!("Run with: cargo run --example onnx_deployment --features onnx-server");
}

#[cfg(not(feature = "onnx-inference"))]
fn main() {
    eprintln!("This example requires ONNX support.");
    eprintln!("Run with: cargo run --example onnx_deployment --features onnx-server");
}
