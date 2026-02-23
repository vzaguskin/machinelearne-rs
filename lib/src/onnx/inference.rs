//! ONNX Runtime inference session.
//!
//! Provides functionality to load and run ONNX models using ONNX Runtime.

use super::error::OnnxError;
use ndarray::{Array1, Array2};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::Path;
use std::sync::Mutex;

/// ONNX inference session for running models.
///
/// Wraps the `ort` crate's Session to provide a simple interface
/// for loading ONNX models and running inference.
pub struct OnnxInferenceSession {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
}

impl OnnxInferenceSession {
    /// Load an ONNX model from a file for CPU inference.
    ///
    /// # Arguments
    /// * `path` - Path to the ONNX model file
    ///
    /// # Example
    /// ```rust,ignore
    /// use machinelearne_rs::onnx::OnnxInferenceSession;
    ///
    /// let session = OnnxInferenceSession::load("model.onnx")?;
    /// let predictions = session.predict(&input)?;
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, OnnxError> {
        let session = Session::builder()
            .map_err(|e| OnnxError::ModelCreationError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| OnnxError::ModelCreationError(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| OnnxError::InferenceError(e.to_string()))?;

        // Get input and output names from the model
        let input_info = session
            .inputs()
            .first()
            .map(|outlet| outlet.name().to_string())
            .ok_or_else(|| OnnxError::MissingField("input".to_string()))?;

        let output_info = session
            .outputs()
            .first()
            .map(|outlet| outlet.name().to_string())
            .ok_or_else(|| OnnxError::MissingField("output".to_string()))?;

        Ok(Self {
            session: Mutex::new(session),
            input_name: input_info,
            output_name: output_info,
        })
    }

    /// Load an ONNX model with GPU support (CUDA).
    ///
    /// # Arguments
    /// * `path` - Path to the ONNX model file
    /// * `device_id` - GPU device ID (typically 0)
    #[cfg(feature = "onnx-cuda")]
    pub fn load_gpu<P: AsRef<Path>>(path: P, _device_id: i32) -> Result<Self, OnnxError> {
        let session = Session::builder()
            .map_err(|e| OnnxError::ModelCreationError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| OnnxError::ModelCreationError(e.to_string()))?
            .with_execution_providers([ort::ep::CUDA::default().build()])
            .map_err(|e| OnnxError::ModelCreationError(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| OnnxError::InferenceError(e.to_string()))?;

        let input_info = session
            .inputs()
            .first()
            .map(|outlet| outlet.name().to_string())
            .ok_or_else(|| OnnxError::MissingField("input".to_string()))?;

        let output_info = session
            .outputs()
            .first()
            .map(|outlet| outlet.name().to_string())
            .ok_or_else(|| OnnxError::MissingField("output".to_string()))?;

        Ok(Self {
            session: Mutex::new(session),
            input_name: input_info,
            output_name: output_info,
        })
    }

    /// Run inference on a batch of samples.
    ///
    /// # Arguments
    /// * `input` - Input data with shape [batch_size, num_features]
    ///
    /// # Returns
    /// Output predictions as a 1D array with shape [batch_size]
    pub fn predict(&self, input: &Array2<f32>) -> Result<Array1<f32>, OnnxError> {
        // Get shape for tensor creation
        let shape: Vec<i64> = input.shape().iter().map(|&d| d as i64).collect();
        let data: Vec<f32> = input.iter().copied().collect();

        // Create input value using the (shape, data) tuple format
        let input_value = ort::value::Value::from_array((shape, data.into_boxed_slice()))
            .map_err(|e| OnnxError::InferenceError(e.to_string()))?;

        // Lock the session and run inference
        let mut session = self
            .session
            .lock()
            .map_err(|_| OnnxError::InferenceError("Failed to lock session".to_string()))?;

        let outputs = session
            .run(ort::inputs![&self.input_name => input_value])
            .map_err(|e| OnnxError::InferenceError(e.to_string()))?;

        // Extract output - try_extract_tensor returns (&Shape, &[f32])
        let (output_shape, output_data) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| OnnxError::InferenceError(e.to_string()))?;

        // Convert to 1D array
        let shape: &[i64] = &*output_shape;
        if shape.len() == 2 && shape[1] == 1 {
            // Output is [batch, 1], flatten to [batch]
            Ok(Array1::from_vec(output_data.to_vec()))
        } else if shape.len() == 1 {
            Ok(Array1::from_vec(output_data.to_vec()))
        } else {
            Err(OnnxError::ShapeMismatch {
                expected: vec![input.nrows()],
                got: shape.iter().map(|&d| d as usize).collect(),
            })
        }
    }

    /// Run inference on a single sample.
    ///
    /// # Arguments
    /// * `input` - Input features as a 1D array
    ///
    /// # Returns
    /// Single prediction value
    pub fn predict_one(&self, input: &Array1<f32>) -> Result<f32, OnnxError> {
        // Reshape to [1, n_features]
        let input_2d = input.clone().insert_axis(ndarray::Axis(0));
        let output = self.predict(&input_2d)?;
        output
            .first()
            .copied()
            .ok_or_else(|| OnnxError::InferenceError("Empty output".to_string()))
    }

    /// Get the input name for the model.
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    /// Get the output name for the model.
    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    /// Get the number of input features expected by the model.
    pub fn n_input_features(&self) -> Result<usize, OnnxError> {
        let session = self
            .session
            .lock()
            .map_err(|_| OnnxError::InferenceError("Failed to lock session".to_string()))?;

        let input_info = session
            .inputs()
            .first()
            .ok_or_else(|| OnnxError::MissingField("input info".to_string()))?;

        // Get shape from input info - dtype is a ValueType enum
        match input_info.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => {
                let dims: &[i64] = &*shape;
                if dims.len() >= 2 {
                    dims.get(1)
                        .copied()
                        .filter(|&d| d > 0)
                        .map(|d| d as usize)
                        .ok_or_else(|| {
                            OnnxError::MissingField("input feature dimension".to_string())
                        })
                } else {
                    Err(OnnxError::InvalidParameters(
                        "Expected 2D input tensor".to_string(),
                    ))
                }
            }
            _ => Err(OnnxError::InvalidParameters(
                "Expected tensor input type".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::backend::{Scalar, Tensor1D};
    use crate::model::linear::{Fitted, LinearModel, LinearParams};
    use crate::onnx::OnnxExportable;
    use ndarray::array;

    fn create_and_export_model() -> tempfile::TempPath {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![2.0, 3.0]),
            bias: Scalar::new(1.0),
        };
        let model = LinearModel::<CpuBackend, Fitted>::new(params);

        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.into_temp_path();
        model.save_onnx(&path).unwrap();
        path
    }

    #[test]
    fn test_load_and_predict() {
        let model_path = create_and_export_model();

        // Load the model
        let session = OnnxInferenceSession::load(&model_path).unwrap();

        // Test input: [1, 2]
        // Expected: 2*1 + 3*2 + 1 = 9
        let input = array![[1.0_f32, 2.0_f32]];
        let output = session.predict(&input).unwrap();

        assert_eq!(output.len(), 1);
        assert!((output[0] - 9.0).abs() < 1e-5);

        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_predict_batch() {
        let model_path = create_and_export_model();
        let session = OnnxInferenceSession::load(&model_path).unwrap();

        // Batch input: [[1, 2], [3, 4]]
        // Expected: [9, 19]
        let input = array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]];
        let output = session.predict(&input).unwrap();

        assert_eq!(output.len(), 2);
        assert!((output[0] - 9.0).abs() < 1e-5);
        assert!((output[1] - 19.0).abs() < 1e-5);

        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_predict_one() {
        let model_path = create_and_export_model();
        let session = OnnxInferenceSession::load(&model_path).unwrap();

        let input = array![1.0_f32, 2.0_f32];
        let output = session.predict_one(&input).unwrap();

        assert!((output - 9.0).abs() < 1e-5);

        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_n_input_features() {
        let model_path = create_and_export_model();
        let session = OnnxInferenceSession::load(&model_path).unwrap();

        let n_features = session.n_input_features().unwrap();
        assert_eq!(n_features, 2);

        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_input_output_names() {
        let model_path = create_and_export_model();
        let session = OnnxInferenceSession::load(&model_path).unwrap();

        // Test accessor methods
        assert!(!session.input_name().is_empty());
        assert!(!session.output_name().is_empty());

        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_load_invalid_file() {
        // Create a temp file with invalid content
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.into_temp_path();
        std::fs::write(&path, b"not a valid onnx model").ok();

        // Should fail to load
        let result = OnnxInferenceSession::load(&path);
        assert!(result.is_err(), "Should fail to load invalid ONNX file");

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_predict_single_output_shape() {
        // Test that output with shape [batch, 1] is properly flattened to [batch]
        let model_path = create_and_export_model();
        let session = OnnxInferenceSession::load(&model_path).unwrap();

        // Single sample should produce 1D output
        let input = array![[1.0_f32, 2.0_f32]];
        let output = session.predict(&input).unwrap();
        assert_eq!(output.shape(), &[1]); // Should be [1] not [1, 1]

        std::fs::remove_file(model_path).ok();
    }

    // ============================================================================
    // Execution Provider Tests (Task 3.4, 3.5)
    // ============================================================================

    /// Test CPU execution provider - always available
    /// Task 3.4: Add tests for CPU provider (always available)
    #[test]
    fn test_cpu_provider_load() {
        let model_path = create_and_export_model();

        // CPU provider is always available - this should never fail
        let session = OnnxInferenceSession::load(&model_path);
        assert!(session.is_ok(), "CPU provider should always be available");

        std::fs::remove_file(model_path).ok();
    }

    /// Test CPU provider prediction accuracy
    /// Task 3.4: Add tests for CPU provider (always available)
    #[test]
    fn test_cpu_provider_prediction_accuracy() {
        let model_path = create_and_export_model();
        let session = OnnxInferenceSession::load(&model_path).unwrap();

        // Multiple test cases to verify CPU provider accuracy
        let test_cases = vec![
            (array![[0.0_f32, 0.0_f32]], 1.0),    // 2*0 + 3*0 + 1 = 1
            (array![[1.0_f32, 0.0_f32]], 3.0),    // 2*1 + 3*0 + 1 = 3
            (array![[0.0_f32, 1.0_f32]], 4.0),    // 2*0 + 3*1 + 1 = 4
            (array![[2.0_f32, 3.0_f32]], 14.0),   // 2*2 + 3*3 + 1 = 14
            (array![[-1.0_f32, -1.0_f32]], -4.0), // 2*(-1) + 3*(-1) + 1 = -4
        ];

        for (input, expected) in test_cases {
            let output = session.predict(&input).unwrap();
            assert!(
                (output[0] - expected).abs() < 1e-5,
                "CPU prediction failed: expected {}, got {}",
                expected,
                output[0]
            );
        }

        std::fs::remove_file(model_path).ok();
    }

    /// Test CPU provider with various batch sizes
    /// Task 3.4: Add tests for CPU provider (always available)
    #[test]
    fn test_cpu_provider_batch_sizes() {
        let model_path = create_and_export_model();
        let session = OnnxInferenceSession::load(&model_path).unwrap();

        // Test with batch size 1
        let input = array![[1.0_f32, 2.0_f32]];
        let output = session.predict(&input).unwrap();
        assert_eq!(output.len(), 1);

        // Test with batch size 10
        let input: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        let input = ndarray::Array2::from_shape_vec((10, 2), input.concat()).unwrap();
        let output = session.predict(&input).unwrap();
        assert_eq!(output.len(), 10);

        // Test with batch size 100
        let input: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        let input = ndarray::Array2::from_shape_vec((100, 2), input.concat()).unwrap();
        let output = session.predict(&input).unwrap();
        assert_eq!(output.len(), 100);

        std::fs::remove_file(model_path).ok();
    }

    /// Test CUDA execution provider - conditional
    /// Task 3.5: Add conditional tests for CUDA provider (skip if unavailable)
    #[cfg(feature = "onnx-cuda")]
    #[test]
    fn test_cuda_provider_load() {
        let model_path = create_and_export_model();

        // Try CUDA provider - may fail if CUDA is not available at runtime
        match OnnxInferenceSession::load_gpu(&model_path, 0) {
            Ok(session) => {
                // CUDA is available, run a simple prediction to verify
                let input = array![[1.0_f32, 2.0_f32]];
                let output = session.predict(&input);
                assert!(
                    output.is_ok(),
                    "CUDA prediction should succeed when provider loads"
                );
            }
            Err(e) => {
                // CUDA not available at runtime - this is expected on systems without GPU
                eprintln!("CUDA not available at runtime, skipping CUDA test: {}", e);
            }
        }

        std::fs::remove_file(model_path).ok();
    }

    /// Test CUDA provider prediction accuracy (if available)
    /// Task 3.5: Add conditional tests for CUDA provider (skip if unavailable)
    #[cfg(feature = "onnx-cuda")]
    #[test]
    fn test_cuda_provider_accuracy_if_available() {
        let model_path = create_and_export_model();

        // Only run accuracy test if CUDA is available
        if let Ok(session) = OnnxInferenceSession::load_gpu(&model_path, 0) {
            // Compare CUDA predictions with expected values
            let input = array![[1.0_f32, 2.0_f32]];
            let output = session.predict(&input).unwrap();

            // Expected: 2*1 + 3*2 + 1 = 9
            assert!(
                (output[0] - 9.0).abs() < 1e-4,
                "CUDA prediction accuracy check failed: expected 9.0, got {}",
                output[0]
            );

            eprintln!("CUDA provider accuracy test passed");
        } else {
            eprintln!("CUDA not available, skipping accuracy test");
        }

        std::fs::remove_file(model_path).ok();
    }

    /// Test that CPU and CUDA providers produce consistent results (if CUDA available)
    /// Task 3.5: Add conditional tests for CUDA provider (skip if unavailable)
    #[cfg(feature = "onnx-cuda")]
    #[test]
    fn test_cpu_cuda_consistency() {
        let model_path = create_and_export_model();

        // Load with CPU
        let cpu_session = OnnxInferenceSession::load(&model_path).unwrap();

        // Try to load with CUDA
        if let Ok(cuda_session) = OnnxInferenceSession::load_gpu(&model_path, 0) {
            // Both available - compare results
            let test_inputs = vec![
                array![[1.0_f32, 2.0_f32]],
                array![[3.0_f32, 4.0_f32]],
                array![[-1.0_f32, 0.5_f32]],
            ];

            for input in test_inputs {
                let cpu_output = cpu_session.predict(&input).unwrap();
                let cuda_output = cuda_session.predict(&input).unwrap();

                assert_eq!(cpu_output.len(), cuda_output.len());

                for (i, (cpu, cuda)) in cpu_output.iter().zip(cuda_output.iter()).enumerate() {
                    let diff = (cpu - cuda).abs();
                    assert!(
                        diff < 1e-4,
                        "CPU/CUDA mismatch at index {}: CPU={}, CUDA={}, diff={}",
                        i,
                        cpu,
                        cuda,
                        diff
                    );
                }
            }

            eprintln!("CPU/CUDA consistency test passed");
        } else {
            eprintln!("CUDA not available, skipping consistency test");
        }

        std::fs::remove_file(model_path).ok();
    }
}
