//! ONNX Runtime inference session.
//!
//! Provides functionality to load and run ONNX models using ONNX Runtime.

use super::error::OnnxError;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ort::session::{GraphOptimizationLevel, Session};
use std::path::Path;

/// ONNX inference session for running models.
///
/// Wraps the `ort` crate's Session to provide a simple interface
/// for loading ONNX models and running inference.
pub struct OnnxInferenceSession {
    session: Session,
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
        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .ok_or_else(|| OnnxError::MissingField("input".to_string()))?;

        let output_name = session
            .outputs
            .first()
            .map(|o| o.name.clone())
            .ok_or_else(|| OnnxError::MissingField("output".to_string()))?;

        Ok(Self {
            session,
            input_name,
            output_name,
        })
    }

    /// Load an ONNX model with GPU support (CUDA).
    ///
    /// # Arguments
    /// * `path` - Path to the ONNX model file
    /// * `device_id` - GPU device ID (typically 0)
    #[cfg(feature = "cuda")]
    pub fn load_gpu<P: AsRef<Path>>(path: P, device_id: i32) -> Result<Self, OnnxError> {
        use ort::execution_providers::CUDAExecutionProvider;

        let session = Session::builder()
            .map_err(|e| OnnxError::ModelCreationError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| OnnxError::ModelCreationError(e.to_string()))?
            .with_execution_providers([CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])
            .map_err(|e| OnnxError::ModelCreationError(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| OnnxError::InferenceError(e.to_string()))?;

        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .ok_or_else(|| OnnxError::MissingField("input".to_string()))?;

        let output_name = session
            .outputs
            .first()
            .map(|o| o.name.clone())
            .ok_or_else(|| OnnxError::MissingField("output".to_string()))?;

        Ok(Self {
            session,
            input_name,
            output_name,
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
        use ort::value::Value;

        // Get shape for tensor creation
        let shape = input.shape().to_vec();
        let data: Vec<f32> = input.iter().copied().collect();

        // Create input value using the (shape, data) tuple format
        let input_value = Value::from_array((IxDyn(&shape), data))
            .map_err(|e| OnnxError::InferenceError(e.to_string()))?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => input_value])
            .map_err(|e| OnnxError::InferenceError(e.to_string()))?;

        // Extract output
        let output = outputs[&self.output_name]
            .try_extract_tensor::<f32>()
            .map_err(|e| OnnxError::InferenceError(e.to_string()))?;

        let output_array = output.view().to_owned();

        // Convert to 1D array
        let shape = output_array.shape();
        if shape.len() == 2 && shape[1] == 1 {
            // Output is [batch, 1], flatten to [batch]
            Ok(output_array
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| OnnxError::InferenceError(e.to_string()))?
                .column(0)
                .to_owned())
        } else if shape.len() == 1 {
            Ok(output_array
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|e| OnnxError::InferenceError(e.to_string()))?
                .to_owned())
        } else {
            Err(OnnxError::ShapeMismatch {
                expected: vec![input.nrows()],
                got: shape.to_vec(),
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
        let input_info = self
            .session
            .inputs
            .first()
            .ok_or_else(|| OnnxError::MissingField("input info".to_string()))?;

        // Get shape from input info
        let shape = &input_info.input_type.tensor_dimensions;
        if shape.len() >= 2 {
            shape
                .get(1)
                .copied()
                .ok_or_else(|| OnnxError::MissingField("input feature dimension".to_string()))
        } else {
            Err(OnnxError::InvalidParameters(
                "Expected 2D input tensor".to_string(),
            ))
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
}
