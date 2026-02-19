//! ONNX export trait and implementations.
//!
//! Provides the `OnnxExportable` trait for exporting models to ONNX format.

use super::error::OnnxError;
use std::path::Path;

/// Trait for exporting models to ONNX format.
///
/// Types implementing this trait can be serialized to ONNX format
/// for portable deployment and inference with ONNX Runtime.
pub trait OnnxExportable {
    /// Export the model to ONNX format as bytes.
    ///
    /// # Arguments
    /// * `opset_version` - ONNX operator set version (default: 17)
    ///
    /// # Returns
    /// Serialized ONNX model bytes.
    fn to_onnx(&self, opset_version: i64) -> Result<Vec<u8>, OnnxError>;

    /// Export the model to ONNX format using the default opset version.
    fn to_onnx_default(&self) -> Result<Vec<u8>, OnnxError> {
        self.to_onnx(super::DEFAULT_OPSET_VERSION)
    }

    /// Save the model to an ONNX file.
    ///
    /// # Arguments
    /// * `path` - Path to save the ONNX file
    fn save_onnx<P: AsRef<Path>>(&self, path: P) -> Result<(), OnnxError> {
        let bytes = self.to_onnx_default()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Save the model to an ONNX file with a specific opset version.
    fn save_onnx_with_version<P: AsRef<Path>>(
        &self,
        path: P,
        opset_version: i64,
    ) -> Result<(), OnnxError> {
        let bytes = self.to_onnx(opset_version)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::OnnxGraphBuilder;

    struct SimpleModel {
        weights: Vec<f32>,
        bias: f32,
    }

    impl OnnxExportable for SimpleModel {
        fn to_onnx(&self, _opset_version: i64) -> Result<Vec<u8>, OnnxError> {
            let mut builder = OnnxGraphBuilder::new("simple_model");

            // Add input: [batch_size, n_features]
            let n_features = self.weights.len();
            builder.add_input_float("input", n_features);

            // Add weights and bias as initializers
            builder.add_float_initializer("weights", &[n_features as i64], &self.weights);
            builder.add_float_initializer("bias", &[1], &[self.bias]);

            // Add output
            builder.add_output_float("output", 1);

            // Build the model
            builder.build()
        }
    }

    #[test]
    fn test_onnx_exportable() {
        let model = SimpleModel {
            weights: vec![1.0, 2.0, 3.0],
            bias: 0.5,
        };

        let bytes = model.to_onnx_default().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_save_onnx() {
        let model = SimpleModel {
            weights: vec![1.0, 2.0],
            bias: 1.0,
        };

        let temp_file = std::env::temp_dir().join("test_simple_model.onnx");
        model.save_onnx(&temp_file).unwrap();

        let bytes = std::fs::read(&temp_file).unwrap();
        assert!(!bytes.is_empty());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_save_onnx_with_version() {
        let model = SimpleModel {
            weights: vec![1.0, 2.0],
            bias: 1.0,
        };

        let temp_file = std::env::temp_dir().join("test_simple_model_v15.onnx");
        model.save_onnx_with_version(&temp_file, 15).unwrap();

        let bytes = std::fs::read(&temp_file).unwrap();
        assert!(!bytes.is_empty());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_to_onnx_custom_version() {
        let model = SimpleModel {
            weights: vec![1.0, 2.0, 3.0],
            bias: 0.5,
        };

        // Test with different opset versions - both should produce valid output
        let bytes_v13 = model.to_onnx(13).unwrap();
        let bytes_v17 = model.to_onnx(17).unwrap();

        assert!(!bytes_v13.is_empty());
        assert!(!bytes_v17.is_empty());
    }
}
