//! Error types for ONNX operations.

use std::io;

/// Error type for ONNX export and inference operations.
#[derive(Debug)]
pub enum OnnxError {
    /// I/O error during file operations.
    IoError(io::Error),
    /// Error during protobuf encoding/decoding.
    ProtobufError(String),
    /// Error during ONNX model creation.
    ModelCreationError(String),
    /// Error during ONNX inference.
    InferenceError(String),
    /// Unsupported operation or type.
    UnsupportedOperation(String),
    /// Invalid model parameters.
    InvalidParameters(String),
    /// Shape mismatch error.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Missing required field.
    MissingField(String),
    /// Unsupported transformer in pipeline.
    UnsupportedTransformer {
        transformer_name: String,
        reason: String,
    },
    /// Error during graph construction.
    GraphConstruction(String),
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxError::IoError(e) => write!(f, "IO error: {}", e),
            OnnxError::ProtobufError(msg) => write!(f, "Protobuf error: {}", msg),
            OnnxError::ModelCreationError(msg) => write!(f, "Model creation error: {}", msg),
            OnnxError::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            OnnxError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            OnnxError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
            OnnxError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            OnnxError::MissingField(field) => write!(f, "Missing required field: {}", field),
            OnnxError::UnsupportedTransformer {
                transformer_name,
                reason,
            } => {
                write!(
                    f,
                    "Unsupported transformer '{}': {}",
                    transformer_name, reason
                )
            }
            OnnxError::GraphConstruction(msg) => {
                write!(f, "Graph construction error: {}", msg)
            }
        }
    }
}

impl std::error::Error for OnnxError {}

impl From<io::Error> for OnnxError {
    fn from(e: io::Error) -> Self {
        OnnxError::IoError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_error_display() {
        let err = OnnxError::IoError(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert!(err.to_string().contains("IO error"));

        let err = OnnxError::ProtobufError("test error".to_string());
        assert!(err.to_string().contains("Protobuf error"));

        let err = OnnxError::ModelCreationError("failed".to_string());
        assert!(err.to_string().contains("Model creation error"));

        let err = OnnxError::InferenceError("runtime error".to_string());
        assert!(err.to_string().contains("Inference error"));

        let err = OnnxError::UnsupportedOperation("op".to_string());
        assert!(err.to_string().contains("Unsupported operation"));

        let err = OnnxError::InvalidParameters("bad params".to_string());
        assert!(err.to_string().contains("Invalid parameters"));

        let err = OnnxError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![1, 2],
        };
        assert!(err.to_string().contains("Shape mismatch"));

        let err = OnnxError::MissingField("field".to_string());
        assert!(err.to_string().contains("Missing required field"));

        let err = OnnxError::UnsupportedTransformer {
            transformer_name: "CustomTransformer".to_string(),
            reason: "Not implemented".to_string(),
        };
        assert!(err.to_string().contains("Unsupported transformer"));
        assert!(err.to_string().contains("CustomTransformer"));

        let err = OnnxError::GraphConstruction("failed to add node".to_string());
        assert!(err.to_string().contains("Graph construction error"));
        assert!(err.to_string().contains("failed to add node"));
    }

    #[test]
    fn test_onnx_error_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let onnx_err: OnnxError = io_err.into();
        assert!(matches!(onnx_err, OnnxError::IoError(_)));
    }
}
