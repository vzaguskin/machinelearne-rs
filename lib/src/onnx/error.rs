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
        }
    }
}

impl std::error::Error for OnnxError {}

impl From<io::Error> for OnnxError {
    fn from(e: io::Error) -> Self {
        OnnxError::IoError(e)
    }
}
