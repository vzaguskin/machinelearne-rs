//! Error types for preprocessing operations.

use std::fmt;

/// Error type for preprocessing operations.
#[derive(Debug)]
pub enum PreprocessingError {
    /// Shape mismatch between expected and actual tensor dimensions.
    InvalidShape { expected: String, got: String },
    /// Numerical computation error (overflow, underflow, etc.).
    NumericalError(String),
    /// Data contains missing values (NaN) when not expected.
    MissingValues(String),
    /// Invalid hyperparameter value.
    InvalidParameter(String),
    /// Serialization or deserialization error.
    SerializationError(String),
    /// I/O error during file operations.
    IoError(String),
    /// Empty data provided where non-empty was required.
    EmptyData(String),
    /// Feature dimension mismatch.
    FeatureMismatch {
        expected_features: usize,
        got_features: usize,
    },
}

impl fmt::Display for PreprocessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PreprocessingError::InvalidShape { expected, got } => {
                write!(f, "Invalid shape: expected {}, got {}", expected, got)
            }
            PreprocessingError::NumericalError(msg) => {
                write!(f, "Numerical error: {}", msg)
            }
            PreprocessingError::MissingValues(msg) => {
                write!(f, "Missing values: {}", msg)
            }
            PreprocessingError::InvalidParameter(msg) => {
                write!(f, "Invalid parameter: {}", msg)
            }
            PreprocessingError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            PreprocessingError::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            }
            PreprocessingError::EmptyData(msg) => {
                write!(f, "Empty data: {}", msg)
            }
            PreprocessingError::FeatureMismatch {
                expected_features,
                got_features,
            } => {
                write!(
                    f,
                    "Feature mismatch: expected {} features, got {}",
                    expected_features, got_features
                )
            }
        }
    }
}

impl std::error::Error for PreprocessingError {}

impl From<std::io::Error> for PreprocessingError {
    fn from(err: std::io::Error) -> Self {
        PreprocessingError::IoError(err.to_string())
    }
}

impl From<bincode::Error> for PreprocessingError {
    fn from(err: bincode::Error) -> Self {
        PreprocessingError::SerializationError(err.to_string())
    }
}
