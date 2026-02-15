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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_invalid_shape() {
        let err = PreprocessingError::InvalidShape {
            expected: "(2, 3)".to_string(),
            got: "(3, 2)".to_string(),
        };
        assert!(err.to_string().contains("Invalid shape"));
    }

    #[test]
    fn test_error_display_numerical_error() {
        let err = PreprocessingError::NumericalError("overflow".to_string());
        assert!(err.to_string().contains("Numerical error"));
    }

    #[test]
    fn test_error_display_missing_values() {
        let err = PreprocessingError::MissingValues("column 0".to_string());
        assert!(err.to_string().contains("Missing values"));
    }

    #[test]
    fn test_error_display_invalid_parameter() {
        let err = PreprocessingError::InvalidParameter("bad param".to_string());
        assert!(err.to_string().contains("Invalid parameter"));
    }

    #[test]
    fn test_error_display_serialization_error() {
        let err = PreprocessingError::SerializationError("failed".to_string());
        assert!(err.to_string().contains("Serialization error"));
    }

    #[test]
    fn test_error_display_io_error() {
        let err = PreprocessingError::IoError("file not found".to_string());
        assert!(err.to_string().contains("I/O error"));
    }

    #[test]
    fn test_error_display_empty_data() {
        let err = PreprocessingError::EmptyData("no rows".to_string());
        assert!(err.to_string().contains("Empty data"));
    }

    #[test]
    fn test_error_display_feature_mismatch() {
        let err = PreprocessingError::FeatureMismatch {
            expected_features: 5,
            got_features: 3,
        };
        assert!(err.to_string().contains("Feature mismatch"));
    }

    #[test]
    fn test_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let err: PreprocessingError = io_err.into();
        assert!(matches!(err, PreprocessingError::IoError(_)));
    }

    #[test]
    fn test_error_is_std_error() {
        let err = PreprocessingError::InvalidParameter("test".to_string());
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_error_from_bincode_error() {
        // Create a bincode error by trying to deserialize invalid data
        let bad_bytes: &[u8] = &[0xff, 0xff, 0xff, 0xff];
        let bincode_result: Result<String, bincode::Error> = bincode::deserialize(bad_bytes);
        if let Err(e) = bincode_result {
            let err: PreprocessingError = e.into();
            assert!(matches!(err, PreprocessingError::SerializationError(_)));
        }
    }
}
