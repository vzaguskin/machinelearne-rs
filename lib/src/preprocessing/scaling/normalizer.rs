//! Normalizer - scales individual samples to unit norm.
//!
//! Normalizes samples individually to unit norm. Each sample (row) is
//! rescaled independently of other samples.
//!
//! Supports three norm types:
//! - L1: Sum of absolute values = 1
//! - L2 (default): Sum of squares = 1 (Euclidean norm)
//! - Max: Maximum absolute value = 1
//!
//! # Example
//! ```ignore
//! use machinelearne_rs::preprocessing::{Transformer, Normalizer, NormType};
//! use machinelearne_rs::backend::CpuBackend;
//!
//! let normalizer = Normalizer::<CpuBackend>::new(NormType::L2);
//! let normalized = normalizer.fit_transform(&data)?;
//! ```

use crate::backend::{Backend, Tensor2D};
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Type of normalization to apply.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum NormType {
    /// L1 norm: sum of absolute values = 1
    L1,
    /// L2 norm (Euclidean): sum of squares = 1
    #[default]
    L2,
    /// Max norm: maximum absolute value = 1
    Max,
}

/// Serializable parameters for a fitted Normalizer (trivial - just norm type).
#[derive(Clone, Serialize, Deserialize)]
pub struct NormalizerParams {
    /// The norm type used.
    pub norm: NormType,
    /// Number of features seen during fit.
    pub n_features: usize,
}

/// Normalizer transformer.
///
/// Scales individual samples to unit norm.
#[derive(Clone)]
pub struct Normalizer<B: Backend> {
    norm: NormType,
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for Normalizer<B> {
    fn default() -> Self {
        Self::new(NormType::default())
    }
}

impl<B: Backend> Normalizer<B> {
    /// Create a new Normalizer with the specified norm type.
    pub fn new(norm: NormType) -> Self {
        Self {
            norm,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Transformer<B> for Normalizer<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = NormalizerParams;
    type Fitted = FittedNormalizer<B>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError> {
        let (rows, cols) = data.shape();

        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit Normalizer on empty data".to_string(),
            ));
        }

        Ok(FittedNormalizer {
            norm: self.norm,
            n_features: cols,
            _backend: PhantomData,
        })
    }

    fn fit_transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        // Normalizer is stateless - just transform directly
        let (rows, cols) = data.shape();

        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot normalize empty data".to_string(),
            ));
        }

        let flat_data = B::to_vec_1d(&B::ravel_2d(&data.data));
        let mut result = vec![0.0; rows * cols];

        for row in 0..rows {
            // Extract row data
            let row_start = row * cols;
            let row_data: Vec<f64> = flat_data[row_start..row_start + cols].to_vec();

            // Compute norm
            let norm = match self.norm {
                NormType::L1 => row_data.iter().map(|x| x.abs()).sum::<f64>(),
                NormType::L2 => (row_data.iter().map(|x| x * x).sum::<f64>()).sqrt(),
                NormType::Max => row_data.iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
            };

            // Normalize
            let scale = if norm == 0.0 { 1.0 } else { 1.0 / norm };
            for col in 0..cols {
                result[row_start + col] = flat_data[row_start + col] * scale;
            }
        }

        Ok(Tensor2D {
            data: B::from_vec_2d(result.iter().map(|&x| x as f32).collect(), rows, cols),
            backend: PhantomData::<B>,
        })
    }
}

/// Fitted Normalizer ready for inference.
#[derive(Clone)]
pub struct FittedNormalizer<B: Backend> {
    norm: NormType,
    n_features: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedNormalizer<B> {
    /// Get the norm type used.
    pub fn norm(&self) -> NormType {
        self.norm
    }
}

impl<B: Backend> FittedTransformer<B> for FittedNormalizer<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = NormalizerParams;

    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let (rows, cols) = data.shape();

        if cols != self.n_features {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features,
                got_features: cols,
            });
        }

        if rows == 0 {
            return Ok(data.clone());
        }

        let flat_data = B::to_vec_1d(&B::ravel_2d(&data.data));
        let mut result = vec![0.0; rows * cols];

        for row in 0..rows {
            let row_start = row * cols;
            let row_data: Vec<f64> = flat_data[row_start..row_start + cols].to_vec();

            let norm = match self.norm {
                NormType::L1 => row_data.iter().map(|x| x.abs()).sum::<f64>(),
                NormType::L2 => (row_data.iter().map(|x| x * x).sum::<f64>()).sqrt(),
                NormType::Max => row_data.iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
            };

            let scale = if norm == 0.0 { 1.0 } else { 1.0 / norm };
            for col in 0..cols {
                result[row_start + col] = flat_data[row_start + col] * scale;
            }
        }

        Ok(Tensor2D {
            data: B::from_vec_2d(result.iter().map(|&x| x as f32).collect(), rows, cols),
            backend: PhantomData::<B>,
        })
    }

    fn inverse_transform(&self, _data: &Self::Output) -> Result<Self::Input, PreprocessingError> {
        Err(PreprocessingError::InvalidParameter(
            "Normalizer does not support inverse_transform (norm information is lost)".to_string(),
        ))
    }

    fn extract_params(&self) -> Self::Params {
        NormalizerParams {
            norm: self.norm,
            n_features: self.n_features,
        }
    }

    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError> {
        Ok(Self {
            norm: params.norm,
            n_features: params.n_features,
            _backend: PhantomData,
        })
    }

    fn n_features_in(&self) -> usize {
        self.n_features
    }

    fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let params = self.extract_params();
        let bytes = bincode::serialize(&params).map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }

    fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, PreprocessingError>
    where
        Self: Sized,
    {
        let bytes = std::fs::read(path)?;
        let params: NormalizerParams = bincode::deserialize(&bytes)
            .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
        Self::from_params(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    fn create_test_data() -> Tensor2D<CpuBackend> {
        // [[3, 4], [1, 0], [0, 0]]
        // Row 0: L2 norm = 5, so normalized = [0.6, 0.8]
        // Row 1: L2 norm = 1, so normalized = [1, 0]
        // Row 2: L2 norm = 0, so stays [0, 0]
        Tensor2D::new(vec![3.0f32, 4.0, 1.0, 0.0, 0.0, 0.0], 3, 2)
    }

    #[test]
    fn test_normalizer_l2() {
        let data = create_test_data();
        let normalizer = Normalizer::<CpuBackend>::new(NormType::L2);
        let normalized = normalizer.fit_transform(&data).unwrap();
        let values = normalized.ravel().to_vec();

        // Row 0: [3, 4] / 5 = [0.6, 0.8]
        assert!((values[0] - 0.6).abs() < 1e-6);
        assert!((values[1] - 0.8).abs() < 1e-6);

        // Row 1: [1, 0] / 1 = [1, 0]
        assert!((values[2] - 1.0).abs() < 1e-6);
        assert!((values[3] - 0.0).abs() < 1e-6);

        // Row 2: [0, 0] / 1 (zero case) = [0, 0]
        assert!((values[4] - 0.0).abs() < 1e-6);
        assert!((values[5] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalizer_l1() {
        let data = create_test_data();
        let normalizer = Normalizer::<CpuBackend>::new(NormType::L1);
        let normalized = normalizer.fit_transform(&data).unwrap();
        let values = normalized.ravel().to_vec();

        // Row 0: [3, 4] / 7 = [0.428..., 0.571...]
        assert!((values[0] - 3.0 / 7.0).abs() < 1e-6);
        assert!((values[1] - 4.0 / 7.0).abs() < 1e-6);

        // Row 1: [1, 0] / 1 = [1, 0]
        assert!((values[2] - 1.0).abs() < 1e-6);
        assert!((values[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalizer_max() {
        let data = create_test_data();
        let normalizer = Normalizer::<CpuBackend>::new(NormType::Max);
        let normalized = normalizer.fit_transform(&data).unwrap();
        let values = normalized.ravel().to_vec();

        // Row 0: [3, 4] / 4 = [0.75, 1]
        assert!((values[0] - 0.75).abs() < 1e-6);
        assert!((values[1] - 1.0).abs() < 1e-6);

        // Row 1: [1, 0] / 1 = [1, 0]
        assert!((values[2] - 1.0).abs() < 1e-6);
        assert!((values[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalizer_transform() {
        let data = create_test_data();
        let normalizer = Normalizer::<CpuBackend>::new(NormType::L2);
        let fitted = normalizer.fit(&data).unwrap();

        let normalized = fitted.transform(&data).unwrap();
        let values = normalized.ravel().to_vec();

        // Same as fit_transform
        assert!((values[0] - 0.6).abs() < 1e-6);
        assert!((values[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalizer_inverse_not_supported() {
        let data = create_test_data();
        let normalizer = Normalizer::<CpuBackend>::new(NormType::L2);
        let fitted = normalizer.fit(&data).unwrap();

        let normalized = fitted.transform(&data).unwrap();
        let result = fitted.inverse_transform(&normalized);

        assert!(matches!(
            result,
            Err(PreprocessingError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_normalizer_serialization() {
        let data = create_test_data();
        let normalizer = Normalizer::<CpuBackend>::new(NormType::L2);
        let fitted = normalizer.fit(&data).unwrap();

        let params = fitted.extract_params();
        let restored = FittedNormalizer::<CpuBackend>::from_params(params).unwrap();

        let normalized1 = fitted.transform(&data).unwrap();
        let normalized2 = restored.transform(&data).unwrap();

        let n1 = normalized1.ravel().to_vec();
        let n2 = normalized2.ravel().to_vec();

        for (a, b) in n1.iter().zip(n2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalizer_feature_mismatch() {
        let data = create_test_data(); // 2 features
        let normalizer = Normalizer::<CpuBackend>::new(NormType::L2);
        let fitted = normalizer.fit(&data).unwrap();

        let wrong_data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 1, 3); // 3 features
        let result = fitted.transform(&wrong_data);

        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch {
                expected_features: 2,
                got_features: 3
            })
        ));
    }
}
