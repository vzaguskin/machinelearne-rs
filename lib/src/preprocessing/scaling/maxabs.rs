//! Max-Abs Scaler.
//!
//! Scales each feature by its maximum absolute value.
//! This estimator scales and translates each feature individually such that
//! the maximal absolute value of each feature in the training set will be 1.0.
//!
//! It does not shift/center the data, and thus does not destroy any sparsity.
//!
//! # Example
//! ```ignore
//! use machinelearne_rs::preprocessing::{Transformer, MaxAbsScaler};
//! use machinelearne_rs::backend::CpuBackend;
//!
//! let scaler = MaxAbsScaler::<CpuBackend>::new();
//! let fitted = scaler.fit(&data)?;
//! let scaled = fitted.transform(&data)?;
//! ```

use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Serializable parameters for a fitted MaxAbsScaler.
#[derive(Clone, Serialize, Deserialize)]
pub struct MaxAbsScalerParams {
    /// Maximum absolute value for each feature.
    pub max_abs_: Vec<f64>,
    /// Scale factor for each feature (1.0 / max_abs).
    pub scale_: Vec<f64>,
    /// Number of features seen during fit.
    pub n_features: usize,
}

/// MaxAbsScaler transformer (unfitted).
///
/// Scales each feature by its maximum absolute value.
#[derive(Clone, Default)]
pub struct MaxAbsScaler<B: Backend> {
    _backend: PhantomData<B>,
}

impl<B: Backend> MaxAbsScaler<B> {
    /// Create a new MaxAbsScaler.
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Transformer<B> for MaxAbsScaler<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = MaxAbsScalerParams;
    type Fitted = FittedMaxAbsScaler<B>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError> {
        let (rows, cols) = data.shape();

        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit MaxAbsScaler on empty data".to_string(),
            ));
        }

        // Get absolute values of data
        let abs_data = B::abs_2d(&data.data);

        // Find max absolute value per column
        let max_abs_raw = B::col_max_2d(&abs_data);
        let max_abs_vals = B::to_vec_1d(&max_abs_raw);

        // Compute scale: 1.0 / max_abs (handle zero max)
        let scale_vals: Vec<f64> = max_abs_vals
            .iter()
            .map(|&m| if m == 0.0 { 1.0 } else { 1.0 / m })
            .collect();

        let max_abs_ = Tensor1D {
            data: B::from_vec_1d(max_abs_vals.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };

        let scale_ = Tensor1D {
            data: B::from_vec_1d(scale_vals.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };

        Ok(FittedMaxAbsScaler {
            max_abs_,
            scale_,
            n_features: cols,
            _backend: PhantomData,
        })
    }

    fn fit_transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let fitted = self.fit(data)?;
        fitted.transform(data)
    }
}

/// Fitted MaxAbsScaler ready for inference.
#[derive(Clone)]
pub struct FittedMaxAbsScaler<B: Backend> {
    max_abs_: Tensor1D<B>,
    scale_: Tensor1D<B>,
    n_features: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedMaxAbsScaler<B> {
    /// Get the maximum absolute values for each feature.
    pub fn max_abs(&self) -> &Tensor1D<B> {
        &self.max_abs_
    }

    /// Get the scale factor for each feature.
    pub fn scale(&self) -> &Tensor1D<B> {
        &self.scale_
    }
}

impl<B: Backend> FittedTransformer<B> for FittedMaxAbsScaler<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = MaxAbsScalerParams;

    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let (_, cols) = data.shape();

        if cols != self.n_features {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features,
                got_features: cols,
            });
        }

        // X_scaled = X * scale_
        let result = B::broadcast_mul_1d_to_2d_rows(&data.data, &self.scale_.data);

        Ok(Tensor2D {
            data: result,
            backend: PhantomData::<B>,
        })
    }

    fn inverse_transform(&self, data: &Self::Output) -> Result<Self::Input, PreprocessingError> {
        let (_, cols) = data.shape();

        if cols != self.n_features {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features,
                got_features: cols,
            });
        }

        // X = X_scaled * max_abs_
        let result = B::broadcast_mul_1d_to_2d_rows(&data.data, &self.max_abs_.data);

        Ok(Tensor2D {
            data: result,
            backend: PhantomData::<B>,
        })
    }

    fn extract_params(&self) -> Self::Params {
        MaxAbsScalerParams {
            max_abs_: self.max_abs_.to_vec(),
            scale_: self.scale_.to_vec(),
            n_features: self.n_features,
        }
    }

    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError> {
        let max_abs_ = Tensor1D {
            data: B::from_vec_1d(params.max_abs_.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };
        let scale_ = Tensor1D {
            data: B::from_vec_1d(params.scale_.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };

        Ok(Self {
            max_abs_,
            scale_,
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
        let params: MaxAbsScalerParams = bincode::deserialize(&bytes)
            .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
        Self::from_params(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    fn create_test_data() -> Tensor2D<CpuBackend> {
        // [[-1, 2], [0, 4], [1, -6]]  -- max abs: [1, 6]
        Tensor2D::new(vec![-1.0f32, 2.0, 0.0, 4.0, 1.0, -6.0], 3, 2)
    }

    #[test]
    fn test_maxabs_scaler_fit() {
        let data = create_test_data();
        let scaler = MaxAbsScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let max_abs = fitted.max_abs().to_vec();
        assert!((max_abs[0] - 1.0).abs() < 1e-6);
        assert!((max_abs[1] - 6.0).abs() < 1e-6);

        let scale = fitted.scale().to_vec();
        assert!((scale[0] - 1.0).abs() < 1e-6);
        assert!((scale[1] - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_maxabs_scaler_transform() {
        let data = create_test_data();
        let scaler = MaxAbsScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();
        let values = transformed.ravel().to_vec();

        // First column: [-1, 0, 1] * 1.0 = [-1, 0, 1]
        assert!((values[0] - (-1.0)).abs() < 1e-6);
        assert!((values[2] - 0.0).abs() < 1e-6);
        assert!((values[4] - 1.0).abs() < 1e-6);

        // Second column: [2, 4, -6] * (1/6) = [0.333, 0.667, -1]
        assert!((values[1] - (2.0 / 6.0)).abs() < 1e-6);
        assert!((values[3] - (4.0 / 6.0)).abs() < 1e-6);
        assert!((values[5] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_maxabs_scaler_inverse_transform() {
        let data = create_test_data();
        let scaler = MaxAbsScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        let original = data.ravel().to_vec();
        let recovered_vec = recovered.ravel().to_vec();

        for (o, r) in original.iter().zip(recovered_vec.iter()) {
            assert!((o - r).abs() < 1e-6, "Expected {}, got {}", o, r);
        }
    }

    #[test]
    fn test_maxabs_scaler_serialization() {
        let data = create_test_data();
        let scaler = MaxAbsScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let params = fitted.extract_params();
        let restored = FittedMaxAbsScaler::<CpuBackend>::from_params(params).unwrap();

        let transformed1 = fitted.transform(&data).unwrap();
        let transformed2 = restored.transform(&data).unwrap();

        let t1 = transformed1.ravel().to_vec();
        let t2 = transformed2.ravel().to_vec();

        for (i, (a, b)) in t1.iter().zip(t2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Mismatch at index {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_maxabs_scaler_feature_mismatch() {
        let data = create_test_data(); // 2 features
        let scaler = MaxAbsScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

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
