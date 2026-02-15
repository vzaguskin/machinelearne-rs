//! Standard Scaler (Z-score normalization).
//!
//! Transforms features by removing the mean and scaling to unit variance.
//!
//! The standard score of a sample `x` is calculated as:
//! ```text
//! z = (x - u) / s
//! ```
//! where `u` is the mean of the training samples, and `s` is the standard deviation.
//!
//! # Example
//! ```ignore
//! use machinelearne_rs::preprocessing::{Transformer, StandardScaler};
//! use machinelearne_rs::backend::CpuBackend;
//!
//! let scaler = StandardScaler::<CpuBackend>::new()
//!     .with_mean(true)
//!     .with_std(true);
//!
//! let fitted = scaler.fit(&data)?;
//! let scaled = fitted.transform(&data)?;
//!
//! // Later, for inference:
//! let loaded = StandardScaler::load_from_file("scaler.bin")?;
//! let new_scaled = loaded.transform(&new_data)?;
//! ```

use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Configuration for StandardScaler.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StandardScalerConfig {
    /// If True, center the data before scaling.
    pub with_mean: bool,
    /// If True, scale the data to unit variance.
    pub with_std: bool,
}

impl Default for StandardScalerConfig {
    fn default() -> Self {
        Self {
            with_mean: true,
            with_std: true,
        }
    }
}

/// Serializable parameters for a fitted StandardScaler.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StandardScalerParams {
    /// Configuration options.
    pub config: StandardScalerConfig,
    /// Mean of each feature (None if with_mean=False).
    pub mean: Vec<f64>,
    /// Standard deviation of each feature (None if with_std=False).
    pub std: Vec<f64>,
    /// Number of features seen during fit.
    pub n_features: usize,
}

// Note: SerializableParams is automatically implemented via blanket impl in serialization.rs
// for types that implement Serialize + Deserialize

/// StandardScaler transformer (unfitted).
///
/// Transforms features by removing the mean and scaling to unit variance.
#[derive(Clone)]
pub struct StandardScaler<B: Backend> {
    config: StandardScalerConfig,
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for StandardScaler<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> StandardScaler<B> {
    /// Create a new StandardScaler with default configuration.
    pub fn new() -> Self {
        Self {
            config: StandardScalerConfig::default(),
            _backend: PhantomData,
        }
    }

    /// Set whether to center data by mean.
    pub fn with_mean(mut self, with_mean: bool) -> Self {
        self.config.with_mean = with_mean;
        self
    }

    /// Set whether to scale data to unit variance.
    pub fn with_std(mut self, with_std: bool) -> Self {
        self.config.with_std = with_std;
        self
    }
}

impl<B: Backend> Transformer<B> for StandardScaler<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = StandardScalerParams;
    type Fitted = FittedStandardScaler<B>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError> {
        let (rows, cols) = data.shape();

        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit StandardScaler on empty data".to_string(),
            ));
        }

        let mean = if self.config.with_mean {
            Tensor1D {
                data: B::col_mean_2d(&data.data),
                backend: PhantomData::<B>,
            }
        } else {
            Tensor1D {
                data: B::zeros_1d(cols),
                backend: PhantomData::<B>,
            }
        };

        let std = if self.config.with_std {
            Tensor1D {
                data: B::col_std_2d(&data.data, 0), // population std (ddof=0)
                backend: PhantomData::<B>,
            }
        } else {
            let ones: Vec<f64> = vec![1.0; cols];
            Tensor1D {
                data: B::from_vec_1d(ones.iter().map(|&x| x as f32).collect()),
                backend: PhantomData::<B>,
            }
        };

        // Handle zero std (constant features)
        let std_vec = std.to_vec();
        let std_adjusted: Vec<f64> = std_vec
            .iter()
            .map(|&s| if s == 0.0 { 1.0 } else { s })
            .collect();
        let std_final = Tensor1D {
            data: B::from_vec_1d(std_adjusted.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };

        Ok(FittedStandardScaler {
            config: self.config.clone(),
            mean,
            std: std_final,
            n_features: cols,
            _backend: PhantomData,
        })
    }

    fn fit_transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let fitted = self.fit(data)?;
        fitted.transform(data)
    }
}

/// Fitted StandardScaler ready for inference.
#[derive(Clone)]
pub struct FittedStandardScaler<B: Backend> {
    config: StandardScalerConfig,
    mean: Tensor1D<B>,
    std: Tensor1D<B>,
    n_features: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedStandardScaler<B> {
    /// Get the mean values for each feature.
    pub fn mean(&self) -> &Tensor1D<B> {
        &self.mean
    }

    /// Get the standard deviation values for each feature.
    pub fn std(&self) -> &Tensor1D<B> {
        &self.std
    }
}

impl<B: Backend> FittedTransformer<B> for FittedStandardScaler<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = StandardScalerParams;

    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let (_, cols) = data.shape();

        if cols != self.n_features {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features,
                got_features: cols,
            });
        }

        let mut result_data = data.data.clone();

        if self.config.with_mean {
            result_data = B::broadcast_sub_1d_to_2d_rows(&result_data, &self.mean.data);
        }

        if self.config.with_std {
            result_data = B::broadcast_div_1d_to_2d_rows(&result_data, &self.std.data);
        }

        Ok(Tensor2D {
            data: result_data,
            backend: PhantomData,
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

        let mut result_data = data.data.clone();

        if self.config.with_std {
            result_data = B::broadcast_mul_1d_to_2d_rows(&result_data, &self.std.data);
        }

        if self.config.with_mean {
            result_data = B::broadcast_add_1d_to_2d_rows(&result_data, &self.mean.data);
        }

        Ok(Tensor2D {
            data: result_data,
            backend: PhantomData,
        })
    }

    fn extract_params(&self) -> Self::Params {
        StandardScalerParams {
            config: self.config.clone(),
            mean: self.mean.to_vec(),
            std: self.std.to_vec(),
            n_features: self.n_features,
        }
    }

    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError> {
        let mean = Tensor1D {
            data: B::from_vec_1d(params.mean.iter().map(|&x| x as f32).collect()),
            backend: PhantomData,
        };
        let std = Tensor1D {
            data: B::from_vec_1d(params.std.iter().map(|&x| x as f32).collect()),
            backend: PhantomData,
        };

        Ok(Self {
            config: params.config,
            mean,
            std,
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
        let params: StandardScalerParams = bincode::deserialize(&bytes)
            .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
        Self::from_params(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    fn create_test_data() -> Tensor2D<CpuBackend> {
        // [[0, 1], [0, 1], [1, 3]]
        Tensor2D::new(vec![0.0f32, 1.0, 0.0, 1.0, 1.0, 3.0], 3, 2)
    }

    #[test]
    fn test_standard_scaler_fit() {
        let data = create_test_data();
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        // Mean: [1/3, 5/3]
        let mean = fitted.mean().to_vec();
        assert!((mean[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((mean[1] - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_standard_scaler_transform() {
        let data = create_test_data();
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();

        // After standardization, each column should have mean≈0 and std≈1
        let mean_vals: Vec<f64> =
            CpuBackend::to_vec_1d(&CpuBackend::col_mean_2d(&transformed.data));
        let std_vals: Vec<f64> =
            CpuBackend::to_vec_1d(&CpuBackend::col_std_2d(&transformed.data, 0));

        assert!(mean_vals[0].abs() < 1e-10, "mean[0] = {}", mean_vals[0]);
        assert!(mean_vals[1].abs() < 1e-10, "mean[1] = {}", mean_vals[1]);
        // Using a slightly larger tolerance for numerical stability
        assert!((std_vals[0] - 1.0).abs() < 1e-8, "std[0] = {}", std_vals[0]);
        assert!((std_vals[1] - 1.0).abs() < 1e-8, "std[1] = {}", std_vals[1]);
    }

    #[test]
    fn test_standard_scaler_inverse_transform() {
        let data = create_test_data();
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        let original = data.ravel().to_vec();
        let recovered_vec = recovered.ravel().to_vec();

        for (o, r) in original.iter().zip(recovered_vec.iter()) {
            assert!((o - r).abs() < 1e-10);
        }
    }

    #[test]
    fn test_standard_scaler_without_mean() {
        let data = create_test_data();
        let scaler = StandardScaler::<CpuBackend>::new().with_mean(false);
        let fitted = scaler.fit(&data).unwrap();

        // Mean should be zeros
        let mean = fitted.mean().to_vec();
        assert!(mean.iter().all(|&m| m == 0.0));
    }

    #[test]
    fn test_standard_scaler_without_std() {
        let data = create_test_data();
        let scaler = StandardScaler::<CpuBackend>::new().with_std(false);
        let fitted = scaler.fit(&data).unwrap();

        // Std should be ones
        let std = fitted.std().to_vec();
        assert!(std.iter().all(|&s| s == 1.0));
    }

    #[test]
    fn test_standard_scaler_serialization() {
        let data = create_test_data();
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let params = fitted.extract_params();
        let restored = FittedStandardScaler::<CpuBackend>::from_params(params).unwrap();

        let transformed1 = fitted.transform(&data).unwrap();
        let transformed2 = restored.transform(&data).unwrap();

        let t1 = transformed1.ravel().to_vec();
        let t2 = transformed2.ravel().to_vec();

        // Use 1e-6 tolerance due to f32->f64 conversion precision
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
    fn test_standard_scaler_feature_mismatch() {
        let data = create_test_data(); // 2 features
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        // Try to transform with wrong number of features
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

    #[test]
    fn test_standard_scaler_fit_transform() {
        let data = create_test_data();
        let scaler = StandardScaler::<CpuBackend>::new();

        let transformed = scaler.fit_transform(&data).unwrap();

        // After standardization, each column should have mean≈0 and std≈1
        let mean_vals: Vec<f64> =
            CpuBackend::to_vec_1d(&CpuBackend::col_mean_2d(&transformed.data));
        let std_vals: Vec<f64> =
            CpuBackend::to_vec_1d(&CpuBackend::col_std_2d(&transformed.data, 0));

        assert!(mean_vals[0].abs() < 1e-10);
        assert!(mean_vals[1].abs() < 1e-10);
    }

    #[test]
    fn test_standard_scaler_empty_data() {
        let data = Tensor2D::<CpuBackend>::zeros(0, 2);
        let scaler = StandardScaler::<CpuBackend>::new();
        let result = scaler.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_standard_scaler_n_features_in() {
        let data = create_test_data();
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();
        assert_eq!(fitted.n_features_in(), 2);
    }

    #[test]
    fn test_standard_scaler_save_load_file() {
        let data = create_test_data();
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let temp_file = std::env::temp_dir().join("test_standard.bin");
        fitted.save_to_file(&temp_file).unwrap();

        let loaded = FittedStandardScaler::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert_eq!(loaded.n_features_in(), fitted.n_features_in());

        let t1 = fitted.transform(&data).unwrap();
        let t2 = loaded.transform(&data).unwrap();

        let v1 = t1.ravel().to_vec();
        let v2 = t2.ravel().to_vec();
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_standard_scaler_constant_feature() {
        // All values in column 0 are the same (constant feature)
        let data = Tensor2D::<CpuBackend>::new(vec![5.0f32, 1.0, 5.0, 2.0, 5.0, 3.0], 3, 2);
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        // Std for constant feature should be 1 (handled internally)
        let std = fitted.std().to_vec();
        assert!((std[0] - 1.0).abs() < 1e-6);

        // Mean should still be 5
        let mean = fitted.mean().to_vec();
        assert!((mean[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_standard_scaler_inverse_feature_mismatch() {
        let data = create_test_data();
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let wrong_data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 1, 3);
        let result = fitted.inverse_transform(&wrong_data);

        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch { .. })
        ));
    }

    #[test]
    fn test_standard_scaler_no_mean_no_std() {
        let data = create_test_data();
        let scaler = StandardScaler::<CpuBackend>::new()
            .with_mean(false)
            .with_std(false);
        let fitted = scaler.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();
        let original = data.ravel().to_vec();
        let result = transformed.ravel().to_vec();

        // Without mean centering or std scaling, data should be unchanged
        for (o, r) in original.iter().zip(result.iter()) {
            assert!((o - r).abs() < 1e-6);
        }
    }
}
