//! Min-Max Scaler.
//!
//! Transforms features by scaling each feature to a given range (default [0, 1]).
//!
//! The transformation is given by:
//! ```text
//! X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
//! ```
//!
//! # Example
//! ```ignore
//! use machinelearne_rs::preprocessing::{Transformer, MinMaxScaler};
//! use machinelearne_rs::backend::CpuBackend;
//!
//! let scaler = MinMaxScaler::<CpuBackend>::new()
//!     .with_range(0.0, 1.0);
//!
//! let fitted = scaler.fit(&data)?;
//! let scaled = fitted.transform(&data)?;
//! ```

use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Configuration for MinMaxScaler.
#[derive(Clone, Serialize, Deserialize)]
pub struct MinMaxScalerConfig {
    /// Minimum value of the target range.
    pub min: f64,
    /// Maximum value of the target range.
    pub max: f64,
}

impl Default for MinMaxScalerConfig {
    fn default() -> Self {
        Self { min: 0.0, max: 1.0 }
    }
}

/// Serializable parameters for a fitted MinMaxScaler.
#[derive(Clone, Serialize, Deserialize)]
pub struct MinMaxScalerParams {
    /// Configuration options.
    pub config: MinMaxScalerConfig,
    /// Minimum of each feature.
    pub min_: Vec<f64>,
    /// Maximum of each feature.
    pub max_: Vec<f64>,
    /// Scale factor for each feature: (max - min) / (feature_max - feature_min).
    pub scale_: Vec<f64>,
    /// Number of features seen during fit.
    pub n_features: usize,
}

/// MinMaxScaler transformer (unfitted).
///
/// Transforms features by scaling each feature to a given range.
#[derive(Clone)]
pub struct MinMaxScaler<B: Backend> {
    config: MinMaxScalerConfig,
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for MinMaxScaler<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MinMaxScaler<B> {
    /// Create a new MinMaxScaler with default range [0, 1].
    pub fn new() -> Self {
        Self {
            config: MinMaxScalerConfig::default(),
            _backend: PhantomData,
        }
    }

    /// Set the target range for scaling.
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        assert!(max > min, "max must be greater than min");
        self.config.min = min;
        self.config.max = max;
        self
    }
}

impl<B: Backend> Transformer<B> for MinMaxScaler<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = MinMaxScalerParams;
    type Fitted = FittedMinMaxScaler<B>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError> {
        let (rows, cols) = data.shape();

        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit MinMaxScaler on empty data".to_string(),
            ));
        }

        let min_vec = B::col_min_2d(&data.data);
        let max_vec = B::col_max_2d(&data.data);

        let min_vals = B::to_vec_1d(&min_vec);
        let max_vals = B::to_vec_1d(&max_vec);

        // Compute scale: (target_max - target_min) / (feature_max - feature_min)
        let target_range = self.config.max - self.config.min;
        let scale_vals: Vec<f64> = min_vals
            .iter()
            .zip(max_vals.iter())
            .map(|(&min, &max)| {
                let range = max - min;
                if range == 0.0 {
                    1.0 // Constant feature: scale by 1 to avoid division by zero
                } else {
                    target_range / range
                }
            })
            .collect();

        let scale_ = Tensor1D {
            data: B::from_vec_1d(scale_vals.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };

        let min_ = Tensor1D {
            data: min_vec,
            backend: PhantomData::<B>,
        };

        Ok(FittedMinMaxScaler {
            config: self.config.clone(),
            min_,
            max_: max_vec,
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

/// Fitted MinMaxScaler ready for inference.
#[derive(Clone)]
pub struct FittedMinMaxScaler<B: Backend> {
    config: MinMaxScalerConfig,
    min_: Tensor1D<B>,
    max_: B::Tensor1D,
    scale_: Tensor1D<B>,
    n_features: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedMinMaxScaler<B> {
    /// Get the minimum values for each feature.
    pub fn min(&self) -> &Tensor1D<B> {
        &self.min_
    }

    /// Get the scale factor for each feature.
    pub fn scale(&self) -> &Tensor1D<B> {
        &self.scale_
    }

    /// Get the data range (max - min) for each feature.
    pub fn data_range(&self) -> Tensor1D<B> {
        let max_tensor = Tensor1D {
            data: self.max_.clone(),
            backend: PhantomData::<B>,
        };
        Tensor1D {
            data: B::sub_1d(&max_tensor.data, &self.min_.data),
            backend: PhantomData::<B>,
        }
    }
}

impl<B: Backend> FittedTransformer<B> for FittedMinMaxScaler<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = MinMaxScalerParams;

    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let (_, cols) = data.shape();

        if cols != self.n_features {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features,
                got_features: cols,
            });
        }

        // X_scaled = (X - X_min) * scale_ + target_min
        // which equals: (X - X_min) / (X_max - X_min) * (target_max - target_min) + target_min
        let centered = B::broadcast_sub_1d_to_2d_rows(&data.data, &self.min_.data);
        let scaled = B::broadcast_mul_1d_to_2d_rows(&centered, &self.scale_.data);
        let result = B::add_scalar_2d(&scaled, &B::scalar_f64(self.config.min));

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

        // X = (X_scaled - target_min) / scale_ + X_min
        let centered = B::add_scalar_2d(&data.data, &B::scalar_f64(-self.config.min));
        let unscaled = B::broadcast_div_1d_to_2d_rows(&centered, &self.scale_.data);
        let result = B::broadcast_add_1d_to_2d_rows(&unscaled, &self.min_.data);

        Ok(Tensor2D {
            data: result,
            backend: PhantomData::<B>,
        })
    }

    fn extract_params(&self) -> Self::Params {
        MinMaxScalerParams {
            config: self.config.clone(),
            min_: self.min_.to_vec(),
            max_: B::to_vec_1d(&self.max_),
            scale_: self.scale_.to_vec(),
            n_features: self.n_features,
        }
    }

    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError> {
        let min_ = Tensor1D {
            data: B::from_vec_1d(params.min_.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };
        let max_ = B::from_vec_1d(params.max_.iter().map(|&x| x as f32).collect());
        let scale_ = Tensor1D {
            data: B::from_vec_1d(params.scale_.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };

        Ok(Self {
            config: params.config,
            min_,
            max_,
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
        let params: MinMaxScalerParams = bincode::deserialize(&bytes)
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
    fn test_minmax_scaler_fit() {
        let data = create_test_data();
        let scaler = MinMaxScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        // Min: [0, 1], Max: [1, 3]
        let min = fitted.min().to_vec();
        assert_eq!(min[0], 0.0);
        assert_eq!(min[1], 1.0);

        // Scale: (1 - 0) / (1 - 0) = 1, (1 - 0) / (3 - 1) = 0.5
        let scale = fitted.scale().to_vec();
        assert!((scale[0] - 1.0).abs() < 1e-10);
        assert!((scale[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_scaler_transform() {
        let data = create_test_data();
        let scaler = MinMaxScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();
        let values = transformed.ravel().to_vec();

        // First column: [0, 0, 1] -> [0, 0, 1]
        assert!((values[0] - 0.0).abs() < 1e-6);
        assert!((values[2] - 0.0).abs() < 1e-6);
        assert!((values[4] - 1.0).abs() < 1e-6);

        // Second column: [1, 1, 3] -> [0, 0, 1]
        assert!((values[1] - 0.0).abs() < 1e-6);
        assert!((values[3] - 0.0).abs() < 1e-6);
        assert!((values[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_scaler_inverse_transform() {
        let data = create_test_data();
        let scaler = MinMaxScaler::<CpuBackend>::new();
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
    fn test_minmax_scaler_custom_range() {
        let data = create_test_data();
        let scaler = MinMaxScaler::<CpuBackend>::new().with_range(-1.0, 1.0);
        let fitted = scaler.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();
        let values = transformed.ravel().to_vec();

        // First column: [0, 0, 1] -> [-1, -1, 1]
        assert!((values[0] - (-1.0)).abs() < 1e-6);
        assert!((values[2] - (-1.0)).abs() < 1e-6);
        assert!((values[4] - 1.0).abs() < 1e-6);

        // Second column: [1, 1, 3] -> [-1, -1, 1]
        assert!((values[1] - (-1.0)).abs() < 1e-6);
        assert!((values[3] - (-1.0)).abs() < 1e-6);
        assert!((values[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_scaler_serialization() {
        let data = create_test_data();
        let scaler = MinMaxScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let params = fitted.extract_params();
        let restored = FittedMinMaxScaler::<CpuBackend>::from_params(params).unwrap();

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
    fn test_minmax_scaler_feature_mismatch() {
        let data = create_test_data(); // 2 features
        let scaler = MinMaxScaler::<CpuBackend>::new();
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

    #[test]
    fn test_minmax_scaler_empty_data() {
        let data = Tensor2D::<CpuBackend>::zeros(0, 2);
        let scaler = MinMaxScaler::<CpuBackend>::new();
        let result = scaler.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_minmax_scaler_n_features_in() {
        let data = create_test_data();
        let scaler = MinMaxScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();
        assert_eq!(fitted.n_features_in(), 2);
    }

    #[test]
    fn test_minmax_scaler_save_load_file() {
        let data = create_test_data();
        let scaler = MinMaxScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let temp_file = std::env::temp_dir().join("test_minmax.bin");
        fitted.save_to_file(&temp_file).unwrap();

        let loaded = FittedMinMaxScaler::<CpuBackend>::load_from_file(&temp_file).unwrap();

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
    fn test_minmax_scaler_fit_transform() {
        let data = create_test_data();
        let scaler = MinMaxScaler::<CpuBackend>::new();
        let transformed = scaler.fit_transform(&data).unwrap();
        let values = transformed.ravel().to_vec();

        // First column: [0, 0, 1] -> [0, 0, 1]
        assert!((values[0] - 0.0).abs() < 1e-6);
        assert!((values[2] - 0.0).abs() < 1e-6);
        assert!((values[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_scaler_data_range() {
        let data = create_test_data();
        let scaler = MinMaxScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let range = fitted.data_range().to_vec();
        assert!((range[0] - 1.0).abs() < 1e-6); // 1 - 0 = 1
        assert!((range[1] - 2.0).abs() < 1e-6); // 3 - 1 = 2
    }

    #[test]
    fn test_minmax_scaler_zero_range() {
        // All values the same - range is 0
        let data = Tensor2D::<CpuBackend>::new(vec![5.0f32, 5.0, 5.0, 5.0], 2, 2);
        let scaler = MinMaxScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        // Should not fail - scale should be 1.0 for zero range
        let transformed = fitted.transform(&data).unwrap();
        let values = transformed.ravel().to_vec();

        // All values should map to target_min (0)
        assert!(values.iter().all(|&v| (v - 0.0).abs() < 1e-6));
    }

    #[test]
    fn test_minmax_scaler_inverse_feature_mismatch() {
        let data = create_test_data();
        let scaler = MinMaxScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let wrong_data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 1, 3);
        let result = fitted.inverse_transform(&wrong_data);

        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch { .. })
        ));
    }
}
