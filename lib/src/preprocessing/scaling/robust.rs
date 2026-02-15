//! Robust Scaler.
//!
//! Scales features using statistics that are robust to outliers.
//! Uses median and interquartile range (IQR) instead of mean and std.
//!
//! The transformation is:
//! ```text
//! X_scaled = (X - median) / IQR
//! ```
//!
//! where IQR is the range between the 1st quartile (25%) and 3rd quartile (75%).
//!
//! # Example
//! ```ignore
//! use machinelearne_rs::preprocessing::{Transformer, RobustScaler};
//! use machinelearne_rs::backend::CpuBackend;
//!
//! let scaler = RobustScaler::<CpuBackend>::new()
//!     .with_centering(true);
//!
//! let fitted = scaler.fit(&data)?;
//! let scaled = fitted.transform(&data)?;
//! ```

use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Configuration for RobustScaler.
#[derive(Clone, Serialize, Deserialize)]
pub struct RobustScalerConfig {
    /// If True, center the data by median before scaling.
    pub with_centering: bool,
    /// If True, scale the data by IQR.
    pub with_scaling: bool,
    /// Quantile range for IQR (default: (25.0, 75.0)).
    pub quantile_range: (f64, f64),
}

impl Default for RobustScalerConfig {
    fn default() -> Self {
        Self {
            with_centering: true,
            with_scaling: true,
            quantile_range: (25.0, 75.0),
        }
    }
}

/// Serializable parameters for a fitted RobustScaler.
#[derive(Clone, Serialize, Deserialize)]
pub struct RobustScalerParams {
    /// Configuration options.
    pub config: RobustScalerConfig,
    /// Center (median) for each feature.
    pub center_: Vec<f64>,
    /// Scale (IQR) for each feature.
    pub scale_: Vec<f64>,
    /// Number of features seen during fit.
    pub n_features: usize,
}

/// RobustScaler transformer (unfitted).
///
/// Scales features using statistics that are robust to outliers.
#[derive(Clone)]
pub struct RobustScaler<B: Backend> {
    config: RobustScalerConfig,
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for RobustScaler<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> RobustScaler<B> {
    /// Create a new RobustScaler with default configuration.
    pub fn new() -> Self {
        Self {
            config: RobustScalerConfig::default(),
            _backend: PhantomData,
        }
    }

    /// Set whether to center data by median.
    pub fn with_centering(mut self, with_centering: bool) -> Self {
        self.config.with_centering = with_centering;
        self
    }

    /// Set whether to scale data by IQR.
    pub fn with_scaling(mut self, with_scaling: bool) -> Self {
        self.config.with_scaling = with_scaling;
        self
    }

    /// Set the quantile range for IQR calculation.
    pub fn with_quantile_range(mut self, min: f64, max: f64) -> Self {
        assert!(
            (0.0..=100.0).contains(&min) && (0.0..=100.0).contains(&max) && min < max,
            "Invalid quantile range: must be 0 <= min < max <= 100"
        );
        self.config.quantile_range = (min, max);
        self
    }
}

/// Compute quantiles for a column of data.
fn compute_quantiles(data: &[f64], q_low: f64, q_high: f64) -> (f64, f64) {
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 0 {
        return (0.0, 1.0);
    }

    // Linear interpolation for quantiles
    let low_idx = (q_low / 100.0 * (n - 1) as f64).min((n - 1) as f64);
    let high_idx = (q_high / 100.0 * (n - 1) as f64).min((n - 1) as f64);

    let low_val = interpolate(&sorted, low_idx);
    let high_val = interpolate(&sorted, high_idx);

    (low_val, high_val)
}

/// Linear interpolation at a fractional index.
fn interpolate(sorted: &[f64], idx: f64) -> f64 {
    let lower = idx.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    let frac = idx - lower as f64;

    sorted[lower] * (1.0 - frac) + sorted[upper] * frac
}

impl<B: Backend> Transformer<B> for RobustScaler<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = RobustScalerParams;
    type Fitted = FittedRobustScaler<B>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError> {
        let (rows, cols) = data.shape();

        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit RobustScaler on empty data".to_string(),
            ));
        }

        // Get the raw data
        let flat_data = B::to_vec_1d(&B::ravel_2d(&data.data));

        // Compute center (median) and scale (IQR) for each column
        let mut center_vals = vec![0.0; cols];
        let mut scale_vals = vec![1.0; cols];

        let (q_low, q_high) = self.config.quantile_range;

        for col in 0..cols {
            // Extract column data
            let column_data: Vec<f64> = (0..rows).map(|row| flat_data[row * cols + col]).collect();

            // Compute quantiles
            let (low, high) = compute_quantiles(&column_data, q_low, q_high);

            if self.config.with_centering {
                // Median is the midpoint of the IQR
                center_vals[col] = (low + high) / 2.0;
            }

            if self.config.with_scaling {
                let iqr = high - low;
                scale_vals[col] = if iqr == 0.0 { 1.0 } else { iqr };
            }
        }

        let center_ = Tensor1D {
            data: B::from_vec_1d(center_vals.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };

        let scale_ = Tensor1D {
            data: B::from_vec_1d(scale_vals.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };

        Ok(FittedRobustScaler {
            config: self.config.clone(),
            center_,
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

/// Fitted RobustScaler ready for inference.
#[derive(Clone)]
pub struct FittedRobustScaler<B: Backend> {
    config: RobustScalerConfig,
    center_: Tensor1D<B>,
    scale_: Tensor1D<B>,
    n_features: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedRobustScaler<B> {
    /// Get the center (median) values for each feature.
    pub fn center(&self) -> &Tensor1D<B> {
        &self.center_
    }

    /// Get the scale (IQR) values for each feature.
    pub fn scale(&self) -> &Tensor1D<B> {
        &self.scale_
    }
}

impl<B: Backend> FittedTransformer<B> for FittedRobustScaler<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = RobustScalerParams;

    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let (_, cols) = data.shape();

        if cols != self.n_features {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features,
                got_features: cols,
            });
        }

        let mut result_data = data.data.clone();

        if self.config.with_centering {
            result_data = B::broadcast_sub_1d_to_2d_rows(&result_data, &self.center_.data);
        }

        if self.config.with_scaling {
            result_data = B::broadcast_div_1d_to_2d_rows(&result_data, &self.scale_.data);
        }

        Ok(Tensor2D {
            data: result_data,
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

        let mut result_data = data.data.clone();

        if self.config.with_scaling {
            result_data = B::broadcast_mul_1d_to_2d_rows(&result_data, &self.scale_.data);
        }

        if self.config.with_centering {
            result_data = B::broadcast_add_1d_to_2d_rows(&result_data, &self.center_.data);
        }

        Ok(Tensor2D {
            data: result_data,
            backend: PhantomData::<B>,
        })
    }

    fn extract_params(&self) -> Self::Params {
        RobustScalerParams {
            config: self.config.clone(),
            center_: self.center_.to_vec(),
            scale_: self.scale_.to_vec(),
            n_features: self.n_features,
        }
    }

    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError> {
        let center_ = Tensor1D {
            data: B::from_vec_1d(params.center_.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };
        let scale_ = Tensor1D {
            data: B::from_vec_1d(params.scale_.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };

        Ok(Self {
            config: params.config,
            center_,
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
        let params: RobustScalerParams = bincode::deserialize(&bytes)
            .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
        Self::from_params(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    fn create_test_data() -> Tensor2D<CpuBackend> {
        // [[1, 2], [2, 4], [3, 6], [4, 8], [5, 100]]  -- second column has outlier
        Tensor2D::new(
            vec![1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 100.0],
            5,
            2,
        )
    }

    #[test]
    fn test_robust_scaler_fit() {
        let data = create_test_data();
        let scaler = RobustScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let center = fitted.center().to_vec();
        let scale = fitted.scale().to_vec();

        // First column: [1, 2, 3, 4, 5], median = 3, Q1 = 2, Q3 = 4, IQR = 2
        assert!((center[0] - 3.0).abs() < 0.1);
        assert!((scale[0] - 2.0).abs() < 0.1);

        // Second column: [2, 4, 6, 8, 100], median = 6, Q1 = 4, Q3 = 8, IQR = 4
        // (outlier 100 doesn't significantly affect median/IQR)
        assert!((center[1] - 6.0).abs() < 0.1);
        assert!((scale[1] - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_robust_scaler_transform() {
        let data = create_test_data();
        let scaler = RobustScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();
        let values = transformed.ravel().to_vec();

        // First column center (median) should be ~3, scale ~2
        // Values: (1-3)/2 = -1, (2-3)/2 = -0.5, (3-3)/2 = 0, (4-3)/2 = 0.5, (5-3)/2 = 1
        assert!((values[0] - (-1.0)).abs() < 0.1);
        assert!((values[2] - (-0.5)).abs() < 0.1);
        assert!((values[4] - 0.0).abs() < 0.1);
        assert!((values[6] - 0.5).abs() < 0.1);
        assert!((values[8] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_robust_scaler_inverse_transform() {
        let data = create_test_data();
        let scaler = RobustScaler::<CpuBackend>::new();
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
    fn test_robust_scaler_without_centering() {
        let data = create_test_data();
        let scaler = RobustScaler::<CpuBackend>::new().with_centering(false);
        let fitted = scaler.fit(&data).unwrap();

        let center = fitted.center().to_vec();
        assert!(center.iter().all(|&c| c == 0.0));
    }

    #[test]
    fn test_robust_scaler_without_scaling() {
        let data = create_test_data();
        let scaler = RobustScaler::<CpuBackend>::new().with_scaling(false);
        let fitted = scaler.fit(&data).unwrap();

        let scale = fitted.scale().to_vec();
        assert!(scale.iter().all(|&s| s == 1.0));
    }

    #[test]
    fn test_robust_scaler_serialization() {
        let data = create_test_data();
        let scaler = RobustScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let params = fitted.extract_params();
        let restored = FittedRobustScaler::<CpuBackend>::from_params(params).unwrap();

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
    fn test_robust_scaler_feature_mismatch() {
        let data = create_test_data(); // 2 features
        let scaler = RobustScaler::<CpuBackend>::new();
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
    fn test_robust_scaler_empty_data() {
        let data = Tensor2D::<CpuBackend>::zeros(0, 2);
        let scaler = RobustScaler::<CpuBackend>::new();
        let result = scaler.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_robust_scaler_n_features_in() {
        let data = create_test_data();
        let scaler = RobustScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();
        assert_eq!(fitted.n_features_in(), 2);
    }

    #[test]
    fn test_robust_scaler_save_load_file() {
        let data = create_test_data();
        let scaler = RobustScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let temp_file = std::env::temp_dir().join("test_robust.bin");
        fitted.save_to_file(&temp_file).unwrap();

        let loaded = FittedRobustScaler::<CpuBackend>::load_from_file(&temp_file).unwrap();

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
    fn test_robust_scaler_fit_transform() {
        let data = create_test_data();
        let scaler = RobustScaler::<CpuBackend>::new();
        let transformed = scaler.fit_transform(&data).unwrap();

        // Should produce valid output
        assert_eq!(transformed.shape(), (5, 2));
    }

    #[test]
    fn test_robust_scaler_with_quantile_range() {
        let data = create_test_data();
        let scaler = RobustScaler::<CpuBackend>::new().with_quantile_range(10.0, 90.0);
        let fitted = scaler.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();
        assert_eq!(transformed.shape(), (5, 2));
    }

    #[test]
    fn test_robust_scaler_no_centering_no_scaling() {
        let data = create_test_data();
        let scaler = RobustScaler::<CpuBackend>::new()
            .with_centering(false)
            .with_scaling(false);
        let fitted = scaler.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();
        let original = data.ravel().to_vec();
        let result = transformed.ravel().to_vec();

        // Without centering or scaling, data should be unchanged
        for (o, r) in original.iter().zip(result.iter()) {
            assert!((o - r).abs() < 1e-6);
        }
    }
}
