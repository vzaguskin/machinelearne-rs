//! Simple Imputer.
//!
//! Imputation transformer for completing missing values.
//! Supports mean, median, most_frequent, and constant strategies.
//!
//! Note: This implementation treats NaN as missing values.
//!
//! # Example
//! ```ignore
//! use machinelearne_rs::preprocessing::{Transformer, SimpleImputer, ImputeStrategy};
//! use machinelearne_rs::backend::CpuBackend;
//!
//! let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Mean);
//! let fitted = imputer.fit(&data)?;
//! let imputed = fitted.transform(&data)?;
//! ```

use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Strategy for imputing missing values.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum ImputeStrategy {
    /// Replace missing values with the mean of each column.
    #[default]
    Mean,
    /// Replace missing values with the median of each column.
    Median,
    /// Replace missing values with the most frequent value of each column.
    MostFrequent,
    /// Replace missing values with a constant value.
    Constant(f64),
}

/// Serializable parameters for a fitted SimpleImputer.
#[derive(Clone, Serialize, Deserialize)]
pub struct SimpleImputerParams {
    /// Strategy used for imputation.
    pub strategy: ImputeStrategy,
    /// Statistics (fill values) for each feature.
    pub statistics_: Vec<f64>,
    /// Number of features seen during fit.
    pub n_features: usize,
}

/// SimpleImputer transformer (unfitted).
///
/// Imputation transformer for completing missing values.
#[derive(Clone)]
pub struct SimpleImputer<B: Backend> {
    strategy: ImputeStrategy,
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for SimpleImputer<B> {
    fn default() -> Self {
        Self::new(ImputeStrategy::default())
    }
}

impl<B: Backend> SimpleImputer<B> {
    /// Create a new SimpleImputer with the specified strategy.
    pub fn new(strategy: ImputeStrategy) -> Self {
        Self {
            strategy,
            _backend: PhantomData,
        }
    }
}

/// Compute statistics for imputation, ignoring NaN values.
fn compute_statistics(
    data: &[f64],
    rows: usize,
    cols: usize,
    strategy: &ImputeStrategy,
) -> Vec<f64> {
    let mut stats = vec![0.0; cols];

    for col in 0..cols {
        // Collect non-NaN values for this column
        let column_values: Vec<f64> = (0..rows)
            .map(|row| data[row * cols + col])
            .filter(|&v| !v.is_nan())
            .collect();

        stats[col] = if column_values.is_empty() {
            0.0 // Default to 0 if all values are missing
        } else {
            match strategy {
                ImputeStrategy::Mean => {
                    column_values.iter().sum::<f64>() / column_values.len() as f64
                }
                ImputeStrategy::Median => {
                    let mut sorted = column_values.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = sorted.len();
                    if n.is_multiple_of(2) {
                        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                    } else {
                        sorted[n / 2]
                    }
                }
                ImputeStrategy::MostFrequent => {
                    // Find the most common value
                    let mut counts = std::collections::HashMap::new();
                    for &v in &column_values {
                        *counts.entry(v.to_bits()).or_insert(0) += 1;
                    }
                    counts
                        .into_iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(bits, _)| f64::from_bits(bits))
                        .unwrap_or(0.0)
                }
                ImputeStrategy::Constant(val) => *val,
            }
        };
    }

    stats
}

impl<B: Backend> Transformer<B> for SimpleImputer<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = SimpleImputerParams;
    type Fitted = FittedSimpleImputer<B>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError> {
        let (rows, cols) = data.shape();

        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit SimpleImputer on empty data".to_string(),
            ));
        }

        let flat_data = B::to_vec_1d(&B::ravel_2d(&data.data));
        let statistics_ = compute_statistics(&flat_data, rows, cols, &self.strategy);

        let statistics_tensor = Tensor1D {
            data: B::from_vec_1d(statistics_.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };

        Ok(FittedSimpleImputer {
            strategy: self.strategy.clone(),
            statistics_: statistics_tensor,
            n_features: cols,
            _backend: PhantomData,
        })
    }

    fn fit_transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let fitted = self.fit(data)?;
        fitted.transform(data)
    }
}

/// Fitted SimpleImputer ready for inference.
#[derive(Clone)]
pub struct FittedSimpleImputer<B: Backend> {
    strategy: ImputeStrategy,
    statistics_: Tensor1D<B>,
    n_features: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedSimpleImputer<B> {
    /// Get the imputation statistics (fill values) for each feature.
    pub fn statistics(&self) -> &Tensor1D<B> {
        &self.statistics_
    }
}

impl<B: Backend> FittedTransformer<B> for FittedSimpleImputer<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = SimpleImputerParams;

    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let (rows, cols) = data.shape();

        if cols != self.n_features {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features,
                got_features: cols,
            });
        }

        let flat_data = B::to_vec_1d(&B::ravel_2d(&data.data));
        let stats = self.statistics_.to_vec();

        // Replace NaN values with statistics
        let mut result = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let val = flat_data[row * cols + col];
                result.push(if val.is_nan() { stats[col] } else { val });
            }
        }

        Ok(Tensor2D {
            data: B::from_vec_2d(result.iter().map(|&x| x as f32).collect(), rows, cols),
            backend: PhantomData::<B>,
        })
    }

    fn inverse_transform(&self, _data: &Self::Output) -> Result<Self::Input, PreprocessingError> {
        Err(PreprocessingError::InvalidParameter(
            "SimpleImputer does not support inverse_transform (missing value information is lost)"
                .to_string(),
        ))
    }

    fn extract_params(&self) -> Self::Params {
        SimpleImputerParams {
            strategy: self.strategy.clone(),
            statistics_: self.statistics_.to_vec(),
            n_features: self.n_features,
        }
    }

    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError> {
        let statistics_ = Tensor1D {
            data: B::from_vec_1d(params.statistics_.iter().map(|&x| x as f32).collect()),
            backend: PhantomData::<B>,
        };

        Ok(Self {
            strategy: params.strategy,
            statistics_,
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
        let params: SimpleImputerParams = bincode::deserialize(&bytes)
            .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
        Self::from_params(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    fn create_test_data_with_missing() -> Tensor2D<CpuBackend> {
        // [[1, NaN], [3, 4], [5, 6]]
        Tensor2D::new(vec![1.0f32, f32::NAN, 3.0, 4.0, 5.0, 6.0], 3, 2)
    }

    #[test]
    fn test_simple_imputer_mean() {
        let data = create_test_data_with_missing();
        let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Mean);
        let fitted = imputer.fit(&data).unwrap();

        let stats = fitted.statistics().to_vec();
        // Column 0: mean of [1, 3, 5] = 3
        // Column 1: mean of [4, 6] = 5 (NaN excluded)
        assert!((stats[0] - 3.0).abs() < 1e-6);
        assert!((stats[1] - 5.0).abs() < 1e-6);

        let imputed = fitted.transform(&data).unwrap();
        let values = imputed.ravel().to_vec();

        // Column 0 unchanged: [1, 3, 5]
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
        assert!((values[4] - 5.0).abs() < 1e-6);

        // Column 1: [5, 4, 6] (NaN replaced with 5)
        assert!((values[1] - 5.0).abs() < 1e-6);
        assert!((values[3] - 4.0).abs() < 1e-6);
        assert!((values[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_simple_imputer_median() {
        let data = create_test_data_with_missing();
        let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Median);
        let fitted = imputer.fit(&data).unwrap();

        let stats = fitted.statistics().to_vec();
        // Column 0: median of [1, 3, 5] = 3
        // Column 1: median of [4, 6] = 5
        assert!((stats[0] - 3.0).abs() < 1e-6);
        assert!((stats[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_simple_imputer_constant() {
        let data = create_test_data_with_missing();
        let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Constant(-1.0));
        let fitted = imputer.fit(&data).unwrap();

        let stats = fitted.statistics().to_vec();
        assert!((stats[0] - (-1.0)).abs() < 1e-6);
        assert!((stats[1] - (-1.0)).abs() < 1e-6);

        let imputed = fitted.transform(&data).unwrap();
        let values = imputed.ravel().to_vec();

        // NaN replaced with -1
        assert!((values[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_simple_imputer_inverse_not_supported() {
        let data = create_test_data_with_missing();
        let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Mean);
        let fitted = imputer.fit(&data).unwrap();

        let imputed = fitted.transform(&data).unwrap();
        let result = fitted.inverse_transform(&imputed);

        assert!(matches!(
            result,
            Err(PreprocessingError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_simple_imputer_serialization() {
        let data = create_test_data_with_missing();
        let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Mean);
        let fitted = imputer.fit(&data).unwrap();

        let params = fitted.extract_params();
        let restored = FittedSimpleImputer::<CpuBackend>::from_params(params).unwrap();

        let imputed1 = fitted.transform(&data).unwrap();
        let imputed2 = restored.transform(&data).unwrap();

        let i1 = imputed1.ravel().to_vec();
        let i2 = imputed2.ravel().to_vec();

        for (a, b) in i1.iter().zip(i2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simple_imputer_feature_mismatch() {
        let data = create_test_data_with_missing(); // 2 features
        let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Mean);
        let fitted = imputer.fit(&data).unwrap();

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
    fn test_simple_imputer_most_frequent() {
        // Create data where most frequent is clear
        let data = Tensor2D::<CpuBackend>::new(
            vec![1.0f32, f32::NAN, 1.0, 2.0, 1.0, 2.0], // col 0: [1, 1, 1], col 1: [NaN, 2, 2]
            3,
            2,
        );

        let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::MostFrequent);
        let fitted = imputer.fit(&data).unwrap();

        let stats = fitted.statistics().to_vec();
        assert!((stats[0] - 1.0).abs() < 1e-6); // 1 is most frequent in col 0
        assert!((stats[1] - 2.0).abs() < 1e-6); // 2 is most frequent in col 1
    }

    #[test]
    fn test_simple_imputer_empty_data() {
        let data = Tensor2D::<CpuBackend>::zeros(0, 2);

        let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Mean);
        let result = imputer.fit(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_simple_imputer_save_load_file() {
        let data = create_test_data_with_missing();
        let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Mean);
        let fitted = imputer.fit(&data).unwrap();

        let temp_file = std::env::temp_dir().join("test_imputer.bin");
        fitted.save_to_file(&temp_file).unwrap();

        let loaded = FittedSimpleImputer::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert_eq!(loaded.n_features_in(), fitted.n_features_in());

        let i1 = fitted.transform(&data).unwrap();
        let i2 = loaded.transform(&data).unwrap();

        let v1 = i1.ravel().to_vec();
        let v2 = i2.ravel().to_vec();
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_simple_imputer_n_features_in() {
        let data = create_test_data_with_missing();
        let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Mean);
        let fitted = imputer.fit(&data).unwrap();

        assert_eq!(fitted.n_features_in(), 2);
    }
}
