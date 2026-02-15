//! One-hot encoding for categorical features.
//!
//! Transforms categorical integer values to one-hot (dummy) encoded vectors.

use crate::backend::{Backend, Tensor2D};
use crate::preprocessing::encoding::HandleUnknown;
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::marker::PhantomData;

/// One-hot encoder for categorical features.
///
/// Converts integer categories to one-hot encoded vectors. Each input column
/// is treated as a categorical feature, and the encoder learns the unique
/// values (categories) present in each column during fitting.
///
/// # Example
/// ```ignore
/// use machinelearne_rs::preprocessing::{OneHotEncoder, Transformer};
/// use machinelearne_rs::backend::CpuBackend;
///
/// // Input: 3 samples with 1 categorical feature each
/// // Categories: 0, 1, 2 (e.g., "red", "green", "blue")
/// let data = Tensor2D::new(vec![0.0, 1.0, 2.0], 3, 1);
///
/// let encoder = OneHotEncoder::<CpuBackend>::new();
/// let fitted = encoder.fit(&data)?;
///
/// // Output: 3x3 one-hot matrix
/// let encoded = fitted.transform(&data)?;
/// // [[1, 0, 0],
/// //  [0, 1, 0],
/// //  [0, 0, 1]]
/// ```
#[derive(Clone, Debug)]
pub struct OneHotEncoder<B: Backend> {
    /// How to handle unknown categories during transform.
    handle_unknown: HandleUnknown,
    _backend: PhantomData<B>,
}

impl<B: Backend> OneHotEncoder<B> {
    /// Create a new OneHotEncoder with default settings.
    pub fn new() -> Self {
        Self {
            handle_unknown: HandleUnknown::default(),
            _backend: PhantomData,
        }
    }

    /// Set the strategy for handling unknown categories.
    pub fn with_handle_unknown(mut self, strategy: HandleUnknown) -> Self {
        self.handle_unknown = strategy;
        self
    }
}

impl<B: Backend> Default for OneHotEncoder<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Output configuration for one-hot encoding.
#[derive(Clone, Debug, Default)]
pub struct OneHotOutput;

/// Serializable parameters for a fitted OneHotEncoder.
#[derive(Clone, Serialize, Deserialize)]
pub struct OneHotEncoderParams {
    /// Categories (unique values) for each input column.
    pub categories_: Vec<Vec<f32>>,
    /// Number of categories per column.
    pub n_values_: Vec<usize>,
    /// Total number of output features.
    pub n_features_out: usize,
    /// Number of input features.
    pub n_features_in: usize,
    /// Handle unknown strategy.
    pub handle_unknown: HandleUnknown,
}

// Note: We rely on the blanket impl of SerializableParams for Serialize + Deserialize

/// Fitted OneHotEncoder ready for inference.
#[derive(Clone)]
pub struct FittedOneHotEncoder<B: Backend> {
    /// Categories (unique sorted values) for each input column.
    categories_: Vec<Vec<f32>>,
    /// Number of categories per column.
    n_values_: Vec<usize>,
    /// Total number of output features (sum of n_values_).
    n_features_out: usize,
    /// Number of input features.
    n_features_in: usize,
    /// Handle unknown strategy.
    handle_unknown: HandleUnknown,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedOneHotEncoder<B> {
    /// Get the categories learned for each feature.
    pub fn categories(&self) -> &[Vec<f32>] {
        &self.categories_
    }

    /// Get the number of output features.
    pub fn n_features_out(&self) -> usize {
        self.n_features_out
    }

    /// Get the number of categories per input feature.
    pub fn n_values(&self) -> &[usize] {
        &self.n_values_
    }
}

impl<B: Backend> Transformer<B> for OneHotEncoder<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = OneHotEncoderParams;
    type Fitted = FittedOneHotEncoder<B>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError> {
        let (rows, cols) = data.shape();

        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit OneHotEncoder on empty data".to_string(),
            ));
        }

        let data_vec = data.ravel().to_vec();

        // Find unique categories for each column
        let mut categories_: Vec<Vec<f32>> = Vec::with_capacity(cols);
        let mut n_values_: Vec<usize> = Vec::with_capacity(cols);

        for col in 0..cols {
            let mut col_cats: HashSet<i32> = HashSet::new();
            for row in 0..rows {
                let val = data_vec[row * cols + col];
                if !val.is_finite() || val < 0.0 || val.fract() != 0.0 {
                    return Err(PreprocessingError::InvalidParameter(format!(
                        "OneHotEncoder expects non-negative integer values, got {} at ({}, {})",
                        val, row, col
                    )));
                }
                col_cats.insert(val as i32);
            }

            // Sort categories
            let mut sorted_cats: Vec<f32> = col_cats.into_iter().map(|x| x as f32).collect();
            sorted_cats.sort_by(|a, b| a.partial_cmp(b).unwrap());

            n_values_.push(sorted_cats.len());
            categories_.push(sorted_cats);
        }

        let n_features_out: usize = n_values_.iter().sum();

        Ok(FittedOneHotEncoder {
            categories_,
            n_values_,
            n_features_out,
            n_features_in: cols,
            handle_unknown: self.handle_unknown,
            _backend: PhantomData,
        })
    }

    fn fit_transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let fitted = self.fit(data)?;
        fitted.transform(data)
    }
}

impl<B: Backend> FittedTransformer<B> for FittedOneHotEncoder<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = OneHotEncoderParams;

    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let (rows, cols) = data.shape();

        if cols != self.n_features_in {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features_in,
                got_features: cols,
            });
        }

        if rows == 0 {
            return Ok(Tensor2D::zeros(0, self.n_features_out));
        }

        let data_vec = data.ravel().to_vec();
        let mut result = vec![0.0f64; rows * self.n_features_out];

        for row in 0..rows {
            let mut out_col_offset = 0;
            for col in 0..cols {
                let val = data_vec[row * cols + col];
                let cats = &self.categories_[col];

                // Find category index
                let cat_idx = cats.iter().position(|&c| (c - val as f32).abs() < 1e-6);

                match cat_idx {
                    Some(idx) => {
                        result[row * self.n_features_out + out_col_offset + idx] = 1.0;
                    }
                    None => {
                        if self.handle_unknown == HandleUnknown::Error {
                            return Err(PreprocessingError::InvalidParameter(format!(
                                "Unknown category {} in column {}",
                                val, col
                            )));
                        }
                        // With Ignore, leave as zeros
                    }
                }

                out_col_offset += self.n_values_[col];
            }
        }

        Ok(Tensor2D::new(
            result.iter().map(|&x| x as f32).collect(),
            rows,
            self.n_features_out,
        ))
    }

    fn inverse_transform(&self, data: &Self::Output) -> Result<Self::Input, PreprocessingError> {
        let (rows, out_cols) = data.shape();

        if out_cols != self.n_features_out {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features_out,
                got_features: out_cols,
            });
        }

        let data_vec = data.ravel().to_vec();
        let mut result = vec![0.0f64; rows * self.n_features_in];

        for row in 0..rows {
            let mut in_col_offset = 0;
            for col in 0..self.n_features_in {
                let n_cats = self.n_values_[col];
                let cats = &self.categories_[col];

                // Find the index of the 1 in this column's one-hot section
                let mut found = false;
                for (i, &cat) in cats.iter().enumerate() {
                    let val = data_vec[row * out_cols + in_col_offset + i];
                    if val > 0.5 {
                        result[row * self.n_features_in + col] = cat as f64;
                        found = true;
                        break;
                    }
                }

                if !found {
                    // No category found - this shouldn't happen for valid one-hot data
                    result[row * self.n_features_in + col] = f64::NAN;
                }

                in_col_offset += n_cats;
            }
        }

        Ok(Tensor2D::new(
            result.iter().map(|&x| x as f32).collect(),
            rows,
            self.n_features_in,
        ))
    }

    fn extract_params(&self) -> Self::Params {
        OneHotEncoderParams {
            categories_: self.categories_.clone(),
            n_values_: self.n_values_.clone(),
            n_features_out: self.n_features_out,
            n_features_in: self.n_features_in,
            handle_unknown: self.handle_unknown,
        }
    }

    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError> {
        Ok(FittedOneHotEncoder {
            categories_: params.categories_,
            n_values_: params.n_values_,
            n_features_out: params.n_features_out,
            n_features_in: params.n_features_in,
            handle_unknown: params.handle_unknown,
            _backend: PhantomData,
        })
    }

    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_one_hot_encoder_single_column() {
        // Input: [[0], [1], [2]]
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0], 3, 1);

        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        assert_eq!(fitted.n_features_in(), 1);
        assert_eq!(fitted.n_features_out(), 3);
        assert_eq!(fitted.categories()[0], vec![0.0f32, 1.0, 2.0]);

        let transformed = fitted.transform(&data).unwrap();
        let vals = transformed.ravel().to_vec();

        // Expected: [[1,0,0], [0,1,0], [0,0,1]]
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 0.0).abs() < 1e-6);
        assert!((vals[2] - 0.0).abs() < 1e-6);
        assert!((vals[3] - 0.0).abs() < 1e-6);
        assert!((vals[4] - 1.0).abs() < 1e-6);
        assert!((vals[5] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_one_hot_encoder_missing_category() {
        // Input: [[0], [2]] - category 1 is missing
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 2.0], 2, 1);

        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        // Should only have 2 categories
        assert_eq!(fitted.n_features_out(), 2);
        assert_eq!(fitted.categories()[0], vec![0.0f32, 2.0]);

        let transformed = fitted.transform(&data).unwrap();
        // [[1,0], [0,1]]
        let vals = transformed.ravel().to_vec();
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 0.0).abs() < 1e-6);
        assert!((vals[2] - 0.0).abs() < 1e-6);
        assert!((vals[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_one_hot_encoder_multiple_columns() {
        // Input: [[0, 1], [1, 0]]
        // Column 0 has categories [0, 1]
        // Column 1 has categories [0, 1]
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 1.0, 0.0], 2, 2);

        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        assert_eq!(fitted.n_features_out(), 4); // 2 + 2

        let transformed = fitted.transform(&data).unwrap();
        // Expected: [[1,0, 0,1], [0,1, 1,0]]
        let vals = transformed.ravel().to_vec();
        assert!((vals[0] - 1.0).abs() < 1e-6); // row 0, col 0 -> [1,0]
        assert!((vals[1] - 0.0).abs() < 1e-6);
        assert!((vals[2] - 0.0).abs() < 1e-6); // row 0, col 1 -> [0,1]
        assert!((vals[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_one_hot_encoder_unknown_error() {
        let train = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let test = Tensor2D::<CpuBackend>::new(vec![2.0f32], 1, 1); // unknown category

        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&train).unwrap();

        let result = fitted.transform(&test);
        assert!(result.is_err());
    }

    #[test]
    fn test_one_hot_encoder_unknown_ignore() {
        let train = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let test = Tensor2D::<CpuBackend>::new(vec![2.0f32], 1, 1); // unknown category

        let encoder = OneHotEncoder::<CpuBackend>::new().with_handle_unknown(HandleUnknown::Ignore);
        let fitted = encoder.fit(&train).unwrap();

        let transformed = fitted.transform(&test).unwrap();
        // Should output [0, 0] for unknown
        let vals = transformed.ravel().to_vec();
        assert!((vals[0] - 0.0).abs() < 1e-6);
        assert!((vals[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_one_hot_encoder_inverse() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0], 3, 1);

        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        let orig = data.ravel().to_vec();
        let rec = recovered.ravel().to_vec();

        for (o, r) in orig.iter().zip(rec.iter()) {
            assert!((o - r).abs() < 1e-6);
        }
    }

    #[test]
    fn test_one_hot_encoder_serialization() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0], 3, 1);

        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        let temp_file = std::env::temp_dir().join("test_onehot.bin");
        fitted.save_to_file(&temp_file).unwrap();

        let loaded = FittedOneHotEncoder::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert_eq!(loaded.n_features_in(), fitted.n_features_in());
        assert_eq!(loaded.n_features_out(), fitted.n_features_out());
        assert_eq!(loaded.categories(), fitted.categories());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_one_hot_encoder_empty_data() {
        let data = Tensor2D::<CpuBackend>::zeros(0, 2);

        let encoder = OneHotEncoder::<CpuBackend>::new();
        let result = encoder.fit(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_one_hot_encoder_feature_mismatch() {
        let train = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0, 3.0], 2, 2);
        let test = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0], 1, 3);

        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&train).unwrap();

        let result = fitted.transform(&test);
        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch { .. })
        ));
    }

    #[test]
    fn test_one_hot_encoder_n_values() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0, 0.0, 1.0, 0.0], 3, 2);

        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        assert_eq!(fitted.n_values(), vec![3, 2]); // col 0: 0,1,2 (3 cats), col 1: 0,1 (2 cats)
    }

    #[test]
    fn test_one_hot_encoder_with_handle_unknown() {
        let encoder = OneHotEncoder::<CpuBackend>::new().with_handle_unknown(HandleUnknown::Ignore);
        assert_eq!(encoder.handle_unknown, HandleUnknown::Ignore);
    }

    #[test]
    fn test_one_hot_encoder_invalid_value() {
        // Negative value should fail
        let data = Tensor2D::<CpuBackend>::new(vec![-1.0f32, 0.0, 1.0], 3, 1);
        let encoder = OneHotEncoder::<CpuBackend>::new();
        let result = encoder.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_one_hot_encoder_non_integer_value() {
        // Non-integer value should fail
        let data = Tensor2D::<CpuBackend>::new(vec![0.5f32, 1.0, 2.0], 3, 1);
        let encoder = OneHotEncoder::<CpuBackend>::new();
        let result = encoder.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_one_hot_encoder_empty_transform() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        // Transform empty data
        let empty = Tensor2D::<CpuBackend>::zeros(0, 1);
        let result = fitted.transform(&empty).unwrap();
        assert_eq!(result.shape(), (0, 2));
    }

    #[test]
    fn test_one_hot_encoder_inverse_empty() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        // Inverse transform empty data
        let empty = Tensor2D::<CpuBackend>::zeros(0, 2);
        let result = fitted.inverse_transform(&empty).unwrap();
        assert_eq!(result.shape(), (0, 1));
    }

    #[test]
    fn test_one_hot_encoder_fit_transform() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0], 3, 1);
        let encoder = OneHotEncoder::<CpuBackend>::new();
        let transformed = encoder.fit_transform(&data).unwrap();
        let vals = transformed.ravel().to_vec();

        assert_eq!(transformed.shape(), (3, 3));
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[4] - 1.0).abs() < 1e-6);
        assert!((vals[8] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_one_hot_encoder_inverse_nan_for_invalid() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        // Create a one-hot like vector with no active category
        let invalid = Tensor2D::<CpuBackend>::new(vec![0.0f32, 0.0], 1, 2);
        let result = fitted.inverse_transform(&invalid).unwrap();
        let vals = result.ravel().to_vec();

        // Should have NaN since no category was found
        assert!(vals[0].is_nan());
    }

    #[test]
    fn test_one_hot_encoder_inverse_feature_mismatch() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        // Wrong number of output features
        let wrong = Tensor2D::<CpuBackend>::new(vec![1.0f32, 0.0, 0.0], 1, 3);
        let result = fitted.inverse_transform(&wrong);

        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch { .. })
        ));
    }

    #[test]
    fn test_one_hot_encoder_extract_params() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0], 3, 1);
        let encoder = OneHotEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        let params = fitted.extract_params();
        assert_eq!(params.n_features_in, 1);
        assert_eq!(params.n_features_out, 3);
        assert_eq!(params.categories_[0], vec![0.0f32, 1.0, 2.0]);
    }

    #[test]
    fn test_one_hot_encoder_from_params() {
        let params = OneHotEncoderParams {
            categories_: vec![vec![0.0f32, 1.0]],
            n_values_: vec![2],
            n_features_out: 2,
            n_features_in: 1,
            handle_unknown: HandleUnknown::Error,
        };

        let fitted = FittedOneHotEncoder::<CpuBackend>::from_params(params).unwrap();
        assert_eq!(fitted.n_features_in(), 1);
        assert_eq!(fitted.n_features_out(), 2);
    }
}
