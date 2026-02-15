//! Ordinal encoding for categorical features.
//!
//! Maps categorical values to integer ordinals (0, 1, 2, ...).

use crate::backend::{Backend, Tensor2D};
use crate::preprocessing::encoding::HandleUnknown;
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Ordinal encoder for categorical features.
///
/// Maps each unique category to an integer ordinal (0, 1, 2, ...).
/// The mapping is learned from the training data, with categories
/// sorted in ascending order.
///
/// # Example
/// ```ignore
/// use machinelearne_rs::preprocessing::{OrdinalEncoder, Transformer};
/// use machinelearne_rs::backend::CpuBackend;
///
/// // Input: [[0], [2], [1]]  (categories: 0, 1, 2)
/// let data = Tensor2D::new(vec![0.0, 2.0, 1.0], 3, 1);
///
/// let encoder = OrdinalEncoder::<CpuBackend>::new();
/// let fitted = encoder.fit(&data)?;
///
/// // Output: [[0], [2], [1]]  (ordinal mapping)
/// let encoded = fitted.transform(&data)?;
/// ```
#[derive(Clone, Debug)]
pub struct OrdinalEncoder<B: Backend> {
    /// How to handle unknown categories during transform.
    handle_unknown: HandleUnknown,
    _backend: PhantomData<B>,
}

impl<B: Backend> OrdinalEncoder<B> {
    /// Create a new OrdinalEncoder with default settings.
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

impl<B: Backend> Default for OrdinalEncoder<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable parameters for a fitted OrdinalEncoder.
#[derive(Clone, Serialize, Deserialize)]
pub struct OrdinalEncoderParams {
    /// Categories (unique sorted values) for each input column.
    pub categories_: Vec<Vec<f32>>,
    /// Mapping from category value to ordinal for each column.
    /// Stored as (category, ordinal) pairs for serialization.
    pub mappings_: Vec<Vec<(f32, usize)>>,
    /// Number of input features.
    pub n_features_in: usize,
    /// Handle unknown strategy.
    pub handle_unknown: HandleUnknown,
}

// Note: We rely on the blanket impl of SerializableParams for Serialize + Deserialize

/// Fitted OrdinalEncoder ready for inference.
#[derive(Clone)]
pub struct FittedOrdinalEncoder<B: Backend> {
    /// Categories (unique sorted values) for each input column.
    categories_: Vec<Vec<f32>>,
    /// Mapping from category value to ordinal index for each column.
    mappings_: Vec<HashMap<i32, usize>>,
    /// Number of input features.
    n_features_in: usize,
    /// Handle unknown strategy.
    handle_unknown: HandleUnknown,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedOrdinalEncoder<B> {
    /// Get the categories learned for each feature.
    pub fn categories(&self) -> &[Vec<f32>] {
        &self.categories_
    }

    /// Get the mapping (category -> ordinal) for a specific feature.
    pub fn mapping(&self, feature_idx: usize) -> Option<&HashMap<i32, usize>> {
        self.mappings_.get(feature_idx)
    }
}

impl<B: Backend> Transformer<B> for OrdinalEncoder<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = OrdinalEncoderParams;
    type Fitted = FittedOrdinalEncoder<B>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError> {
        let (rows, cols) = data.shape();

        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit OrdinalEncoder on empty data".to_string(),
            ));
        }

        let data_vec = data.ravel().to_vec();

        // Find unique categories for each column
        let mut categories_: Vec<Vec<f32>> = Vec::with_capacity(cols);
        let mut mappings_: Vec<HashMap<i32, usize>> = Vec::with_capacity(cols);

        for col in 0..cols {
            let mut col_cats: std::collections::HashSet<i32> = std::collections::HashSet::new();
            for row in 0..rows {
                let val = data_vec[row * cols + col];
                if !val.is_finite() {
                    return Err(PreprocessingError::InvalidParameter(format!(
                        "OrdinalEncoder expects finite values, got {} at ({}, {})",
                        val, row, col
                    )));
                }
                // For ordinal encoding, we allow non-integer values too
                // (they'll be rounded to i32 for the mapping key)
                col_cats.insert(val.round() as i32);
            }

            // Sort categories
            let mut sorted_cats: Vec<i32> = col_cats.into_iter().collect();
            sorted_cats.sort();

            // Create mapping
            let mut mapping = HashMap::new();
            for (idx, &cat) in sorted_cats.iter().enumerate() {
                mapping.insert(cat, idx);
            }

            categories_.push(sorted_cats.into_iter().map(|x| x as f32).collect());
            mappings_.push(mapping);
        }

        Ok(FittedOrdinalEncoder {
            categories_,
            mappings_,
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

impl<B: Backend> FittedTransformer<B> for FittedOrdinalEncoder<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = OrdinalEncoderParams;

    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let (rows, cols) = data.shape();

        if cols != self.n_features_in {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features_in,
                got_features: cols,
            });
        }

        if rows == 0 {
            return Ok(Tensor2D::zeros(0, cols));
        }

        let data_vec = data.ravel().to_vec();
        let mut result = vec![0.0f64; rows * cols];

        for row in 0..rows {
            for col in 0..cols {
                let val = data_vec[row * cols + col];
                let key = val.round() as i32;

                match self.mappings_[col].get(&key) {
                    Some(&ordinal) => {
                        result[row * cols + col] = ordinal as f64;
                    }
                    None => {
                        if self.handle_unknown == HandleUnknown::Error {
                            return Err(PreprocessingError::InvalidParameter(format!(
                                "Unknown category {} in column {}",
                                val, col
                            )));
                        }
                        // With Ignore, output NaN
                        result[row * cols + col] = f64::NAN;
                    }
                }
            }
        }

        Ok(Tensor2D::new(
            result.iter().map(|&x| x as f32).collect(),
            rows,
            cols,
        ))
    }

    fn inverse_transform(&self, data: &Self::Output) -> Result<Self::Input, PreprocessingError> {
        let (rows, cols) = data.shape();

        if cols != self.n_features_in {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features_in,
                got_features: cols,
            });
        }

        let data_vec = data.ravel().to_vec();
        let mut result = vec![0.0f64; rows * cols];

        for row in 0..rows {
            for col in 0..cols {
                let ordinal = data_vec[row * cols + col] as usize;
                if ordinal < self.categories_[col].len() {
                    result[row * cols + col] = self.categories_[col][ordinal] as f64;
                } else {
                    result[row * cols + col] = f64::NAN;
                }
            }
        }

        Ok(Tensor2D::new(
            result.iter().map(|&x| x as f32).collect(),
            rows,
            cols,
        ))
    }

    fn extract_params(&self) -> Self::Params {
        // Convert HashMaps to Vec<(f32, usize)> for serialization
        let mappings_: Vec<Vec<(f32, usize)>> = self
            .mappings_
            .iter()
            .map(|m| {
                let mut pairs: Vec<(f32, usize)> = m.iter().map(|(&k, &v)| (k as f32, v)).collect();
                pairs.sort_by_key(|&(_, v)| v);
                pairs
            })
            .collect();

        OrdinalEncoderParams {
            categories_: self.categories_.clone(),
            mappings_,
            n_features_in: self.n_features_in,
            handle_unknown: self.handle_unknown,
        }
    }

    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError> {
        // Convert Vec<(f32, usize)> back to HashMap
        let mappings_: Vec<HashMap<i32, usize>> = params
            .mappings_
            .iter()
            .map(|pairs| pairs.iter().map(|&(k, v)| (k.round() as i32, v)).collect())
            .collect();

        Ok(FittedOrdinalEncoder {
            categories_: params.categories_,
            mappings_,
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
    fn test_ordinal_encoder_basic() {
        // Input: [[0], [2], [1]]
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 2.0, 1.0], 3, 1);

        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        assert_eq!(fitted.n_features_in(), 1);
        assert_eq!(fitted.categories()[0], vec![0.0f32, 1.0, 2.0]);

        let transformed = fitted.transform(&data).unwrap();
        let vals = transformed.ravel().to_vec();

        // Categories sorted: 0->0, 1->1, 2->2
        assert!((vals[0] - 0.0).abs() < 1e-6); // 0 -> 0
        assert!((vals[1] - 2.0).abs() < 1e-6); // 2 -> 2
        assert!((vals[2] - 1.0).abs() < 1e-6); // 1 -> 1
    }

    #[test]
    fn test_ordinal_encoder_non_contiguous() {
        // Input: [[0], [5], [10]]
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 5.0, 10.0], 3, 1);

        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        assert_eq!(fitted.categories()[0], vec![0.0f32, 5.0, 10.0]);

        let transformed = fitted.transform(&data).unwrap();
        let vals = transformed.ravel().to_vec();

        // Ordinals: 0->0, 5->1, 10->2
        assert!((vals[0] - 0.0).abs() < 1e-6);
        assert!((vals[1] - 1.0).abs() < 1e-6);
        assert!((vals[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_ordinal_encoder_inverse() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 5.0, 10.0], 3, 1);

        let encoder = OrdinalEncoder::<CpuBackend>::new();
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
    fn test_ordinal_encoder_unknown_error() {
        let train = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let test = Tensor2D::<CpuBackend>::new(vec![2.0f32], 1, 1);

        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&train).unwrap();

        let result = fitted.transform(&test);
        assert!(result.is_err());
    }

    #[test]
    fn test_ordinal_encoder_unknown_ignore() {
        let train = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let test = Tensor2D::<CpuBackend>::new(vec![2.0f32], 1, 1);

        let encoder =
            OrdinalEncoder::<CpuBackend>::new().with_handle_unknown(HandleUnknown::Ignore);
        let fitted = encoder.fit(&train).unwrap();

        let transformed = fitted.transform(&test).unwrap();
        let vals = transformed.ravel().to_vec();
        assert!(vals[0].is_nan());
    }

    #[test]
    fn test_ordinal_encoder_serialization() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 5.0, 10.0], 3, 1);

        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        let temp_file = std::env::temp_dir().join("test_ordinal.bin");
        fitted.save_to_file(&temp_file).unwrap();

        let loaded = FittedOrdinalEncoder::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert_eq!(loaded.n_features_in(), fitted.n_features_in());
        assert_eq!(loaded.categories(), fitted.categories());

        // Verify transform gives same result
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
    fn test_ordinal_encoder_empty_data() {
        let data = Tensor2D::<CpuBackend>::zeros(0, 2);

        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let result = encoder.fit(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_ordinal_encoder_feature_mismatch() {
        let train = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0, 3.0], 2, 2);
        let test = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0], 1, 3);

        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&train).unwrap();

        let result = fitted.transform(&test);
        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch { .. })
        ));
    }

    #[test]
    fn test_ordinal_encoder_with_handle_unknown() {
        let encoder =
            OrdinalEncoder::<CpuBackend>::new().with_handle_unknown(HandleUnknown::Ignore);
        assert_eq!(encoder.handle_unknown, HandleUnknown::Ignore);
    }

    #[test]
    fn test_ordinal_encoder_n_features_in() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0, 3.0], 2, 2);

        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        assert_eq!(fitted.n_features_in(), 2);
    }

    #[test]
    fn test_ordinal_encoder_fit_transform() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 2.0, 1.0], 3, 1);
        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let transformed = encoder.fit_transform(&data).unwrap();
        let vals = transformed.ravel().to_vec();

        assert!((vals[0] - 0.0).abs() < 1e-6);
        assert!((vals[1] - 2.0).abs() < 1e-6);
        assert!((vals[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ordinal_encoder_empty_transform() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        // Transform empty data
        let empty = Tensor2D::<CpuBackend>::zeros(0, 1);
        let result = fitted.transform(&empty).unwrap();
        assert_eq!(result.shape(), (0, 1));
    }

    #[test]
    fn test_ordinal_encoder_mapping() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 5.0, 10.0], 3, 1);
        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        let mapping = fitted.mapping(0).unwrap();
        assert_eq!(mapping.get(&0), Some(&0));
        assert_eq!(mapping.get(&5), Some(&1));
        assert_eq!(mapping.get(&10), Some(&2));
    }

    #[test]
    fn test_ordinal_encoder_invalid_value() {
        // NaN should fail
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, f32::NAN, 1.0], 3, 1);
        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let result = encoder.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_ordinal_encoder_extract_params() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 5.0, 10.0], 3, 1);
        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        let params = fitted.extract_params();
        assert_eq!(params.n_features_in, 1);
        assert_eq!(params.categories_[0], vec![0.0f32, 5.0, 10.0]);
    }

    #[test]
    fn test_ordinal_encoder_from_params() {
        let params = OrdinalEncoderParams {
            categories_: vec![vec![0.0f32, 1.0, 2.0]],
            mappings_: vec![vec![(0.0, 0), (1.0, 1), (2.0, 2)]],
            n_features_in: 1,
            handle_unknown: HandleUnknown::Error,
        };

        let fitted = FittedOrdinalEncoder::<CpuBackend>::from_params(params).unwrap();
        assert_eq!(fitted.n_features_in(), 1);
        assert_eq!(fitted.categories()[0], vec![0.0f32, 1.0, 2.0]);
    }

    #[test]
    fn test_ordinal_encoder_inverse_out_of_bounds() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        // Index 5 is out of bounds (only 2 categories)
        let bad_indices = Tensor2D::<CpuBackend>::new(vec![5.0f32], 1, 1);
        let result = fitted.inverse_transform(&bad_indices).unwrap();
        let vals = result.ravel().to_vec();

        // Should be NaN for out of bounds index
        assert!(vals[0].is_nan());
    }

    #[test]
    fn test_ordinal_encoder_inverse_feature_mismatch() {
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let encoder = OrdinalEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&data).unwrap();

        let wrong = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0], 1, 3);
        let result = fitted.inverse_transform(&wrong);

        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch { .. })
        ));
    }

    #[test]
    fn test_ordinal_encoder_default() {
        let encoder = OrdinalEncoder::<CpuBackend>::default();
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 2, 1);
        let result = encoder.fit(&data);
        assert!(result.is_ok());
    }
}
