//! Polynomial feature generation.
//!
//! Generates polynomial and interaction features from input data.

use crate::backend::{Backend, Tensor2D};
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// PolynomialFeatures transformer for generating polynomial and interaction features.
///
/// Generates a new feature matrix consisting of all polynomial combinations
/// of the features with degree less than or equal to the specified degree.
///
/// For example, if an input sample is two dimensional and of the form [a, b],
/// the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
///
/// # Example
/// ```ignore
/// use machinelearne_rs::preprocessing::{PolynomialFeatures, Transformer};
/// use machinelearne_rs::backend::CpuBackend;
///
/// // Input: [[1, 2], [3, 4]]
/// let data = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
///
/// let poly = PolynomialFeatures::<CpuBackend>::new().with_degree(2);
/// let fitted = poly.fit(&data)?;
///
/// // Output: [[1, 1, 2, 1, 2, 4], [1, 3, 4, 9, 12, 16]]
/// let transformed = fitted.transform(&data)?;
/// ```
#[derive(Clone, Debug)]
pub struct PolynomialFeatures<B: Backend> {
    /// Maximum degree of polynomial features.
    degree: usize,
    /// If True, include a bias column (feature of all 1s).
    include_bias: bool,
    /// If True, only produce interaction features (products of distinct features).
    interaction_only: bool,
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for PolynomialFeatures<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> PolynomialFeatures<B> {
    /// Create a new PolynomialFeatures with default settings (degree=2).
    pub fn new() -> Self {
        Self {
            degree: 2,
            include_bias: true,
            interaction_only: false,
            _backend: PhantomData,
        }
    }

    /// Set the maximum degree of polynomial features.
    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set whether to include a bias column (all 1s).
    pub fn with_include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Set whether to only produce interaction features.
    pub fn with_interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }
}

/// Serializable parameters for fitted PolynomialFeatures.
#[derive(Clone, Serialize, Deserialize)]
pub struct PolynomialFeaturesParams {
    /// Maximum degree of polynomial features.
    pub degree: usize,
    /// If True, include a bias column.
    pub include_bias: bool,
    /// If True, only produce interaction features.
    pub interaction_only: bool,
    /// Number of input features.
    pub n_features_in: usize,
    /// Number of output features.
    pub n_features_out: usize,
    /// List of (degree, feature indices) combinations for output features.
    /// For example, [(0, [])] for bias, [(1, [0])] for first input feature,
    /// [(2, [0, 0])] for square of first feature, [(2, [0, 1])] for interaction.
    pub output_combinations: Vec<(usize, Vec<usize>)>,
}

/// Fitted PolynomialFeatures ready for inference.
#[derive(Clone)]
pub struct FittedPolynomialFeatures<B: Backend> {
    /// Maximum degree of polynomial features.
    degree: usize,
    /// If True, include a bias column.
    include_bias: bool,
    /// If True, only produce interaction features.
    interaction_only: bool,
    /// Number of input features.
    n_features_in: usize,
    /// Number of output features.
    n_features_out: usize,
    /// Output feature combinations (degree, indices).
    output_combinations: Vec<(usize, Vec<usize>)>,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedPolynomialFeatures<B> {
    /// Get the number of output features.
    pub fn n_features_out(&self) -> usize {
        self.n_features_out
    }

    /// Get the output feature combinations.
    pub fn output_combinations(&self) -> &[(usize, Vec<usize>)] {
        &self.output_combinations
    }
}

impl<B: Backend> Transformer<B> for PolynomialFeatures<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = PolynomialFeaturesParams;
    type Fitted = FittedPolynomialFeatures<B>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError> {
        let (rows, cols) = data.shape();

        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit PolynomialFeatures on empty data".to_string(),
            ));
        }

        if cols == 0 {
            return Err(PreprocessingError::InvalidParameter(
                "Cannot fit PolynomialFeatures on data with no features".to_string(),
            ));
        }

        // Generate output combinations
        let output_combinations = generate_polynomial_combinations(
            cols,
            self.degree,
            self.include_bias,
            self.interaction_only,
        );

        let n_features_out = output_combinations.len();

        Ok(FittedPolynomialFeatures {
            degree: self.degree,
            include_bias: self.include_bias,
            interaction_only: self.interaction_only,
            n_features_in: cols,
            n_features_out,
            output_combinations,
            _backend: PhantomData,
        })
    }

    fn fit_transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let fitted = self.fit(data)?;
        fitted.transform(data)
    }
}

impl<B: Backend> FittedTransformer<B> for FittedPolynomialFeatures<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = PolynomialFeaturesParams;

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
        let mut result = Vec::with_capacity(rows * self.n_features_out);

        for row in 0..rows {
            let row_start = row * cols;
            let row_data: Vec<f64> = (0..cols).map(|c| data_vec[row_start + c]).collect();

            for (_degree, indices) in &self.output_combinations {
                let mut val = 1.0f64;
                for &idx in indices {
                    val *= row_data[idx];
                }
                result.push(val);
            }
        }

        Ok(Tensor2D::new(
            result.iter().map(|&x| x as f32).collect(),
            rows,
            self.n_features_out,
        ))
    }

    fn inverse_transform(&self, _data: &Self::Output) -> Result<Self::Input, PreprocessingError> {
        Err(PreprocessingError::InvalidParameter(
            "PolynomialFeatures does not support inverse_transform".to_string(),
        ))
    }

    fn extract_params(&self) -> Self::Params {
        PolynomialFeaturesParams {
            degree: self.degree,
            include_bias: self.include_bias,
            interaction_only: self.interaction_only,
            n_features_in: self.n_features_in,
            n_features_out: self.n_features_out,
            output_combinations: self.output_combinations.clone(),
        }
    }

    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError> {
        Ok(FittedPolynomialFeatures {
            degree: params.degree,
            include_bias: params.include_bias,
            interaction_only: params.interaction_only,
            n_features_in: params.n_features_in,
            n_features_out: params.n_features_out,
            output_combinations: params.output_combinations,
            _backend: PhantomData,
        })
    }

    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

/// Generate all polynomial combinations up to given degree.
fn generate_polynomial_combinations(
    n_features: usize,
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
) -> Vec<(usize, Vec<usize>)> {
    let mut combinations = Vec::new();

    // Bias term (degree 0)
    if include_bias {
        combinations.push((0, Vec::new()));
    }

    // Generate combinations for degrees 1 to max_degree
    for d in 1..=degree {
        generate_degree_combinations(
            n_features,
            d,
            d,
            interaction_only,
            &mut Vec::new(),
            &mut combinations,
        );
    }

    combinations
}

/// Recursively generate combinations for a specific degree.
fn generate_degree_combinations(
    n_features: usize,
    target_degree: usize,
    remaining_degree: usize,
    interaction_only: bool,
    current: &mut Vec<usize>,
    result: &mut Vec<(usize, Vec<usize>)>,
) {
    if remaining_degree == 0 {
        // Check interaction_only constraint
        if interaction_only {
            // All indices must be distinct for interaction_only
            let mut sorted = current.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() != current.len() {
                return; // Has duplicates, skip
            }
        }
        result.push((target_degree, current.clone()));
        return;
    }

    // Determine starting index to avoid duplicates and maintain ordering
    let start = if current.is_empty() {
        0
    } else {
        *current.last().unwrap()
    };

    for i in start..n_features {
        current.push(i);
        generate_degree_combinations(
            n_features,
            target_degree,
            remaining_degree - 1,
            interaction_only,
            current,
            result,
        );
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_polynomial_features_basic() {
        // Input: [[1, 2]] -> [1, a, b, a^2, ab, b^2]
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);

        let poly = PolynomialFeatures::<CpuBackend>::new()
            .with_degree(2)
            .with_include_bias(true);
        let fitted = poly.fit(&data).unwrap();

        assert_eq!(fitted.n_features_in(), 2);
        assert_eq!(fitted.n_features_out(), 6); // bias + 2 linear + 3 quadratic

        let transformed = fitted.transform(&data).unwrap();
        let vals = transformed.ravel().to_vec();

        // [1, 1, 2, 1, 2, 4]
        assert!((vals[0] - 1.0).abs() < 1e-6); // bias
        assert!((vals[1] - 1.0).abs() < 1e-6); // a
        assert!((vals[2] - 2.0).abs() < 1e-6); // b
        assert!((vals[3] - 1.0).abs() < 1e-6); // a^2
        assert!((vals[4] - 2.0).abs() < 1e-6); // ab
        assert!((vals[5] - 4.0).abs() < 1e-6); // b^2
    }

    #[test]
    fn test_polynomial_features_no_bias() {
        // Input: [[1, 2]] -> [a, b, a^2, ab, b^2]
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);

        let poly = PolynomialFeatures::<CpuBackend>::new()
            .with_degree(2)
            .with_include_bias(false);
        let fitted = poly.fit(&data).unwrap();

        assert_eq!(fitted.n_features_out(), 5); // 2 linear + 3 quadratic

        let transformed = fitted.transform(&data).unwrap();
        let vals = transformed.ravel().to_vec();

        // [1, 2, 1, 2, 4]
        assert!((vals[0] - 1.0).abs() < 1e-6); // a
        assert!((vals[1] - 2.0).abs() < 1e-6); // b
        assert!((vals[2] - 1.0).abs() < 1e-6); // a^2
        assert!((vals[3] - 2.0).abs() < 1e-6); // ab
        assert!((vals[4] - 4.0).abs() < 1e-6); // b^2
    }

    #[test]
    fn test_polynomial_features_interaction_only() {
        // Input: [[1, 2, 3]] -> [a, b, c, ab, ac, bc]
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 1, 3);

        let poly = PolynomialFeatures::<CpuBackend>::new()
            .with_degree(2)
            .with_include_bias(false)
            .with_interaction_only(true);
        let fitted = poly.fit(&data).unwrap();

        // 3 linear + 3 interaction (no a^2, b^2, c^2)
        assert_eq!(fitted.n_features_out(), 6);

        let transformed = fitted.transform(&data).unwrap();
        let vals = transformed.ravel().to_vec();

        // [1, 2, 3, 2, 3, 6]
        assert!((vals[0] - 1.0).abs() < 1e-6); // a
        assert!((vals[1] - 2.0).abs() < 1e-6); // b
        assert!((vals[2] - 3.0).abs() < 1e-6); // c
        assert!((vals[3] - 2.0).abs() < 1e-6); // ab
        assert!((vals[4] - 3.0).abs() < 1e-6); // ac
        assert!((vals[5] - 6.0).abs() < 1e-6); // bc
    }

    #[test]
    fn test_polynomial_features_degree_3() {
        // Input: [[1, 2]]
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);

        let poly = PolynomialFeatures::<CpuBackend>::new()
            .with_degree(3)
            .with_include_bias(true);
        let fitted = poly.fit(&data).unwrap();

        // bias + 2 linear + 3 quadratic + 4 cubic = 10
        assert_eq!(fitted.n_features_out(), 10);

        let transformed = fitted.transform(&data).unwrap();
        let vals = transformed.ravel().to_vec();

        // [1, 1, 2, 1, 2, 4, 1, 2, 4, 8]
        // degree 0: 1
        // degree 1: a, b
        // degree 2: a^2, ab, b^2
        // degree 3: a^3, a^2*b, a*b^2, b^3
        assert!((vals[0] - 1.0).abs() < 1e-6); // bias
        assert!((vals[6] - 1.0).abs() < 1e-6); // a^3
        assert!((vals[9] - 8.0).abs() < 1e-6); // b^3
    }

    #[test]
    fn test_polynomial_features_multiple_rows() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);

        let poly = PolynomialFeatures::<CpuBackend>::new()
            .with_degree(2)
            .with_include_bias(true);
        let fitted = poly.fit(&data).unwrap();

        let transformed = fitted.transform(&data).unwrap();
        assert_eq!(transformed.shape(), (2, 6));

        let vals = transformed.ravel().to_vec();

        // Row 0: [1, 1, 2, 1, 2, 4]
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[4] - 2.0).abs() < 1e-6);
        assert!((vals[5] - 4.0).abs() < 1e-6);

        // Row 1: [1, 3, 4, 9, 12, 16]
        assert!((vals[6] - 1.0).abs() < 1e-6);
        assert!((vals[10] - 12.0).abs() < 1e-6);
        assert!((vals[11] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_polynomial_features_serialization() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);

        let poly = PolynomialFeatures::<CpuBackend>::new()
            .with_degree(2)
            .with_include_bias(true);
        let fitted = poly.fit(&data).unwrap();

        let temp_file = std::env::temp_dir().join("test_poly.bin");
        fitted.save_to_file(&temp_file).unwrap();

        let loaded = FittedPolynomialFeatures::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert_eq!(loaded.n_features_in(), fitted.n_features_in());
        assert_eq!(loaded.n_features_out(), fitted.n_features_out());

        let t1 = fitted.transform(&data).unwrap();
        let t2 = loaded.transform(&data).unwrap();

        let v1 = t1.ravel().to_vec();
        let v2 = t2.ravel().to_vec();
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(temp_file).ok();
    }
}
