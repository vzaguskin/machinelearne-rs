//! Linear models for regression and classification.
//!
//! This module implements a type-safe linear model with compile-time state tracking:
//! - [`LinearRegression`] = `LinearModel<Unfitted>` — used during training.
//! - [`LinearModel<Fitted>`] — inference-only, serializable predictor.
//!
//! The design follows [ADR-0001: separate-trainer-losses](https://github.com/vzaguskin/machinelearne-rs/issues/1):
//! > *"Fitted model is free from training hyperparameters."*
//!
//! Supports L1/L2 regularization via loss functions and works with any backend implementing [`Backend`].
//!
pub use crate::backend::scalar::{Scalar, ScalarOps};
pub use crate::backend::tensor1d::Tensor1D;
pub use crate::backend::tensor2d::Tensor2D;
pub use crate::backend::Backend;
use crate::loss::TensorLike;
pub use crate::model::{Fitted, InferenceModel, ParamOps, TrainableModel, Unfitted};
use std::marker::PhantomData;

/// Trainable parameters of a linear model: weights and bias.
///
/// Used internally by both [`TrainableModel`] and [`InferenceModel`] implementations.
/// Implements [`ParamOps`] to support optimizer updates (e.g., SGD).
#[derive(Clone)]
pub struct LinearParams<B: Backend>
where
    Tensor1D<B>: Clone,
    Scalar<B>: Clone,
{
    pub weights: Tensor1D<B>,
    pub bias: Scalar<B>,
}

/// Serializable representation of linear model parameters.
///
/// Converts internal backend-specific tensors into plain `Vec<f32>` for storage.
/// Used by [`InferenceModel::save_to_file`] and [`InferenceModel::load_from_file`].
///
/// ⚠️ Currently uses `f32` for compactness
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SerializableLinearParams {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl<B: Backend> From<&LinearParams<B>> for SerializableLinearParams {
    fn from(params: &LinearParams<B>) -> Self {
        // Предполагаем, что эти методы существуют в вашем бэкенде
        let weights = params
            .weights
            .to_vec()
            .into_iter()
            .map(|x| x as f32)
            .collect(); // Vec<f32>
        let bias = params.bias.data.to_f64() as f32; // f32
        Self { weights, bias }
    }
}

impl<B: Backend> TryFrom<SerializableLinearParams> for LinearParams<B> {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: SerializableLinearParams) -> Result<Self, Self::Error> {
        let weights = Tensor1D::<B>::new(value.weights);
        let bias = Scalar::<B>::new(value.bias as f64);
        Ok(Self { weights, bias })
    }
}

impl<B> ParamOps<B> for LinearParams<B>
where
    B: Backend,
{
    fn add(&self, other: &Self) -> Self {
        let w = self.weights.add(&other.weights);
        let b = self.bias + other.bias;
        Self {
            weights: w,
            bias: b,
        }
    }
    fn scale(&self, scalar: Scalar<B>) -> Self {
        let w = self.weights.scale(&scalar);
        let b = self.bias * scalar;
        Self {
            weights: w,
            bias: b,
        }
    }
}

/// A linear model with state encoded at the type level.
///
/// - When `S = Unfitted`: implements [`TrainableModel`] — used during training.
/// - When `S = Fitted`: implements [`InferenceModel`] — used for prediction and serialization.
///
/// This enforces, at compile time, that you cannot call `predict()` on an untrained model.
pub struct LinearModel<B: Backend, S> {
    params: LinearParams<B>,
    _state: std::marker::PhantomData<S>,
}

impl<B: Backend> LinearModel<B, Fitted> {
    /// Creates a new fitted linear model from trained parameters.
    ///
    /// Typically called internally by [`TrainableModel::into_fitted`].
    /// Useful for manual model construction or loading from external sources.
    pub fn new(params: LinearParams<B>) -> Self {
        Self {
            params,
            _state: std::marker::PhantomData::<Fitted>,
        }
    }
}

/// Implements inference for a trained linear model: `y = w^T x + b`.
///
/// - Single-sample input: [`Tensor1D<B>`] → output: [`Scalar<B>`]
/// - Batch input: [`Tensor2D<B>`] → output: [`Tensor1D<B>`]
///
/// Serialization uses [`SerializableLinearParams`] (see `save_to_file`/`load_from_file`).
impl<B: Backend> InferenceModel<B> for LinearModel<B, Fitted> {
    type InputSingle = Tensor1D<B>;
    type InputBatch = Tensor2D<B>;
    type OutputSingle = Scalar<B>;
    type OutputBatch = Tensor1D<B>;
    type ParamsRepr = SerializableLinearParams;

    /// Predict on a single sample (feature vector).
    fn predict(&self, input: &Self::InputSingle) -> Self::OutputSingle {
        self.params.weights.dot(input) + self.params.bias
    }

    fn predict_batch(&self, input: &Self::InputBatch) -> Self::OutputBatch {
        input
            .dot(&self.params.weights)
            .add_scalar(&self.params.bias)
    }

    fn extract_params(&self) -> Self::ParamsRepr {
        (&self.params).into()
    }

    fn from_params(params: Self::ParamsRepr) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        let internal_params = LinearParams::<B>::try_from(params)?;
        Ok(Self::new(internal_params))
    }
}

/// Implements training interface for linear regression.
///
/// Forward pass: `X @ w + b`  
/// Backward pass: computes gradients ∇w = X^T · grad, ∇b = sum(grad)
///
/// After training, convert to inference model via [`Self::into_fitted`].
impl<B: Backend> TrainableModel<B> for LinearModel<B, Unfitted> {
    type Params = LinearParams<B>;
    type Gradients = LinearParams<B>;
    type Prediction = Tensor1D<B>;
    type Input = Tensor2D<B>;
    type Output = LinearModel<B, Fitted>;

    fn forward(&self, x: &Self::Input) -> Self::Prediction {
        x.dot(&self.params.weights).add_scalar(&self.params.bias)
    }

    fn params(&self) -> &Self::Params {
        &self.params
    }

    fn update_params(&mut self, params: &Self::Params) {
        self.params = params.clone();
    }

    fn into_fitted(self) -> LinearModel<B, Fitted> {
        LinearModel::<B, Fitted>::new(self.params)
    }

    fn backward(&self, x: &Self::Input, grad_output: &Self::Prediction) -> Self::Gradients {
        let grad_weights = x.tdot(grad_output);
        let grad_bias = grad_output.sum();
        LinearParams {
            weights: grad_weights,
            bias: grad_bias,
        }
    }
}

/// Alias for an **unfitted** linear regression model.
///
/// Equivalent to `LinearModel<Unfitted>`. Use this type when constructing
/// a model for training with [`Trainer`].
pub type LinearRegression<B> = LinearModel<B, Unfitted>;

impl<B: Backend> LinearRegression<B> {
    // Creates a new linear regression model with zero-initialized weights.
    ///
    /// # Parameters
    /// - `n_features`: number of input features (dimensionality of `x`).
    pub fn new(n_features: usize) -> Self {
        let params = LinearParams {
            weights: Tensor1D::<B>::zeros(n_features),
            bias: Scalar::<B>::new(0.),
        };
        Self {
            params,
            _state: PhantomData,
        }
    }

    /// Constructs a model from explicit parameters (e.g., for testing or warm start).
    pub fn from_params(params: LinearParams<B>) -> Self {
        Self {
            params,
            _state: PhantomData,
        }
    }
}

// Convenient alias for CPU-based linear regression.
///
/// Example:
/// ```rust
/// use machinelearne_rs::model::linear::LinearRegressor;
/// let model = LinearRegressor::new(10); // 10 features
/// ```
pub type LinearRegressor = LinearRegression<crate::backend::CpuBackend>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    // === ParamOps Tests ===

    #[test]
    fn test_param_ops_add() {
        let p1 = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let p2 = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.5, 1.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };

        let result = p1.add(&p2);

        assert_eq!(result.weights.to_vec(), vec![1.5, 3.0]);
        assert_eq!(result.bias.data.to_f64(), 1.0);
    }

    #[test]
    fn test_param_ops_add_negative() {
        let p1 = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0]),
            bias: Scalar::<CpuBackend>::new(1.0),
        };
        let p2 = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![-0.5]),
            bias: Scalar::<CpuBackend>::new(-2.0),
        };

        let result = p1.add(&p2);

        assert_eq!(result.weights.to_vec(), vec![0.5]);
        assert_eq!(result.bias.data.to_f64(), -1.0);
    }

    #[test]
    fn test_param_ops_scale() {
        let p = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![2.0, 4.0]),
            bias: Scalar::<CpuBackend>::new(1.0),
        };

        let result = p.scale(Scalar::<CpuBackend>::new(0.5));

        assert_eq!(result.weights.to_vec(), vec![1.0, 2.0]);
        assert_eq!(result.bias.data.to_f64(), 0.5);
    }

    #[test]
    fn test_param_ops_scale_negative() {
        let p = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![2.0]),
            bias: Scalar::<CpuBackend>::new(1.0),
        };

        let result = p.scale(Scalar::<CpuBackend>::new(-1.0));

        assert_eq!(result.weights.to_vec(), vec![-2.0]);
        assert_eq!(result.bias.data.to_f64(), -1.0);
    }

    #[test]
    fn test_param_ops_scale_zero() {
        let p = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![5.0, 10.0]),
            bias: Scalar::<CpuBackend>::new(100.0),
        };

        let result = p.scale(Scalar::<CpuBackend>::new(0.0));

        assert_eq!(result.weights.to_vec(), vec![0.0, 0.0]);
        assert_eq!(result.bias.data.to_f64(), 0.0);
    }

    // === LinearRegression (Unfitted) Tests ===

    #[test]
    fn test_linear_regression_new_zero_initialized() {
        let model = LinearRegression::<CpuBackend>::new(3);

        let params = model.params();
        assert_eq!(params.weights.to_vec(), vec![0.0, 0.0, 0.0]);
        assert_eq!(params.bias.data.to_f64(), 0.0);
    }

    #[test]
    fn test_linear_regression_new_single_feature() {
        let model = LinearRegression::<CpuBackend>::new(1);

        assert_eq!(model.params().weights.to_vec(), vec![0.0]);
        assert_eq!(model.params().bias.data.to_f64(), 0.0);
    }

    #[test]
    fn test_linear_regression_new_large_features() {
        let n_features = 1000;
        let model = LinearRegression::<CpuBackend>::new(n_features);

        assert_eq!(model.params().weights.to_vec().len(), n_features);
        assert!(model.params().weights.to_vec().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_linear_regression_from_params() {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };

        let model = LinearRegression::<CpuBackend>::from_params(params.clone());

        assert_eq!(model.params().weights.to_vec(), params.weights.to_vec());
        assert_eq!(model.params().bias.data.to_f64(), params.bias.data.to_f64());
    }

    #[test]
    fn test_linear_regression_update_params() {
        let mut model = LinearRegression::<CpuBackend>::new(2);

        let new_params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };

        model.update_params(&new_params);

        assert_eq!(model.params().weights.to_vec(), vec![1.0, 2.0]);
        assert_eq!(model.params().bias.data.to_f64(), 0.5);
    }

    #[test]
    fn test_linear_regression_forward_with_zero_params() {
        let model = LinearRegression::<CpuBackend>::new(2);

        let x = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let pred = model.forward(&x);

        // With zero weights and bias: X @ w + b = 0 + 0 = 0
        assert_eq!(pred.to_vec(), vec![0.0, 0.0]);
    }

    #[test]
    fn test_linear_regression_forward_correctness() {
        // Create model with known weights
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![2.0, 3.0]),
            bias: Scalar::<CpuBackend>::new(1.0),
        };
        let model = LinearRegression::<CpuBackend>::from_params(params);

        // Input: [[1, 0], [0, 1]]
        // Expected: [[2*1 + 3*0 + 1], [2*0 + 3*1 + 1]] = [3, 4]
        let x = Tensor2D::<CpuBackend>::new(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let pred = model.forward(&x);

        assert_eq!(pred.to_vec(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_linear_regression_forward_batch() {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(3.0),
        };
        let model = LinearRegression::<CpuBackend>::from_params(params);

        // Input: [[1, 1], [2, 2]]
        // Expected: [[1*1 + 2*1 + 3], [1*2 + 2*2 + 3]] = [6, 9]
        let x = Tensor2D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0], 2, 2);
        let pred = model.forward(&x);

        assert_eq!(pred.to_vec(), vec![6.0, 9.0]);
    }

    #[test]
    fn test_linear_regression_backward_single_sample() {
        let model = LinearRegression::<CpuBackend>::new(1);
        let x = Tensor2D::<CpuBackend>::new(vec![2.0], 1, 1);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![0.5]);

        let grads = model.backward(&x, &grad_output);

        // grad_w = X^T @ grad = [2] * [0.5] = 1.0
        // grad_b = sum(grad) = 0.5
        assert_eq!(grads.weights.to_vec(), vec![1.0]);
        assert_eq!(grads.bias.data.to_f64(), 0.5);
    }

    #[test]
    fn test_linear_regression_backward_batch() {
        let model = LinearRegression::<CpuBackend>::new(2);
        // Input: [[1, 2], [3, 4]]
        let x = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        // grad_output: [0.5, 0.25]
        let grad_output = Tensor1D::<CpuBackend>::new(vec![0.5, 0.25]);

        let grads = model.backward(&x, &grad_output);

        // grad_w = X^T @ grad = [[1, 3], [2, 4]] @ [0.5, 0.25]^T
        //       = [1*0.5 + 3*0.25, 2*0.5 + 4*0.25]
        //       = [0.5 + 0.75, 1.0 + 1.0] = [1.25, 2.0]
        // grad_b = sum([0.5, 0.25]) = 0.75
        assert!((grads.weights.to_vec()[0] - 1.25).abs() < 1e-10);
        assert!((grads.weights.to_vec()[1] - 2.0).abs() < 1e-10);
        assert!((grads.bias.data.to_f64() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_linear_regression_backward_negative_gradients() {
        let model = LinearRegression::<CpuBackend>::new(1);
        let x = Tensor2D::<CpuBackend>::new(vec![1.0], 1, 1);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![-0.5]);

        let grads = model.backward(&x, &grad_output);

        assert_eq!(grads.weights.to_vec(), vec![-0.5]);
        assert_eq!(grads.bias.data.to_f64(), -0.5);
    }

    #[test]
    fn test_linear_regression_backward_zero_gradients() {
        let model = LinearRegression::<CpuBackend>::new(2);
        let x = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![0.0, 0.0]);

        let grads = model.backward(&x, &grad_output);

        assert_eq!(grads.weights.to_vec(), vec![0.0, 0.0]);
        assert_eq!(grads.bias.data.to_f64(), 0.0);
    }

    // === Fitted Model Tests ===

    #[test]
    fn test_linear_model_fitted_new() {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };

        let model = LinearModel::<CpuBackend, Fitted>::new(params);

        assert_eq!(model.params.weights.to_vec(), vec![1.0, 2.0]);
        assert_eq!(model.params.bias.data.to_f64(), 0.5);
    }

    #[test]
    fn test_linear_model_predict_single_sample() {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![2.0, 3.0]),
            bias: Scalar::<CpuBackend>::new(1.0),
        };
        let model = LinearModel::<CpuBackend, Fitted>::new(params);

        let input = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]);
        let pred = model.predict(&input);

        // y = 2*1 + 3*2 + 1 = 2 + 6 + 1 = 9
        assert_eq!(pred.data.to_f64(), 9.0);
    }

    #[test]
    fn test_linear_model_predict_batch() {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(3.0),
        };
        let model = LinearModel::<CpuBackend, Fitted>::new(params);

        let batch = Tensor2D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0], 2, 2);
        let pred = model.predict_batch(&batch);

        // y1 = 1*1 + 2*1 + 3 = 6
        // y2 = 1*2 + 2*2 + 3 = 9
        assert_eq!(pred.to_vec(), vec![6.0, 9.0]);
    }

    #[test]
    fn test_linear_model_predict_single_feature() {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![5.0]),
            bias: Scalar::<CpuBackend>::new(0.0),
        };
        let model = LinearModel::<CpuBackend, Fitted>::new(params);

        let input = Tensor1D::<CpuBackend>::new(vec![2.0]);
        let pred = model.predict(&input);

        assert_eq!(pred.data.to_f64(), 10.0);
    }

    #[test]
    fn test_linear_model_predict_zero_bias() {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 1.0]),
            bias: Scalar::<CpuBackend>::new(0.0),
        };
        let model = LinearModel::<CpuBackend, Fitted>::new(params);

        let input = Tensor1D::<CpuBackend>::new(vec![3.0, 4.0]);
        let pred = model.predict(&input);

        assert_eq!(pred.data.to_f64(), 7.0);
    }

    #[test]
    fn test_linear_model_predict_negative_weights() {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![-1.0, -2.0]),
            bias: Scalar::<CpuBackend>::new(5.0),
        };
        let model = LinearModel::<CpuBackend, Fitted>::new(params);

        let input = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0]);
        let pred = model.predict(&input);

        // y = -1*1 + -2*1 + 5 = 2
        assert_eq!(pred.data.to_f64(), 2.0);
    }

    #[test]
    fn test_linear_model_extract_params() {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.5, 2.5, 3.5]),
            bias: Scalar::<CpuBackend>::new(0.25),
        };
        let model = LinearModel::<CpuBackend, Fitted>::new(params);

        let extracted = model.extract_params();

        assert_eq!(extracted.weights.len(), 3);
        assert!((extracted.weights[0] - 1.5).abs() < 1e-6);
        assert!((extracted.weights[1] - 2.5).abs() < 1e-6);
        assert!((extracted.weights[2] - 3.5).abs() < 1e-6);
        assert!((extracted.bias - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_linear_model_from_params() -> Result<(), Box<dyn std::error::Error>> {
        let serial = SerializableLinearParams {
            weights: vec![1.0, 2.0, 3.0],
            bias: 0.5,
        };

        let model = LinearModel::<CpuBackend, Fitted>::from_params(serial)?;

        assert_eq!(model.params.weights.to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(model.params.bias.data.to_f64(), 0.5);

        Ok(())
    }

    #[test]
    fn test_linear_model_into_fitted() {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let unfitted_model = LinearRegression::<CpuBackend>::from_params(params);

        let fitted_model: LinearModel<CpuBackend, Fitted> = unfitted_model.into_fitted();

        assert_eq!(fitted_model.params.weights.to_vec(), vec![1.0]);
        assert_eq!(fitted_model.params.bias.data.to_f64(), 0.5);
    }

    #[test]
    fn test_linear_model_predict_does_not_mutate_input() {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 1.0]),
            bias: Scalar::<CpuBackend>::new(0.0),
        };
        let model = LinearModel::<CpuBackend, Fitted>::new(params);

        let input = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]);
        let original = input.clone();

        let _ = model.predict(&input);

        // Input should be unchanged
        assert_eq!(input.to_vec(), original.to_vec());
    }

    // === Serialization Tests ===

    #[test]
    fn test_linear_model_save_load() -> Result<(), Box<dyn std::error::Error>> {
        let weights = vec![1.0, 2.0, 3.0];
        let bias = 0.5;
        let serial = SerializableLinearParams { weights, bias };
        let params = LinearParams::<CpuBackend>::try_from(serial)?;
        let model = LinearModel::<CpuBackend, Fitted>::new(params);

        // Save
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("model.bin");
        model.save_to_file(&path)?;

        // Load
        let loaded = LinearModel::<CpuBackend, Fitted>::load_from_file(&path)?;

        // Compare
        let orig_repr = model.extract_params();
        let loaded_repr = loaded.extract_params();
        assert_eq!(orig_repr.weights, loaded_repr.weights);
        assert_eq!(orig_repr.bias, loaded_repr.bias);

        Ok(())
    }

    #[test]
    fn test_serialization_params_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let original = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.1, 0.2, 0.3]),
            bias: Scalar::<CpuBackend>::new(0.05),
        };

        // Convert to serializable
        let serial: SerializableLinearParams = (&original).into();

        // Convert back
        let restored = LinearParams::<CpuBackend>::try_from(serial)?;

        // Verify with tolerance due to f32/f64 conversion precision loss
        for (orig, rest) in original
            .weights
            .to_vec()
            .iter()
            .zip(restored.weights.to_vec().iter())
        {
            assert!((orig - rest).abs() < 1e-6);
        }
        assert!((original.bias.data.to_f64() - restored.bias.data.to_f64()).abs() < 1e-6);

        Ok(())
    }
}
