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
pub use crate::backend::backend::Backend;
pub use crate::backend::scalar::{Scalar, ScalarOps};
pub use crate::backend::tensor1d::Tensor1D;
pub use crate::backend::tensor2d::Tensor2D;
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
        let b = self.bias.clone() + other.bias.clone();
        Self {
            weights: w,
            bias: b,
        }
    }
    fn scale(&self, scalar: Scalar<B>) -> Self {
        let w = self.weights.scale(&scalar);
        let b = self.bias.clone() * scalar.clone();
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
        self.params.weights.dot(input) + self.params.bias.clone()
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
}
