//! Core model abstractions: training vs inference separation.
//!
//! This module defines the two central traits:
//! - [`TrainableModel`]: used during training; owns mutable parameters and computes gradients.
//! - [`InferenceModel`]: lightweight, serializable, stateless predictor for production use.
//!
//! This design ensures that a fitted model carries **only** what is needed for prediction,
//! with no optimizer state, loss functions, or training hyperparameters.
//!
//! See [`linear::LinearRegressor`] for a concrete example.

pub mod state;
pub use state::{Fitted, Unfitted};

pub mod linear;
pub use crate::backend::backend::Backend;
pub use crate::backend::scalar::{Scalar, ScalarOps};
use crate::serialization::SerializableParams;

/// A model that can be trained: it computes forward passes, gradients, and updates its parameters.
///
/// This trait is used **only during training**. After training, it is converted into an
/// [`InferenceModel`] via [`Self::into_fitted`], which strips away all training-related state.
///
/// # Type Parameters
/// - `B`: The backend (e.g., `CpuBackend`) used for computation.
/// - `Input`: Input data type (e.g., `Tensor1D<B>`).
/// - `Prediction`: Output of the forward pass (e.g., scalar for regression).
/// - `Params`: Internal trainable parameters (e.g., weights + bias).
/// - `Gradients`: Gradient structure matching `Params`.
/// - `Output`: The corresponding [`InferenceModel`] type.
///
/// # Safety & Invariants
/// - `backward` must be called **after** `forward` with the same input.
/// - `update_params` must preserve the shape/structure of parameters.
///
/// # Example
/// See [`linear::LinearRegressor`] for a full implementation.
pub trait TrainableModel<B: Backend> {
    type Input;
    type Prediction;
    type Params;
    type Gradients;
    type Output;

    fn forward(&self, input: &Self::Input) -> Self::Prediction;
    fn backward(&self, input: &Self::Input, grad_output: &Self::Prediction) -> Self::Gradients;
    fn params(&self) -> &Self::Params;
    fn update_params(&mut self, new_params: &Self::Params);

    fn into_fitted(self) -> Self::Output;
}
/// Operations required to update model parameters during optimization.
///
/// Optimizers like SGD rely on these operations to compute weight updates:
/// ```text
/// w_new = w_old + (-lr) * grad
/// ```
///
/// Implementations must be **element-wise** and preserve parameter structure.
///
pub trait ParamOps<B: Backend>: Clone {
    fn add(&self, other: &Self) -> Self;
    fn scale(&self, scalar: Scalar<B>) -> Self;
}
/// A lightweight, serializable model for inference only.
///
/// This trait represents the **final product** of training: it contains no optimizer state,
/// loss functions, or batch metadata. It is designed for:
/// - Fast prediction (`predict`, `predict_batch`)
/// - Serialization (`save_to_file`, `load_from_file`)
/// - Deployment in production environments
///
/// # Type Parameters
/// - `InputSingle` / `OutputSingle`: types for single-sample prediction.
/// - `InputBatch` / `OutputBatch`: types for batched prediction (optional optimization).
/// - `ParamsRepr`: a serializable representation of internal parameters (e.g., struct with `Vec<f64>`).
///
/// # Guarantees
/// - `extract_params()` + `from_params()` is a round-trip.
/// - `save_to_file` / `load_from_file` are compatible across platforms (uses `bincode` internally).
///
/// # Example
/// ```rust
/// use machinelearne_rs::{CpuBackend,
///                         Tensor1D,
///                         model::linear::LinearModel,
///                         model::linear::LinearParams,
///                         model::linear::Fitted,
///                         model::InferenceModel,
///                         backend::Scalar};
/// let weights = Tensor1D::<CpuBackend>::new(vec![3.0f32, 4.0]);
/// let bias = Scalar::<CpuBackend>::new(1.0);
/// let params = LinearParams { weights, bias };

/// let model = LinearModel::<CpuBackend, Fitted>::from_params((&params).into()).unwrap();
/// let input = Tensor1D::<CpuBackend>::new(vec![3.0, 4.0]);
/// let pred = model.predict(&input); // ≈ 1*3 + 2*4 + 0.5 = 11.5
/// ```
pub trait InferenceModel<B: Backend> {
    type InputSingle;
    type OutputSingle;
    type InputBatch;
    type OutputBatch;
    type ParamsRepr: SerializableParams;
    fn predict(&self, input: &Self::InputSingle) -> Self::OutputSingle;
    fn predict_batch(&self, input: &Self::InputBatch) -> Self::OutputBatch;
    fn extract_params(&self) -> Self::ParamsRepr;
    fn from_params(params: Self::ParamsRepr) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;

    fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let bytes = self
            .extract_params()
            .to_bytes()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, bytes)
    }

    fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        let bytes = std::fs::read(path)?;
        let params = Self::ParamsRepr::from_bytes(&bytes)
            .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
        Self::from_params(params)
    }
}
