//! Core traits for preprocessing transformers.
//!
//! This module defines the two central traits:
//! - [`Transformer`]: Used during fitting; has hyperparameters and can learn from data.
//! - [`FittedTransformer`]: After fitting; ready for inference and serialization.

use crate::backend::Backend;
use crate::preprocessing::error::PreprocessingError;
use crate::serialization::SerializableParams;

/// Trait for unfitted transformers with hyperparameters.
///
/// A transformer learns parameters from training data and can then transform
/// new data using those learned parameters. This trait represents the
/// configurable, unfitted state.
///
/// # Type Parameters
/// - `B`: The backend (e.g., `CpuBackend`) used for computation.
/// - `Input`: Input data type (typically `Tensor2D<B>`).
/// - `Output`: Output data type (typically `Tensor2D<B>`).
/// - `Params`: Serializable representation of learned parameters.
/// - `Fitted`: The corresponding fitted transformer type.
///
/// # Example
/// ```ignore
/// use machinelearne_rs::preprocessing::{Transformer, StandardScaler};
/// use machinelearne_rs::backend::CpuBackend;
///
/// let scaler = StandardScaler::<CpuBackend>::new();
/// let fitted = scaler.fit(&data)?;
/// let transformed = fitted.transform(&new_data)?;
/// ```
pub trait Transformer<B: Backend>: Clone {
    /// Input data type for transformation.
    type Input;
    /// Output data type after transformation.
    type Output;
    /// Serializable representation of learned parameters.
    type Params: SerializableParams;
    /// The fitted transformer type ready for inference.
    type Fitted: FittedTransformer<
        B,
        Params = Self::Params,
        Input = Self::Input,
        Output = Self::Output,
    >;

    /// Fit the transformer to the training data.
    ///
    /// Learns parameters (e.g., mean and std for StandardScaler) from the data.
    ///
    /// # Errors
    /// Returns [`PreprocessingError`] if:
    /// - Data is empty
    /// - Data contains invalid values (NaN, Inf)
    /// - Shape is incompatible with the transformer
    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError>;

    /// Fit the transformer and transform the data in one step.
    ///
    /// This is often more efficient than calling `fit` followed by `transform`,
    /// especially for transformers that can compute the transform during fitting.
    fn fit_transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError>;
}

/// Trait for fitted transformers ready for inference.
///
/// After fitting, a transformer contains learned parameters (e.g., mean_, std_
/// for StandardScaler) and can transform new data. It can also be serialized
/// and deserialized for deployment.
///
/// # Type Parameters
/// - `B`: The backend used for computation.
/// - `Params`: Serializable representation of learned parameters.
///
/// # Guarantees
/// - `extract_params()` + `from_params()` is a round-trip.
/// - `save_to_file` / `load_from_file` are cross-platform compatible.
pub trait FittedTransformer<B: Backend>: Clone {
    /// Input data type for transformation.
    type Input;
    /// Output data type after transformation.
    type Output;
    /// Serializable representation of learned parameters.
    type Params: SerializableParams;

    /// Transform data using learned parameters.
    ///
    /// # Errors
    /// Returns [`PreprocessingError`] if:
    /// - Input shape doesn't match expected number of features
    /// - Input contains invalid values
    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError>;

    /// Reverse the transformation (if supported).
    ///
    /// Not all transformers support inverse transformation.
    /// For example, OneHotEncoder cannot be inverted.
    ///
    /// # Errors
    /// Returns [`PreprocessingError`] if inverse transform is not supported
    /// or if the data cannot be inverse transformed.
    fn inverse_transform(&self, data: &Self::Output) -> Result<Self::Input, PreprocessingError>;

    /// Extract learned parameters as a serializable representation.
    fn extract_params(&self) -> Self::Params;

    /// Reconstruct a fitted transformer from parameters.
    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError>
    where
        Self: Sized;

    /// Save the fitted transformer to a file.
    fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let params = self.extract_params();
        let bytes = params.to_bytes().map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }

    /// Load a fitted transformer from a file.
    fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, PreprocessingError>
    where
        Self: Sized,
    {
        let bytes = std::fs::read(path)?;
        let params = Self::Params::from_bytes(&bytes)
            .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
        Self::from_params(params)
    }

    /// Returns the number of features seen during fit.
    fn n_features_in(&self) -> usize;
}

/// Marker trait for transformers that don't require fitting.
///
/// Stateless transformers (like Normalizer) can transform data without
/// learning any parameters. They implement both `Transformer` and this trait.
pub trait StatelessTransformer<B: Backend>: Transformer<B> {
    /// Transform data without fitting.
    ///
    /// For stateless transformers, this is equivalent to `fit_transform`
    /// but communicates that no learning occurs.
    fn transform_direct(data: &Self::Input) -> Result<Self::Output, PreprocessingError>;
}
