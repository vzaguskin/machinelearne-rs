//! Trait object interface for pipeline steps.
//!
//! This provides a unified interface for all fitted transformers,
//! allowing them to be stored as `Box<dyn FittedStep<B>>`.

use crate::backend::{Backend, Tensor2D};
use crate::preprocessing::error::PreprocessingError;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::any::type_name;

/// Trait for fitted transformers that can be used in a pipeline.
///
/// This is the trait object interface that allows storing heterogeneous
/// transformers in a Vec without enums.
///
/// # Type Parameters
/// - `InputState`: The state required by this step (e.g., `Imputed` for Scaler)
/// - `OutputState`: The state produced by this step (e.g., `Scaled` for Scaler)
pub trait FittedStep<B: Backend> {
    /// Transform the data.
    fn transform(&self, data: &Tensor2D<B>) -> Result<Tensor2D<B>, PreprocessingError>;

    /// Inverse transform the data (if supported).
    fn inverse_transform(&self, data: &Tensor2D<B>) -> Result<Tensor2D<B>, PreprocessingError>;

    /// Get the step name for debugging.
    fn step_name(&self) -> &'static str;

    /// Get the number of input features.
    fn n_features_in(&self) -> usize;

    /// Get the number of output features.
    fn n_features_out(&self) -> usize;

    /// Whether this step supports inverse transform.
    fn supports_inverse(&self) -> bool {
        true
    }

    /// Serialize this step to bytes (for saving).
    fn to_bytes(&self) -> Result<Vec<u8>, PreprocessingError>;

    /// Get the type name of the underlying transformer.
    fn type_name(&self) -> &'static str {
        type_name::<Self>()
    }
}

/// Blanket implementation of FittedStep for any FittedTransformer.
///
/// This allows any transformer that implements `FittedTransformer` to be
/// automatically usable as a pipeline step via trait object.
impl<B, T> FittedStep<B> for T
where
    B: Backend,
    T: crate::preprocessing::traits::FittedTransformer<
            B,
            Input = Tensor2D<B>,
            Output = Tensor2D<B>,
        > + Clone
        + 'static,
    T::Params: Serialize + DeserializeOwned,
{
    fn transform(&self, data: &Tensor2D<B>) -> Result<Tensor2D<B>, PreprocessingError> {
        crate::preprocessing::traits::FittedTransformer::transform(self, data)
    }

    fn inverse_transform(&self, data: &Tensor2D<B>) -> Result<Tensor2D<B>, PreprocessingError> {
        crate::preprocessing::traits::FittedTransformer::inverse_transform(self, data)
    }

    fn step_name(&self) -> &'static str {
        // Extract a friendly name from the type name
        let full_name = type_name::<T>();
        // Type name looks like: "machinelearne_rs::preprocessing::scaling::standard::FittedStandardScaler<...>"
        // We want to extract "FittedStandardScaler"

        // Find the last "::" before any "<"
        let before_generic = if let Some(pos) = full_name.find('<') {
            &full_name[..pos]
        } else {
            full_name
        };

        // Now find the last "::" to get just the struct name
        if let Some(pos) = before_generic.rfind("::") {
            &before_generic[pos + 2..]
        } else {
            before_generic
        }
    }

    fn n_features_in(&self) -> usize {
        crate::preprocessing::traits::FittedTransformer::n_features_in(self)
    }

    fn n_features_out(&self) -> usize {
        // For most transformers, output features = input features
        // OneHotEncoder overrides this
        crate::preprocessing::traits::FittedTransformer::n_features_in(self)
    }

    fn to_bytes(&self) -> Result<Vec<u8>, PreprocessingError> {
        let params = self.extract_params();
        bincode::serialize(&params)
            .map_err(|e| PreprocessingError::SerializationError(e.to_string()))
    }
}

/// A wrapper that stores a step with its state transitions.
///
/// This is used internally by the pipeline to track what state
/// transitions each step performs.
pub struct StepWithState<B: Backend> {
    /// The actual step implementation.
    pub step: Box<dyn FittedStep<B>>,
    /// Human-readable name.
    pub name: &'static str,
}

impl<B: Backend> StepWithState<B> {
    /// Create a new step wrapper.
    pub fn new<S: FittedStep<B> + 'static>(step: S) -> Self {
        let name = step.step_name();
        Self {
            step: Box::new(step),
            name,
        }
    }

    /// Create from a boxed step.
    pub fn from_boxed(step: Box<dyn FittedStep<B>>, name: &'static str) -> Self {
        Self { step, name }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::preprocessing::scaling::{FittedStandardScaler, StandardScaler};
    use crate::preprocessing::traits::Transformer;

    #[test]
    fn test_step_name_extraction() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        let name = fitted.step_name();
        assert!(
            name.contains("StandardScaler") || name.contains("Scaler"),
            "step_name was: {}",
            name
        );
    }

    #[test]
    fn test_step_as_trait_object() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted = scaler.fit(&data).unwrap();

        // Store as trait object
        let step: Box<dyn FittedStep<CpuBackend>> = Box::new(fitted);

        // Use through trait object (virtual dispatch)
        let result = step.transform(&data).unwrap();
        assert_eq!(result.shape(), (2, 2));
    }
}
