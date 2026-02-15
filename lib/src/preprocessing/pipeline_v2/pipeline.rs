//! Pipeline V2 implementation with trait objects and type-state enforcement.

use crate::backend::{Backend, Tensor2D};
use crate::preprocessing::encoding::{FittedOneHotEncoder, FittedOrdinalEncoder};
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::imputation::FittedSimpleImputer;
use crate::preprocessing::scaling::{
    FittedMaxAbsScaler, FittedMinMaxScaler, FittedNormalizer, FittedRobustScaler,
    FittedStandardScaler,
};
use std::marker::PhantomData;

use super::state::{Encoded, Imputed, Raw, Scaled, State};
use super::step::{FittedStep, StepWithState};

/// Unfitted pipeline builder with type-state tracking.
///
/// The state parameter `S` tracks what preprocessing has been added,
/// enforcing valid order at compile time.
pub struct PipelineV2<B: Backend, S: State = Raw> {
    steps: Vec<StepWithState<B>>,
    n_features: usize,
    _state: PhantomData<S>,
}

impl<B: Backend> PipelineV2<B, Raw> {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            n_features: 0,
            _state: PhantomData,
        }
    }
}

impl<B: Backend> Default for PipelineV2<B, Raw> {
    fn default() -> Self {
        Self::new()
    }
}

// Common methods for all states
impl<B: Backend, S: State> PipelineV2<B, S> {
    /// Get the number of steps in the pipeline.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get the number of input features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get step names for debugging.
    pub fn step_names(&self) -> Vec<&'static str> {
        self.steps.iter().map(|s| s.name).collect()
    }
}

// Methods available at Raw state (can add imputer)
impl<B: Backend> PipelineV2<B, Raw> {
    /// Add an imputer step (handles missing values).
    ///
    /// Transitions the pipeline from `Raw` to `Imputed` state.
    pub fn add_imputer(mut self, imputer: FittedSimpleImputer<B>) -> PipelineV2<B, Imputed> {
        if self.steps.is_empty() {
            self.n_features = imputer.n_features_in();
        }
        self.steps.push(StepWithState::new(imputer));
        PipelineV2 {
            steps: self.steps,
            n_features: self.n_features,
            _state: PhantomData,
        }
    }

    /// Add a scaler step directly (if data has no missing values).
    ///
    /// This skips the imputation step, transitioning directly from `Raw` to `Scaled`.
    pub fn add_scaler_direct(
        mut self,
        scaler: impl FittedStep<B> + 'static,
    ) -> PipelineV2<B, Scaled> {
        if self.steps.is_empty() {
            self.n_features = scaler.n_features_in();
        }
        let name = scaler.step_name();
        self.steps
            .push(StepWithState::from_boxed(Box::new(scaler), name));
        PipelineV2 {
            steps: self.steps,
            n_features: self.n_features,
            _state: PhantomData,
        }
    }

    /// Fit the pipeline on raw data and return a fitted pipeline.
    pub fn fit(self, data: &Tensor2D<B>) -> Result<FittedPipelineV2<B, Raw>, PreprocessingError> {
        if self.steps.is_empty() {
            return Err(PreprocessingError::InvalidParameter(
                "Cannot fit empty pipeline".to_string(),
            ));
        }
        // For unfitted steps, we'd need to fit them here
        // For now, assume steps are already fitted
        Ok(FittedPipelineV2 {
            steps: self.steps,
            n_features_in: data.shape().1,
            n_features_out: data.shape().1, // Will be updated based on transformers
            _state: PhantomData,
        })
    }
}

// Methods available at Imputed state (can add scaler)
impl<B: Backend> PipelineV2<B, Imputed> {
    /// Add a scaler step.
    ///
    /// Transitions the pipeline from `Imputed` to `Scaled` state.
    pub fn add_scaler(self, scaler: impl FittedStep<B> + 'static) -> PipelineV2<B, Scaled> {
        let mut pipeline: PipelineV2<B, Scaled> = PipelineV2 {
            steps: self.steps,
            n_features: self.n_features,
            _state: PhantomData,
        };
        let name = scaler.step_name();
        pipeline
            .steps
            .push(StepWithState::from_boxed(Box::new(scaler), name));
        pipeline
    }

    /// Add StandardScaler.
    pub fn add_standard_scaler(self, scaler: FittedStandardScaler<B>) -> PipelineV2<B, Scaled> {
        self.add_scaler(scaler)
    }

    /// Add MinMaxScaler.
    pub fn add_minmax_scaler(self, scaler: FittedMinMaxScaler<B>) -> PipelineV2<B, Scaled> {
        self.add_scaler(scaler)
    }

    /// Add RobustScaler.
    pub fn add_robust_scaler(self, scaler: FittedRobustScaler<B>) -> PipelineV2<B, Scaled> {
        self.add_scaler(scaler)
    }

    /// Add MaxAbsScaler.
    pub fn add_maxabs_scaler(self, scaler: FittedMaxAbsScaler<B>) -> PipelineV2<B, Scaled> {
        self.add_scaler(scaler)
    }

    /// Add Normalizer.
    pub fn add_normalizer(self, normalizer: FittedNormalizer<B>) -> PipelineV2<B, Scaled> {
        self.add_scaler(normalizer)
    }

    /// Convert to fitted pipeline (without scaling).
    pub fn into_fitted(self) -> FittedPipelineV2<B, Imputed> {
        FittedPipelineV2 {
            steps: self.steps,
            n_features_in: self.n_features,
            n_features_out: self.n_features,
            _state: PhantomData,
        }
    }
}

// Methods available at Scaled state (can add encoder or finalize)
impl<B: Backend> PipelineV2<B, Scaled> {
    /// Add an encoder step.
    ///
    /// Transitions the pipeline from `Scaled` to `Encoded` state.
    pub fn add_encoder(self, encoder: impl FittedStep<B> + 'static) -> PipelineV2<B, Encoded> {
        let mut pipeline: PipelineV2<B, Encoded> = PipelineV2 {
            steps: self.steps,
            n_features: self.n_features,
            _state: PhantomData,
        };
        let name = encoder.step_name();
        pipeline
            .steps
            .push(StepWithState::from_boxed(Box::new(encoder), name));
        pipeline
    }

    /// Add OneHotEncoder.
    pub fn add_one_hot_encoder(self, encoder: FittedOneHotEncoder<B>) -> PipelineV2<B, Encoded> {
        self.add_encoder(encoder)
    }

    /// Add OrdinalEncoder.
    pub fn add_ordinal_encoder(self, encoder: FittedOrdinalEncoder<B>) -> PipelineV2<B, Encoded> {
        self.add_encoder(encoder)
    }

    /// Convert to fitted pipeline.
    pub fn into_fitted(self) -> FittedPipelineV2<B, Scaled> {
        FittedPipelineV2 {
            steps: self.steps,
            n_features_in: self.n_features,
            n_features_out: self.n_features,
            _state: PhantomData,
        }
    }
}

// Methods available at Encoded state (ready for model)
impl<B: Backend> PipelineV2<B, Encoded> {
    /// Convert to fitted pipeline.
    pub fn into_fitted(self) -> FittedPipelineV2<B, Encoded> {
        FittedPipelineV2 {
            steps: self.steps,
            n_features_in: self.n_features,
            n_features_out: self.n_features,
            _state: PhantomData,
        }
    }
}

/// Fitted pipeline ready for inference.
///
/// The state parameter `S` indicates what preprocessing has been applied,
/// which determines what kind of models can use this pipeline.
pub struct FittedPipelineV2<B: Backend, S: State> {
    steps: Vec<StepWithState<B>>,
    n_features_in: usize,
    n_features_out: usize,
    _state: PhantomData<S>,
}

impl<B: Backend, S: State> FittedPipelineV2<B, S> {
    /// Transform data using all pipeline steps.
    ///
    /// This uses virtual dispatch (vtable) to call each step's transform method.
    pub fn transform(&self, data: &Tensor2D<B>) -> Result<Tensor2D<B>, PreprocessingError> {
        if data.shape().1 != self.n_features_in {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features_in,
                got_features: data.shape().1,
            });
        }

        let mut current = data.clone();
        for step in &self.steps {
            current = step.step.transform(&current)?;
        }
        Ok(current)
    }

    /// Inverse transform data through all pipeline steps (in reverse).
    pub fn inverse_transform(&self, data: &Tensor2D<B>) -> Result<Tensor2D<B>, PreprocessingError> {
        let mut current = data.clone();
        for step in self.steps.iter().rev() {
            if !step.step.supports_inverse() {
                return Err(PreprocessingError::InvalidParameter(format!(
                    "Step '{}' does not support inverse transform",
                    step.name
                )));
            }
            current = step.step.inverse_transform(&current)?;
        }
        Ok(current)
    }

    /// Get the number of input features.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Get the number of output features.
    pub fn n_features_out(&self) -> usize {
        self.n_features_out
    }

    /// Get step names.
    pub fn step_names(&self) -> Vec<&'static str> {
        self.steps.iter().map(|s| s.name).collect()
    }
}

// Additional methods for Scaled state (ready for most models)
impl<B: Backend> FittedPipelineV2<B, Scaled> {
    /// Create a PredictivePipeline by combining with a model.
    pub fn with_model<M>(self, model: M) -> PredictivePipelineV2<B, M>
    where
        M: crate::model::InferenceModel<B>,
    {
        PredictivePipelineV2 {
            pipeline: self,
            model,
            _backend: PhantomData,
        }
    }
}

/// A complete predictive pipeline (preprocessing + model).
///
/// This is the final product that can be saved, loaded, and used for predictions.
pub struct PredictivePipelineV2<B: Backend, M>
where
    M: crate::model::InferenceModel<B>,
{
    pipeline: FittedPipelineV2<B, Scaled>,
    model: M,
    _backend: PhantomData<B>,
}

impl<B: Backend, M> PredictivePipelineV2<B, M>
where
    M: crate::model::InferenceModel<
        B,
        InputBatch = Tensor2D<B>,
        OutputBatch = crate::backend::Tensor1D<B>,
    >,
{
    /// Make predictions on new data (batch mode).
    pub fn predict(
        &self,
        data: &Tensor2D<B>,
    ) -> Result<crate::backend::Tensor1D<B>, PreprocessingError> {
        let preprocessed = self.pipeline.transform(data)?;
        Ok(self.model.predict_batch(&preprocessed))
    }

    /// Get reference to the underlying model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get reference to the preprocessing pipeline.
    pub fn pipeline(&self) -> &FittedPipelineV2<B, Scaled> {
        &self.pipeline
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::preprocessing::imputation::{ImputeStrategy, SimpleImputer};
    use crate::preprocessing::scaling::StandardScaler;
    use crate::preprocessing::traits::Transformer;

    fn create_test_data() -> Tensor2D<CpuBackend> {
        Tensor2D::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2)
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline: PipelineV2<CpuBackend, Raw> = PipelineV2::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
    }

    #[test]
    fn test_pipeline_state_transition() {
        let data = create_test_data();

        // Create imputer
        let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Mean);
        let fitted_imputer = imputer.fit(&data).unwrap();

        // Create scaler
        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted_scaler = scaler.fit(&data).unwrap();

        // Build pipeline with state transitions
        let pipeline = PipelineV2::new()
            .add_imputer(fitted_imputer) // Raw -> Imputed
            .add_standard_scaler(fitted_scaler); // Imputed -> Scaled

        assert_eq!(pipeline.len(), 2);
        assert_eq!(
            pipeline.step_names(),
            vec!["FittedSimpleImputer", "FittedStandardScaler"]
        );
    }

    #[test]
    fn test_fitted_pipeline_transform() {
        let data = create_test_data();

        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted_scaler = scaler.fit(&data).unwrap();

        let pipeline = PipelineV2::<CpuBackend, Raw>::new()
            .add_scaler_direct(fitted_scaler)
            .into_fitted();

        let result = pipeline.transform(&data).unwrap();
        assert_eq!(result.shape(), data.shape());
    }

    #[test]
    fn test_fitted_pipeline_feature_mismatch() {
        let data = create_test_data();

        let scaler = StandardScaler::<CpuBackend>::new();
        let fitted_scaler = scaler.fit(&data).unwrap();

        let pipeline = PipelineV2::<CpuBackend, Raw>::new()
            .add_scaler_direct(fitted_scaler)
            .into_fitted();

        let wrong_data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 1, 3);
        let result = pipeline.transform(&wrong_data);

        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch { .. })
        ));
    }
}
