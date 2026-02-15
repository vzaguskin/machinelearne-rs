//! Pipeline for chaining transformers.
//!
//! A Pipeline allows chaining multiple transformers together, where the output
//! of one transformer becomes the input to the next.
//!
//! # Example
//! ```ignore
//! use machinelearne_rs::preprocessing::{
//!     Pipeline, StandardScaler, MinMaxScaler, Transformer
//! };
//! use machinelearne_rs::backend::CpuBackend;
//!
//! let pipeline = Pipeline::<CpuBackend>::new()
//!     .add(StandardScaler::new())
//!     .add(MinMaxScaler::new().with_range(0.0, 1.0));
//!
//! let fitted = pipeline.fit(&data)?;
//! let transformed = fitted.transform(&data)?;
//! ```

use crate::backend::{Backend, Tensor2D};
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::imputation::{FittedSimpleImputer, SimpleImputer, SimpleImputerParams};
use crate::preprocessing::scaling::{
    FittedMaxAbsScaler, FittedMinMaxScaler, FittedNormalizer, FittedRobustScaler,
    FittedStandardScaler, MaxAbsScaler, MaxAbsScalerParams, MinMaxScaler, MinMaxScalerParams,
    Normalizer, NormalizerParams, RobustScaler, RobustScalerParams, StandardScaler,
    StandardScalerParams,
};
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// A trait for fitted transformers that can be part of a pipeline.
///
/// This trait is automatically implemented for any fitted transformer
/// that works with Tensor2D input/output.
pub trait PipelineStep<B: Backend>: Clone {
    /// Transform the data.
    fn transform_step(&self, data: &Tensor2D<B>) -> Result<Tensor2D<B>, PreprocessingError>;
    /// Inverse transform the data (if supported).
    fn inverse_transform_step(&self, data: &Tensor2D<B>)
        -> Result<Tensor2D<B>, PreprocessingError>;
    /// Get the step name for debugging.
    fn step_name(&self) -> &'static str;
}

/// Serializable representation of a fitted pipeline.
#[derive(Clone, Serialize, Deserialize)]
pub struct PipelineParams {
    /// Number of steps in the pipeline.
    pub n_steps: usize,
    /// Number of features seen during fit.
    pub n_features: usize,
}

/// A step in the pipeline that can be serialized.
#[derive(Clone)]
pub enum PipelineStepEnum<B: Backend> {
    /// StandardScaler step.
    StandardScaler(FittedStandardScaler<B>),
    /// MinMaxScaler step.
    MinMaxScaler(FittedMinMaxScaler<B>),
    /// RobustScaler step.
    RobustScaler(FittedRobustScaler<B>),
    /// MaxAbsScaler step.
    MaxAbsScaler(FittedMaxAbsScaler<B>),
    /// Normalizer step.
    Normalizer(FittedNormalizer<B>),
    /// SimpleImputer step.
    SimpleImputer(FittedSimpleImputer<B>),
}

impl<B: Backend> PipelineStep<B> for PipelineStepEnum<B> {
    fn transform_step(&self, data: &Tensor2D<B>) -> Result<Tensor2D<B>, PreprocessingError> {
        match self {
            PipelineStepEnum::StandardScaler(t) => t.transform(data),
            PipelineStepEnum::MinMaxScaler(t) => t.transform(data),
            PipelineStepEnum::RobustScaler(t) => t.transform(data),
            PipelineStepEnum::MaxAbsScaler(t) => t.transform(data),
            PipelineStepEnum::Normalizer(t) => t.transform(data),
            PipelineStepEnum::SimpleImputer(t) => t.transform(data),
        }
    }

    fn inverse_transform_step(
        &self,
        data: &Tensor2D<B>,
    ) -> Result<Tensor2D<B>, PreprocessingError> {
        match self {
            PipelineStepEnum::StandardScaler(t) => t.inverse_transform(data),
            PipelineStepEnum::MinMaxScaler(t) => t.inverse_transform(data),
            PipelineStepEnum::RobustScaler(t) => t.inverse_transform(data),
            PipelineStepEnum::MaxAbsScaler(t) => t.inverse_transform(data),
            PipelineStepEnum::Normalizer(t) => t.inverse_transform(data),
            PipelineStepEnum::SimpleImputer(t) => t.inverse_transform(data),
        }
    }

    fn step_name(&self) -> &'static str {
        match self {
            PipelineStepEnum::StandardScaler(_) => "StandardScaler",
            PipelineStepEnum::MinMaxScaler(_) => "MinMaxScaler",
            PipelineStepEnum::RobustScaler(_) => "RobustScaler",
            PipelineStepEnum::MaxAbsScaler(_) => "MaxAbsScaler",
            PipelineStepEnum::Normalizer(_) => "Normalizer",
            PipelineStepEnum::SimpleImputer(_) => "SimpleImputer",
        }
    }
}

/// Builder for a fitted step (used during pipeline construction).
trait FittedStepBuilder<B: Backend>: Clone {
    type Fitted: PipelineStep<B>;
    fn fit(&self, data: &Tensor2D<B>) -> Result<Self::Fitted, PreprocessingError>;
}

/// A step in the unfitted pipeline.
#[derive(Clone)]
pub enum UnfittedStepEnum<B: Backend> {
    StandardScaler(StandardScaler<B>),
    MinMaxScaler(MinMaxScaler<B>),
    RobustScaler(RobustScaler<B>),
    MaxAbsScaler(MaxAbsScaler<B>),
    Normalizer(Normalizer<B>),
    SimpleImputer(SimpleImputer<B>),
}

impl<B: Backend> FittedStepBuilder<B> for UnfittedStepEnum<B> {
    type Fitted = PipelineStepEnum<B>;

    fn fit(&self, data: &Tensor2D<B>) -> Result<Self::Fitted, PreprocessingError> {
        match self {
            UnfittedStepEnum::StandardScaler(t) => {
                t.fit(data).map(PipelineStepEnum::StandardScaler)
            }
            UnfittedStepEnum::MinMaxScaler(t) => t.fit(data).map(PipelineStepEnum::MinMaxScaler),
            UnfittedStepEnum::RobustScaler(t) => t.fit(data).map(PipelineStepEnum::RobustScaler),
            UnfittedStepEnum::MaxAbsScaler(t) => t.fit(data).map(PipelineStepEnum::MaxAbsScaler),
            UnfittedStepEnum::Normalizer(t) => t.fit(data).map(PipelineStepEnum::Normalizer),
            UnfittedStepEnum::SimpleImputer(t) => t.fit(data).map(PipelineStepEnum::SimpleImputer),
        }
    }
}

/// Pipeline transformer (unfitted).
///
/// Chains multiple transformers together.
#[derive(Clone)]
pub struct Pipeline<B: Backend> {
    steps: Vec<UnfittedStepEnum<B>>,
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for Pipeline<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Pipeline<B> {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            _backend: PhantomData,
        }
    }

    /// Add a StandardScaler to the pipeline.
    pub fn add_standard_scaler(mut self, scaler: StandardScaler<B>) -> Self {
        self.steps.push(UnfittedStepEnum::StandardScaler(scaler));
        self
    }

    /// Add a MinMaxScaler to the pipeline.
    pub fn add_minmax_scaler(mut self, scaler: MinMaxScaler<B>) -> Self {
        self.steps.push(UnfittedStepEnum::MinMaxScaler(scaler));
        self
    }

    /// Add a RobustScaler to the pipeline.
    pub fn add_robust_scaler(mut self, scaler: RobustScaler<B>) -> Self {
        self.steps.push(UnfittedStepEnum::RobustScaler(scaler));
        self
    }

    /// Add a MaxAbsScaler to the pipeline.
    pub fn add_maxabs_scaler(mut self, scaler: MaxAbsScaler<B>) -> Self {
        self.steps.push(UnfittedStepEnum::MaxAbsScaler(scaler));
        self
    }

    /// Add a Normalizer to the pipeline.
    pub fn add_normalizer(mut self, normalizer: Normalizer<B>) -> Self {
        self.steps.push(UnfittedStepEnum::Normalizer(normalizer));
        self
    }

    /// Add a SimpleImputer to the pipeline.
    pub fn add_simple_imputer(mut self, imputer: SimpleImputer<B>) -> Self {
        self.steps.push(UnfittedStepEnum::SimpleImputer(imputer));
        self
    }

    /// Get the number of steps in the pipeline.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl<B: Backend> Transformer<B> for Pipeline<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = PipelineParams;
    type Fitted = FittedPipeline<B>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError> {
        if self.steps.is_empty() {
            return Err(PreprocessingError::InvalidParameter(
                "Cannot fit an empty pipeline".to_string(),
            ));
        }

        let (rows, cols) = data.shape();
        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit pipeline on empty data".to_string(),
            ));
        }

        let mut fitted_steps = Vec::with_capacity(self.steps.len());
        let mut current_data = data.clone();

        for step in &self.steps {
            let fitted = step.fit(&current_data)?;
            current_data = fitted.transform_step(&current_data)?;
            fitted_steps.push(fitted);
        }

        Ok(FittedPipeline {
            steps: fitted_steps,
            n_features: cols,
            _backend: PhantomData,
        })
    }

    fn fit_transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let fitted = self.fit(data)?;
        fitted.transform(data)
    }
}

/// Fitted Pipeline ready for inference.
#[derive(Clone)]
pub struct FittedPipeline<B: Backend> {
    steps: Vec<PipelineStepEnum<B>>,
    n_features: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedPipeline<B> {
    /// Get the number of steps in the pipeline.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get the names of all steps in the pipeline.
    pub fn step_names(&self) -> Vec<&'static str> {
        self.steps.iter().map(|s| s.step_name()).collect()
    }
}

impl<B: Backend> FittedTransformer<B> for FittedPipeline<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = PipelineParams;

    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let (_, cols) = data.shape();

        if cols != self.n_features {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features,
                got_features: cols,
            });
        }

        let mut result = data.clone();
        for step in &self.steps {
            result = step.transform_step(&result)?;
        }
        Ok(result)
    }

    fn inverse_transform(&self, data: &Self::Output) -> Result<Self::Input, PreprocessingError> {
        let (_, cols) = data.shape();

        if cols != self.n_features {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features,
                got_features: cols,
            });
        }

        let mut result = data.clone();
        // Apply inverse transforms in reverse order
        for step in self.steps.iter().rev() {
            result = step.inverse_transform_step(&result)?;
        }
        Ok(result)
    }

    fn extract_params(&self) -> Self::Params {
        PipelineParams {
            n_steps: self.steps.len(),
            n_features: self.n_features,
        }
    }

    fn from_params(_params: Self::Params) -> Result<Self, PreprocessingError> {
        Err(PreprocessingError::InvalidParameter(
            "Pipeline does not support from_params - use save_to_file/load_from_file instead"
                .to_string(),
        ))
    }

    fn n_features_in(&self) -> usize {
        self.n_features
    }

    fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        // Save each step's params in sequence
        let mut step_params = Vec::new();
        for step in &self.steps {
            let (name, bytes) = match step {
                PipelineStepEnum::StandardScaler(t) => (
                    "StandardScaler",
                    bincode::serialize(&t.extract_params()).map_err(std::io::Error::other)?,
                ),
                PipelineStepEnum::MinMaxScaler(t) => (
                    "MinMaxScaler",
                    bincode::serialize(&t.extract_params()).map_err(std::io::Error::other)?,
                ),
                PipelineStepEnum::RobustScaler(t) => (
                    "RobustScaler",
                    bincode::serialize(&t.extract_params()).map_err(std::io::Error::other)?,
                ),
                PipelineStepEnum::MaxAbsScaler(t) => (
                    "MaxAbsScaler",
                    bincode::serialize(&t.extract_params()).map_err(std::io::Error::other)?,
                ),
                PipelineStepEnum::Normalizer(t) => (
                    "Normalizer",
                    bincode::serialize(&t.extract_params()).map_err(std::io::Error::other)?,
                ),
                PipelineStepEnum::SimpleImputer(t) => (
                    "SimpleImputer",
                    bincode::serialize(&t.extract_params()).map_err(std::io::Error::other)?,
                ),
            };
            step_params.push((name.to_string(), bytes));
        }

        let serialized =
            bincode::serialize(&(self.n_features, step_params)).map_err(std::io::Error::other)?;
        std::fs::write(path, serialized)
    }

    fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, PreprocessingError>
    where
        Self: Sized,
    {
        let bytes = std::fs::read(path)?;
        let (n_features, step_params): (usize, Vec<(String, Vec<u8>)>) =
            bincode::deserialize(&bytes)
                .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;

        let mut steps = Vec::new();
        for (name, step_bytes) in step_params {
            let step = match name.as_str() {
                "StandardScaler" => {
                    let params: StandardScalerParams = bincode::deserialize(&step_bytes)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    PipelineStepEnum::StandardScaler(FittedStandardScaler::from_params(params)?)
                }
                "MinMaxScaler" => {
                    let params: MinMaxScalerParams = bincode::deserialize(&step_bytes)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    PipelineStepEnum::MinMaxScaler(FittedMinMaxScaler::from_params(params)?)
                }
                "RobustScaler" => {
                    let params: RobustScalerParams = bincode::deserialize(&step_bytes)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    PipelineStepEnum::RobustScaler(FittedRobustScaler::from_params(params)?)
                }
                "MaxAbsScaler" => {
                    let params: MaxAbsScalerParams = bincode::deserialize(&step_bytes)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    PipelineStepEnum::MaxAbsScaler(FittedMaxAbsScaler::from_params(params)?)
                }
                "Normalizer" => {
                    let params: NormalizerParams = bincode::deserialize(&step_bytes)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    PipelineStepEnum::Normalizer(FittedNormalizer::from_params(params)?)
                }
                "SimpleImputer" => {
                    let params: SimpleImputerParams = bincode::deserialize(&step_bytes)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    PipelineStepEnum::SimpleImputer(FittedSimpleImputer::from_params(params)?)
                }
                _ => {
                    return Err(PreprocessingError::SerializationError(format!(
                        "Unknown step type: {}",
                        name
                    )))
                }
            };
            steps.push(step);
        }

        Ok(Self {
            steps,
            n_features,
            _backend: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::preprocessing::scaling::NormType;

    fn create_test_data() -> Tensor2D<CpuBackend> {
        // [[0, 1], [0, 1], [1, 3]]
        Tensor2D::new(vec![0.0f32, 1.0, 0.0, 1.0, 1.0, 3.0], 3, 2)
    }

    #[test]
    fn test_pipeline_single_step() {
        let data = create_test_data();
        let pipeline = Pipeline::<CpuBackend>::new().add_standard_scaler(StandardScaler::new());

        let fitted = pipeline.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        // Should be standardized
        let values = transformed.ravel().to_vec();
        assert!(values.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_pipeline_multiple_steps() {
        let data = create_test_data();
        let pipeline = Pipeline::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new())
            .add_minmax_scaler(MinMaxScaler::new().with_range(0.0, 1.0));

        let fitted = pipeline.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        // After StandardScaler + MinMaxScaler, values should be in [0, 1]
        let values = transformed.ravel().to_vec();
        assert!(values.iter().all(|&v| v >= -1e-6 && v <= 1.0 + 1e-6));
    }

    #[test]
    fn test_pipeline_inverse_transform() {
        let data = create_test_data();
        let pipeline = Pipeline::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new())
            .add_minmax_scaler(MinMaxScaler::new().with_range(0.0, 1.0));

        let fitted = pipeline.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        let original = data.ravel().to_vec();
        let recovered_vec = recovered.ravel().to_vec();

        for (o, r) in original.iter().zip(recovered_vec.iter()) {
            assert!((o - r).abs() < 1e-5, "Expected {}, got {}", o, r);
        }
    }

    #[test]
    fn test_pipeline_with_normalizer() {
        let data = create_test_data();
        let pipeline = Pipeline::<CpuBackend>::new().add_normalizer(Normalizer::new(NormType::L2));

        let fitted = pipeline.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        // Each row should have L2 norm = 1
        let values = transformed.ravel().to_vec();
        for row in 0..3 {
            let row_norm = (values[row * 2].powi(2) + values[row * 2 + 1].powi(2)).sqrt();
            assert!(
                (row_norm - 1.0).abs() < 1e-5,
                "Row {} L2 norm: {}",
                row,
                row_norm
            );
        }
    }

    #[test]
    fn test_pipeline_serialization() {
        let data = create_test_data();
        let pipeline = Pipeline::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new())
            .add_minmax_scaler(MinMaxScaler::new().with_range(0.0, 1.0));

        let fitted = pipeline.fit(&data).unwrap();

        // Save to file
        let temp_file = std::env::temp_dir().join("test_pipeline.bin");
        fitted.save_to_file(&temp_file).unwrap();

        // Load from file
        let loaded = FittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();

        // Compare results
        let transformed1 = fitted.transform(&data).unwrap();
        let transformed2 = loaded.transform(&data).unwrap();

        let t1 = transformed1.ravel().to_vec();
        let t2 = transformed2.ravel().to_vec();

        for (a, b) in t1.iter().zip(t2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        // Clean up
        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_pipeline_feature_mismatch() {
        let data = create_test_data(); // 2 features
        let pipeline = Pipeline::<CpuBackend>::new().add_standard_scaler(StandardScaler::new());

        let fitted = pipeline.fit(&data).unwrap();

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
    fn test_pipeline_empty() {
        let pipeline = Pipeline::<CpuBackend>::new();
        let data = create_test_data();

        let result = pipeline.fit(&data);
        assert!(matches!(
            result,
            Err(PreprocessingError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_pipeline_step_names() {
        let pipeline = Pipeline::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new())
            .add_minmax_scaler(MinMaxScaler::new());

        let data = create_test_data();
        let fitted = pipeline.fit(&data).unwrap();

        let names = fitted.step_names();
        assert_eq!(names, vec!["StandardScaler", "MinMaxScaler"]);
    }
}
