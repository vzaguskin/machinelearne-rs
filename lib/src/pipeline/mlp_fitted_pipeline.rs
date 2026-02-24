//! Fitted pipeline for MLP models with end-to-end inference.
//!
//! Combines preprocessing transformers with a trained MLP model for unified
//! prediction and serialization.

use std::io;
use std::marker::PhantomData;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::model::mlp::{MLPModel, SerializableMLPParams};
use crate::model::{Fitted, InferenceModel};
use crate::preprocessing::feature_engineering::{
    FittedPolynomialFeatures, PolynomialFeaturesParams,
};
use crate::preprocessing::pipeline::{FittedPipeline as PreprocPipeline, PipelineStepEnum};
use crate::preprocessing::traits::FittedTransformer;

/// Error type for MLP pipeline operations.
#[derive(Debug)]
pub enum MLPPipelineError {
    /// I/O error during file operations.
    IoError(io::Error),
    /// Serialization/deserialization error.
    SerializationError(String),
    /// Preprocessing error.
    PreprocessingError(crate::preprocessing::error::PreprocessingError),
    /// Feature mismatch error.
    FeatureMismatch { expected: usize, got: usize },
    /// Invalid pipeline state.
    InvalidState(String),
}

impl std::fmt::Display for MLPPipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MLPPipelineError::IoError(e) => write!(f, "IO error: {}", e),
            MLPPipelineError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            MLPPipelineError::PreprocessingError(e) => write!(f, "Preprocessing error: {:?}", e),
            MLPPipelineError::FeatureMismatch { expected, got } => {
                write!(f, "Feature mismatch: expected {}, got {}", expected, got)
            }
            MLPPipelineError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
        }
    }
}

impl std::error::Error for MLPPipelineError {}

impl From<io::Error> for MLPPipelineError {
    fn from(e: io::Error) -> Self {
        MLPPipelineError::IoError(e)
    }
}

impl From<bincode::Error> for MLPPipelineError {
    fn from(e: bincode::Error) -> Self {
        MLPPipelineError::SerializationError(e.to_string())
    }
}

impl From<crate::preprocessing::error::PreprocessingError> for MLPPipelineError {
    fn from(e: crate::preprocessing::error::PreprocessingError) -> Self {
        MLPPipelineError::PreprocessingError(e)
    }
}

/// Metadata for a fitted MLP pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MLPPipelineMetadata {
    /// Number of input features (before preprocessing).
    pub n_features_in: usize,
    /// Number of features after preprocessing (before polynomial).
    pub n_features_after_preproc: usize,
    /// Number of features after polynomial expansion (model input).
    pub n_features_model: usize,
    /// Number of output features.
    pub n_outputs: usize,
    /// Whether polynomial features are applied.
    pub has_polynomial: bool,
    /// Polynomial degree (1 = no expansion).
    pub poly_degree: usize,
    /// Number of preprocessing steps.
    pub n_preproc_steps: usize,
    /// Pipeline version for compatibility.
    pub version: u32,
    /// Layer sizes of the MLP.
    pub layer_sizes: Vec<usize>,
}

impl Default for MLPPipelineMetadata {
    fn default() -> Self {
        Self {
            n_features_in: 0,
            n_features_after_preproc: 0,
            n_features_model: 0,
            n_outputs: 1,
            has_polynomial: false,
            poly_degree: 1,
            n_preproc_steps: 0,
            version: 1,
            layer_sizes: vec![],
        }
    }
}

/// Serializable representation of a fitted MLP pipeline.
#[derive(Serialize, Deserialize)]
struct SerializableMLPPipeline {
    metadata: MLPPipelineMetadata,
    /// Serialized preprocessing pipeline steps.
    preproc_data: Vec<(String, Vec<u8>)>,
    /// Serialized polynomial features (if present).
    poly_data: Option<Vec<u8>>,
    /// Serialized MLP model parameters.
    model_data: Vec<u8>,
}

/// A fitted pipeline combining preprocessing and MLP model for end-to-end inference.
///
/// This struct holds:
/// - A preprocessing pipeline (scalers, imputers, encoders)
/// - Optional polynomial feature expansion
/// - A trained MLP model
///
/// It provides unified `predict` methods that apply the full transformation
/// chain and model inference.
pub struct MLPFittedPipeline<B: Backend> {
    /// Preprocessing pipeline (scalers, imputers, etc.).
    preprocessor: Option<PreprocPipeline<B>>,
    /// Optional polynomial features.
    polynomial: Option<FittedPolynomialFeatures<B>>,
    /// Trained MLP model.
    model: MLPModel<B, Fitted>,
    /// Pipeline metadata.
    metadata: MLPPipelineMetadata,
    _backend: PhantomData<B>,
}

impl<B: Backend> MLPFittedPipeline<B> {
    /// Create a new fitted pipeline from components.
    ///
    /// # Arguments
    /// * `preprocessor` - Optional preprocessing pipeline
    /// * `polynomial` - Optional polynomial feature transformer
    /// * `model` - Trained MLP model
    pub fn new(
        preprocessor: Option<PreprocPipeline<B>>,
        polynomial: Option<FittedPolynomialFeatures<B>>,
        model: MLPModel<B, Fitted>,
    ) -> Self {
        let n_features_in = preprocessor
            .as_ref()
            .map(|p| p.n_features_in())
            .unwrap_or_else(|| {
                polynomial
                    .as_ref()
                    .map(|p| p.n_features_in())
                    .unwrap_or_else(|| model.layer_sizes().first().copied().unwrap_or(0))
            });

        let n_features_after_preproc = if let Some(ref preproc) = preprocessor {
            preproc
                .steps()
                .last()
                .map(|s| s.n_features_in())
                .unwrap_or(n_features_in)
        } else {
            n_features_in
        };

        let (n_features_model, has_polynomial, poly_degree) = if let Some(ref poly) = polynomial {
            (poly.n_features_out(), true, poly.extract_params().degree)
        } else {
            (n_features_after_preproc, false, 1)
        };

        let n_outputs = *model.layer_sizes().last().unwrap_or(&1);
        let n_preproc_steps = preprocessor.as_ref().map(|p| p.len()).unwrap_or(0);
        let layer_sizes = model.layer_sizes().to_vec();

        let metadata = MLPPipelineMetadata {
            n_features_in,
            n_features_after_preproc,
            n_features_model,
            n_outputs,
            has_polynomial,
            poly_degree,
            n_preproc_steps,
            version: 1,
            layer_sizes,
        };

        Self {
            preprocessor,
            polynomial,
            model,
            metadata,
            _backend: PhantomData,
        }
    }

    /// Create a pipeline with just a model (no preprocessing).
    pub fn from_model(model: MLPModel<B, Fitted>) -> Self {
        let layer_sizes = model.layer_sizes().to_vec();
        let n_features = layer_sizes.first().copied().unwrap_or(0);
        let n_outputs = layer_sizes.last().copied().unwrap_or(1);

        let metadata = MLPPipelineMetadata {
            n_features_in: n_features,
            n_features_after_preproc: n_features,
            n_features_model: n_features,
            n_outputs,
            has_polynomial: false,
            poly_degree: 1,
            n_preproc_steps: 0,
            version: 1,
            layer_sizes,
        };

        Self {
            preprocessor: None,
            polynomial: None,
            model,
            metadata,
            _backend: PhantomData,
        }
    }

    /// Predict on raw data (applies full transformation chain).
    ///
    /// This method:
    /// 1. Applies preprocessing transformations (if any)
    /// 2. Applies polynomial feature expansion (if any)
    /// 3. Runs MLP model inference
    ///
    /// # Arguments
    /// * `raw_data` - Raw input features (n_samples x n_features_in)
    ///
    /// # Returns
    /// Predictions as a 2D tensor (n_samples x n_outputs)
    pub fn predict(&self, raw_data: &Tensor2D<B>) -> Result<Tensor2D<B>, MLPPipelineError> {
        let (_, cols) = raw_data.shape();

        if cols != self.metadata.n_features_in {
            return Err(MLPPipelineError::FeatureMismatch {
                expected: self.metadata.n_features_in,
                got: cols,
            });
        }

        // Step 1: Apply preprocessing
        let mut data = if let Some(ref preproc) = self.preprocessor {
            preproc.transform(raw_data)?
        } else {
            raw_data.clone()
        };

        // Step 2: Apply polynomial features
        if let Some(ref poly) = self.polynomial {
            data = poly.transform(&data)?;
        }

        // Step 3: Model prediction
        Ok(self.model.predict_batch(&data))
    }

    /// Get predictions flattened to 1D (for single-output models).
    pub fn predict_flattened(
        &self,
        raw_data: &Tensor2D<B>,
    ) -> Result<Tensor1D<B>, MLPPipelineError> {
        let predictions = self.predict(raw_data)?;
        let flat_data = predictions.ravel().to_vec();
        Ok(Tensor1D::new(
            flat_data.into_iter().map(|x| x as f32).collect(),
        ))
    }

    /// Get the number of input features.
    pub fn n_features_in(&self) -> usize {
        self.metadata.n_features_in
    }

    /// Get the number of output features.
    pub fn n_outputs(&self) -> usize {
        self.metadata.n_outputs
    }

    /// Get the metadata.
    pub fn metadata(&self) -> &MLPPipelineMetadata {
        &self.metadata
    }

    /// Get a reference to the preprocessing pipeline.
    pub fn preprocessor(&self) -> Option<&PreprocPipeline<B>> {
        self.preprocessor.as_ref()
    }

    /// Get a reference to the polynomial features.
    pub fn polynomial(&self) -> Option<&FittedPolynomialFeatures<B>> {
        self.polynomial.as_ref()
    }

    /// Get a reference to the model.
    pub fn model(&self) -> &MLPModel<B, Fitted> {
        &self.model
    }

    /// Save the pipeline to a file.
    ///
    /// The file format is:
    /// - Magic bytes "MLPM" (4 bytes) - MLP Pipeline marker
    /// - Version (4 bytes u32)
    /// - Serialized pipeline data
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), MLPPipelineError> {
        // Serialize preprocessing steps
        let mut preproc_data = Vec::new();
        if let Some(ref preproc) = self.preprocessor {
            for step in preproc.steps() {
                let (name, bytes) = match step {
                    PipelineStepEnum::StandardScaler(t) => {
                        ("StandardScaler", bincode::serialize(&t.extract_params())?)
                    }
                    PipelineStepEnum::MinMaxScaler(t) => {
                        ("MinMaxScaler", bincode::serialize(&t.extract_params())?)
                    }
                    PipelineStepEnum::RobustScaler(t) => {
                        ("RobustScaler", bincode::serialize(&t.extract_params())?)
                    }
                    PipelineStepEnum::MaxAbsScaler(t) => {
                        ("MaxAbsScaler", bincode::serialize(&t.extract_params())?)
                    }
                    PipelineStepEnum::Normalizer(t) => {
                        ("Normalizer", bincode::serialize(&t.extract_params())?)
                    }
                    PipelineStepEnum::SimpleImputer(t) => {
                        ("SimpleImputer", bincode::serialize(&t.extract_params())?)
                    }
                    PipelineStepEnum::OneHotEncoder(t) => {
                        ("OneHotEncoder", bincode::serialize(&t.extract_params())?)
                    }
                    PipelineStepEnum::OrdinalEncoder(t) => {
                        ("OrdinalEncoder", bincode::serialize(&t.extract_params())?)
                    }
                };
                preproc_data.push((name.to_string(), bytes));
            }
        }

        // Serialize polynomial features
        let poly_data = self
            .polynomial
            .as_ref()
            .map(|p| bincode::serialize(&p.extract_params()))
            .transpose()?;

        // Serialize model
        let model_data = bincode::serialize(&self.model.extract_params())?;

        // Create serializable struct
        let serializable = SerializableMLPPipeline {
            metadata: self.metadata.clone(),
            preproc_data,
            poly_data,
            model_data,
        };

        // Serialize and write
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MLPM"); // Magic bytes for MLP Pipeline
        bytes.extend_from_slice(&1u32.to_le_bytes()); // Version
        bytes.extend_from_slice(&bincode::serialize(&serializable)?);

        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a pipeline from a file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, MLPPipelineError> {
        use crate::preprocessing::encoding::{
            FittedOneHotEncoder, FittedOrdinalEncoder, OneHotEncoderParams, OrdinalEncoderParams,
        };
        use crate::preprocessing::imputation::{FittedSimpleImputer, SimpleImputerParams};
        use crate::preprocessing::scaling::{
            FittedMaxAbsScaler, FittedMinMaxScaler, FittedNormalizer, FittedRobustScaler,
            FittedStandardScaler, MaxAbsScalerParams, MinMaxScalerParams, NormalizerParams,
            RobustScalerParams, StandardScalerParams,
        };

        let bytes = std::fs::read(path)?;

        // Verify magic bytes
        if bytes.len() < 8 || &bytes[0..4] != b"MLPM" {
            return Err(MLPPipelineError::InvalidState(
                "Invalid file format: expected MLP pipeline".to_string(),
            ));
        }

        // Read version
        let _version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        // Deserialize
        let serializable: SerializableMLPPipeline = bincode::deserialize(&bytes[8..])?;

        // Reconstruct preprocessing pipeline
        let mut preproc_steps = Vec::new();

        for (name, step_bytes) in &serializable.preproc_data {
            let step = match name.as_str() {
                "StandardScaler" => {
                    let params: StandardScalerParams = bincode::deserialize(step_bytes)?;
                    let fitted = FittedStandardScaler::from_params(params)?;
                    PipelineStepEnum::StandardScaler(fitted)
                }
                "MinMaxScaler" => {
                    let params: MinMaxScalerParams = bincode::deserialize(step_bytes)?;
                    let fitted = FittedMinMaxScaler::from_params(params)?;
                    PipelineStepEnum::MinMaxScaler(fitted)
                }
                "RobustScaler" => {
                    let params: RobustScalerParams = bincode::deserialize(step_bytes)?;
                    let fitted = FittedRobustScaler::from_params(params)?;
                    PipelineStepEnum::RobustScaler(fitted)
                }
                "MaxAbsScaler" => {
                    let params: MaxAbsScalerParams = bincode::deserialize(step_bytes)?;
                    let fitted = FittedMaxAbsScaler::from_params(params)?;
                    PipelineStepEnum::MaxAbsScaler(fitted)
                }
                "Normalizer" => {
                    let params: NormalizerParams = bincode::deserialize(step_bytes)?;
                    let fitted = FittedNormalizer::from_params(params)?;
                    PipelineStepEnum::Normalizer(fitted)
                }
                "SimpleImputer" => {
                    let params: SimpleImputerParams = bincode::deserialize(step_bytes)?;
                    let fitted = FittedSimpleImputer::from_params(params)?;
                    PipelineStepEnum::SimpleImputer(fitted)
                }
                "OneHotEncoder" => {
                    let params: OneHotEncoderParams = bincode::deserialize(step_bytes)?;
                    let fitted = FittedOneHotEncoder::from_params(params)?;
                    PipelineStepEnum::OneHotEncoder(fitted)
                }
                "OrdinalEncoder" => {
                    let params: OrdinalEncoderParams = bincode::deserialize(step_bytes)?;
                    let fitted = FittedOrdinalEncoder::from_params(params)?;
                    PipelineStepEnum::OrdinalEncoder(fitted)
                }
                _ => {
                    return Err(MLPPipelineError::InvalidState(format!(
                        "Unknown step type: {}",
                        name
                    )))
                }
            };
            preproc_steps.push(step);
        }

        let preprocessor = if preproc_steps.is_empty() {
            None
        } else {
            Some(PreprocPipeline::from_steps(
                preproc_steps,
                serializable.metadata.n_features_in,
            ))
        };

        // Reconstruct polynomial features
        let polynomial = if let Some(poly_bytes) = serializable.poly_data {
            let params: PolynomialFeaturesParams = bincode::deserialize(&poly_bytes)?;
            Some(FittedPolynomialFeatures::from_params(params)?)
        } else {
            None
        };

        // Reconstruct model
        let model_params: SerializableMLPParams = bincode::deserialize(&serializable.model_data)?;
        let model = MLPModel::<B, Fitted>::from_params(model_params)
            .map_err(|e| MLPPipelineError::SerializationError(e.to_string()))?;

        Ok(Self {
            preprocessor,
            polynomial,
            model,
            metadata: serializable.metadata,
            _backend: PhantomData,
        })
    }
}

impl<B: Backend> Clone for MLPFittedPipeline<B> {
    fn clone(&self) -> Self {
        Self {
            preprocessor: self.preprocessor.clone(),
            polynomial: self.polynomial.clone(),
            model: self.model.clone(),
            metadata: self.metadata.clone(),
            _backend: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::model::{Activation, TrainableModel, MLP};
    use crate::preprocessing::feature_engineering::PolynomialFeatures;
    use crate::preprocessing::pipeline::Pipeline;
    use crate::preprocessing::scaling::StandardScaler;
    use crate::preprocessing::traits::Transformer;

    fn create_simple_mlp_model() -> MLPModel<CpuBackend, Fitted> {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        model.into_fitted()
    }

    #[test]
    fn test_mlp_fitted_pipeline_from_model() {
        let model = create_simple_mlp_model();
        let pipeline = MLPFittedPipeline::from_model(model);

        assert_eq!(pipeline.n_features_in(), 2);
        assert_eq!(pipeline.n_outputs(), 1);
        assert!(pipeline.preprocessor().is_none());
        assert!(pipeline.polynomial().is_none());
    }

    #[test]
    fn test_mlp_fitted_pipeline_predict() {
        let model = create_simple_mlp_model();
        let pipeline = MLPFittedPipeline::from_model(model);

        // Input: [[1, 2], [3, 4]]
        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let predictions = pipeline.predict(&input).unwrap();

        // Output shape: (2, 1)
        assert_eq!(predictions.shape(), (2, 1));
    }

    #[test]
    fn test_mlp_fitted_pipeline_predict_flattened() {
        let model = create_simple_mlp_model();
        let pipeline = MLPFittedPipeline::from_model(model);

        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);
        let predictions = pipeline.predict_flattened(&input).unwrap();

        assert_eq!(predictions.len(), 1);
    }

    #[test]
    fn test_mlp_fitted_pipeline_feature_mismatch() {
        let model = create_simple_mlp_model();
        let pipeline = MLPFittedPipeline::from_model(model);

        // Input with wrong number of features
        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 1, 3);
        let result = pipeline.predict(&input);

        assert!(matches!(
            result,
            Err(MLPPipelineError::FeatureMismatch {
                expected: 2,
                got: 3
            })
        ));
    }

    #[test]
    fn test_mlp_fitted_pipeline_with_preprocessing() {
        // Create preprocessing pipeline
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let preproc = Pipeline::<CpuBackend>::new().add_standard_scaler(StandardScaler::new());
        let fitted_preproc = preproc.fit(&data).unwrap();

        // Create model
        let model = create_simple_mlp_model();

        // Create pipeline
        let pipeline = MLPFittedPipeline::new(Some(fitted_preproc), None, model);

        assert_eq!(pipeline.n_features_in(), 2);
        assert!(pipeline.preprocessor().is_some());
    }

    #[test]
    fn test_mlp_fitted_pipeline_serialization() {
        let model = create_simple_mlp_model();
        let pipeline = MLPFittedPipeline::from_model(model);

        let temp_file = std::env::temp_dir().join("test_mlp_fitted_pipeline.bin");
        pipeline.save_to_file(&temp_file).unwrap();

        let loaded = MLPFittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();

        // Compare predictions
        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);
        let pred1 = pipeline.predict(&input).unwrap();
        let pred2 = loaded.predict(&input).unwrap();

        let p1_data = pred1.ravel().to_vec();
        let p2_data = pred2.ravel().to_vec();
        assert!((p1_data[0] - p2_data[0]).abs() < 1e-6);

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_mlp_fitted_pipeline_serialization_with_preprocessing() {
        // Create preprocessing pipeline
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let preproc = Pipeline::<CpuBackend>::new().add_standard_scaler(StandardScaler::new());
        let fitted_preproc = preproc.fit(&data).unwrap();

        let model = create_simple_mlp_model();
        let pipeline = MLPFittedPipeline::new(Some(fitted_preproc), None, model);

        let temp_file = std::env::temp_dir().join("test_mlp_fitted_pipeline_preproc.bin");
        pipeline.save_to_file(&temp_file).unwrap();

        let loaded = MLPFittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert_eq!(loaded.metadata().n_preproc_steps, 1);

        // Compare predictions
        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let pred1 = pipeline.predict(&input).unwrap();
        let pred2 = loaded.predict(&input).unwrap();

        for (a, b) in pred1
            .ravel()
            .to_vec()
            .iter()
            .zip(pred2.ravel().to_vec().iter())
        {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_mlp_fitted_pipeline_multi_output() {
        // Create a model with 2 outputs
        let model = MLP::<CpuBackend>::new(&[3, 4, 2], &[Activation::ReLU, Activation::Identity]);
        let fitted = model.into_fitted();
        let pipeline = MLPFittedPipeline::from_model(fitted);

        assert_eq!(pipeline.n_outputs(), 2);

        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 1, 3);
        let predictions = pipeline.predict(&input).unwrap();

        assert_eq!(predictions.shape(), (1, 2));
    }

    #[test]
    fn test_mlp_fitted_pipeline_invalid_file_format() {
        let temp_file = std::env::temp_dir().join("test_invalid_mlp_format.bin");
        std::fs::write(&temp_file, b"invalid data").unwrap();

        let result = MLPFittedPipeline::<CpuBackend>::load_from_file(&temp_file);
        assert!(matches!(result, Err(MLPPipelineError::InvalidState(_))));

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_mlp_fitted_pipeline_clone() {
        let model = create_simple_mlp_model();
        let pipeline = MLPFittedPipeline::from_model(model);

        let cloned = pipeline.clone();

        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);
        let pred1 = pipeline.predict(&input).unwrap();
        let pred2 = cloned.predict(&input).unwrap();

        for (a, b) in pred1
            .ravel()
            .to_vec()
            .iter()
            .zip(pred2.ravel().to_vec().iter())
        {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mlp_pipeline_error_display() {
        let err = MLPPipelineError::FeatureMismatch {
            expected: 5,
            got: 3,
        };
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("3"));

        let err = MLPPipelineError::InvalidState("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_mlp_pipeline_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: MLPPipelineError = io_err.into();
        assert!(matches!(err, MLPPipelineError::IoError(_)));
    }
}
