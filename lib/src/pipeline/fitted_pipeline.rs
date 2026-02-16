//! Fitted pipeline for end-to-end inference.
//!
//! Combines preprocessing transformers with a trained model for unified
//! prediction and serialization.

use std::io;
use std::marker::PhantomData;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::model::linear::{Fitted, LinearModel, SerializableLinearParams};
use crate::model::InferenceModel;
use crate::preprocessing::feature_engineering::{
    FittedPolynomialFeatures, PolynomialFeaturesParams,
};
use crate::preprocessing::pipeline::{FittedPipeline as PreprocPipeline, PipelineStepEnum};
use crate::preprocessing::traits::FittedTransformer;

/// Error type for pipeline operations.
#[derive(Debug)]
pub enum PipelineError {
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

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::IoError(e) => write!(f, "IO error: {}", e),
            PipelineError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            PipelineError::PreprocessingError(e) => write!(f, "Preprocessing error: {:?}", e),
            PipelineError::FeatureMismatch { expected, got } => {
                write!(f, "Feature mismatch: expected {}, got {}", expected, got)
            }
            PipelineError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<io::Error> for PipelineError {
    fn from(e: io::Error) -> Self {
        PipelineError::IoError(e)
    }
}

impl From<bincode::Error> for PipelineError {
    fn from(e: bincode::Error) -> Self {
        PipelineError::SerializationError(e.to_string())
    }
}

impl From<crate::preprocessing::error::PreprocessingError> for PipelineError {
    fn from(e: crate::preprocessing::error::PreprocessingError) -> Self {
        PipelineError::PreprocessingError(e)
    }
}

/// Metadata for a fitted pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineMetadata {
    /// Number of input features (before preprocessing).
    pub n_features_in: usize,
    /// Number of features after preprocessing (before polynomial).
    pub n_features_after_preproc: usize,
    /// Number of features after polynomial expansion (model input).
    pub n_features_model: usize,
    /// Whether polynomial features are applied.
    pub has_polynomial: bool,
    /// Polynomial degree (1 = no expansion).
    pub poly_degree: usize,
    /// Number of preprocessing steps.
    pub n_preproc_steps: usize,
    /// Pipeline version for compatibility.
    pub version: u32,
}

impl Default for PipelineMetadata {
    fn default() -> Self {
        Self {
            n_features_in: 0,
            n_features_after_preproc: 0,
            n_features_model: 0,
            has_polynomial: false,
            poly_degree: 1,
            n_preproc_steps: 0,
            version: 1,
        }
    }
}

/// Serializable representation of a fitted pipeline.
#[derive(Serialize, Deserialize)]
struct SerializablePipeline {
    metadata: PipelineMetadata,
    /// Serialized preprocessing pipeline steps.
    preproc_data: Vec<(String, Vec<u8>)>,
    /// Serialized polynomial features (if present).
    poly_data: Option<Vec<u8>>,
    /// Serialized model parameters.
    model_data: Vec<u8>,
}

/// A fitted pipeline combining preprocessing and model for end-to-end inference.
///
/// This struct holds:
/// - A preprocessing pipeline (scalers, imputers, encoders)
/// - Optional polynomial feature expansion
/// - A trained linear model
///
/// It provides unified `predict` methods that apply the full transformation
/// chain and model inference.
pub struct FittedPipeline<B: Backend> {
    /// Preprocessing pipeline (scalers, imputers, etc.).
    preprocessor: Option<PreprocPipeline<B>>,
    /// Optional polynomial features.
    polynomial: Option<FittedPolynomialFeatures<B>>,
    /// Trained linear model.
    model: LinearModel<B, Fitted>,
    /// Pipeline metadata.
    metadata: PipelineMetadata,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedPipeline<B> {
    /// Create a new fitted pipeline from components.
    ///
    /// # Arguments
    /// * `preprocessor` - Optional preprocessing pipeline
    /// * `polynomial` - Optional polynomial feature transformer
    /// * `model` - Trained linear model
    pub fn new(
        preprocessor: Option<PreprocPipeline<B>>,
        polynomial: Option<FittedPolynomialFeatures<B>>,
        model: LinearModel<B, Fitted>,
    ) -> Self {
        let n_features_in = preprocessor
            .as_ref()
            .map(|p| p.n_features_in())
            .unwrap_or_else(|| {
                polynomial
                    .as_ref()
                    .map(|p| p.n_features_in())
                    .unwrap_or_else(|| model.extract_params().weights.len())
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

        let n_preproc_steps = preprocessor.as_ref().map(|p| p.len()).unwrap_or(0);

        let metadata = PipelineMetadata {
            n_features_in,
            n_features_after_preproc,
            n_features_model,
            has_polynomial,
            poly_degree,
            n_preproc_steps,
            version: 1,
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
    pub fn from_model(model: LinearModel<B, Fitted>) -> Self {
        let n_features = model.extract_params().weights.len();
        let metadata = PipelineMetadata {
            n_features_in: n_features,
            n_features_after_preproc: n_features,
            n_features_model: n_features,
            has_polynomial: false,
            poly_degree: 1,
            n_preproc_steps: 0,
            version: 1,
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
    /// 3. Runs model inference
    ///
    /// # Arguments
    /// * `raw_data` - Raw input features (n_samples x n_features_in)
    ///
    /// # Returns
    /// Predictions as a 1D tensor
    pub fn predict(&self, raw_data: &Tensor2D<B>) -> Result<Tensor1D<B>, PipelineError> {
        let (_, cols) = raw_data.shape();

        if cols != self.metadata.n_features_in {
            return Err(PipelineError::FeatureMismatch {
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

    /// Get the number of input features.
    pub fn n_features_in(&self) -> usize {
        self.metadata.n_features_in
    }

    /// Get the metadata.
    pub fn metadata(&self) -> &PipelineMetadata {
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
    pub fn model(&self) -> &LinearModel<B, Fitted> {
        &self.model
    }

    /// Save the pipeline to a file.
    ///
    /// The file format is:
    /// - Magic bytes "MLPL" (4 bytes)
    /// - Version (4 bytes u32)
    /// - Metadata (JSON)
    /// - Preprocessing steps
    /// - Polynomial features (optional)
    /// - Model parameters
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), PipelineError> {
        use crate::preprocessing::traits::FittedTransformer;

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
        let serializable = SerializablePipeline {
            metadata: self.metadata.clone(),
            preproc_data,
            poly_data,
            model_data,
        };

        // Serialize and write
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MLPL"); // Magic bytes
        bytes.extend_from_slice(&1u32.to_le_bytes()); // Version
        bytes.extend_from_slice(&bincode::serialize(&serializable)?);

        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a pipeline from a file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, PipelineError> {
        use crate::preprocessing::encoding::{
            FittedOneHotEncoder, FittedOrdinalEncoder, OneHotEncoderParams, OrdinalEncoderParams,
        };
        use crate::preprocessing::imputation::{FittedSimpleImputer, SimpleImputerParams};
        use crate::preprocessing::scaling::{
            FittedMaxAbsScaler, FittedMinMaxScaler, FittedNormalizer, FittedRobustScaler,
            FittedStandardScaler, MaxAbsScalerParams, MinMaxScalerParams, NormalizerParams,
            RobustScalerParams, StandardScalerParams,
        };
        use crate::preprocessing::traits::FittedTransformer;

        let bytes = std::fs::read(path)?;

        // Verify magic bytes
        if bytes.len() < 8 || &bytes[0..4] != b"MLPL" {
            return Err(PipelineError::InvalidState(
                "Invalid file format".to_string(),
            ));
        }

        // Read version
        let _version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        // Deserialize
        let serializable: SerializablePipeline = bincode::deserialize(&bytes[8..])?;

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
                    return Err(PipelineError::InvalidState(format!(
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
        let model_params: SerializableLinearParams =
            bincode::deserialize(&serializable.model_data)?;
        let model = LinearModel::<B, Fitted>::from_params(model_params)
            .map_err(|e| PipelineError::SerializationError(e.to_string()))?;

        Ok(Self {
            preprocessor,
            polynomial,
            model,
            metadata: serializable.metadata,
            _backend: PhantomData,
        })
    }
}

impl<B: Backend> Clone for FittedPipeline<B> {
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
    use crate::model::linear::LinearRegression;
    use crate::model::TrainableModel;
    use crate::preprocessing::feature_engineering::PolynomialFeatures;
    use crate::preprocessing::pipeline::Pipeline;
    use crate::preprocessing::scaling::StandardScaler;
    use crate::preprocessing::traits::Transformer;

    fn create_simple_model() -> LinearModel<CpuBackend, Fitted> {
        // Create a simple model: y = 2*x1 + 3*x2 + 1
        let model = LinearRegression::<CpuBackend>::new(2);
        // In a real scenario, we'd train the model, but for testing we'll use from_params
        let params = crate::model::linear::LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![2.0, 3.0]),
            bias: crate::backend::scalar::Scalar::new(1.0),
        };
        LinearRegression::from_params(params).into_fitted()
    }

    #[test]
    fn test_fitted_pipeline_from_model() {
        let model = create_simple_model();
        let pipeline = FittedPipeline::from_model(model);

        assert_eq!(pipeline.n_features_in(), 2);
        assert!(pipeline.preprocessor().is_none());
        assert!(pipeline.polynomial().is_none());
    }

    #[test]
    fn test_fitted_pipeline_predict_simple() {
        let model = create_simple_model();
        let pipeline = FittedPipeline::from_model(model);

        // Input: [[1, 2], [3, 4]]
        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let predictions = pipeline.predict(&input).unwrap();

        // y1 = 2*1 + 3*2 + 1 = 9
        // y2 = 2*3 + 3*4 + 1 = 19
        let vals = predictions.to_vec();
        assert!((vals[0] - 9.0).abs() < 1e-6);
        assert!((vals[1] - 19.0).abs() < 1e-6);
    }

    #[test]
    fn test_fitted_pipeline_feature_mismatch() {
        let model = create_simple_model();
        let pipeline = FittedPipeline::from_model(model);

        // Input with wrong number of features
        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 1, 3);
        let result = pipeline.predict(&input);

        assert!(matches!(
            result,
            Err(PipelineError::FeatureMismatch {
                expected: 2,
                got: 3
            })
        ));
    }

    #[test]
    fn test_fitted_pipeline_with_preprocessing() {
        // Create preprocessing pipeline
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let preproc = Pipeline::<CpuBackend>::new().add_standard_scaler(StandardScaler::new());
        let fitted_preproc = preproc.fit(&data).unwrap();

        // Create model for 2 features
        let model = create_simple_model();

        // Create pipeline
        let pipeline = FittedPipeline::new(Some(fitted_preproc), None, model);

        assert_eq!(pipeline.n_features_in(), 2);
        assert!(pipeline.preprocessor().is_some());
    }

    #[test]
    fn test_fitted_pipeline_with_polynomial() {
        // Create polynomial features
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);
        let poly = PolynomialFeatures::<CpuBackend>::new()
            .with_degree(2)
            .with_include_bias(false);
        let fitted_poly = poly.fit(&data).unwrap();

        // Create model for 5 features (2 linear + 3 quadratic)
        let model_params = crate::model::linear::LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]),
            bias: crate::backend::scalar::Scalar::new(0.0),
        };
        let model = LinearRegression::<CpuBackend>::from_params(model_params).into_fitted();

        // Create pipeline
        let pipeline = FittedPipeline::new(None, Some(fitted_poly), model);

        assert_eq!(pipeline.n_features_in(), 2);
        assert!(pipeline.polynomial().is_some());
        assert_eq!(pipeline.metadata().n_features_model, 5);
    }

    #[test]
    fn test_fitted_pipeline_serialization() {
        let model = create_simple_model();
        let pipeline = FittedPipeline::from_model(model);

        let temp_file = std::env::temp_dir().join("test_fitted_pipeline.bin");
        pipeline.save_to_file(&temp_file).unwrap();

        let loaded = FittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();

        // Compare predictions
        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);
        let pred1 = pipeline.predict(&input).unwrap();
        let pred2 = loaded.predict(&input).unwrap();

        assert!((pred1.to_vec()[0] - pred2.to_vec()[0]).abs() < 1e-6);

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_fitted_pipeline_serialization_with_preprocessing() {
        // Create preprocessing pipeline
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let preproc = Pipeline::<CpuBackend>::new().add_standard_scaler(StandardScaler::new());
        let fitted_preproc = preproc.fit(&data).unwrap();

        let model = create_simple_model();
        let pipeline = FittedPipeline::new(Some(fitted_preproc), None, model);

        let temp_file = std::env::temp_dir().join("test_fitted_pipeline_preproc.bin");
        pipeline.save_to_file(&temp_file).unwrap();

        let loaded = FittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert_eq!(loaded.metadata().n_preproc_steps, 1);

        // Compare predictions
        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let pred1 = pipeline.predict(&input).unwrap();
        let pred2 = loaded.predict(&input).unwrap();

        for (a, b) in pred1.to_vec().iter().zip(pred2.to_vec().iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_fitted_pipeline_serialization_with_polynomial() {
        // Create polynomial features
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);
        let poly = PolynomialFeatures::<CpuBackend>::new()
            .with_degree(2)
            .with_include_bias(false);
        let fitted_poly = poly.fit(&data).unwrap();

        // Create model for 5 features
        let model_params = crate::model::linear::LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]),
            bias: crate::backend::scalar::Scalar::new(0.0),
        };
        let model = LinearRegression::<CpuBackend>::from_params(model_params).into_fitted();

        let pipeline = FittedPipeline::new(None, Some(fitted_poly), model);

        let temp_file = std::env::temp_dir().join("test_fitted_pipeline_poly.bin");
        pipeline.save_to_file(&temp_file).unwrap();

        let loaded = FittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert!(loaded.polynomial().is_some());
        assert_eq!(loaded.metadata().has_polynomial, true);
        assert_eq!(loaded.metadata().poly_degree, 2);

        // Compare predictions
        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);
        let pred1 = pipeline.predict(&input).unwrap();
        let pred2 = loaded.predict(&input).unwrap();

        assert!((pred1.to_vec()[0] - pred2.to_vec()[0]).abs() < 1e-6);

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_fitted_pipeline_clone() {
        let model = create_simple_model();
        let pipeline = FittedPipeline::from_model(model);

        let cloned = pipeline.clone();

        let input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);
        let pred1 = pipeline.predict(&input).unwrap();
        let pred2 = cloned.predict(&input).unwrap();

        assert!((pred1.to_vec()[0] - pred2.to_vec()[0]).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_error_display() {
        let err = PipelineError::FeatureMismatch {
            expected: 5,
            got: 3,
        };
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("3"));

        let err = PipelineError::InvalidState("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_pipeline_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: PipelineError = io_err.into();
        assert!(matches!(err, PipelineError::IoError(_)));
    }

    #[test]
    fn test_fitted_pipeline_metadata() {
        let model = create_simple_model();
        let pipeline = FittedPipeline::from_model(model);

        let metadata = pipeline.metadata();
        assert_eq!(metadata.n_features_in, 2);
        assert_eq!(metadata.has_polynomial, false);
        assert_eq!(metadata.poly_degree, 1);
        assert_eq!(metadata.n_preproc_steps, 0);
    }

    #[test]
    fn test_fitted_pipeline_model_accessor() {
        let model = create_simple_model();
        let pipeline = FittedPipeline::from_model(model.clone());

        let model_ref = pipeline.model();
        let orig_params = model.extract_params();
        let ref_params = model_ref.extract_params();

        for (a, b) in orig_params.weights.iter().zip(ref_params.weights.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fitted_pipeline_with_full_chain() {
        // Create preprocessing pipeline with scaler
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let preproc = Pipeline::<CpuBackend>::new().add_standard_scaler(StandardScaler::new());
        let fitted_preproc = preproc.fit(&data).unwrap();

        // Create polynomial features
        let poly = PolynomialFeatures::<CpuBackend>::new()
            .with_degree(2)
            .with_include_bias(false);
        let fitted_poly = poly.fit(&data).unwrap();

        // Create model for 5 features
        let model_params = crate::model::linear::LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]),
            bias: crate::backend::scalar::Scalar::new(0.0),
        };
        let model = LinearRegression::<CpuBackend>::from_params(model_params).into_fitted();

        // Create pipeline with all components
        let pipeline = FittedPipeline::new(Some(fitted_preproc), Some(fitted_poly), model);

        assert!(pipeline.preprocessor().is_some());
        assert!(pipeline.polynomial().is_some());
        assert_eq!(pipeline.metadata().n_preproc_steps, 1);
        assert_eq!(pipeline.metadata().has_polynomial, true);

        // Test prediction
        let pred = pipeline.predict(&data).unwrap();
        assert_eq!(pred.to_vec().len(), 3);
    }

    #[test]
    fn test_fitted_pipeline_invalid_file_format() {
        let temp_file = std::env::temp_dir().join("test_invalid_format.bin");
        std::fs::write(&temp_file, b"invalid data").unwrap();

        let result = FittedPipeline::<CpuBackend>::load_from_file(&temp_file);
        assert!(matches!(result, Err(PipelineError::InvalidState(_))));

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_fitted_pipeline_serialization_full_chain() {
        // Create preprocessing pipeline
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let preproc = Pipeline::<CpuBackend>::new().add_standard_scaler(StandardScaler::new());
        let fitted_preproc = preproc.fit(&data).unwrap();

        // Create polynomial features
        let poly = PolynomialFeatures::<CpuBackend>::new()
            .with_degree(2)
            .with_include_bias(false);
        let fitted_poly = poly.fit(&data).unwrap();

        // Create model
        let model_params = crate::model::linear::LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]),
            bias: crate::backend::scalar::Scalar::new(0.0),
        };
        let model = LinearRegression::<CpuBackend>::from_params(model_params).into_fitted();

        let pipeline = FittedPipeline::new(Some(fitted_preproc), Some(fitted_poly), model);

        let temp_file = std::env::temp_dir().join("test_fitted_pipeline_full.bin");
        pipeline.save_to_file(&temp_file).unwrap();

        let loaded = FittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();

        // Compare predictions
        let pred1 = pipeline.predict(&data).unwrap();
        let pred2 = loaded.predict(&data).unwrap();

        for (a, b) in pred1.to_vec().iter().zip(pred2.to_vec().iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_pipeline_error_serialization() {
        let err = PipelineError::SerializationError("test error".to_string());
        assert!(err.to_string().contains("Serialization error"));
    }

    #[test]
    fn test_fitted_pipeline_serialization_minmax() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let preproc = Pipeline::<CpuBackend>::new()
            .add_minmax_scaler(crate::preprocessing::scaling::MinMaxScaler::new());
        let fitted_preproc = preproc.fit(&data).unwrap();
        let model = create_simple_model();
        let pipeline = FittedPipeline::new(Some(fitted_preproc), None, model);

        let temp_file = std::env::temp_dir().join("test_fitted_pipeline_minmax.bin");
        pipeline.save_to_file(&temp_file).unwrap();

        let loaded = FittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();
        assert!(loaded.preprocessor().is_some());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_fitted_pipeline_serialization_robust() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let preproc = Pipeline::<CpuBackend>::new()
            .add_robust_scaler(crate::preprocessing::scaling::RobustScaler::new());
        let fitted_preproc = preproc.fit(&data).unwrap();
        let model = create_simple_model();
        let pipeline = FittedPipeline::new(Some(fitted_preproc), None, model);

        let temp_file = std::env::temp_dir().join("test_fitted_pipeline_robust.bin");
        pipeline.save_to_file(&temp_file).unwrap();

        let loaded = FittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();
        assert!(loaded.preprocessor().is_some());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_fitted_pipeline_serialization_maxabs() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let preproc = Pipeline::<CpuBackend>::new()
            .add_maxabs_scaler(crate::preprocessing::scaling::MaxAbsScaler::new());
        let fitted_preproc = preproc.fit(&data).unwrap();
        let model = create_simple_model();
        let pipeline = FittedPipeline::new(Some(fitted_preproc), None, model);

        let temp_file = std::env::temp_dir().join("test_fitted_pipeline_maxabs.bin");
        pipeline.save_to_file(&temp_file).unwrap();

        let loaded = FittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();
        assert!(loaded.preprocessor().is_some());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_fitted_pipeline_serialization_normalizer() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let preproc = Pipeline::<CpuBackend>::new().add_normalizer(
            crate::preprocessing::scaling::Normalizer::new(
                crate::preprocessing::scaling::NormType::L2,
            ),
        );
        let fitted_preproc = preproc.fit(&data).unwrap();
        let model = create_simple_model();
        let pipeline = FittedPipeline::new(Some(fitted_preproc), None, model);

        let temp_file = std::env::temp_dir().join("test_fitted_pipeline_normalizer.bin");
        pipeline.save_to_file(&temp_file).unwrap();

        let loaded = FittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();
        assert!(loaded.preprocessor().is_some());

        std::fs::remove_file(temp_file).ok();
    }
}
