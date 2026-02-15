//! Predictive pipeline combining preprocessing and model inference.
//!
//! This module provides a unified wrapper for a complete ML pipeline that
//! combines preprocessing transformers with a trained model.

use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::model::InferenceModel;
use crate::preprocessing::column_transformer::FittedColumnTransformer;
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::feature_engineering::FittedPolynomialFeatures;
use crate::preprocessing::traits::FittedTransformer;
use crate::serialization::SerializableParams;
use serde::{Deserialize, Serialize};
use std::io;
use std::marker::PhantomData;

/// Serializable parameters for the predictive pipeline.
#[derive(Clone, Serialize, Deserialize)]
pub struct PredictivePipelineParams {
    /// Preprocessor parameters.
    pub preprocessor: Vec<u8>,
    /// Polynomial features parameters (optional).
    pub poly: Option<Vec<u8>>,
    /// Model parameters.
    pub model: Vec<u8>,
    /// Number of input features.
    pub n_features_in: usize,
}

/// Predictive pipeline combining preprocessing and model inference.
///
/// This provides a unified interface for:
/// 1. Preprocessing new data using a fitted ColumnTransformer
/// 2. Optionally generating polynomial features
/// 3. Making predictions with a trained model
///
/// The entire pipeline can be serialized and loaded for deployment.
pub struct PredictivePipeline<
    B: Backend,
    M: InferenceModel<B, InputBatch = Tensor2D<B>, OutputBatch = Tensor1D<B>>,
> {
    /// Fitted column transformer for preprocessing.
    preprocessor: FittedColumnTransformer<B>,
    /// Optional polynomial features transformer.
    poly: Option<FittedPolynomialFeatures<B>>,
    /// Trained model for inference.
    model: M,
    /// Number of input features expected.
    n_features_in: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend, M: InferenceModel<B, InputBatch = Tensor2D<B>, OutputBatch = Tensor1D<B>>>
    PredictivePipeline<B, M>
{
    /// Create a new predictive pipeline.
    pub fn new(
        preprocessor: FittedColumnTransformer<B>,
        poly: Option<FittedPolynomialFeatures<B>>,
        model: M,
    ) -> Self {
        let n_features_in = preprocessor.n_features_in();
        Self {
            preprocessor,
            poly,
            model,
            n_features_in,
            _backend: PhantomData,
        }
    }

    /// Get the number of input features.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Preprocess data (transform using preprocessor and polynomial features).
    pub fn preprocess(&self, data: &Tensor2D<B>) -> Result<Tensor2D<B>, PreprocessingError> {
        let mut processed = self.preprocessor.transform(data)?;

        if let Some(ref poly) = self.poly {
            processed = poly.transform(&processed)?;
        }

        Ok(processed)
    }

    /// Make predictions on new data.
    pub fn predict(&self, data: &Tensor2D<B>) -> Result<Tensor1D<B>, PreprocessingError> {
        let (_, cols) = data.shape();

        if cols != self.n_features_in {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features_in,
                got_features: cols,
            });
        }

        let processed = self.preprocess(data)?;
        let predictions = self.model.predict_batch(&processed);

        Ok(predictions)
    }

    /// Save the entire pipeline to a file.
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> io::Result<()> {
        let params = self.extract_params();
        let bytes = params.to_bytes().map_err(io::Error::other)?;
        std::fs::write(path, bytes)
    }

    /// Load a pipeline from a file.
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
        model_loader: impl FnOnce(&[u8]) -> Result<M, PreprocessingError>,
    ) -> Result<Self, PreprocessingError> {
        let bytes = std::fs::read(path).map_err(|e| {
            PreprocessingError::SerializationError(format!("Failed to read file: {}", e))
        })?;
        let params = PredictivePipelineParams::from_bytes(&bytes)
            .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
        Self::from_params(params, model_loader)
    }

    /// Extract parameters for serialization.
    pub fn extract_params(&self) -> PredictivePipelineParams {
        let preprocessor_bytes = self.preprocessor.extract_params().to_bytes().unwrap();
        let poly_bytes = self
            .poly
            .as_ref()
            .map(|p| p.extract_params().to_bytes().unwrap());
        let model_bytes = self.model.extract_params().to_bytes().unwrap();

        PredictivePipelineParams {
            preprocessor: preprocessor_bytes,
            poly: poly_bytes,
            model: model_bytes,
            n_features_in: self.n_features_in,
        }
    }

    /// Reconstruct from parameters.
    pub fn from_params(
        params: PredictivePipelineParams,
        model_loader: impl FnOnce(&[u8]) -> Result<M, PreprocessingError>,
    ) -> Result<Self, PreprocessingError> {
        let preprocessor_params =
            crate::preprocessing::column_transformer::ColumnTransformerParams::from_bytes(
                &params.preprocessor,
            )
            .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
        let preprocessor = FittedColumnTransformer::from_params(preprocessor_params)?;

        let poly = if let Some(poly_bytes) = params.poly {
            let poly_params =
                crate::preprocessing::feature_engineering::PolynomialFeaturesParams::from_bytes(
                    &poly_bytes,
                )
                .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
            Some(FittedPolynomialFeatures::from_params(poly_params)?)
        } else {
            None
        };

        let model = model_loader(&params.model)?;

        Ok(Self {
            preprocessor,
            poly,
            model,
            n_features_in: params.n_features_in,
            _backend: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::model::linear::{LinearModel, LinearParams, SerializableLinearParams};
    use crate::model::state::Fitted;
    use crate::preprocessing::{
        ColumnSpec, ColumnTransformer, PolynomialFeatures, StandardScaler, Transformer,
    };

    // Helper to create a simple fitted model for testing
    fn create_test_model(n_features: usize) -> LinearModel<CpuBackend, Fitted> {
        let serial_params = SerializableLinearParams {
            weights: vec![1.0; n_features],
            bias: 0.0,
        };
        let params = LinearParams::try_from(serial_params).expect("Failed to convert params");
        <LinearModel<CpuBackend, Fitted>>::new(params)
    }

    #[test]
    fn test_predictive_pipeline_basic() {
        // Create simple data
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);

        // Build preprocessor
        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::All);
        let fitted_ct = ct.fit(&data).unwrap();

        // Create model
        let model = create_test_model(2);

        // Build pipeline
        let pipeline = PredictivePipeline::new(fitted_ct, None, model);

        // Predict
        let predictions = pipeline.predict(&data).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_predictive_pipeline_with_polynomial() {
        // Create simple data
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);

        // Build preprocessor
        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::All);
        let fitted_ct = ct.fit(&data).unwrap();

        // Transform to get intermediate data for poly fitting
        let preprocessed = fitted_ct.transform(&data).unwrap();

        // Build polynomial features
        let poly = PolynomialFeatures::<CpuBackend>::new()
            .with_degree(2)
            .with_include_bias(false);
        let fitted_poly = poly.fit(&preprocessed).unwrap();

        // Create model with correct number of features
        let n_poly_features = fitted_poly.n_features_out();
        let model = create_test_model(n_poly_features);

        // Build pipeline
        let pipeline = PredictivePipeline::new(fitted_ct, Some(fitted_poly), model);

        // Predict
        let predictions = pipeline.predict(&data).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_predictive_pipeline_preprocess() {
        // Create simple data
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);

        // Build preprocessor
        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::All);
        let fitted_ct = ct.fit(&data).unwrap();

        // Create model
        let model = create_test_model(2);

        // Build pipeline
        let pipeline = PredictivePipeline::new(fitted_ct, None, model);

        // Preprocess
        let processed = pipeline.preprocess(&data).unwrap();
        assert_eq!(processed.shape(), (2, 2));
    }

    #[test]
    fn test_predictive_pipeline_feature_mismatch() {
        // Create data with 2 features
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);

        // Build preprocessor for 2 features
        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::All);
        let fitted_ct = ct.fit(&data).unwrap();

        // Create model
        let model = create_test_model(2);

        // Build pipeline
        let pipeline = PredictivePipeline::new(fitted_ct, None, model);

        // Try to predict with wrong number of features
        let bad_data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 1, 3);
        let result = pipeline.predict(&bad_data);

        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch { .. })
        ));
    }

    #[test]
    fn test_predictive_pipeline_serialization() {
        // Create simple data
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);

        // Build preprocessor
        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::All);
        let fitted_ct = ct.fit(&data).unwrap();

        // Create model
        let model = create_test_model(2);

        // Build pipeline
        let pipeline = PredictivePipeline::new(fitted_ct, None, model);

        // Save
        let temp_file = std::env::temp_dir().join("test_predictive_pipeline.bin");
        pipeline.save_to_file(&temp_file).unwrap();

        // Load with model loader
        let loaded =
            PredictivePipeline::<CpuBackend, LinearModel<CpuBackend, Fitted>>::load_from_file(
                &temp_file,
                |bytes| {
                    let serial_params: SerializableLinearParams = bincode::deserialize(bytes)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    let params = LinearParams::try_from(serial_params)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    Ok(<LinearModel<CpuBackend, Fitted>>::new(params))
                },
            )
            .unwrap();

        // Verify
        assert_eq!(loaded.n_features_in(), pipeline.n_features_in());

        let p1 = pipeline.predict(&data).unwrap();
        let p2 = loaded.predict(&data).unwrap();

        let v1 = p1.to_vec();
        let v2 = p2.to_vec();
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(temp_file).ok();
    }
}
