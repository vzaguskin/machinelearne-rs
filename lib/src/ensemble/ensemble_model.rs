//! Gradient Boosted Model for inference.
//!
//! A fitted gradient boosted ensemble that can make predictions and be serialized.
//! Follows the InferenceModel pattern from ADR-0001: contains only prediction
//! parameters (weak learners), no training state.

use crate::backend::{Backend, Scalar, Tensor1D, Tensor2D};
use crate::model::InferenceModel;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use super::decision_stump::FittedStump;

/// Serializable representation of a GradientBoostedModel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientBoostedModelParams {
    /// Initial prediction (F_0)
    pub initial_prediction: f64,
    /// Learning rate (shrinkage)
    pub learning_rate: f64,
    /// Number of features expected in input
    pub n_features: usize,
    /// Weak learners (decision stumps)
    pub weak_learners: Vec<FittedStump>,
}

/// Trait for types that can predict from raw feature vectors.
///
/// This abstraction allows the ensemble to work with any weak learner type.
pub trait StumpPredictor: Clone + Send + Sync + 'static {
    /// Predict for a single sample.
    fn predict_one(&self, features: &[f64]) -> f64;

    /// Predict for multiple samples.
    fn predict_batch<B: Backend>(&self, features: &Tensor2D<B>) -> Vec<f64>;
}

impl StumpPredictor for FittedStump {
    fn predict_one(&self, features: &[f64]) -> f64 {
        FittedStump::predict_one(self, features)
    }

    fn predict_batch<B: Backend>(&self, features: &Tensor2D<B>) -> Vec<f64> {
        FittedStump::predict_batch(self, features)
    }
}

/// A fitted gradient boosted ensemble model.
///
/// This model is the result of training with `GradientBoostingTrainer`.
/// It contains only the weak learners and learning rate needed for prediction.
///
/// # Type Parameters
/// - `B`: The backend for tensor operations
/// - `H`: The type of weak learner (e.g., `FittedStump`)
pub struct GradientBoostedModel<B: Backend, H = FittedStump> {
    /// Initial prediction (F_0)
    initial_prediction: Scalar<B>,
    /// Learning rate (shrinkage)
    learning_rate: f64,
    /// Number of features expected in input
    n_features: usize,
    /// Weak learners
    weak_learners: Vec<H>,
    _phantom: PhantomData<B>,
}

impl<B: Backend, H: Clone + StumpPredictor> GradientBoostedModel<B, H> {
    /// Create a new gradient boosted model from components.
    pub fn new(
        initial_prediction: f64,
        learning_rate: f64,
        n_features: usize,
        weak_learners: Vec<H>,
    ) -> Self {
        Self {
            initial_prediction: Scalar::new(initial_prediction),
            learning_rate,
            n_features,
            weak_learners,
            _phantom: PhantomData,
        }
    }

    /// Get the number of weak learners.
    pub fn n_estimators(&self) -> usize {
        self.weak_learners.len()
    }

    /// Get the learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Get the initial prediction.
    pub fn initial_prediction(&self) -> f64 {
        self.initial_prediction.to_f64()
    }

    /// Predict for a single sample.
    pub fn predict(&self, input: &Tensor1D<B>) -> Scalar<B> {
        let features: Vec<f64> = input.to_vec();
        let mut pred = self.initial_prediction.to_f64();

        for learner in &self.weak_learners {
            pred += self.learning_rate * learner.predict_one(&features);
        }

        Scalar::new(pred)
    }

    /// Predict for multiple samples.
    pub fn predict_batch(&self, input: &Tensor2D<B>) -> Tensor1D<B> {
        let n_samples = input.shape().0;
        let learner_preds: Vec<Vec<f64>> = self
            .weak_learners
            .iter()
            .map(|learner| learner.predict_batch(input))
            .collect();

        let predictions: Vec<f32> = (0..n_samples)
            .map(|i| {
                let mut pred = self.initial_prediction.to_f64();
                for learner_pred in &learner_preds {
                    pred += self.learning_rate * learner_pred[i];
                }
                pred as f32
            })
            .collect();

        Tensor1D::new(predictions)
    }
}

// Implement InferenceModel only for the concrete FittedStump case
// This allows us to properly handle serialization
impl<B: Backend> InferenceModel<B> for GradientBoostedModel<B, FittedStump> {
    type InputSingle = Tensor1D<B>;
    type OutputSingle = Scalar<B>;
    type InputBatch = Tensor2D<B>;
    type OutputBatch = Tensor1D<B>;
    type ParamsRepr = GradientBoostedModelParams;

    fn predict(&self, input: &Self::InputSingle) -> Self::OutputSingle {
        GradientBoostedModel::predict(self, input)
    }

    fn predict_batch(&self, input: &Self::InputBatch) -> Self::OutputBatch {
        GradientBoostedModel::predict_batch(self, input)
    }

    fn extract_params(&self) -> Self::ParamsRepr {
        GradientBoostedModelParams {
            initial_prediction: self.initial_prediction.to_f64(),
            learning_rate: self.learning_rate,
            n_features: self.n_features,
            weak_learners: self.weak_learners.clone(),
        }
    }

    fn from_params(params: Self::ParamsRepr) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::new(
            params.initial_prediction,
            params.learning_rate,
            params.n_features,
            params.weak_learners,
        ))
    }
}

/// A gradient boosted regressor (convenience type alias).
pub type GradientBoostedRegressor<B> = GradientBoostedModel<B, FittedStump>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::serialization::SerializableParams;

    #[test]
    fn test_gradient_boosted_model_predict_single() {
        let stumps = vec![
            FittedStump {
                feature_idx: 0,
                threshold: 1.5,
                left_value: 1.0,
                right_value: 2.0,
            },
            FittedStump {
                feature_idx: 0,
                threshold: 0.5,
                left_value: 0.5,
                right_value: 1.5,
            },
        ];

        let model: GradientBoostedModel<CpuBackend> =
            GradientBoostedModel::new(0.0, 0.5, 1, stumps);

        // For input [0.0]:
        // Stump 1: 0.0 < 1.5 -> 1.0, contribution: 0.5 * 1.0 = 0.5
        // Stump 2: 0.0 < 0.5 -> 0.5, contribution: 0.5 * 0.5 = 0.25
        // Total: 0.0 + 0.5 + 0.25 = 0.75
        let input = Tensor1D::<CpuBackend>::new(vec![0.0]);
        let pred = model.predict(&input);
        assert!((pred.to_f64() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_boosted_model_predict_batch() {
        let stumps = vec![FittedStump {
            feature_idx: 0,
            threshold: 1.5,
            left_value: 10.0,
            right_value: 20.0,
        }];

        let model: GradientBoostedModel<CpuBackend> =
            GradientBoostedModel::new(0.0, 1.0, 1, stumps);

        let input = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let predictions = model.predict_batch(&input);
        let pred_vec = predictions.to_vec();

        assert!((pred_vec[0] - 10.0).abs() < 1e-10);
        assert!((pred_vec[1] - 10.0).abs() < 1e-10);
        assert!((pred_vec[2] - 20.0).abs() < 1e-10);
        assert!((pred_vec[3] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_boosted_model_serialization() {
        let stumps = vec![
            FittedStump {
                feature_idx: 0,
                threshold: 1.5,
                left_value: 1.0,
                right_value: 2.0,
            },
            FittedStump {
                feature_idx: 1,
                threshold: 0.5,
                left_value: 0.5,
                right_value: 1.5,
            },
        ];

        let model: GradientBoostedModel<CpuBackend> =
            GradientBoostedModel::new(0.5, 0.1, 2, stumps);

        // Extract and restore params
        let params = model.extract_params();
        let restored: GradientBoostedModel<CpuBackend> =
            GradientBoostedModel::from_params(params).unwrap();

        assert_eq!(restored.n_estimators(), 2);
        assert!((restored.learning_rate() - 0.1).abs() < 1e-10);
        assert!((restored.initial_prediction() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_boosted_model_save_load_roundtrip() {
        let stumps = vec![FittedStump {
            feature_idx: 0,
            threshold: 1.0,
            left_value: 5.0,
            right_value: 10.0,
        }];

        let model: GradientBoostedModel<CpuBackend> =
            GradientBoostedModel::new(2.0, 0.5, 1, stumps);

        // Save to bytes
        let bytes = model.extract_params().to_bytes().unwrap();

        // Load from bytes
        let params = GradientBoostedModelParams::from_bytes(&bytes).unwrap();
        let loaded: GradientBoostedModel<CpuBackend> =
            GradientBoostedModel::from_params(params).unwrap();

        // Verify predictions match
        let input = Tensor1D::<CpuBackend>::new(vec![0.5]);
        let orig_pred = model.predict(&input);
        let loaded_pred = loaded.predict(&input);

        assert!((orig_pred.to_f64() - loaded_pred.to_f64()).abs() < 1e-10);
    }

    #[test]
    fn test_inference_model_trait() {
        let stumps = vec![FittedStump {
            feature_idx: 0,
            threshold: 1.0,
            left_value: 5.0,
            right_value: 10.0,
        }];

        let model: GradientBoostedRegressor<CpuBackend> =
            GradientBoostedModel::new(2.0, 0.5, 1, stumps);

        // Use InferenceModel trait methods
        // For input [0.5]:
        // - 0.5 < 1.0, so stump predicts 5.0
        // - contribution = learning_rate * 5.0 = 0.5 * 5.0 = 2.5
        // - total = initial + contribution = 2.0 + 2.5 = 4.5
        let input = Tensor1D::<CpuBackend>::new(vec![0.5]);
        let pred = <GradientBoostedModel<CpuBackend> as InferenceModel<CpuBackend>>::predict(
            &model, &input,
        );
        assert!((pred.to_f64() - 4.5).abs() < 1e-10);
    }
}
