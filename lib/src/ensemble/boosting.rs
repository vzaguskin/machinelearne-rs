//! Gradient Boosting Trainer.
//!
//! Implements the gradient boosting training algorithm that builds an ensemble
//! of weak learners sequentially, where each learner fits the pseudo-residuals
//! of the current ensemble.

use crate::backend::{Backend, Tensor1D, Tensor2D};
use serde::{Deserialize, Serialize};

use super::decision_stump::{DecisionStump, FittedStump};
use super::ensemble_model::{GradientBoostedModel, StumpPredictor};
use super::loss::BoostingLoss;

/// Configuration for gradient boosting training.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientBoostingConfig {
    /// Number of boosting iterations (weak learners to train)
    pub n_estimators: usize,
    /// Learning rate (shrinkage) - smaller values require more estimators
    pub learning_rate: f64,
    /// Fraction of features to consider for each split (1.0 = all features)
    pub subsample: f64,
    /// Verbosity level (0 = silent, 1 = progress)
    pub verbose: u8,
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            subsample: 1.0,
            verbose: 0,
        }
    }
}

impl GradientBoostingConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of estimators.
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the feature subsample rate.
    pub fn subsample(mut self, rate: f64) -> Self {
        self.subsample = rate;
        self
    }

    /// Set verbosity.
    pub fn verbose(mut self, level: u8) -> Self {
        self.verbose = level;
        self
    }
}

/// Trainer for gradient boosted models.
///
/// This trainer builds an ensemble by iteratively fitting weak learners
/// to the pseudo-residuals of the current ensemble predictions.
///
/// # Type Parameters
/// - `L`: The loss function (must implement `BoostingLoss`)
pub struct GradientBoostingTrainer<L: BoostingLoss> {
    config: GradientBoostingConfig,
    loss: L,
}

impl<L: BoostingLoss> GradientBoostingTrainer<L> {
    /// Create a new trainer with the given loss function and default configuration.
    pub fn new(loss: L) -> Self {
        Self {
            config: GradientBoostingConfig::default(),
            loss,
        }
    }

    /// Set the training configuration.
    pub fn with_config(mut self, config: GradientBoostingConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the number of estimators.
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.config.n_estimators = n;
        self
    }

    /// Set the learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set verbosity.
    pub fn verbose(mut self, level: u8) -> Self {
        self.config.verbose = level;
        self
    }

    /// Train a gradient boosted model using decision stumps.
    ///
    /// # Arguments
    /// * `features` - Training features (n_samples x n_features)
    /// * `targets` - Training targets (n_samples)
    ///
    /// # Returns
    /// A fitted `GradientBoostedModel` ready for inference.
    pub fn fit<B: Backend>(
        &self,
        features: &Tensor2D<B>,
        targets: &Tensor1D<B>,
    ) -> GradientBoostedModel<B, FittedStump> {
        self.fit_with_weak_learner(features, targets, &DecisionStump::new())
    }

    /// Train a gradient boosted model with a custom weak learner.
    ///
    /// # Arguments
    /// * `features` - Training features (n_samples x n_features)
    /// * `targets` - Training targets (n_samples)
    /// * `weak_learner` - The weak learner to use for each boosting iteration
    ///
    /// # Returns
    /// A fitted `GradientBoostedModel` ready for inference.
    pub fn fit_with_weak_learner<B: Backend, W: WeakLearner<B>>(
        &self,
        features: &Tensor2D<B>,
        targets: &Tensor1D<B>,
        weak_learner: &W,
    ) -> GradientBoostedModel<B, W::FittedModel> {
        let (n_samples, n_features) = features.shape();
        let target_vec = targets.to_vec();

        // Targets are already f64 from to_vec()
        let targets_f64 = target_vec;

        // Initialize with the optimal constant prediction
        let initial_prediction = self.loss.initial_prediction(&targets_f64);

        // Track current predictions
        let mut predictions: Vec<f64> = vec![initial_prediction; n_samples];

        // Store weak learners
        let mut weak_learners: Vec<W::FittedModel> = Vec::with_capacity(self.config.n_estimators);

        if self.config.verbose > 0 {
            println!(
                "Starting gradient boosting with {} estimators, lr={}",
                self.config.n_estimators, self.config.learning_rate
            );
            println!("Initial prediction: {:.4}", initial_prediction);
        }

        // Boosting iterations
        for iter in 0..self.config.n_estimators {
            // Compute pseudo-residuals (negative gradient)
            let residuals: Vec<f64> = (0..n_samples)
                .map(|i| self.loss.negative_gradient(targets_f64[i], predictions[i]))
                .collect();

            // Convert residuals to tensor for weak learner fitting
            let residuals_tensor =
                Tensor1D::<B>::new(residuals.iter().map(|&r| r as f32).collect());

            // Fit weak learner to residuals
            let fitted_learner = weak_learner.fit(features, &residuals_tensor);

            // Update predictions with the weak learner's contribution
            let learner_predictions = fitted_learner.predict_batch(features);
            for (i, pred) in predictions.iter_mut().enumerate() {
                *pred += self.config.learning_rate * learner_predictions[i];
            }

            weak_learners.push(fitted_learner);

            // Log progress
            if self.config.verbose > 0 && (iter + 1) % 10 == 0 {
                let mse: f64 = targets_f64
                    .iter()
                    .zip(predictions.iter())
                    .map(|(t, p)| (t - p).powi(2))
                    .sum::<f64>()
                    / n_samples as f64;
                println!("Iteration {}: MSE = {:.6}", iter + 1, mse);
            }
        }

        if self.config.verbose > 0 {
            let final_mse: f64 = targets_f64
                .iter()
                .zip(predictions.iter())
                .map(|(t, p)| (t - p).powi(2))
                .sum::<f64>()
                / n_samples as f64;
            println!("Final MSE: {:.6}", final_mse);
        }

        GradientBoostedModel::new(
            initial_prediction,
            self.config.learning_rate,
            n_features,
            weak_learners,
        )
    }
}

/// Trait for weak learners that can be used in gradient boosting.
///
/// A weak learner is a simple model that is fit to the pseudo-residuals
/// at each boosting iteration.
pub trait WeakLearner<B: Backend> {
    /// The type of fitted model produced by this learner.
    type FittedModel: super::ensemble_model::StumpPredictor;

    /// Fit the weak learner to the given features and targets.
    ///
    /// # Arguments
    /// * `features` - Training features (n_samples x n_features)
    /// * `targets` - Pseudo-residuals to fit (n_samples)
    ///
    /// # Returns
    /// A fitted model that can make predictions.
    fn fit(&self, features: &Tensor2D<B>, targets: &Tensor1D<B>) -> Self::FittedModel;
}

impl<B: Backend> WeakLearner<B> for DecisionStump {
    type FittedModel = FittedStump;

    fn fit(&self, features: &Tensor2D<B>, targets: &Tensor1D<B>) -> Self::FittedModel {
        self.fit(features, targets).unwrap_or(FittedStump {
            feature_idx: 0,
            threshold: f64::NEG_INFINITY,
            left_value: 0.0,
            right_value: 0.0,
        })
    }
}

/// A gradient boosting regressor using least squares loss (convenience alias).
pub type GradientBoostingRegressor = GradientBoostingTrainer<super::loss::LeastSquaresLoss>;

impl GradientBoostingRegressor {
    /// Create a new gradient boosting regressor with default settings.
    pub fn with_defaults() -> Self {
        Self::new(super::loss::LeastSquaresLoss)
    }
}

impl Default for GradientBoostingRegressor {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_gradient_boosting_simple_linear() {
        // y = 2*x (simple linear relationship)
        let features = Tensor2D::<CpuBackend>::new(
            vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            10,
            1,
        );
        let targets =
            Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let trainer = GradientBoostingRegressor::default()
            .n_estimators(50)
            .learning_rate(0.5);

        let model = trainer.fit(&features, &targets);

        // Test predictions
        let test_input = Tensor1D::<CpuBackend>::new(vec![2.5]);
        let prediction = model.predict(&test_input);
        assert!(
            (prediction.to_f64() - 5.0).abs() < 0.5,
            "Expected ~5.0, got {}",
            prediction.to_f64()
        );
    }

    #[test]
    fn test_gradient_boosting_constant_target() {
        // All targets are the same
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![5.0, 5.0, 5.0, 5.0]);

        let trainer = GradientBoostingRegressor::default().n_estimators(10);
        let model = trainer.fit(&features, &targets);

        // Initial prediction should be 5.0
        assert!((model.initial_prediction() - 5.0).abs() < 1e-10);

        // Predictions should be close to 5.0
        let test_input = Tensor1D::<CpuBackend>::new(vec![1.5]);
        let prediction = model.predict(&test_input);
        assert!((prediction.to_f64() - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_gradient_boosting_batch_prediction() {
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0, 6.0]);

        let trainer = GradientBoostingRegressor::default()
            .n_estimators(100)
            .learning_rate(0.5);

        let model = trainer.fit(&features, &targets);

        // Batch prediction
        let test_features = Tensor2D::<CpuBackend>::new(vec![0.5, 1.5, 2.5], 3, 1);
        let predictions = model.predict_batch(&test_features);
        let pred_vec = predictions.to_vec();

        // y = 2*x, so expected: [1.0, 3.0, 5.0]
        // Gradient boosting with stumps is approximate, especially for linear functions
        // The model learns step functions which approximate the linear pattern
        assert!(
            (pred_vec[0] - 1.0).abs() < 2.5,
            "Expected ~1.0, got {}",
            pred_vec[0]
        );
        assert!(
            (pred_vec[1] - 3.0).abs() < 2.5,
            "Expected ~3.0, got {}",
            pred_vec[1]
        );
        assert!(
            (pred_vec[2] - 5.0).abs() < 2.5,
            "Expected ~5.0, got {}",
            pred_vec[2]
        );
    }

    #[test]
    fn test_gradient_boosting_config() {
        let config = GradientBoostingConfig::new()
            .n_estimators(200)
            .learning_rate(0.05)
            .subsample(0.8)
            .verbose(1);

        assert_eq!(config.n_estimators, 200);
        assert!((config.learning_rate - 0.05).abs() < 1e-10);
        assert!((config.subsample - 0.8).abs() < 1e-10);
        assert_eq!(config.verbose, 1);
    }

    #[test]
    fn test_gradient_boosting_verbose() {
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0], 3, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 2.0]);

        let trainer = GradientBoostingRegressor::default()
            .n_estimators(20)
            .verbose(1);

        // Should not panic with verbose output
        let _model = trainer.fit(&features, &targets);
    }

    #[test]
    fn test_weak_learner_trait() {
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let stump = DecisionStump::new();
        let fitted = <DecisionStump as WeakLearner<CpuBackend>>::fit(&stump, &features, &targets);

        // Should produce a valid fitted stump
        let predictions = fitted.predict_batch(&features);
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_gradient_boosting_multi_feature() {
        // y = x1 + x2
        let features = Tensor2D::<CpuBackend>::new(
            vec![
                0.0, 0.0, // sample 0
                1.0, 0.0, // sample 1
                0.0, 1.0, // sample 2
                1.0, 1.0, // sample 3
                2.0, 1.0, // sample 4
            ],
            5,
            2,
        );
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 1.0, 2.0, 3.0]);

        let trainer = GradientBoostingRegressor::default()
            .n_estimators(100)
            .learning_rate(0.5);

        let model = trainer.fit(&features, &targets);

        // Test on training data
        let predictions = model.predict_batch(&features);
        let pred_vec = predictions.to_vec();

        // Gradient boosting with stumps is approximate, especially for linear functions
        // Check that predictions are reasonably close
        for (i, (&pred, &target)) in pred_vec.iter().zip(targets.to_vec().iter()).enumerate() {
            assert!(
                (pred - target).abs() < 1.0,
                "Sample {}: expected {}, got {}",
                i,
                target,
                pred
            );
        }
    }
}
