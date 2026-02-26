//! Gradient Boosting Trainer.
//!
//! Implements the gradient boosting training algorithm that builds an ensemble
//! of weak learners sequentially, where each learner fits the pseudo-residuals
//! of the current ensemble.

use crate::backend::{Backend, Tensor1D, Tensor2D};
use rand::prelude::*;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

use super::decision_stump::{DecisionStump, FittedStump};
use super::ensemble_model::{GradientBoostedModel, StumpPredictor};
use super::loss::BoostingLoss;

/// Configuration for early stopping during training.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Fraction of training data to use for validation (default: 0.1)
    pub validation_fraction: f64,
    /// Number of iterations with no improvement to wait before stopping (default: 10)
    pub n_iter_no_change: usize,
    /// Minimum improvement to qualify as an improvement (default: 1e-4)
    pub tol: f64,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            validation_fraction: 0.1,
            n_iter_no_change: 10,
            tol: 1e-4,
        }
    }
}

impl EarlyStoppingConfig {
    /// Create a new early stopping configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the validation fraction.
    pub fn validation_fraction(mut self, fraction: f64) -> Self {
        self.validation_fraction = fraction.clamp(0.0, 0.5);
        self
    }

    /// Set the number of iterations with no improvement before stopping.
    pub fn n_iter_no_change(mut self, n: usize) -> Self {
        self.n_iter_no_change = n.max(1);
        self
    }

    /// Set the minimum improvement tolerance.
    pub fn tol(mut self, tolerance: f64) -> Self {
        self.tol = tolerance.max(0.0);
        self
    }
}

/// Configuration for gradient boosting training.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientBoostingConfig {
    /// Number of boosting iterations (weak learners to train)
    pub n_estimators: usize,
    /// Learning rate (shrinkage) - smaller values require more estimators
    pub learning_rate: f64,
    /// Fraction of features to consider for each split (1.0 = all features)
    pub subsample: f64,
    /// Fraction of features to use for each tree (1.0 = all features)
    pub colsample_bytree: f64,
    /// Random seed for reproducibility (None = random seed)
    pub random_seed: Option<u64>,
    /// Early stopping configuration (None = disabled)
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Verbosity level (0 = silent, 1 = progress)
    pub verbose: u8,
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            subsample: 1.0,
            colsample_bytree: 1.0,
            random_seed: None,
            early_stopping: None,
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

    /// Set the column subsample rate per tree.
    pub fn colsample_bytree(mut self, rate: f64) -> Self {
        self.colsample_bytree = rate.clamp(0.0, 1.0);
        self
    }

    /// Set the random seed for reproducibility.
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Enable early stopping with the given configuration.
    pub fn early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.early_stopping = Some(config);
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

    /// Set the column subsample rate per tree.
    pub fn colsample_bytree(mut self, rate: f64) -> Self {
        self.config.colsample_bytree = rate.clamp(0.0, 1.0);
        self
    }

    /// Set the random seed for reproducibility.
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Enable early stopping with the given configuration.
    pub fn early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.config.early_stopping = Some(config);
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
        let feature_data: Vec<f64> = features.ravel().to_vec();
        let target_vec = targets.to_vec();

        // Check if early stopping is enabled
        let early_stopping_config = self.config.early_stopping.as_ref();

        // Split data into train/validation if early stopping is enabled
        let (train_indices, val_indices) = if let Some(es_config) = early_stopping_config {
            let n_val = ((n_samples as f64) * es_config.validation_fraction).floor() as usize;
            let n_val = n_val.max(1).min(n_samples / 2); // At least 1, at most half
            let n_train = n_samples - n_val;

            // Use last n_val samples for validation
            let train: Vec<usize> = (0..n_train).collect();
            let val: Vec<usize> = (n_train..n_samples).collect();
            (train, val)
        } else {
            ((0..n_samples).collect(), Vec::new())
        };

        // Extract train and validation data
        let train_targets: Vec<f64> = train_indices.iter().map(|&i| target_vec[i]).collect();
        let val_targets: Vec<f64> = val_indices.iter().map(|&i| target_vec[i]).collect();

        // Initialize with the optimal constant prediction (using training data only)
        let initial_prediction = self.loss.initial_prediction(&train_targets);

        // Track current predictions for training data
        let mut train_predictions: Vec<f64> = vec![initial_prediction; train_indices.len()];

        // Track current predictions for validation data (if early stopping enabled)
        let mut val_predictions: Vec<f64> = if !val_indices.is_empty() {
            vec![initial_prediction; val_indices.len()]
        } else {
            Vec::new()
        };

        // Store weak learners
        let mut weak_learners: Vec<W::FittedModel> = Vec::with_capacity(self.config.n_estimators);

        // Early stopping state
        let mut best_val_loss: Option<f64> = None;
        let mut best_iteration: usize = 0;
        let mut no_improvement_count: usize = 0;

        // Initialize RNG for feature subsampling
        let mut rng: SmallRng = match self.config.random_seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_entropy(),
        };

        if self.config.verbose > 0 {
            println!(
                "Starting gradient boosting with {} estimators, lr={}",
                self.config.n_estimators, self.config.learning_rate
            );
            if early_stopping_config.is_some() {
                println!(
                    "Early stopping enabled: validation_fraction={}, n_iter_no_change={}, tol={}",
                    early_stopping_config.as_ref().unwrap().validation_fraction,
                    early_stopping_config.as_ref().unwrap().n_iter_no_change,
                    early_stopping_config.as_ref().unwrap().tol
                );
            }
            println!("Initial prediction: {:.4}", initial_prediction);
        }

        // Boosting iterations
        let mut stopped_early = false;
        for iter in 0..self.config.n_estimators {
            // Compute pseudo-residuals (negative gradient) for training data
            let residuals: Vec<f64> = (0..train_indices.len())
                .map(|i| {
                    self.loss
                        .negative_gradient(train_targets[i], train_predictions[i])
                })
                .collect();

            // Create training features tensor
            let mut train_features_data: Vec<f32> =
                Vec::with_capacity(train_indices.len() * n_features);
            for &i in &train_indices {
                for j in 0..n_features {
                    train_features_data.push(feature_data[i * n_features + j] as f32);
                }
            }
            let train_features =
                Tensor2D::<B>::new(train_features_data, train_indices.len(), n_features);
            let residuals_tensor =
                Tensor1D::<B>::new(residuals.iter().map(|&r| r as f32).collect());

            // Generate feature mask for this iteration (colsample_bytree)
            let feature_mask = self.generate_feature_mask(n_features, &mut rng);

            // Fit weak learner to residuals with feature mask
            let fitted_learner =
                weak_learner.fit(&train_features, &residuals_tensor, feature_mask.as_deref());

            // Update training predictions with the weak learner's contribution
            let train_learner_predictions = fitted_learner.predict_batch(&train_features);
            for (i, pred) in train_predictions.iter_mut().enumerate() {
                *pred += self.config.learning_rate * train_learner_predictions[i];
            }

            // Update validation predictions if early stopping is enabled
            if !val_indices.is_empty() {
                // Create validation features tensor
                let mut val_features_data: Vec<f32> =
                    Vec::with_capacity(val_indices.len() * n_features);
                for &i in &val_indices {
                    for j in 0..n_features {
                        val_features_data.push(feature_data[i * n_features + j] as f32);
                    }
                }
                let val_features =
                    Tensor2D::<B>::new(val_features_data, val_indices.len(), n_features);

                let val_learner_predictions = fitted_learner.predict_batch(&val_features);
                for (i, pred) in val_predictions.iter_mut().enumerate() {
                    *pred += self.config.learning_rate * val_learner_predictions[i];
                }
            }

            weak_learners.push(fitted_learner);

            // Early stopping check
            if let Some(es_config) = early_stopping_config {
                // Compute validation loss
                let val_loss: f64 = if !val_predictions.is_empty() {
                    val_targets
                        .iter()
                        .zip(val_predictions.iter())
                        .map(|(t, p)| (t - p).powi(2))
                        .sum::<f64>()
                        / val_predictions.len() as f64
                } else {
                    // Fallback to training loss if no validation set
                    train_targets
                        .iter()
                        .zip(train_predictions.iter())
                        .map(|(t, p)| (t - p).powi(2))
                        .sum::<f64>()
                        / train_predictions.len() as f64
                };

                // Check for improvement
                let improved = match best_val_loss {
                    None => true,
                    Some(best) => val_loss < best - es_config.tol,
                };

                if improved {
                    best_val_loss = Some(val_loss);
                    best_iteration = iter;
                    no_improvement_count = 0;

                    if self.config.verbose > 0 {
                        println!(
                            "Iteration {}: Validation MSE = {:.6} (best)",
                            iter + 1,
                            val_loss
                        );
                    }
                } else {
                    no_improvement_count += 1;

                    if self.config.verbose > 0 && (iter + 1) % 10 == 0 {
                        println!(
                            "Iteration {}: Validation MSE = {:.6} (no improvement for {} iterations)",
                            iter + 1,
                            val_loss,
                            no_improvement_count
                        );
                    }

                    // Check if we should stop
                    if no_improvement_count >= es_config.n_iter_no_change {
                        if self.config.verbose > 0 {
                            println!(
                                "Early stopping at iteration {}. Best iteration: {}",
                                iter + 1,
                                best_iteration + 1
                            );
                        }
                        stopped_early = true;
                        break;
                    }
                }
            } else {
                // Log progress without early stopping
                if self.config.verbose > 0 && (iter + 1) % 10 == 0 {
                    let mse: f64 = train_targets
                        .iter()
                        .zip(train_predictions.iter())
                        .map(|(t, p)| (t - p).powi(2))
                        .sum::<f64>()
                        / train_predictions.len() as f64;
                    println!("Iteration {}: MSE = {:.6}", iter + 1, mse);
                }
            }
        }

        // Restore best model if early stopping was triggered
        if stopped_early && best_iteration + 1 < weak_learners.len() {
            weak_learners.truncate(best_iteration + 1);
            if self.config.verbose > 0 {
                println!(
                    "Restored model to iteration {} ({} weak learners)",
                    best_iteration + 1,
                    weak_learners.len()
                );
            }
        }

        if self.config.verbose > 0 {
            let final_mse: f64 = train_targets
                .iter()
                .zip(train_predictions.iter())
                .map(|(t, p)| (t - p).powi(2))
                .sum::<f64>()
                / train_predictions.len() as f64;
            println!("Final MSE: {:.6}", final_mse);
        }

        GradientBoostedModel::new(
            initial_prediction,
            self.config.learning_rate,
            n_features,
            weak_learners,
        )
    }

    /// Generate a feature mask for column subsampling.
    ///
    /// Returns Some(mask) if colsample_bytree < 1.0, None otherwise.
    fn generate_feature_mask(&self, n_features: usize, rng: &mut SmallRng) -> Option<Vec<usize>> {
        if self.config.colsample_bytree >= 1.0 || n_features == 0 {
            return None;
        }

        // Compute number of features to select (at least 1)
        let n_select = ((n_features as f64) * self.config.colsample_bytree)
            .ceil()
            .min(n_features as f64)
            .max(1.0) as usize;

        // Create shuffled list of feature indices
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.partial_shuffle(rng, n_select);

        Some(indices.into_iter().take(n_select).collect())
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
    /// * `feature_mask` - Optional indices of features to consider (None = all features)
    ///
    /// # Returns
    /// A fitted model that can make predictions.
    fn fit(
        &self,
        features: &Tensor2D<B>,
        targets: &Tensor1D<B>,
        feature_mask: Option<&[usize]>,
    ) -> Self::FittedModel;
}

impl<B: Backend> WeakLearner<B> for DecisionStump {
    type FittedModel = FittedStump;

    fn fit(
        &self,
        features: &Tensor2D<B>,
        targets: &Tensor1D<B>,
        feature_mask: Option<&[usize]>,
    ) -> Self::FittedModel {
        self.fit_with_mask(features, targets, feature_mask)
            .unwrap_or(FittedStump {
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
        let fitted =
            <DecisionStump as WeakLearner<CpuBackend>>::fit(&stump, &features, &targets, None);

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

    #[test]
    fn test_feature_subsampling_reproducible() {
        // With the same seed, feature subsampling should produce identical results
        let features = Tensor2D::<CpuBackend>::new(
            vec![
                0.0, 0.0, 1.0, // sample 0
                1.0, 0.0, 2.0, // sample 1
                0.0, 1.0, 3.0, // sample 2
                1.0, 1.0, 4.0, // sample 3
            ],
            4,
            3,
        );
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0]);

        // Train two models with the same seed and colsample_bytree
        let trainer1 = GradientBoostingRegressor::default()
            .n_estimators(20)
            .learning_rate(0.5)
            .colsample_bytree(0.5)
            .random_seed(42);

        let trainer2 = GradientBoostingRegressor::default()
            .n_estimators(20)
            .learning_rate(0.5)
            .colsample_bytree(0.5)
            .random_seed(42);

        let model1 = trainer1.fit(&features, &targets);
        let model2 = trainer2.fit(&features, &targets);

        // Predictions should be identical
        let test_input = Tensor1D::<CpuBackend>::new(vec![0.5, 0.5, 2.5]);
        let pred1 = model1.predict(&test_input);
        let pred2 = model2.predict(&test_input);

        assert!(
            (pred1.to_f64() - pred2.to_f64()).abs() < 1e-10,
            "Predictions should be identical with same seed"
        );
    }

    #[test]
    fn test_feature_subsampling_different_seeds() {
        // With different seeds, feature subsampling may produce different results
        let features = Tensor2D::<CpuBackend>::new(
            vec![
                0.0, 0.0, 1.0, // sample 0
                1.0, 0.0, 2.0, // sample 1
                0.0, 1.0, 3.0, // sample 2
                1.0, 1.0, 4.0, // sample 3
            ],
            4,
            3,
        );
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0]);

        // Train two models with different seeds
        let trainer1 = GradientBoostingRegressor::default()
            .n_estimators(20)
            .learning_rate(0.5)
            .colsample_bytree(0.5)
            .random_seed(42);

        let trainer2 = GradientBoostingRegressor::default()
            .n_estimators(20)
            .learning_rate(0.5)
            .colsample_bytree(0.5)
            .random_seed(123);

        let model1 = trainer1.fit(&features, &targets);
        let model2 = trainer2.fit(&features, &targets);

        // Predictions may be different (though not guaranteed for small examples)
        // At minimum, both should produce valid predictions
        let test_input = Tensor1D::<CpuBackend>::new(vec![0.5, 0.5, 2.5]);
        let pred1 = model1.predict(&test_input).to_f64();
        let pred2 = model2.predict(&test_input).to_f64();

        // Both should be finite numbers
        assert!(pred1.is_finite() && pred2.is_finite());
    }

    #[test]
    fn test_feature_subsampling_full_features() {
        // colsample_bytree = 1.0 should use all features
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0, 6.0]);

        let trainer = GradientBoostingRegressor::default()
            .n_estimators(20)
            .learning_rate(0.5)
            .colsample_bytree(1.0);

        let model = trainer.fit(&features, &targets);

        // Should still train successfully
        let test_input = Tensor1D::<CpuBackend>::new(vec![2.0]);
        let pred = model.predict(&test_input);
        assert!((pred.to_f64() - 4.0).abs() < 2.0);
    }

    #[test]
    fn test_feature_mask_respected_by_stump() {
        // Test that the feature mask is correctly passed to DecisionStump
        use super::WeakLearner;

        let features = Tensor2D::<CpuBackend>::new(
            vec![
                0.0, 0.0, // sample 0: feat0=0, feat1=0
                1.0, 0.0, // sample 1: feat0=1, feat1=0
                0.0, 1.0, // sample 2: feat0=0, feat1=1
                1.0, 1.0, // sample 3: feat0=1, feat1=1
            ],
            4,
            2,
        );
        // Targets only vary with feature 0 (x1): 0, 1, 0, 1
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 0.0, 1.0]);

        let stump = DecisionStump::new();

        // With only feature 0 allowed, should find the perfect split
        let fitted_with_mask = WeakLearner::fit(&stump, &features, &targets, Some(&[0]));
        let preds = fitted_with_mask.predict_batch(&features);

        // All predictions should match targets
        for (pred, &target) in preds.iter().zip([0.0, 1.0, 0.0, 1.0].iter()) {
            assert!((pred - target).abs() < 0.1);
        }

        // With only feature 1 allowed, predictions won't be as good
        let fitted_with_wrong_mask = WeakLearner::fit(&stump, &features, &targets, Some(&[1]));
        let preds_wrong = fitted_with_wrong_mask.predict_batch(&features);

        // Predictions should differ from targets since we can't split on the informative feature
        let perfect_match: bool = preds_wrong
            .iter()
            .zip([0.0, 1.0, 0.0, 1.0].iter())
            .all(|(p, &t)| (p - t).abs() < 0.1);
        assert!(
            !perfect_match,
            "Should not match targets with wrong feature mask"
        );
    }

    #[test]
    fn test_early_stopping_config() {
        let config = EarlyStoppingConfig::new()
            .validation_fraction(0.2)
            .n_iter_no_change(5)
            .tol(0.001);

        assert!((config.validation_fraction - 0.2).abs() < 1e-10);
        assert_eq!(config.n_iter_no_change, 5);
        assert!((config.tol - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_early_stopping_triggers() {
        // Create a simple dataset with limited samples to trigger early stopping
        // With small data, the model will quickly overfit to training data
        // and validation loss will plateau
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        // Enable early stopping with aggressive patience and high tolerance
        let early_stopping = EarlyStoppingConfig::new()
            .validation_fraction(0.33) // 2 samples for validation
            .n_iter_no_change(2) // Stop after 2 iterations with no improvement
            .tol(0.1); // Require 0.1 improvement

        let trainer = GradientBoostingRegressor::default()
            .n_estimators(100)
            .learning_rate(0.3) // Higher learning rate to converge faster
            .early_stopping(early_stopping);

        let model = trainer.fit(&features, &targets);

        // Model should have fewer than 100 estimators due to early stopping
        assert!(
            model.n_estimators() < 100,
            "Expected early stopping to trigger, but got {} estimators",
            model.n_estimators()
        );
    }

    #[test]
    fn test_early_stopping_disabled() {
        // Without early stopping, should train all estimators
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0, 6.0]);

        let trainer = GradientBoostingRegressor::default()
            .n_estimators(50)
            .learning_rate(0.5);

        let model = trainer.fit(&features, &targets);

        // Should have all 50 estimators
        assert_eq!(model.n_estimators(), 50);
    }

    #[test]
    fn test_early_stopping_verbose() {
        // Test that verbose early stopping doesn't panic
        let features = Tensor2D::<CpuBackend>::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            10,
            1,
        );
        let targets =
            Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let early_stopping = EarlyStoppingConfig::new()
            .validation_fraction(0.2)
            .n_iter_no_change(3);

        let trainer = GradientBoostingRegressor::default()
            .n_estimators(50)
            .learning_rate(0.1)
            .early_stopping(early_stopping)
            .verbose(1);

        // Should not panic with verbose output
        let _model = trainer.fit(&features, &targets);
    }
}
