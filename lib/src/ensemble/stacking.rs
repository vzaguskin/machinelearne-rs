//! Stacking ensemble for combining predictions from multiple models.
//!
//! A stacking ensemble trains a meta-learner on the predictions of base models
//! to produce a final prediction. This is useful for combining different
//! types of models (e.g., Linear, MLP, GradientBoosting) that may capture
//! different patterns in the data.

use crate::backend::{Backend, Tensor2D};
use crate::ensemble::ensemble_model::StumpPredictor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

/// A trait for models that can be used in stacking.
pub trait Stackable<B: Backend> {
    /// Get predictions for a batch of samples.
    fn predict_batch(&self, features: &Tensor2D<B>) -> Vec<f64>;
}

/// Implementation of Stackable for GradientBoostedModel which returns Tensor1D.
impl<B: Backend, H: StumpPredictor> Stackable<B> for crate::ensemble::GradientBoostedModel<B, H> {
    fn predict_batch(&self, features: &Tensor2D<B>) -> Vec<f64> {
        crate::ensemble::GradientBoostedModel::predict_batch(self, features).to_vec()
    }
}

/// Configuration for stacking ensemble.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StackingConfig {
    /// Whether to use out-of-fold predictions for training meta-learner.
    /// When true, the validation set is used for meta-learner training.
    pub use_validation: bool,
    /// Verbosity level (0 = silent, 1 = progress)
    pub verbose: u8,
}

impl Default for StackingConfig {
    fn default() -> Self {
        Self {
            use_validation: true,
            verbose: 0,
        }
    }
}

/// A stacking ensemble combining multiple base models via a meta-learner.
pub struct StackingEnsemble<B: Backend, M: Stackable<B>> {
    /// Base models with their names.
    base_models: HashMap<String, M>,
    /// The meta-learner for combining predictions.
    meta_learner: M,
    /// Configuration.
    config: StackingConfig,
    /// Number of base models.
    n_base_models: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend, M: Stackable<B>> StackingEnsemble<B, M> {
    /// Create a new stacking ensemble.
    pub fn new(meta_learner: M) -> Self {
        Self {
            base_models: HashMap::new(),
            meta_learner,
            config: StackingConfig::default(),
            n_base_models: 0,
            _backend: PhantomData,
        }
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: StackingConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a base model to the ensemble.
    pub fn add_model(mut self, name: &str, model: M) -> Self {
        self.base_models.insert(name.to_string(), model);
        self.n_base_models = self.base_models.len();
        self
    }

    /// Get the number of base models.
    pub fn num_models(&self) -> usize {
        self.n_base_models
    }

    /// Predict for multiple samples.
    pub fn predict_batch(&self, features: &Tensor2D<B>) -> Vec<f64> {
        let (n_samples, _n_features) = features.shape();

        // Collect base model predictions for all samples
        let mut all_base_preds: Vec<Vec<f64>> = vec![Vec::new(); n_samples];
        for sample_preds in &mut all_base_preds {
            sample_preds.reserve(self.n_base_models);
        }

        let mut model_names: Vec<&str> = self.base_models.keys().map(|s| s.as_str()).collect();
        model_names.sort();

        for name in &model_names {
            if let Some(model) = self.base_models.get(*name) {
                let preds = model.predict_batch(features);
                for (i, &pred) in preds.iter().enumerate() {
                    all_base_preds[i].push(pred);
                }
            }
        }

        // Convert to tensor for meta-learner
        let meta_features: Vec<f32> = all_base_preds
            .iter()
            .flat_map(|v| v.iter().map(|&x| x as f32))
            .collect();

        let meta_tensor = Tensor2D::<B>::new(meta_features, n_samples, self.n_base_models);
        self.meta_learner.predict_batch(&meta_tensor)
    }
}

/// Builder for creating stacking ensembles.
pub struct StackingBuilder<B: Backend, M: Stackable<B>> {
    base_models: HashMap<String, M>,
    meta_learner: Option<M>,
    config: StackingConfig,
    _backend: PhantomData<B>,
}

impl<B: Backend, M: Stackable<B>> StackingBuilder<B, M> {
    /// Create a new stacking builder.
    pub fn new() -> Self {
        Self {
            base_models: HashMap::new(),
            meta_learner: None,
            config: StackingConfig::default(),
            _backend: PhantomData,
        }
    }

    /// Add a base model.
    pub fn add_model(mut self, name: &str, model: M) -> Self {
        self.base_models.insert(name.to_string(), model);
        self
    }

    /// Set the meta-learner.
    pub fn meta_learner(mut self, model: M) -> Self {
        self.meta_learner = Some(model);
        self
    }

    /// Set the configuration.
    pub fn config(mut self, config: StackingConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the stacking ensemble.
    pub fn build(self) -> StackingEnsemble<B, M> {
        let meta = self.meta_learner.expect("Meta-learner is required");
        let n_base_models = self.base_models.len();
        StackingEnsemble {
            base_models: self.base_models,
            meta_learner: meta,
            config: self.config,
            n_base_models,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend, M: Stackable<B>> Default for StackingBuilder<B, M> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::ensemble::GradientBoostingRegressor;

    #[test]
    fn test_stacking_config_default() {
        let config = StackingConfig::default();
        assert!(config.use_validation);
        assert_eq!(config.verbose, 0);
    }

    #[test]
    fn test_stacking_builder() {
        // Create training data
        let features = crate::Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1);
        let targets = crate::Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);

        // Fit a gradient boosting model
        let gb_trainer = GradientBoostingRegressor::default().n_estimators(10);
        let gb_model = gb_trainer.fit(&features, &targets);

        // Fit a meta-learner model
        let meta_trainer = GradientBoostingRegressor::default().n_estimators(5);
        // For stacking, meta-learner needs to be pre-fitted on base model predictions
        // Here we just use the same data for simplicity
        let meta_model = meta_trainer.fit(&features, &targets);

        let mut builder = StackingBuilder::new();
        builder = builder.add_model("gb", gb_model);
        builder = builder.meta_learner(meta_model);

        let ensemble = builder.build();
        assert_eq!(ensemble.num_models(), 1);
    }

    #[test]
    fn test_stacking_ensemble_predict_batch() {
        // Create training data
        let features = crate::Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1);
        let targets = crate::Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);

        // Fit base model
        let gb_trainer = GradientBoostingRegressor::default().n_estimators(10);
        let gb_model = gb_trainer.fit(&features, &targets);

        // Fit meta-learner
        let meta_trainer = GradientBoostingRegressor::default().n_estimators(5);
        let meta_model = meta_trainer.fit(&features, &targets);

        // Build ensemble
        let ensemble = StackingBuilder::new()
            .add_model("gb", gb_model)
            .meta_learner(meta_model)
            .build();

        // Test predict_batch
        let predictions = ensemble.predict_batch(&features);
        assert_eq!(predictions.len(), 6);

        // Predictions should be finite
        for pred in &predictions {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_stacking_ensemble_with_config() {
        let config = StackingConfig {
            use_validation: false,
            verbose: 1,
        };

        let features = crate::Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0], 3, 1);
        let targets = crate::Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0]);

        let gb_model = GradientBoostingRegressor::default()
            .n_estimators(5)
            .fit(&features, &targets);
        let meta_model = GradientBoostingRegressor::default()
            .n_estimators(3)
            .fit(&features, &targets);

        let ensemble = StackingBuilder::new()
            .add_model("gb", gb_model)
            .meta_learner(meta_model)
            .config(config.clone())
            .build();

        assert_eq!(ensemble.num_models(), 1);
    }

    #[test]
    fn test_stacking_builder_default() {
        let builder: StackingBuilder<
            CpuBackend,
            crate::ensemble::GradientBoostedModel<CpuBackend>,
        > = StackingBuilder::default();
        assert!(builder.base_models.is_empty());
        assert!(builder.meta_learner.is_none());
    }

    #[test]
    fn test_stacking_ensemble_new_direct() {
        let features = crate::Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0], 3, 1);
        let targets = crate::Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0]);

        let meta_model = GradientBoostingRegressor::default()
            .n_estimators(5)
            .fit(&features, &targets);

        let ensemble = StackingEnsemble::new(meta_model);
        assert_eq!(ensemble.num_models(), 0);

        // Add a model
        let gb_model = GradientBoostingRegressor::default()
            .n_estimators(5)
            .fit(&features, &targets);
        let ensemble = ensemble.add_model("gb", gb_model);
        assert_eq!(ensemble.num_models(), 1);
    }

    #[test]
    fn test_stacking_ensemble_multiple_base_models() {
        let features = crate::Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = crate::Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0, 6.0]);

        let model1 = GradientBoostingRegressor::default()
            .n_estimators(5)
            .fit(&features, &targets);
        let model2 = GradientBoostingRegressor::default()
            .n_estimators(10)
            .fit(&features, &targets);
        let meta_model = GradientBoostingRegressor::default()
            .n_estimators(3)
            .fit(&features, &targets);

        let ensemble = StackingBuilder::new()
            .add_model("gb1", model1)
            .add_model("gb2", model2)
            .meta_learner(meta_model)
            .build();

        assert_eq!(ensemble.num_models(), 2);

        // Test predictions work with multiple base models
        let predictions = ensemble.predict_batch(&features);
        assert_eq!(predictions.len(), 4);
    }
}
