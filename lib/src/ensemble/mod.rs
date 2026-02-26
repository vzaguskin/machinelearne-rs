//! Gradient Boosting ensemble methods for regression and classification.
//!
//! This module implements gradient boosting decision trees (GBDT) following
//! the architecture decision in ADR-0010. Unlike the gradient descent-based
//! models in `model/`, gradient boosting builds an ensemble of weak learners
//! sequentially, where each learner fits the pseudo-residuals of the current
//! ensemble.
//!
//! # Architecture
//!
//! The module is organized into:
//! - **Boosting losses**: `BoostingLoss` trait with `LeastSquaresLoss`, `LogisticLoss`, etc.
//! - **Weak learners**: `DecisionStump` for simple single-split trees
//! - **Ensemble model**: `GradientBoostedModel` implementing `InferenceModel`
//! - **Trainer**: `GradientBoostingTrainer` for building ensembles
//!
//! # Example
//!
//! ```rust
//! use machinelearne_rs::ensemble::{GradientBoostingRegressor, GradientBoostedModel};
//! use machinelearne_rs::backend::CpuBackend;
//! use machinelearne_rs::{Tensor1D, Tensor2D};
//! use machinelearne_rs::model::InferenceModel;
//!
//! // Create training data: y = 2*x
//! let features = Tensor2D::<CpuBackend>::new(
//!     vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
//!     10, 1
//! );
//! let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
//!
//! // Train gradient boosting model
//! let trainer = GradientBoostingRegressor::default()
//!     .n_estimators(100)
//!     .learning_rate(0.1);
//!
//! let model = trainer.fit(&features, &targets);
//!
//! // Make predictions
//! let test_input = Tensor1D::<CpuBackend>::new(vec![2.5]);
//! let prediction = model.predict(&test_input);
//! println!("Prediction: {}", prediction.to_f64());
//!
//! // Batch prediction
//! let test_batch = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0], 3, 1);
//! let predictions = model.predict_batch(&test_batch);
//! ```
//!
//! # Design Notes
//!
//! ## Why not use the existing `Trainer`?
//!
//! Gradient boosting has fundamentally different training dynamics:
//! - It builds sequential ensembles, not parameter updates
//! - Each iteration produces a new weak learner, not updated weights
//! - Targets change (pseudo-residuals), not fixed
//!
//! The `GradientBoostingTrainer` follows similar API conventions but is
//! specialized for boosting semantics.
//!
//! ## GPU Acceleration Path
//!
//! Phase 1 (current): CPU-only with exact split finding
//! Phase 3+: Histogram-based trees with WGPU acceleration
//!
//! Histogram-based algorithms (LightGBM, XGBoost gpu_hist) are GPU-friendly
//! because they parallelize bin aggregation across thousands of GPU cores.
//!
//! # References
//!
//! - ADR-0010: Gradient Boosting Architecture
//! - Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine"
//! - Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"

pub mod boosting;
pub mod comparison;
pub mod decision_stump;
pub mod decision_tree;
pub mod ensemble_model;
pub mod loss;
pub mod stacking;
pub mod tuning;

// Re-export main types for convenience
pub use boosting::{
    EarlyStoppingConfig, GradientBoostingConfig, GradientBoostingRegressor,
    GradientBoostingTrainer, WeakLearner,
};
pub use comparison::{Evaluable, ModelComparison, ModelMetrics, ModelResult};
pub use decision_stump::{DecisionStump, FittedStump};
pub use decision_tree::{DecisionTree, DecisionTreeConfig, FittedTree, TreeNode};
pub use ensemble_model::{
    GradientBoostedModel, GradientBoostedModelParams, GradientBoostedRegressor as GBRegressor,
    StumpPredictor,
};
pub use loss::{BoostingLoss, LeastSquaresLoss, LogisticLoss, QuantileLoss};
pub use stacking::{Stackable, StackingBuilder, StackingConfig, StackingEnsemble};
pub use tuning::{EvalResult, GridSearchConfig, GridSearchGB, GridSearchResults, HyperParamConfig};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::model::InferenceModel;
    use crate::serialization::SerializableParams;
    use crate::{Tensor1D, Tensor2D};

    #[test]
    fn test_full_pipeline() {
        // Train a model that learns y = 2*x
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], 5, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0, 6.0, 8.0]);

        // Train
        let trainer = GradientBoostingRegressor::default()
            .n_estimators(100)
            .learning_rate(0.5);

        let model = trainer.fit(&features, &targets);

        // Verify training - gradient boosting with stumps is approximate
        let train_preds = model.predict_batch(&features);
        for (pred, target) in train_preds.to_vec().iter().zip(targets.to_vec().iter()) {
            assert!((pred - target).abs() < 1.5);
        }

        // Test inference on new data
        let test_input = Tensor1D::<CpuBackend>::new(vec![2.5]);
        let pred = model.predict(&test_input);
        assert!((pred.to_f64() - 5.0).abs() < 2.0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0], 3, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0]);

        let trainer = GradientBoostingRegressor::default().n_estimators(10);
        let original_model = trainer.fit(&features, &targets);

        // Serialize
        let bytes = original_model.extract_params().to_bytes().unwrap();

        // Deserialize
        let params = GradientBoostedModelParams::from_bytes(&bytes).unwrap();
        let loaded_model: GradientBoostedModel<CpuBackend> =
            GradientBoostedModel::from_params(params).unwrap();

        // Verify predictions match
        let test_input = Tensor1D::<CpuBackend>::new(vec![1.5]);
        let orig_pred = original_model.predict(&test_input);
        let loaded_pred = loaded_model.predict(&test_input);

        assert!((orig_pred.to_f64() - loaded_pred.to_f64()).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_regression() {
        // Test with quantile loss
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], 5, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0, 6.0, 8.0]);

        let trainer = GradientBoostingTrainer::new(QuantileLoss::new(0.5))
            .n_estimators(30)
            .learning_rate(0.3);

        let model = trainer.fit(&features, &targets);

        // Should still learn the pattern
        let test_input = Tensor1D::<CpuBackend>::new(vec![2.0]);
        let pred = model.predict(&test_input);
        // Allow larger tolerance for quantile regression
        assert!((pred.to_f64() - 4.0).abs() < 2.0);
    }
}
