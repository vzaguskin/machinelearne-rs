//! Model comparison utilities for evaluating and comparing ensemble models.
//!
//! Provides metrics computation (MSE, MAE, R²) and utilities
//! for comparing multiple fitted models on test data.

use crate::backend::{Backend, Tensor1D, Tensor2D};

/// Metrics for evaluating regression models.
#[derive(Clone, Debug, Default)]
pub struct ModelMetrics {
    /// Mean Squared Error
    pub mse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// R-squared (coefficient of determination)
    pub r2: f64,
}

impl ModelMetrics {
    /// Compute metrics from predictions and targets.
    pub fn compute(predictions: &[f64], targets: &[f64]) -> Self {
        let n = predictions.len();
        if n == 0 || n != targets.len() {
            return Self::default();
        }

        let mut mse = 0.0;
        let mut mae = 0.0;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        let target_mean: f64 = targets.iter().sum::<f64>() / n as f64;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let error = pred - target;
            mse += error * error;
            mae += error.abs();
            ss_tot += (target - target_mean).powi(2);
            ss_res += error * error;
        }

        mse /= n as f64;
        mae /= n as f64;

        // R² = 1 - SS_res / SS_tot
        let r2 = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0 // Perfect fit or no variance
        };

        Self { mse, mae, r2 }
    }
}

/// A model that can be evaluated on test data.
pub trait Evaluable<B: Backend> {
    /// Predict for a batch of samples.
    fn predict_batch(&self, features: &Tensor2D<B>) -> Vec<f64>;
}

/// Result of evaluating a single model.
#[derive(Clone, Debug)]
pub struct ModelResult {
    /// Name of the model.
    pub name: String,
    /// Computed metrics.
    pub metrics: ModelMetrics,
}

/// Compare multiple models on test data.
pub struct ModelComparison<B: Backend> {
    /// Test features.
    features: Tensor2D<B>,
    /// Test targets.
    targets: Tensor1D<B>,
    /// Results for each model.
    results: Vec<ModelResult>,
}

impl<B: Backend> ModelComparison<B> {
    /// Create a new comparison with test data.
    pub fn new(features: Tensor2D<B>, targets: Tensor1D<B>) -> Self {
        Self {
            features,
            targets,
            results: Vec::new(),
        }
    }

    /// Evaluate a model and add to comparison.
    pub fn evaluate<M: Evaluable<B>>(&mut self, name: &str, model: &M) -> &mut Self {
        let predictions = model.predict_batch(&self.features);
        let targets_vec = self.targets.to_vec();
        let metrics = ModelMetrics::compute(&predictions, &targets_vec);
        self.results.push(ModelResult {
            name: name.to_string(),
            metrics,
        });
        self
    }

    /// Get the model with the best (lowest) MSE.
    pub fn best_model(&self) -> Option<&ModelResult> {
        self.results.iter().min_by(|a, b| {
            a.metrics
                .mse
                .partial_cmp(&b.metrics.mse)
                .unwrap_or(std::cmp::Ordering::Less)
        })
    }

    /// Get all results sorted by MSE (best first).
    pub fn ranked(&self) -> Vec<&ModelResult> {
        let mut results: Vec<_> = self.results.iter().collect();
        results.sort_by(|a, b| {
            a.metrics
                .mse
                .partial_cmp(&b.metrics.mse)
                .unwrap_or(std::cmp::Ordering::Less)
        });
        results
    }

    /// Print a formatted summary table.
    pub fn summary(&self) {
        println!("\n{:20} {:>15} {:>15} {:>15}", "Model", "MSE", "MAE", "R²");
        println!("{}", "-".repeat(65));
        for result in self.ranked() {
            println!(
                "{:20} {:>15.6} {:>15.6} {:>15.6}",
                result.name, result.metrics.mse, result.metrics.mae, result.metrics.r2
            );
        }
        if let Some(best) = self.best_model() {
            println!("\nBest model: {} (MSE: {:.6})", best.name, best.metrics.mse);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_model_metrics_perfect_prediction() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 2.0, 3.0];
        let metrics = ModelMetrics::compute(&predictions, &targets);
        assert!((metrics.mse).abs() < 1e-10);
        assert!((metrics.mae).abs() < 1e-10);
        assert!((metrics.r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_model_metrics_with_error() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.5, 2.5, 3.5];
        let metrics = ModelMetrics::compute(&predictions, &targets);
        assert!(metrics.mse > 0.0);
        assert!(metrics.mae > 0.0);
        assert!(metrics.r2 < 1.0);
    }

    #[test]
    fn test_model_metrics_empty() {
        let predictions: Vec<f64> = vec![];
        let targets: Vec<f64> = vec![];
        let metrics = ModelMetrics::compute(&predictions, &targets);
        assert!((metrics.mse).abs() < 1e-10);
        assert!((metrics.mae).abs() < 1e-10);
        assert!((metrics.r2).abs() < 1e-10);
    }

    #[test]
    fn test_model_metrics_mismatched_lengths() {
        let predictions = vec![1.0, 2.0];
        let targets = vec![1.0, 2.0, 3.0];
        let metrics = ModelMetrics::compute(&predictions, &targets);
        assert!((metrics.mse).abs() < 1e-10); // Returns default
    }

    #[test]
    fn test_model_metrics_no_variance() {
        // All targets the same - no variance
        let predictions = vec![1.0, 1.0, 1.0];
        let targets = vec![1.0, 1.0, 1.0];
        let metrics = ModelMetrics::compute(&predictions, &targets);
        assert!((metrics.r2).abs() < 1e-10); // r2 should be 0 when ss_tot is 0
    }

    #[test]
    fn test_model_comparison() {
        let features = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0]);

        struct MockModel;
        impl Evaluable<CpuBackend> for MockModel {
            fn predict_batch(&self, _features: &Tensor2D<CpuBackend>) -> Vec<f64> {
                vec![1.1, 2.1, 3.1, 4.1]
            }
        }

        let mut comparison = ModelComparison::new(features, targets);
        comparison.evaluate("MockModel", &MockModel);

        assert_eq!(comparison.results.len(), 1);
        let best = comparison.best_model().unwrap();
        assert_eq!(best.name, "MockModel");
    }

    #[test]
    fn test_model_comparison_multiple_models() {
        let features = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0]);

        struct GoodModel;
        impl Evaluable<CpuBackend> for GoodModel {
            fn predict_batch(&self, _features: &Tensor2D<CpuBackend>) -> Vec<f64> {
                vec![1.01, 2.01, 3.01, 4.01] // Better predictions
            }
        }

        struct BadModel;
        impl Evaluable<CpuBackend> for BadModel {
            fn predict_batch(&self, _features: &Tensor2D<CpuBackend>) -> Vec<f64> {
                vec![2.0, 3.0, 4.0, 5.0] // Worse predictions
            }
        }

        let mut comparison = ModelComparison::new(features, targets);
        comparison.evaluate("BadModel", &BadModel);
        comparison.evaluate("GoodModel", &GoodModel);

        // Test ranked returns best first
        let ranked = comparison.ranked();
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].name, "GoodModel"); // Better model first

        // Test best_model returns the one with lowest MSE
        let best = comparison.best_model().unwrap();
        assert_eq!(best.name, "GoodModel");
    }

    #[test]
    fn test_model_comparison_no_models() {
        let features = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0]);

        let comparison = ModelComparison::new(features, targets);
        assert!(comparison.best_model().is_none());
        assert!(comparison.ranked().is_empty());
    }

    #[test]
    fn test_model_comparison_chaining() {
        let features = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0]);

        struct MockModel;
        impl Evaluable<CpuBackend> for MockModel {
            fn predict_batch(&self, _features: &Tensor2D<CpuBackend>) -> Vec<f64> {
                vec![1.0, 2.0, 3.0, 4.0]
            }
        }

        let mut comparison = ModelComparison::new(features, targets);
        comparison
            .evaluate("Model1", &MockModel)
            .evaluate("Model2", &MockModel);

        assert_eq!(comparison.results.len(), 2);
    }

    #[test]
    fn test_model_comparison_summary() {
        let features = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0]);

        struct MockModel;
        impl Evaluable<CpuBackend> for MockModel {
            fn predict_batch(&self, _features: &Tensor2D<CpuBackend>) -> Vec<f64> {
                vec![1.1, 2.1, 3.1, 4.1]
            }
        }

        let mut comparison = ModelComparison::new(features, targets);
        comparison.evaluate("MockModel", &MockModel);

        // Call summary to test the function
        comparison.summary();
    }

    #[test]
    fn test_model_comparison_nan_mse() {
        // Test with NaN values to trigger unwrap_or branch
        let features = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0]);

        struct NanModel;
        impl Evaluable<CpuBackend> for NanModel {
            fn predict_batch(&self, _features: &Tensor2D<CpuBackend>) -> Vec<f64> {
                vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN]
            }
        }

        let mut comparison = ModelComparison::new(features, targets);
        comparison.evaluate("NanModel", &NanModel);

        // best_model should still return something (using unwrap_or)
        let best = comparison.best_model();
        assert!(best.is_some());

        // ranked should also handle NaN
        let ranked = comparison.ranked();
        assert_eq!(ranked.len(), 1);
    }

    #[test]
    fn test_model_result_debug_clone() {
        let result = ModelResult {
            name: "TestModel".to_string(),
            metrics: ModelMetrics {
                mse: 0.5,
                mae: 0.6,
                r2: 0.9,
            },
        };

        // Test Clone
        let cloned = result.clone();
        assert_eq!(cloned.name, "TestModel");

        // Test Debug (just make sure it doesn't panic)
        let _debug = format!("{:?}", result);
    }

    #[test]
    fn test_model_metrics_default() {
        let metrics = ModelMetrics::default();
        assert!((metrics.mse).abs() < 1e-10);
        assert!((metrics.mae).abs() < 1e-10);
        assert!((metrics.r2).abs() < 1e-10);
    }

    #[test]
    fn test_model_metrics_clone_debug() {
        let metrics = ModelMetrics {
            mse: 1.0,
            mae: 0.5,
            r2: 0.9,
        };

        // Test Clone
        let cloned = metrics.clone();
        assert!((cloned.mse - 1.0).abs() < 1e-10);

        // Test Debug
        let _debug = format!("{:?}", metrics);
    }
}
