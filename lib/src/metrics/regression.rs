//! Regression metrics for evaluating model performance.
//!
//! Provides common regression metrics (MSE, RMSE, MAE, R²) with both:
//! - Standalone functions for simple usage
//! - [`RegressionMetric`] enum implementing [`Scorer`] for use with model selection

use crate::backend::{Backend, Tensor1D};

/// Trait for scoring model predictions.
///
/// A scorer computes a metric value from predictions and targets.
/// Higher scores are better (metrics like MSE/MAE are negated internally).
pub trait Scorer<B: Backend>: Clone {
    /// The type of predictions (e.g., `Tensor1D<B>` or `Vec<f32>`).
    type Prediction;
    /// The type of targets (e.g., `Tensor1D<B>` or `Vec<f32>`).
    type Target;

    /// Compute the score from predictions and targets.
    ///
    /// # Returns
    /// A score value where **higher is better**.
    fn score(&self, prediction: &Self::Prediction, target: &Self::Target) -> f64;

    /// Returns the name of the scoring metric.
    fn name(&self) -> &'static str;
}

/// Regression metrics that can be used as scorers.
///
/// All metrics are negated so that higher values are better,
/// consistent with sklearn's convention.
#[derive(Clone, Copy, Debug, Default)]
pub enum RegressionMetric {
    /// Negative Mean Squared Error (higher is better, max is 0).
    NegMSE,
    /// Negative Root Mean Squared Error (higher is better, max is 0).
    NegRMSE,
    /// Negative Mean Absolute Error (higher is better, max is 0).
    NegMAE,
    /// R² score (coefficient of determination, higher is better, max is 1).
    #[default]
    R2,
}

impl RegressionMetric {
    /// Calculate Mean Squared Error (MSE).
    ///
    /// MSE = mean((y_true - y_pred)^2)
    pub fn mse(y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "Arrays must have the same length"
        );

        if y_true.is_empty() {
            return 0.0;
        }

        let sum_sq: f32 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .sum();

        sum_sq / y_true.len() as f32
    }

    /// Calculate Root Mean Squared Error (RMSE).
    ///
    /// RMSE = sqrt(MSE)
    pub fn rmse(y_true: &[f32], y_pred: &[f32]) -> f32 {
        Self::mse(y_true, y_pred).sqrt()
    }

    /// Calculate Mean Absolute Error (MAE).
    ///
    /// MAE = mean(|y_true - y_pred|)
    pub fn mae(y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "Arrays must have the same length"
        );

        if y_true.is_empty() {
            return 0.0;
        }

        let sum_abs: f32 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).abs())
            .sum();

        sum_abs / y_true.len() as f32
    }

    /// Calculate R² (coefficient of determination).
    ///
    /// R² = 1 - (SS_res / SS_tot)
    ///
    /// R² ranges from 0 to 1, where 1 indicates perfect prediction.
    /// Values can be negative if the model is arbitrarily worse than the mean.
    pub fn r_squared(y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "Arrays must have the same length"
        );

        if y_true.is_empty() {
            return 0.0;
        }

        let mean_true: f32 = y_true.iter().copied().sum::<f32>() / y_true.len() as f32;

        let ss_res: f32 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .sum();

        let ss_tot: f32 = y_true.iter().map(|&t| (t - mean_true).powi(2)).sum();

        if ss_tot == 0.0 {
            // All values are the same, perfect prediction if predictions are also the same
            return if ss_res == 0.0 { 1.0 } else { 0.0 };
        }

        1.0 - (ss_res / ss_tot)
    }

    /// Compute the metric value from slices (higher is better).
    pub fn score_slices(&self, y_true: &[f32], y_pred: &[f32]) -> f64 {
        match self {
            RegressionMetric::NegMSE => -(Self::mse(y_true, y_pred) as f64),
            RegressionMetric::NegRMSE => -(Self::rmse(y_true, y_pred) as f64),
            RegressionMetric::NegMAE => -(Self::mae(y_true, y_pred) as f64),
            RegressionMetric::R2 => Self::r_squared(y_true, y_pred) as f64,
        }
    }

    /// Returns the name of the metric.
    pub fn metric_name(&self) -> &'static str {
        match self {
            RegressionMetric::NegMSE => "neg_mean_squared_error",
            RegressionMetric::NegRMSE => "neg_root_mean_squared_error",
            RegressionMetric::NegMAE => "neg_mean_absolute_error",
            RegressionMetric::R2 => "r2",
        }
    }
}

impl<B: Backend> Scorer<B> for RegressionMetric {
    type Prediction = Tensor1D<B>;
    type Target = Tensor1D<B>;

    fn score(&self, prediction: &Self::Prediction, target: &Self::Target) -> f64 {
        // Convert Vec<f64> to Vec<f32> for metric computation
        let pred_vec: Vec<f32> = prediction.to_vec().iter().map(|&x| x as f32).collect();
        let target_vec: Vec<f32> = target.to_vec().iter().map(|&x| x as f32).collect();
        self.score_slices(&target_vec, &pred_vec)
    }

    fn name(&self) -> &'static str {
        self.metric_name()
    }
}

/// Convenience struct holding all regression metrics.
#[derive(Debug, Clone, Copy)]
pub struct RegressionMetrics {
    /// Mean Squared Error
    pub mse: f32,
    /// Root Mean Squared Error
    pub rmse: f32,
    /// Mean Absolute Error
    pub mae: f32,
    /// R² score
    pub r_squared: f32,
}

impl RegressionMetrics {
    /// Create a new RegressionMetrics instance from all components.
    pub fn new(mse: f32, mae: f32, r_squared: f32) -> Self {
        Self {
            mse,
            rmse: mse.sqrt(),
            mae,
            r_squared,
        }
    }

    /// Calculate all metrics at once.
    pub fn calculate(y_true: &[f32], y_pred: &[f32]) -> Self {
        Self::new(
            RegressionMetric::mse(y_true, y_pred),
            RegressionMetric::mae(y_true, y_pred),
            RegressionMetric::r_squared(y_true, y_pred),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_perfect() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        assert!((RegressionMetric::mse(&y_true, &y_pred) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_error() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![2.0, 3.0, 4.0, 5.0];
        // Errors: [-1, -1, -1, -1], squared: [1, 1, 1, 1], mean: 1.0
        assert!((RegressionMetric::mse(&y_true, &y_pred) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mae() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![2.0, 3.0, 4.0, 5.0];
        // Errors: [-1, -1, -1, -1], abs: [1, 1, 1, 1], mean: 1.0
        assert!((RegressionMetric::mae(&y_true, &y_pred) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_r_squared_perfect() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        assert!((RegressionMetric::r_squared(&y_true, &y_pred) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_r_squared_mean() {
        let y_true = vec![2.0, 2.0, 2.0, 2.0];
        let y_pred = vec![2.0, 2.0, 2.0, 2.0];
        // All predictions equal the mean (which is 2.0), so R² should be 1.0
        assert!((RegressionMetric::r_squared(&y_true, &y_pred) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_r_squared_poor_prediction() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![4.0, 3.0, 2.0, 1.0]; // Completely wrong
        let r2 = RegressionMetric::r_squared(&y_true, &y_pred);
        // Should be negative for very poor predictions
        assert!(r2 < 0.0);
    }

    #[test]
    fn test_neg_mse_scorer() {
        let metric = RegressionMetric::NegMSE;
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 3.0];
        let score = metric.score_slices(&y_true, &y_pred);
        assert!((score - 0.0).abs() < 1e-6);

        let y_pred_bad = vec![2.0, 3.0, 4.0];
        let score_bad = metric.score_slices(&y_true, &y_pred_bad);
        assert!(score_bad < 0.0); // Negated MSE should be negative
    }

    #[test]
    fn test_r2_scorer() {
        let metric = RegressionMetric::R2;
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 3.0];
        let score = metric.score_slices(&y_true, &y_pred);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_metric_names() {
        assert_eq!(
            RegressionMetric::NegMSE.metric_name(),
            "neg_mean_squared_error"
        );
        assert_eq!(
            RegressionMetric::NegRMSE.metric_name(),
            "neg_root_mean_squared_error"
        );
        assert_eq!(
            RegressionMetric::NegMAE.metric_name(),
            "neg_mean_absolute_error"
        );
        assert_eq!(RegressionMetric::R2.metric_name(), "r2");
    }

    #[test]
    fn test_regression_metrics_calculate() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        let metrics = RegressionMetrics::calculate(&y_true, &y_pred);
        assert!((metrics.mse - 0.0).abs() < 1e-6);
        assert!((metrics.mae - 0.0).abs() < 1e-6);
        assert!((metrics.r_squared - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scorer_with_tensor() {
        use crate::backend::CpuBackend;

        let metric = RegressionMetric::R2;
        let y_true = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]);
        let y_pred = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]);

        let score = metric.score(&y_pred, &y_true);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scorer_neg_mse_with_tensor() {
        use crate::backend::CpuBackend;

        let metric = RegressionMetric::NegMSE;
        let y_true = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]);
        let y_pred = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]);

        let score = metric.score(&y_pred, &y_true);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scorer_neg_rmse_with_tensor() {
        use crate::backend::CpuBackend;

        let metric = RegressionMetric::NegRMSE;
        let y_true = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]);
        let y_pred = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]);

        let score = metric.score(&y_pred, &y_true);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scorer_neg_mae_with_tensor() {
        use crate::backend::CpuBackend;

        let metric = RegressionMetric::NegMAE;
        let y_true = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]);
        let y_pred = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]);

        let score = metric.score(&y_pred, &y_true);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_empty() {
        let y_true: Vec<f32> = vec![];
        let y_pred: Vec<f32> = vec![];
        assert!((RegressionMetric::mse(&y_true, &y_pred) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mae_empty() {
        let y_true: Vec<f32> = vec![];
        let y_pred: Vec<f32> = vec![];
        assert!((RegressionMetric::mae(&y_true, &y_pred) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_r_squared_empty() {
        let y_true: Vec<f32> = vec![];
        let y_pred: Vec<f32> = vec![];
        assert!((RegressionMetric::r_squared(&y_true, &y_pred) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_metric() {
        let metric = RegressionMetric::default();
        assert!(matches!(metric, RegressionMetric::R2));
    }
}
