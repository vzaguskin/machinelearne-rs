//! Boosting-specific loss functions.
//!
//! Unlike gradient descent losses that compute `grad_wrt_prediction`, boosting losses
//! compute the negative gradient of the loss with respect to the current predictions,
//! which becomes the target for the next weak learner.

use serde::{Deserialize, Serialize};

/// Trait for loss functions used in gradient boosting.
///
/// Boosting losses differ from gradient descent losses:
/// - They compute the "pseudo-residuals" (negative gradient of loss w.r.t. prediction)
/// - They provide an initial prediction (e.g., mean for least squares)
/// - They work with scalar target/prediction pairs, not tensors
///
/// The negative gradient becomes the target for fitting the next weak learner.
pub trait BoostingLoss: Clone + Default {
    /// Compute the initial prediction for the ensemble (before any weak learners).
    ///
    /// For MSE, this is the mean of targets.
    /// For logistic loss, this is log(positive_rate / (1 - positive_rate)).
    fn initial_prediction(&self, targets: &[f64]) -> f64;

    /// Compute the negative gradient of the loss w.r.t. prediction.
    ///
    /// This is the "pseudo-residual" that the next weak learner will fit.
    /// For MSE: -(0.5 * (pred - target)) = (target - pred)
    /// For logistic: target - sigmoid(pred)
    fn negative_gradient(&self, target: f64, prediction: f64) -> f64;

    /// Optional: Compute the loss value for monitoring/logging.
    fn loss(&self, target: f64, prediction: f64) -> f64 {
        let _ = (target, prediction);
        0.0 // Default: no loss computation
    }
}

/// Least Squares loss for regression: L = 0.5 * (y - F(x))^2
///
/// This is the canonical loss for gradient boosting regression.
///
/// - Initial prediction: mean(targets)
/// - Negative gradient: y - F(x) (the residual)
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct LeastSquaresLoss;

impl BoostingLoss for LeastSquaresLoss {
    fn initial_prediction(&self, targets: &[f64]) -> f64 {
        if targets.is_empty() {
            return 0.0;
        }
        targets.iter().sum::<f64>() / targets.len() as f64
    }

    fn negative_gradient(&self, target: f64, prediction: f64) -> f64 {
        target - prediction
    }

    fn loss(&self, target: f64, prediction: f64) -> f64 {
        0.5 * (target - prediction).powi(2)
    }
}

/// Quantile loss for robust regression.
///
/// Produces prediction at the specified quantile (0.5 = median).
///
/// - Initial prediction: quantile of targets
/// - Negative gradient: sign(residual) based on quantile
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct QuantileLoss {
    /// The quantile to predict (0.0 to 1.0)
    pub alpha: f64,
}

impl Default for QuantileLoss {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

impl QuantileLoss {
    /// Create a new quantile loss for the specified quantile.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
        }
    }
}

impl BoostingLoss for QuantileLoss {
    fn initial_prediction(&self, targets: &[f64]) -> f64 {
        if targets.is_empty() {
            return 0.0;
        }
        let mut sorted = targets.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() - 1) as f64 * self.alpha).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    fn negative_gradient(&self, target: f64, prediction: f64) -> f64 {
        let residual = target - prediction;
        if residual > 0.0 {
            self.alpha
        } else if residual < 0.0 {
            self.alpha - 1.0
        } else {
            0.0
        }
    }

    fn loss(&self, target: f64, prediction: f64) -> f64 {
        let residual = target - prediction;
        if residual >= 0.0 {
            self.alpha * residual
        } else {
            (self.alpha - 1.0) * residual
        }
    }
}

/// Logistic (binomial deviance) loss for binary classification.
///
/// Targets should be 0.0 or 1.0.
/// Predictions are in log-odds space (logits).
///
/// - Initial prediction: log(p / (1-p)) where p = mean(targets)
/// - Negative gradient: y - sigmoid(F(x))
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct LogisticLoss;

impl LogisticLoss {
    /// Sigmoid function
    fn sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        }
    }
}

impl BoostingLoss for LogisticLoss {
    fn initial_prediction(&self, targets: &[f64]) -> f64 {
        if targets.is_empty() {
            return 0.0;
        }
        let sum: f64 = targets.iter().sum();
        let n = targets.len() as f64;
        let p = (sum / n).clamp(1e-10, 1.0 - 1e-10);
        (p / (1.0 - p)).ln()
    }

    fn negative_gradient(&self, target: f64, prediction: f64) -> f64 {
        target - Self::sigmoid(prediction)
    }

    fn loss(&self, target: f64, prediction: f64) -> f64 {
        // Numerically stable log loss
        let prob = Self::sigmoid(prediction);
        -(target * prob.ln() + (1.0 - target) * (1.0 - prob).ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_least_squares_initial_prediction() {
        let loss = LeastSquaresLoss;
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let init = loss.initial_prediction(&targets);
        assert!((init - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_least_squares_negative_gradient() {
        let loss = LeastSquaresLoss;
        // residual = target - prediction
        assert!((loss.negative_gradient(5.0, 3.0) - 2.0).abs() < 1e-10);
        assert!((loss.negative_gradient(3.0, 5.0) - (-2.0)).abs() < 1e-10);
        assert!((loss.negative_gradient(3.0, 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_least_squares_loss() {
        let loss = LeastSquaresLoss;
        assert!((loss.loss(5.0, 3.0) - 2.0).abs() < 1e-10); // 0.5 * 4 = 2
        assert!((loss.loss(3.0, 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_initial_prediction_median() {
        let loss = QuantileLoss::new(0.5);
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let init = loss.initial_prediction(&targets);
        assert!((init - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_initial_prediction_q25() {
        let loss = QuantileLoss::new(0.25);
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let init = loss.initial_prediction(&targets);
        // 25% of 4 indices = 1, so sorted[1] = 2.0
        assert!((init - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_negative_gradient() {
        let loss = QuantileLoss::new(0.75);

        // target > prediction: gradient = alpha = 0.75
        assert!((loss.negative_gradient(5.0, 3.0) - 0.75).abs() < 1e-10);

        // target < prediction: gradient = alpha - 1 = -0.25
        assert!((loss.negative_gradient(3.0, 5.0) - (-0.25)).abs() < 1e-10);

        // target == prediction: gradient = 0
        assert!((loss.negative_gradient(3.0, 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_logistic_initial_prediction() {
        let loss = LogisticLoss;

        // 50% positive rate -> log(0.5/0.5) = log(1) = 0
        let targets = vec![0.0, 1.0, 0.0, 1.0];
        let init = loss.initial_prediction(&targets);
        assert!(init.abs() < 1e-10);

        // 100% positive rate -> log(1/eps) -> large positive
        let targets = vec![1.0, 1.0, 1.0];
        let init = loss.initial_prediction(&targets);
        assert!(init > 10.0);
    }

    #[test]
    fn test_logistic_negative_gradient() {
        let loss = LogisticLoss;

        // At prediction=0, sigmoid(0)=0.5
        // For target=1: gradient = 1 - 0.5 = 0.5
        assert!((loss.negative_gradient(1.0, 0.0) - 0.5).abs() < 1e-10);

        // For target=0: gradient = 0 - 0.5 = -0.5
        assert!((loss.negative_gradient(0.0, 0.0) - (-0.5)).abs() < 1e-10);

        // At large positive prediction, sigmoid ~ 1
        // For target=1: gradient ~ 0
        assert!(loss.negative_gradient(1.0, 10.0).abs() < 0.01);

        // For target=0: gradient ~ -1
        assert!((loss.negative_gradient(0.0, 10.0) - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_logistic_loss() {
        let loss = LogisticLoss;

        // At prediction=0, prob=0.5, loss should be ln(2) for both classes
        let loss_0 = loss.loss(0.0, 0.0);
        let loss_1 = loss.loss(1.0, 0.0);
        assert!((loss_0 - 0.693).abs() < 0.01);
        assert!((loss_1 - 0.693).abs() < 0.01);
    }

    #[test]
    fn test_empty_targets() {
        let loss = LeastSquaresLoss;
        assert_eq!(loss.initial_prediction(&[]), 0.0);
    }
}
