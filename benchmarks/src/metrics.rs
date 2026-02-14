/// Metrics for evaluating regression models.
pub struct Metrics;

impl Metrics {
    /// Calculate Mean Squared Error (MSE).
    ///
    /// MSE = mean((y_true - y_pred)^2)
    ///
    /// # Arguments
    ///
    /// * `y_true` - Ground truth values
    /// * `y_pred` - Predicted values
    ///
    /// # Returns
    ///
    /// The MSE value (lower is better)
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
    ///
    /// # Arguments
    ///
    /// * `y_true` - Ground truth values
    /// * `y_pred` - Predicted values
    ///
    /// # Returns
    ///
    /// The RMSE value (lower is better, in same units as the target)
    pub fn rmse(y_true: &[f32], y_pred: &[f32]) -> f32 {
        Self::mse(y_true, y_pred).sqrt()
    }

    /// Calculate Mean Absolute Error (MAE).
    ///
    /// MAE = mean(|y_true - y_pred|)
    ///
    /// # Arguments
    ///
    /// * `y_true` - Ground truth values
    /// * `y_pred` - Predicted values
    ///
    /// # Returns
    ///
    /// The MAE value (lower is better)
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
    /// where:
    /// - SS_res = sum((y_true - y_pred)^2)  (residual sum of squares)
    /// - SS_tot = sum((y_true - mean(y_true))^2)  (total sum of squares)
    ///
    /// R² ranges from 0 to 1, where 1 indicates perfect prediction.
    /// Values can be negative if the model is arbitrarily worse than the mean.
    ///
    /// # Arguments
    ///
    /// * `y_true` - Ground truth values
    /// * `y_pred` - Predicted values
    ///
    /// # Returns
    ///
    /// The R² value (higher is better)
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

    /// Calculate all metrics at once.
    ///
    /// # Arguments
    ///
    /// * `y_true` - Ground truth values
    /// * `y_pred` - Predicted values
    ///
    /// # Returns
    ///
    /// A struct containing MSE, RMSE, MAE, and R²
    pub fn calculate_all(y_true: &[f32], y_pred: &[f32]) -> RegressionMetrics {
        RegressionMetrics {
            mse: Self::mse(y_true, y_pred),
            rmse: Self::rmse(y_true, y_pred),
            mae: Self::mae(y_true, y_pred),
            r_squared: Self::r_squared(y_true, y_pred),
        }
    }
}

/// Struct to hold all regression metrics.
#[derive(Debug, Clone, Copy)]
pub struct RegressionMetrics {
    pub mse: f32,
    pub rmse: f32,
    pub mae: f32,
    pub r_squared: f32,
}

impl RegressionMetrics {
    /// Create a new RegressionMetrics instance.
    pub fn new(mse: f32, mae: f32, r_squared: f32) -> Self {
        Self {
            mse,
            rmse: mse.sqrt(),
            mae,
            r_squared,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_perfect() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        assert!((Metrics::mse(&y_true, &y_pred) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_error() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![2.0, 3.0, 4.0, 5.0];
        // Errors: [-1, -1, -1, -1], squared: [1, 1, 1, 1], mean: 1.0
        assert!((Metrics::mse(&y_true, &y_pred) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mae() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![2.0, 3.0, 4.0, 5.0];
        // Errors: [-1, -1, -1, -1], abs: [1, 1, 1, 1], mean: 1.0
        assert!((Metrics::mae(&y_true, &y_pred) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_r_squared_perfect() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        assert!((Metrics::r_squared(&y_true, &y_pred) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_r_squared_mean() {
        let y_true = vec![2.0, 2.0, 2.0, 2.0];
        let y_pred = vec![2.0, 2.0, 2.0, 2.0];
        // All predictions equal the mean (which is 2.0), so R² should be 1.0
        assert!((Metrics::r_squared(&y_true, &y_pred) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_all() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        let metrics = Metrics::calculate_all(&y_true, &y_pred);
        assert!((metrics.mse - 0.0).abs() < 1e-6);
        assert!((metrics.mae - 0.0).abs() < 1e-6);
        assert!((metrics.r_squared - 1.0).abs() < 1e-6);
    }
}
