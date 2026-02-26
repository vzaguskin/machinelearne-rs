//! Decision stump - the simplest weak learner for gradient boosting.
//!
//! A decision stump is a single-level decision tree that splits on one feature
//! at one threshold. Despite its simplicity, ensembles of stumps (via gradient
//! boosting) can model complex relationships.

use crate::backend::{Backend, Tensor1D, Tensor2D};
use serde::{Deserialize, Serialize};

/// A fitted decision stump that predicts based on a single feature threshold.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FittedStump {
    /// Index of the feature to split on
    pub feature_idx: usize,
    /// Threshold value for the split
    pub threshold: f64,
    /// Prediction for samples where feature < threshold (left child)
    pub left_value: f64,
    /// Prediction for samples where feature >= threshold (right child)
    pub right_value: f64,
}

impl FittedStump {
    /// Predict for a single sample.
    pub fn predict_one(&self, features: &[f64]) -> f64 {
        if features[self.feature_idx] < self.threshold {
            self.left_value
        } else {
            self.right_value
        }
    }

    /// Predict for multiple samples.
    pub fn predict_batch<B: Backend>(&self, features: &Tensor2D<B>) -> Vec<f64> {
        let (n_samples, n_features) = features.shape();
        let data = features.ravel().to_vec();

        (0..n_samples)
            .map(|i| {
                let sample: Vec<f64> = (0..n_features).map(|j| data[i * n_features + j]).collect();
                self.predict_one(&sample)
            })
            .collect()
    }
}

/// A decision stump weak learner.
///
/// Finds the best single-feature split to minimize squared error on the targets
/// (which are typically pseudo-residuals in gradient boosting).
#[derive(Clone, Debug, Default)]
pub struct DecisionStump {
    /// Minimum number of samples required to consider a split
    pub min_samples_split: usize,
    /// Minimum improvement in loss to accept a split (optional)
    pub min_impurity_decrease: Option<f64>,
}

impl DecisionStump {
    /// Create a new decision stump with default parameters.
    pub fn new() -> Self {
        Self {
            min_samples_split: 2,
            min_impurity_decrease: None,
        }
    }

    /// Set the minimum samples required to split.
    pub fn with_min_samples_split(mut self, min_samples: usize) -> Self {
        self.min_samples_split = min_samples;
        self
    }

    /// Set the minimum impurity decrease required for a split.
    pub fn with_min_impurity_decrease(mut self, min_decrease: f64) -> Self {
        self.min_impurity_decrease = Some(min_decrease);
        self
    }

    /// Fit a decision stump to the data.
    ///
    /// Returns the fitted stump, or None if no valid split was found.
    pub fn fit<B: Backend>(
        &self,
        features: &Tensor2D<B>,
        targets: &Tensor1D<B>,
    ) -> Option<FittedStump> {
        let (n_samples, n_features) = features.shape();
        let feature_data: Vec<f64> = features.ravel().to_vec();
        let target_data: Vec<f64> = targets.to_vec();

        if n_samples < self.min_samples_split {
            // Return a constant predictor (mean of targets)
            let mean = if target_data.is_empty() {
                0.0
            } else {
                target_data.iter().sum::<f64>() / target_data.len() as f64
            };
            return Some(FittedStump {
                feature_idx: 0,
                threshold: f64::NEG_INFINITY,
                left_value: mean,
                right_value: mean,
            });
        }

        // Find the best split across all features
        let mut best_split: Option<SplitCandidate> = None;

        for feat_idx in 0..n_features {
            if let Some(split) = self.find_best_split_for_feature(
                feat_idx,
                &feature_data,
                &target_data,
                n_samples,
                n_features,
            ) {
                if let Some(ref best) = best_split {
                    if split.loss < best.loss {
                        best_split = Some(split);
                    }
                } else {
                    best_split = Some(split);
                }
            }
        }

        // Check minimum impurity decrease
        if let Some(split) = best_split {
            if let Some(min_decrease) = self.min_impurity_decrease {
                // Compute parent loss (variance * n)
                let parent_mean: f64 = target_data.iter().sum::<f64>() / n_samples as f64;
                let parent_loss: f64 = target_data.iter().map(|&t| (t - parent_mean).powi(2)).sum();

                let improvement = parent_loss - split.loss;
                if improvement < min_decrease {
                    // Return constant predictor
                    return Some(FittedStump {
                        feature_idx: 0,
                        threshold: f64::NEG_INFINITY,
                        left_value: parent_mean,
                        right_value: parent_mean,
                    });
                }
            }

            Some(FittedStump {
                feature_idx: split.feature_idx,
                threshold: split.threshold,
                left_value: split.left_value,
                right_value: split.right_value,
            })
        } else {
            // No valid split found, return constant predictor
            let mean = target_data.iter().sum::<f64>() / target_data.len() as f64;
            Some(FittedStump {
                feature_idx: 0,
                threshold: f64::NEG_INFINITY,
                left_value: mean,
                right_value: mean,
            })
        }
    }

    fn find_best_split_for_feature(
        &self,
        feat_idx: usize,
        feature_data: &[f64],
        target_data: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> Option<SplitCandidate> {
        // Collect (feature_value, target) pairs
        let mut pairs: Vec<(f64, f64)> = (0..n_samples)
            .map(|i| (feature_data[i * n_features + feat_idx], target_data[i]))
            .collect();

        // Sort by feature value
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Remove duplicates and find unique split points
        let mut unique_values: Vec<f64> = pairs.iter().map(|(f, _)| *f).collect();
        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_values.dedup();

        if unique_values.len() < 2 {
            return None; // Can't split on a constant feature
        }

        // Compute cumulative sums for efficient split evaluation
        let n = pairs.len();
        let mut cumsum: f64 = 0.0;
        let mut cumsum_sq: f64 = 0.0;
        let mut cumsums: Vec<(f64, f64)> = vec![(0.0, 0.0); n + 1];

        for (i, &(_, target)) in pairs.iter().enumerate() {
            cumsum += target;
            cumsum_sq += target * target;
            cumsums[i + 1] = (cumsum, cumsum_sq);
        }

        let total_sum = cumsum;
        let total_sum_sq = cumsum_sq;

        // Try each split point
        let mut best_loss = f64::INFINITY;
        let mut best_split: Option<SplitCandidate> = None;

        for i in 1..n {
            // Skip if feature value is the same as previous
            if pairs[i].0 == pairs[i - 1].0 {
                continue;
            }

            let threshold = (pairs[i - 1].0 + pairs[i].0) / 2.0;

            let left_n = i as f64;
            let right_n = (n - i) as f64;

            // Skip if split would result in too few samples
            if left_n < self.min_samples_split as f64 || right_n < self.min_samples_split as f64 {
                continue;
            }

            let left_sum = cumsums[i].0;
            let left_sum_sq = cumsums[i].1;
            let right_sum = total_sum - left_sum;
            let right_sum_sq = total_sum_sq - left_sum_sq;

            // Compute loss as sum of squared errors
            // MSE_left * n_left + MSE_right * n_right
            // = (sum_sq_left - sum_left^2/n_left) + (sum_sq_right - sum_right^2/n_right)
            let left_loss = left_sum_sq - left_sum * left_sum / left_n;
            let right_loss = right_sum_sq - right_sum * right_sum / right_n;
            let total_loss = left_loss + right_loss;

            if total_loss < best_loss {
                best_loss = total_loss;
                best_split = Some(SplitCandidate {
                    feature_idx: feat_idx,
                    threshold,
                    left_value: left_sum / left_n,
                    right_value: right_sum / right_n,
                    loss: total_loss,
                });
            }
        }

        best_split
    }
}

struct SplitCandidate {
    feature_idx: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
    loss: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_decision_stump_perfect_split() {
        // Feature 0 perfectly separates: x < 1.5 -> y=1, x >= 1.5 -> y=2
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let stump = DecisionStump::new();
        let fitted = stump.fit(&features, &targets).unwrap();

        assert_eq!(fitted.feature_idx, 0);
        assert!((fitted.left_value - 1.0).abs() < 0.01);
        assert!((fitted.right_value - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_decision_stump_predict() {
        let fitted = FittedStump {
            feature_idx: 0,
            threshold: 1.5,
            left_value: 1.0,
            right_value: 2.0,
        };

        assert!((fitted.predict_one(&[0.0]) - 1.0).abs() < 1e-10);
        assert!((fitted.predict_one(&[1.0]) - 1.0).abs() < 1e-10);
        assert!((fitted.predict_one(&[2.0]) - 2.0).abs() < 1e-10);
        assert!((fitted.predict_one(&[3.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_decision_stump_multi_feature() {
        // Feature 1 has the better split
        let features = Tensor2D::<CpuBackend>::new(
            vec![
                0.0, 0.0, // sample 0
                1.0, 0.0, // sample 1
                0.0, 1.0, // sample 2
                1.0, 1.0, // sample 3
            ],
            4,
            2,
        );
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 0.0, 1.0, 1.0]);

        let stump = DecisionStump::new();
        let fitted = stump.fit(&features, &targets).unwrap();

        // Should pick feature 1 (index 1) as the split feature
        assert_eq!(fitted.feature_idx, 1);
    }

    #[test]
    fn test_decision_stump_constant_target() {
        // All targets are the same - should return constant predictor
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![5.0, 5.0, 5.0, 5.0]);

        let stump = DecisionStump::new();
        let fitted = stump.fit(&features, &targets).unwrap();

        // Should return constant predictions
        assert!((fitted.left_value - 5.0).abs() < 1e-10);
        assert!((fitted.right_value - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_decision_stump_min_samples_split() {
        // 4 samples with min_samples_split=2 means at least 2 on each side
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let stump = DecisionStump::new().with_min_samples_split(2);
        let fitted = stump.fit(&features, &targets).unwrap();

        // Should find a valid split at threshold 1.5
        assert!((fitted.left_value - 1.0).abs() < 0.1);
        assert!((fitted.right_value - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_decision_stump_single_sample() {
        // Single sample should return constant predictor
        let features = Tensor2D::<CpuBackend>::new(vec![1.0], 1, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![5.0]);

        let stump = DecisionStump::new();
        let fitted = stump.fit(&features, &targets).unwrap();

        assert!((fitted.left_value - 5.0).abs() < 1e-10);
        assert!((fitted.right_value - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_decision_stump_predict_batch() {
        let fitted = FittedStump {
            feature_idx: 0,
            threshold: 1.5,
            left_value: 10.0,
            right_value: 20.0,
        };

        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let predictions = fitted.predict_batch(&features);

        assert_eq!(predictions.len(), 4);
        assert!((predictions[0] - 10.0).abs() < 1e-10);
        assert!((predictions[1] - 10.0).abs() < 1e-10);
        assert!((predictions[2] - 20.0).abs() < 1e-10);
        assert!((predictions[3] - 20.0).abs() < 1e-10);
    }
}
