//! Histogram-Based Decision Tree for efficient gradient boosting.
//!
//! This module implements histogram-based split finding like LightGBM/XGBoost `gpu_hist`:
//! - Discretizes continuous features into k bins (typically 256)
//! - Builds histograms: aggregate gradients per bin
//! - Finds splits by scanning bins instead of data points
//! - Uses bin subtraction trick for faster child histogram computation
//!
//! # Complexity
//! - Exact tree: O(#data × #features) per node
//! - Histogram tree: O(#bins × #features) per node - much faster for large datasets
//!
//! # Example
//!
//! ```rust
//! use machinelearne_rs::ensemble::{HistogramTree, GradientBoostingRegressor, WeakLearner};
//! use machinelearne_rs::backend::CpuBackend;
//! use machinelearne_rs::{Tensor1D, Tensor2D};
//!
//! let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
//! let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0, 6.0]);
//!
//! let tree = HistogramTree::new();
//! let fitted = tree.fit(&features, &targets);
//!
//! let pred = fitted.predict_one(&[1.5]);
//! ```

use crate::backend::{Backend, Tensor1D, Tensor2D};
use serde::{Deserialize, Serialize};

use super::boosting::WeakLearner;
use super::decision_tree::TreeNode;
use super::ensemble_model::StumpPredictor;

// Implement WeakLearner for HistogramTree to enable use in gradient boosting
impl<B: Backend> WeakLearner<B> for HistogramTree {
    type FittedModel = FittedHistogramTree;

    fn fit(
        &self,
        features: &Tensor2D<B>,
        targets: &Tensor1D<B>,
        feature_mask: Option<&[usize]>,
    ) -> Self::FittedModel {
        self.fit_with_mask(features, targets, feature_mask)
    }
}

/// Configuration for building a histogram-based decision tree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistogramTreeConfig {
    /// Maximum depth of the tree (default: 6)
    pub max_depth: usize,
    /// Minimum samples required to consider splitting a node (default: 2)
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node (default: 1)
    pub min_samples_leaf: usize,
    /// Number of bins for histogram (default: 256)
    pub num_bins: usize,
    /// Maximum features to consider for each split (None = all features)
    pub max_features: Option<usize>,
    /// Minimum split gain to consider a split valid (default: 0.0)
    pub min_split_gain: f64,
}

impl Default for HistogramTreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 6,
            min_samples_split: 2,
            min_samples_leaf: 1,
            num_bins: 256,
            max_features: None,
            min_split_gain: 0.0,
        }
    }
}

impl HistogramTreeConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum depth of the tree.
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth.clamp(1, 20);
        self
    }

    /// Set the minimum samples required to split a node.
    pub fn min_samples_split(mut self, min: usize) -> Self {
        self.min_samples_split = min.max(2);
        self
    }

    /// Set the minimum samples required in a leaf.
    pub fn min_samples_leaf(mut self, min: usize) -> Self {
        self.min_samples_leaf = min.max(1);
        self
    }

    /// Set the number of bins for histogram discretization.
    pub fn num_bins(mut self, bins: usize) -> Self {
        self.num_bins = bins.clamp(2, 1024);
        self
    }

    /// Set the maximum features to consider per split.
    pub fn max_features(mut self, max: usize) -> Self {
        self.max_features = Some(max.max(1));
        self
    }

    /// Set the minimum split gain threshold.
    pub fn min_split_gain(mut self, gain: f64) -> Self {
        self.min_split_gain = gain.max(0.0);
        self
    }
}

/// Bins continuous features into discrete values.
///
/// Uses quantile-based binning to create bins with roughly equal numbers of samples.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureBinner {
    /// Bin edges for each feature (n_features x num_bins+1)
    /// Values <= edges[0] go to bin 0
    /// Values > edges[i] and <= edges[i+1] go to bin i+1
    /// Values > edges[num_bins-1] go to bin num_bins-1
    feature_bins: Vec<Vec<f64>>,
    /// Number of features
    n_features: usize,
    /// Number of bins per feature
    num_bins: usize,
}

impl FeatureBinner {
    /// Create a fitter from training data.
    ///
    /// Computes quantile-based bin edges for each feature.
    pub fn from_data(data: &[f64], n_samples: usize, n_features: usize, num_bins: usize) -> Self {
        let mut feature_bins = Vec::with_capacity(n_features);

        for feat_idx in 0..n_features {
            // Extract feature values
            let mut values: Vec<f64> = (0..n_samples)
                .map(|i| data[i * n_features + feat_idx])
                .filter(|&v| v.is_finite())
                .collect();

            if values.is_empty() {
                // Handle empty/invalid feature
                feature_bins.push(vec![0.0; num_bins]);
                continue;
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Remove duplicates for quantile computation
            values.dedup();

            let n = values.len();
            let mut edges = Vec::with_capacity(num_bins);

            if n <= num_bins {
                // Fewer unique values than bins: use all unique values
                for &v in &values {
                    edges.push(v);
                }
                // Pad with the last value
                while edges.len() < num_bins {
                    edges.push(*edges.last().unwrap_or(&0.0));
                }
            } else {
                // Compute quantile edges
                for i in 1..=num_bins {
                    let rank = (i as f64 / (num_bins + 1) as f64) * (n - 1) as f64;
                    let lower = rank.floor() as usize;
                    let upper = (lower + 1).min(n - 1);
                    let frac = rank - lower as f64;
                    let edge = values[lower] * (1.0 - frac) + values[upper] * frac;
                    edges.push(edge);
                }
            }

            feature_bins.push(edges);
        }

        Self {
            feature_bins,
            n_features,
            num_bins,
        }
    }

    /// Get the bin index for a feature value.
    pub fn bin_for_value(&self, feature_idx: usize, value: f64) -> usize {
        if feature_idx >= self.n_features || self.feature_bins[feature_idx].is_empty() {
            return 0;
        }

        let edges = &self.feature_bins[feature_idx];

        // Binary search for the bin
        // Bin i is for values in (edges[i-1], edges[i]]
        let mut bin = 0;
        for (i, &edge) in edges.iter().enumerate() {
            if value <= edge {
                bin = i;
                break;
            }
            bin = i;
        }

        bin.min(self.num_bins - 1)
    }

    /// Transform all samples to bin indices.
    ///
    /// Returns a Vec of shape (n_samples x n_features) with bin indices.
    pub fn transform(&self, data: &[f64], n_samples: usize, n_features: usize) -> Vec<usize> {
        let mut bin_indices = Vec::with_capacity(n_samples * n_features);

        for i in 0..n_samples {
            for j in 0..n_features {
                let value = data[i * n_features + j];
                bin_indices.push(self.bin_for_value(j, value));
            }
        }

        bin_indices
    }

    /// Get the threshold value for a bin boundary.
    ///
    /// Returns the midpoint between bin_idx and bin_idx+1 edges, which provides
    /// a clean separation for the split. This avoids issues with samples that
    /// have values exactly equal to the bin edge.
    pub fn get_threshold(&self, feature_idx: usize, bin_idx: usize) -> f64 {
        if feature_idx >= self.n_features || bin_idx >= self.num_bins {
            return 0.0;
        }

        let edges = &self.feature_bins[feature_idx];
        let current_edge = edges[bin_idx];

        // Get the next edge, or use current + small epsilon if at the last bin
        let next_edge = if bin_idx + 1 < edges.len() {
            edges[bin_idx + 1]
        } else {
            current_edge + 1.0
        };

        // Return midpoint to ensure clean separation
        (current_edge + next_edge) / 2.0
    }

    /// Get number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get number of bins.
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }
}

/// A histogram for a single feature.
///
/// Stores gradient sum and sample count per bin.
#[derive(Clone, Debug)]
pub struct Histogram {
    /// Gradient sum per bin
    grad_sum: Vec<f64>,
    /// Sample count per bin
    count: Vec<usize>,
    /// Number of bins
    num_bins: usize,
}

impl Histogram {
    /// Create a new empty histogram.
    pub fn new(num_bins: usize) -> Self {
        Self {
            grad_sum: vec![0.0; num_bins],
            count: vec![0; num_bins],
            num_bins,
        }
    }

    /// Add a sample to the histogram.
    pub fn add(&mut self, bin_idx: usize, gradient: f64) {
        if bin_idx < self.num_bins {
            self.grad_sum[bin_idx] += gradient;
            self.count[bin_idx] += 1;
        }
    }

    /// Subtract another histogram from this one.
    ///
    /// Used for the bin subtraction trick:
    /// larger_child_hist = parent_hist - smaller_child_hist
    pub fn subtract(&self, other: &Histogram) -> Histogram {
        let mut result = Histogram::new(self.num_bins);
        for i in 0..self.num_bins {
            result.grad_sum[i] = self.grad_sum[i] - other.grad_sum[i];
            result.count[i] = self.count[i].saturating_sub(other.count[i]);
        }
        result
    }

    /// Get total gradient sum.
    pub fn total_grad(&self) -> f64 {
        self.grad_sum.iter().sum()
    }

    /// Get total sample count.
    pub fn total_count(&self) -> usize {
        self.count.iter().sum()
    }

    /// Compute cumulative statistics for split finding.
    ///
    /// Returns (cumulative_grad, cumulative_count) for each bin boundary.
    pub fn cumulative_stats(&self) -> (Vec<f64>, Vec<usize>) {
        let n = self.num_bins;
        let mut cumul_grad = Vec::with_capacity(n);
        let mut cumul_count = Vec::with_capacity(n);

        let mut grad = 0.0;
        let mut cnt = 0;

        for i in 0..n {
            grad += self.grad_sum[i];
            cnt += self.count[i];
            cumul_grad.push(grad);
            cumul_count.push(cnt);
        }

        (cumul_grad, cumul_count)
    }

    /// Get gradient sum for a bin.
    pub fn grad_sum(&self, bin_idx: usize) -> f64 {
        if bin_idx < self.num_bins {
            self.grad_sum[bin_idx]
        } else {
            0.0
        }
    }

    /// Get count for a bin.
    pub fn count(&self, bin_idx: usize) -> usize {
        if bin_idx < self.num_bins {
            self.count[bin_idx]
        } else {
            0
        }
    }
}

/// Build histograms for all features.
///
/// # Arguments
/// * `bin_indices` - Pre-computed bin indices (n_samples x n_features)
/// * `gradients` - Target gradients/residuals (n_samples)
/// * `sample_indices` - Indices of samples to include (for node-level histograms)
/// * `n_features` - Number of features
/// * `num_bins` - Number of bins per feature
///
/// # Returns
/// Vector of histograms, one per feature
pub fn build_histograms(
    bin_indices: &[usize],
    gradients: &[f64],
    sample_indices: &[usize],
    n_features: usize,
    num_bins: usize,
) -> Vec<Histogram> {
    let mut histograms: Vec<Histogram> =
        (0..n_features).map(|_| Histogram::new(num_bins)).collect();

    for &sample_idx in sample_indices {
        let gradient = gradients[sample_idx];
        for feat_idx in 0..n_features {
            let bin_idx = bin_indices[sample_idx * n_features + feat_idx];
            histograms[feat_idx].add(bin_idx, gradient);
        }
    }

    histograms
}

/// Find the best split across all features using histograms.
///
/// Returns (feature_idx, bin_idx, gain, left_value, right_value) if a good split is found.
pub fn find_best_split_from_histograms(
    histograms: &[Histogram],
    _binner: &FeatureBinner,
    total_grad: f64,
    total_count: usize,
    features_to_try: &[usize],
    min_samples_leaf: usize,
    min_split_gain: f64,
) -> Option<(usize, usize, f64, f64, f64)> {
    if total_count == 0 {
        return None;
    }

    let mut best: Option<(usize, usize, f64, f64, f64)> = None;
    let mut best_gain = f64::NEG_INFINITY;

    for &feat_idx in features_to_try {
        if feat_idx >= histograms.len() {
            continue;
        }

        let hist = &histograms[feat_idx];
        let (cumul_grad, cumul_count) = hist.cumulative_stats();

        // Try split after each bin (except the last)
        for bin_idx in 0..(hist.num_bins - 1) {
            let left_count = cumul_count[bin_idx];
            let right_count = total_count - left_count;

            // Check min_samples_leaf constraint
            if left_count < min_samples_leaf || right_count < min_samples_leaf {
                continue;
            }

            let left_grad = cumul_grad[bin_idx];
            let right_grad = total_grad - left_grad;

            // Compute split gain using variance reduction
            // Gain = Var(parent) - Var(left) - Var(right)
            // For gradient boosting, we approximate with:
            // Gain = (left_grad^2 / left_n + right_grad^2 / right_n) - total_grad^2 / total_n
            // This is proportional to the reduction in loss

            let left_n = left_count as f64;
            let right_n = right_count as f64;
            let total_n = total_count as f64;

            // Skip if either side is empty
            if left_n == 0.0 || right_n == 0.0 {
                continue;
            }

            let gain = (left_grad * left_grad / left_n) + (right_grad * right_grad / right_n)
                - (total_grad * total_grad / total_n);

            if gain > best_gain && gain > min_split_gain {
                best_gain = gain;
                let left_value = left_grad / left_n;
                let right_value = right_grad / right_n;
                best = Some((feat_idx, bin_idx, gain, left_value, right_value));
            }
        }
    }

    best
}

/// A fitted histogram-based decision tree ready for prediction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FittedHistogramTree {
    /// Root node of the tree.
    root: TreeNode,
    /// Number of features the tree was trained on.
    n_features: usize,
    /// Number of bins used (for reference).
    num_bins: usize,
}

impl FittedHistogramTree {
    /// Create a new fitted histogram tree.
    pub fn new(root: TreeNode, n_features: usize, num_bins: usize) -> Self {
        Self {
            root,
            n_features,
            num_bins,
        }
    }

    /// Predict for a single sample.
    pub fn predict_one(&self, features: &[f64]) -> f64 {
        self.root.predict_one(features)
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

    /// Get the depth of the tree.
    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    /// Get the total number of nodes.
    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }

    /// Get the number of leaf nodes.
    pub fn leaf_count(&self) -> usize {
        self.root.leaf_count()
    }

    /// Get the number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get the number of bins.
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }
}

// Implement StumpPredictor for FittedHistogramTree to integrate with gradient boosting
impl StumpPredictor for FittedHistogramTree {
    fn predict_one(&self, features: &[f64]) -> f64 {
        FittedHistogramTree::predict_one(self, features)
    }

    fn predict_batch<B: Backend>(&self, features: &Tensor2D<B>) -> Vec<f64> {
        FittedHistogramTree::predict_batch(self, features)
    }
}

/// A histogram-based decision tree weak learner.
#[derive(Clone, Debug, Default)]
pub struct HistogramTree {
    config: HistogramTreeConfig,
}

impl HistogramTree {
    /// Create a new histogram tree with default configuration.
    pub fn new() -> Self {
        Self {
            config: HistogramTreeConfig::default(),
        }
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: HistogramTreeConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the maximum depth.
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.config.max_depth = depth.clamp(1, 20);
        self
    }

    /// Set the minimum samples to split.
    pub fn min_samples_split(mut self, min: usize) -> Self {
        self.config.min_samples_split = min.max(2);
        self
    }

    /// Set the minimum samples in a leaf.
    pub fn min_samples_leaf(mut self, min: usize) -> Self {
        self.config.min_samples_leaf = min.max(1);
        self
    }

    /// Set the number of bins.
    pub fn num_bins(mut self, bins: usize) -> Self {
        self.config.num_bins = bins.clamp(2, 1024);
        self
    }

    /// Fit a histogram tree to the data.
    pub fn fit<B: Backend>(
        &self,
        features: &Tensor2D<B>,
        targets: &Tensor1D<B>,
    ) -> FittedHistogramTree {
        self.fit_with_mask(features, targets, None)
    }

    /// Fit a histogram tree to the data with an optional feature mask.
    ///
    /// # Arguments
    /// * `features` - Training features (n_samples x n_features)
    /// * `targets` - Targets to fit (pseudo-residuals in boosting)
    /// * `feature_mask` - Optional indices of features to consider (None = all features)
    ///
    /// # Returns
    /// A fitted tree ready for prediction.
    pub fn fit_with_mask<B: Backend>(
        &self,
        features: &Tensor2D<B>,
        targets: &Tensor1D<B>,
        feature_mask: Option<&[usize]>,
    ) -> FittedHistogramTree {
        let (n_samples, n_features) = features.shape();
        let feature_data: Vec<f64> = features.ravel().to_vec();
        let target_data: Vec<f64> = targets.to_vec();

        // Determine which features to consider
        let effective_features: Vec<usize> = match (feature_mask, self.config.max_features) {
            (Some(mask), Some(max_feat)) => mask.iter().cloned().take(max_feat).collect(),
            (Some(mask), None) if !mask.is_empty() => mask.to_vec(),
            (None, Some(max_feat)) => (0..max_feat.min(n_features)).collect(),
            _ => (0..n_features).collect(),
        };

        // Create feature binner from data
        let binner =
            FeatureBinner::from_data(&feature_data, n_samples, n_features, self.config.num_bins);

        // Transform data to bin indices
        let bin_indices = binner.transform(&feature_data, n_samples, n_features);

        // All sample indices
        let sample_indices: Vec<usize> = (0..n_samples).collect();

        // Build root histograms
        let root_histograms = build_histograms(
            &bin_indices,
            &target_data,
            &sample_indices,
            n_features,
            self.config.num_bins,
        );

        // Build the tree recursively
        let root = self.build_tree(
            &bin_indices,
            &target_data,
            n_samples,
            n_features,
            0, // current depth
            &effective_features,
            &binner,
            &root_histograms,
            &sample_indices,
        );

        FittedHistogramTree::new(root, n_features, self.config.num_bins)
    }

    /// Recursively build the tree using histograms.
    #[allow(clippy::too_many_arguments)]
    fn build_tree(
        &self,
        bin_indices: &[usize],
        target_data: &[f64],
        _n_samples: usize,
        n_features: usize,
        current_depth: usize,
        features_to_try: &[usize],
        binner: &FeatureBinner,
        histograms: &[Histogram],
        sample_indices: &[usize],
    ) -> TreeNode {
        let total_count: usize = sample_indices.len();
        let total_grad: f64 = sample_indices.iter().map(|&i| target_data[i]).sum();

        // Check stopping conditions
        if current_depth >= self.config.max_depth
            || total_count < self.config.min_samples_split
            || total_count < 2 * self.config.min_samples_leaf
        {
            return TreeNode::Leaf {
                value: if total_count > 0 {
                    total_grad / total_count as f64
                } else {
                    0.0
                },
            };
        }

        // Find the best split from histograms
        let best_split = find_best_split_from_histograms(
            histograms,
            binner,
            total_grad,
            total_count,
            features_to_try,
            self.config.min_samples_leaf,
            self.config.min_split_gain,
        );

        match best_split {
            Some((feat_idx, bin_idx, _gain, _left_value, _right_value)) => {
                // Get the threshold for this bin
                let threshold = binner.get_threshold(feat_idx, bin_idx);

                // Partition samples based on bin index
                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = sample_indices
                    .iter()
                    .cloned()
                    .partition(|&i| bin_indices[i * n_features + feat_idx] <= bin_idx);

                // Check min_samples_leaf constraint again
                if left_indices.len() < self.config.min_samples_leaf
                    || right_indices.len() < self.config.min_samples_leaf
                {
                    return TreeNode::Leaf {
                        value: total_grad / total_count as f64,
                    };
                }

                // Build child histograms using bin subtraction trick
                // Build the smaller child's histogram, then subtract from parent
                let (smaller_indices, _larger_indices) =
                    if left_indices.len() <= right_indices.len() {
                        (&left_indices, &right_indices)
                    } else {
                        (&right_indices, &left_indices)
                    };

                let smaller_histograms = build_histograms(
                    bin_indices,
                    target_data,
                    smaller_indices,
                    n_features,
                    self.config.num_bins,
                );

                let larger_histograms: Vec<Histogram> = histograms
                    .iter()
                    .zip(smaller_histograms.iter())
                    .map(|(parent, smaller)| parent.subtract(smaller))
                    .collect();

                let (left_histograms, right_histograms, left_samples, right_samples) =
                    if left_indices.len() <= right_indices.len() {
                        (
                            smaller_histograms.clone(),
                            larger_histograms,
                            left_indices.clone(),
                            right_indices.clone(),
                        )
                    } else {
                        (
                            larger_histograms,
                            smaller_histograms.clone(),
                            left_indices.clone(),
                            right_indices.clone(),
                        )
                    };

                // Recursively build children
                let left = self.build_tree(
                    bin_indices,
                    target_data,
                    _n_samples,
                    n_features,
                    current_depth + 1,
                    features_to_try,
                    binner,
                    &left_histograms,
                    &left_samples,
                );
                let right = self.build_tree(
                    bin_indices,
                    target_data,
                    _n_samples,
                    n_features,
                    current_depth + 1,
                    features_to_try,
                    binner,
                    &right_histograms,
                    &right_samples,
                );

                TreeNode::Split {
                    feature_idx: feat_idx,
                    threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
            None => TreeNode::Leaf {
                value: if total_count > 0 {
                    total_grad / total_count as f64
                } else {
                    0.0
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_feature_binner_quantiles() {
        // Create data with 100 samples, 1 feature
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let binner = FeatureBinner::from_data(&data, 100, 1, 10);

        assert_eq!(binner.n_features(), 1);
        assert_eq!(binner.num_bins(), 10);

        // First value should be in bin 0
        assert_eq!(binner.bin_for_value(0, 0.0), 0);

        // Last value should be in the last bin
        let last_bin = binner.bin_for_value(0, 99.0);
        assert!(last_bin < 10);

        // Middle value should be around bin 5
        let mid_bin = binner.bin_for_value(0, 50.0);
        assert!(mid_bin > 2 && mid_bin < 8);
    }

    #[test]
    fn test_feature_binner_transform() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let binner = FeatureBinner::from_data(&data, 10, 1, 4);

        let bins = binner.transform(&data, 10, 1);
        assert_eq!(bins.len(), 10);

        // All bins should be valid
        for &bin in &bins {
            assert!(bin < 4);
        }
    }

    #[test]
    fn test_histogram_add_subtract() {
        let mut hist1 = Histogram::new(4);
        hist1.add(0, 1.0);
        hist1.add(1, 2.0);
        hist1.add(2, 3.0);
        hist1.add(3, 4.0);

        assert_eq!(hist1.total_count(), 4);
        assert!((hist1.total_grad() - 10.0).abs() < 1e-10);

        let mut hist2 = Histogram::new(4);
        hist2.add(0, 1.0);
        hist2.add(1, 2.0);

        let hist3 = hist1.subtract(&hist2);
        assert_eq!(hist3.count(0), 0);
        assert_eq!(hist3.count(1), 0);
        assert_eq!(hist3.count(2), 1);
        assert_eq!(hist3.count(3), 1);
        assert!((hist3.grad_sum(2) - 3.0).abs() < 1e-10);
        assert!((hist3.grad_sum(3) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_cumulative_stats() {
        let mut hist = Histogram::new(4);
        hist.add(0, 1.0);
        hist.add(1, 2.0);
        hist.add(2, 3.0);
        hist.add(3, 4.0);

        let (cumul_grad, cumul_count) = hist.cumulative_stats();

        assert_eq!(cumul_count, vec![1, 2, 3, 4]);
        assert!((cumul_grad[0] - 1.0).abs() < 1e-10);
        assert!((cumul_grad[1] - 3.0).abs() < 1e-10);
        assert!((cumul_grad[2] - 6.0).abs() < 1e-10);
        assert!((cumul_grad[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_finding_basic() {
        // Create data where a clear split exists
        // Feature: [0, 0, 1, 1], Target: [1, 1, 2, 2]
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let targets = vec![1.0, 1.0, 2.0, 2.0];

        let binner = FeatureBinner::from_data(&data, 4, 1, 4);
        let bins = binner.transform(&data, 4, 1);

        let sample_indices: Vec<usize> = (0..4).collect();
        let histograms = build_histograms(&bins, &targets, &sample_indices, 1, 4);

        let total_grad: f64 = targets.iter().sum();
        let split =
            find_best_split_from_histograms(&histograms, &binner, total_grad, 4, &[0], 1, 0.0);

        assert!(split.is_some());
        let (feat_idx, _bin_idx, _gain, left_value, right_value) = split.unwrap();
        assert_eq!(feat_idx, 0);
        assert!((left_value - 1.0).abs() < 0.5);
        assert!((right_value - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_histogram_tree_basic() {
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let tree = HistogramTree::new().max_depth(2);
        let fitted = tree.fit(&features, &targets);

        // Should have depth at most 2
        assert!(fitted.depth() <= 2);

        // Predictions should be reasonable
        let pred = fitted.predict_one(&[0.5]);
        assert!((pred - 1.0).abs() < 1.0);

        let pred = fitted.predict_one(&[2.5]);
        assert!((pred - 2.0).abs() < 1.0);
    }

    #[test]
    fn test_histogram_tree_batch_prediction() {
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let tree = HistogramTree::new().max_depth(3).num_bins(4);
        let fitted = tree.fit(&features, &targets);

        let predictions = fitted.predict_batch(&features);
        assert_eq!(predictions.len(), 4);

        // Histogram-based trees approximate the data
        for (pred, &target) in predictions.iter().zip([1.0, 1.0, 2.0, 2.0].iter()) {
            assert!(
                (pred - target).abs() < 0.5,
                "Expected ~{}, got {}",
                target,
                pred
            );
        }
    }

    #[test]
    fn test_histogram_tree_min_samples_leaf() {
        // With min_samples_leaf=3, shouldn't split 4 samples
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let tree = HistogramTree::new().min_samples_leaf(3);
        let fitted = tree.fit(&features, &targets);

        // Should be a single leaf (can't split with min 3 per leaf)
        assert_eq!(fitted.leaf_count(), 1);
    }

    #[test]
    fn test_histogram_tree_constant_target() {
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![5.0, 5.0, 5.0, 5.0]);

        let tree = HistogramTree::new().max_depth(3);
        let fitted = tree.fit(&features, &targets);

        // Should predict the mean
        let pred = fitted.predict_one(&[1.5]);
        assert!((pred - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_tree_multi_feature() {
        let features =
            Tensor2D::<CpuBackend>::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], 4, 2);
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 1.0, 2.0]);

        let tree = HistogramTree::new().max_depth(2);
        let fitted = tree.fit(&features, &targets);

        // Verify tree structure
        assert!(fitted.depth() <= 2);
        assert!(fitted.node_count() >= 1);
    }

    #[test]
    fn test_histogram_tree_weak_learner_trait() {
        use super::super::boosting::WeakLearner;

        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let tree = HistogramTree::new();
        let fitted = WeakLearner::<CpuBackend>::fit(&tree, &features, &targets, None);

        let predictions = fitted.predict_batch(&features);
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_histogram_tree_with_feature_mask() {
        // y = x1 (feature 0 is informative, feature 1 is noise)
        let features = Tensor2D::<CpuBackend>::new(
            vec![
                0.0, 0.5, // sample 0
                1.0, 0.3, // sample 1
                2.0, 0.7, // sample 2
                3.0, 0.1, // sample 3
            ],
            4,
            2,
        );
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0]);

        let tree = HistogramTree::new().max_depth(2);

        // With only feature 0 allowed
        let fitted = tree.fit_with_mask(&features, &targets, Some(&[0]));
        let pred = fitted.predict_one(&[1.5, 0.0]);
        assert!((pred - 1.5).abs() < 1.0);

        // With only feature 1 allowed (noise)
        let fitted_noise = tree.fit_with_mask(&features, &targets, Some(&[1]));
        let pred_noise = fitted_noise.predict_one(&[1.5, 0.0]);
        // Should still produce a valid prediction, but may not be accurate
        assert!(pred_noise.is_finite());
    }

    #[test]
    fn test_histogram_tree_config() {
        let config = HistogramTreeConfig::new()
            .max_depth(5)
            .min_samples_leaf(2)
            .num_bins(128)
            .min_split_gain(0.01);

        assert_eq!(config.max_depth, 5);
        assert_eq!(config.min_samples_leaf, 2);
        assert_eq!(config.num_bins, 128);
        assert!((config.min_split_gain - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_tree_depth_limit() {
        let config = HistogramTreeConfig::new().max_depth(10);
        assert_eq!(config.max_depth, 10);

        // Should clamp to max 20
        let config = HistogramTreeConfig::new().max_depth(30);
        assert_eq!(config.max_depth, 20);

        // Should clamp to min 1
        let config = HistogramTreeConfig::new().max_depth(0);
        assert_eq!(config.max_depth, 1);
    }

    #[test]
    fn test_stump_predictor_trait() {
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let tree = HistogramTree::new().max_depth(2);
        let fitted = tree.fit(&features, &targets);

        // Use StumpPredictor trait
        let pred = StumpPredictor::predict_one(&fitted, &[0.5]);
        assert!(pred.is_finite());

        let preds = StumpPredictor::predict_batch::<CpuBackend>(&fitted, &features);
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_gradient_boosting_with_histogram_tree() {
        use super::super::boosting::{GradientBoostingTrainer, WeakLearner};
        use super::super::loss::LeastSquaresLoss;

        // Train a model that learns y = 2*x
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], 5, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 2.0, 4.0, 6.0, 8.0]);

        // Use histogram tree as weak learner
        let histogram_tree = HistogramTree::new().max_depth(3).num_bins(16);

        let trainer = GradientBoostingTrainer::new(LeastSquaresLoss)
            .n_estimators(50)
            .learning_rate(0.5);

        let model = trainer.fit_with_weak_learner(&features, &targets, &histogram_tree);

        // Verify training works
        let train_preds = model.predict_batch(&features);
        for (i, (pred, target)) in train_preds
            .to_vec()
            .iter()
            .zip(targets.to_vec().iter())
            .enumerate()
        {
            assert!(
                (pred - target).abs() < 2.0,
                "Sample {}: expected {}, got {}",
                i,
                target,
                pred
            );
        }

        // Test inference on new data
        let test_input = Tensor1D::<CpuBackend>::new(vec![2.5]);
        let pred = model.predict(&test_input);
        assert!(
            (pred.to_f64() - 5.0).abs() < 2.0,
            "Expected ~5.0, got {}",
            pred.to_f64()
        );
    }

    #[test]
    fn test_histogram_tree_vs_exact_tree() {
        // Compare histogram tree with exact decision tree
        use super::super::decision_tree::DecisionTree;

        let features = Tensor2D::<CpuBackend>::new(
            vec![
                0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0,
                8.0, 0.0, 9.0, 0.0,
            ],
            10,
            2,
        );
        let targets =
            Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Train exact tree
        let exact_tree = DecisionTree::new().max_depth(3);
        let exact_fitted = exact_tree.fit(&features, &targets);

        // Train histogram tree
        let hist_tree = HistogramTree::new().max_depth(3).num_bins(32);
        let hist_fitted = hist_tree.fit(&features, &targets);

        // Both should produce reasonable predictions
        let exact_pred = exact_fitted.predict_one(&[4.5, 0.0]);
        let hist_pred = hist_fitted.predict_one(&[4.5, 0.0]);

        // Both should be close to 4.5 (the actual value)
        assert!(
            (exact_pred - 4.5).abs() < 1.0,
            "Exact tree: expected ~4.5, got {}",
            exact_pred
        );
        assert!(
            (hist_pred - 4.5).abs() < 1.5,
            "Histogram tree: expected ~4.5, got {}",
            hist_pred
        );
    }
}
