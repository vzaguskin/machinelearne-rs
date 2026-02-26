//! Decision Tree - a multi-level weak learner for gradient boosting.
//!
//! A decision tree recursively splits the feature space to minimize prediction error.
//! Unlike `DecisionStump` (single split), trees can capture more complex patterns
//! through hierarchical splitting.

use crate::backend::{Backend, Tensor1D, Tensor2D};
use serde::{Deserialize, Serialize};

use super::boosting::WeakLearner;
use super::ensemble_model::StumpPredictor;

// Implement WeakLearner for DecisionTree to enable use in gradient boosting
impl<B: Backend> WeakLearner<B> for DecisionTree {
    type FittedModel = FittedTree;

    fn fit(
        &self,
        features: &Tensor2D<B>,
        targets: &Tensor1D<B>,
        feature_mask: Option<&[usize]>,
    ) -> Self::FittedModel {
        self.fit_with_mask(features, targets, feature_mask)
    }
}

/// Configuration for building a decision tree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionTreeConfig {
    /// Maximum depth of the tree (default: 3)
    pub max_depth: usize,
    /// Minimum samples required to consider splitting a node (default: 2)
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node (default: 1)
    pub min_samples_leaf: usize,
    /// Maximum features to consider for each split (None = all features)
    pub max_features: Option<usize>,
}

impl Default for DecisionTreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
        }
    }
}

impl DecisionTreeConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum depth of the tree.
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth.clamp(1, 10);
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

    /// Set the maximum features to consider per split.
    pub fn max_features(mut self, max: usize) -> Self {
        self.max_features = Some(max.max(1));
        self
    }
}

/// A node in the decision tree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TreeNode {
    /// Leaf node with a prediction value.
    Leaf { value: f64 },
    /// Internal node with a split condition and children.
    Split {
        /// Feature index to split on.
        feature_idx: usize,
        /// Threshold value for the split.
        threshold: f64,
        /// Left child (feature < threshold).
        left: Box<TreeNode>,
        /// Right child (feature >= threshold).
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    /// Predict for a single sample by traversing the tree.
    pub fn predict_one(&self, features: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
            } => {
                if features[*feature_idx] < *threshold {
                    left.predict_one(features)
                } else {
                    right.predict_one(features)
                }
            }
        }
    }

    /// Get the depth of this node.
    pub fn depth(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 0,
            TreeNode::Split { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }

    /// Count the number of nodes in this subtree.
    pub fn node_count(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => 1 + left.node_count() + right.node_count(),
        }
    }

    /// Count the number of leaves in this subtree.
    pub fn leaf_count(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => left.leaf_count() + right.leaf_count(),
        }
    }
}

/// A fitted decision tree ready for prediction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FittedTree {
    /// Root node of the tree.
    root: TreeNode,
    /// Number of features the tree was trained on.
    n_features: usize,
}

impl FittedTree {
    /// Create a new fitted tree.
    pub fn new(root: TreeNode, n_features: usize) -> Self {
        Self { root, n_features }
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
}

/// A decision tree weak learner.
#[derive(Clone, Debug, Default)]
pub struct DecisionTree {
    config: DecisionTreeConfig,
}

impl DecisionTree {
    /// Create a new decision tree with default configuration.
    pub fn new() -> Self {
        Self {
            config: DecisionTreeConfig::default(),
        }
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: DecisionTreeConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the maximum depth.
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.config.max_depth = depth.clamp(1, 10);
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

    /// Fit a decision tree to the data.
    pub fn fit<B: Backend>(&self, features: &Tensor2D<B>, targets: &Tensor1D<B>) -> FittedTree {
        self.fit_with_mask(features, targets, None)
    }

    /// Fit a decision tree to the data with an optional feature mask.
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
    ) -> FittedTree {
        let (n_samples, n_features) = features.shape();
        let feature_data: Vec<f64> = features.ravel().to_vec();
        let target_data: Vec<f64> = targets.to_vec();

        // Determine which features to consider
        // If both config.max_features and external mask are set, use the intersection
        let effective_features: Vec<usize> = match (feature_mask, self.config.max_features) {
            (Some(mask), Some(max_feat)) => mask.iter().cloned().take(max_feat).collect(),
            (Some(mask), None) if !mask.is_empty() => mask.to_vec(),
            (None, Some(max_feat)) => (0..max_feat.min(n_features)).collect(),
            _ => (0..n_features).collect(),
        };

        // Build the tree recursively
        let root = self.build_tree(
            &feature_data,
            &target_data,
            n_samples,
            n_features,
            0, // current depth
            &effective_features,
        );

        FittedTree::new(root, n_features)
    }

    /// Recursively build the tree.
    fn build_tree(
        &self,
        feature_data: &[f64],
        target_data: &[f64],
        n_samples: usize,
        n_features: usize,
        current_depth: usize,
        features_to_try: &[usize],
    ) -> TreeNode {
        // Check stopping conditions
        if current_depth >= self.config.max_depth
            || n_samples < self.config.min_samples_split
            || n_samples < 2 * self.config.min_samples_leaf
        {
            return TreeNode::Leaf {
                value: self.compute_leaf_value(target_data),
            };
        }

        // Find the best split
        let best_split = self.find_best_split(
            feature_data,
            target_data,
            n_samples,
            n_features,
            features_to_try,
        );

        match best_split {
            Some(split) => {
                // Partition the data
                let (left_features, left_targets, left_n) =
                    self.partition_left(&split, feature_data, target_data, n_samples, n_features);
                let (right_features, right_targets, right_n) =
                    self.partition_right(&split, feature_data, target_data, n_samples, n_features);

                // Check min_samples_leaf constraint
                if left_n < self.config.min_samples_leaf || right_n < self.config.min_samples_leaf {
                    return TreeNode::Leaf {
                        value: self.compute_leaf_value(target_data),
                    };
                }

                // Recursively build children
                let left = self.build_tree(
                    &left_features,
                    &left_targets,
                    left_n,
                    n_features,
                    current_depth + 1,
                    features_to_try,
                );
                let right = self.build_tree(
                    &right_features,
                    &right_targets,
                    right_n,
                    n_features,
                    current_depth + 1,
                    features_to_try,
                );

                TreeNode::Split {
                    feature_idx: split.feature_idx,
                    threshold: split.threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
            None => TreeNode::Leaf {
                value: self.compute_leaf_value(target_data),
            },
        }
    }

    /// Compute the leaf value (mean of targets).
    fn compute_leaf_value(&self, targets: &[f64]) -> f64 {
        if targets.is_empty() {
            0.0
        } else {
            targets.iter().sum::<f64>() / targets.len() as f64
        }
    }

    /// Find the best split across the allowed features.
    fn find_best_split(
        &self,
        feature_data: &[f64],
        target_data: &[f64],
        n_samples: usize,
        n_features: usize,
        features_to_try: &[usize],
    ) -> Option<SplitCandidate> {
        let mut best_split: Option<SplitCandidate> = None;

        for &feat_idx in features_to_try {
            if feat_idx >= n_features {
                continue;
            }
            if let Some(split) = self.find_best_split_for_feature(
                feat_idx,
                feature_data,
                target_data,
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

        best_split
    }

    /// Find the best split for a single feature.
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

        // Check if we have at least 2 unique values
        let unique_count = pairs.windows(2).filter(|w| w[0].0 != w[1].0).count() + 1;
        if unique_count < 2 {
            return None;
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

            let left_n = i;
            let right_n = n - i;

            // Check min_samples_leaf constraint
            if left_n < self.config.min_samples_leaf || right_n < self.config.min_samples_leaf {
                continue;
            }

            let threshold = (pairs[i - 1].0 + pairs[i].0) / 2.0;

            let left_sum = cumsums[i].0;
            let left_sum_sq = cumsums[i].1;
            let right_sum = total_sum - left_sum;
            let right_sum_sq = total_sum_sq - left_sum_sq;

            let left_n_f = left_n as f64;
            let right_n_f = right_n as f64;

            // Compute loss as sum of squared errors
            let left_loss = left_sum_sq - left_sum * left_sum / left_n_f;
            let right_loss = right_sum_sq - right_sum * right_sum / right_n_f;
            let total_loss = left_loss + right_loss;

            if total_loss < best_loss {
                best_loss = total_loss;
                best_split = Some(SplitCandidate {
                    feature_idx: feat_idx,
                    threshold,
                    _left_value: left_sum / left_n_f,
                    _right_value: right_sum / right_n_f,
                    loss: total_loss,
                });
            }
        }

        best_split
    }

    /// Partition data for left child.
    fn partition_left(
        &self,
        split: &SplitCandidate,
        feature_data: &[f64],
        target_data: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> (Vec<f64>, Vec<f64>, usize) {
        let mut left_features = Vec::new();
        let mut left_targets = Vec::new();

        for i in 0..n_samples {
            if feature_data[i * n_features + split.feature_idx] < split.threshold {
                for j in 0..n_features {
                    left_features.push(feature_data[i * n_features + j]);
                }
                left_targets.push(target_data[i]);
            }
        }

        let left_n = left_targets.len();
        (left_features, left_targets, left_n)
    }

    /// Partition data for right child.
    fn partition_right(
        &self,
        split: &SplitCandidate,
        feature_data: &[f64],
        target_data: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> (Vec<f64>, Vec<f64>, usize) {
        let mut right_features = Vec::new();
        let mut right_targets = Vec::new();

        for i in 0..n_samples {
            if feature_data[i * n_features + split.feature_idx] >= split.threshold {
                for j in 0..n_features {
                    right_features.push(feature_data[i * n_features + j]);
                }
                right_targets.push(target_data[i]);
            }
        }

        let right_n = right_targets.len();
        (right_features, right_targets, right_n)
    }
}

#[allow(dead_code)]
struct SplitCandidate {
    feature_idx: usize,
    threshold: f64,
    _left_value: f64,
    _right_value: f64,
    loss: f64,
}

// Implement StumpPredictor for FittedTree to integrate with gradient boosting
impl StumpPredictor for FittedTree {
    fn predict_one(&self, features: &[f64]) -> f64 {
        FittedTree::predict_one(self, features)
    }

    fn predict_batch<B: Backend>(&self, features: &Tensor2D<B>) -> Vec<f64> {
        FittedTree::predict_batch(self, features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_decision_tree_depth_1_is_stump() {
        // With max_depth=1, tree should behave like a stump
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let tree = DecisionTree::new().max_depth(1);
        let fitted = tree.fit(&features, &targets);

        // Should have depth 1 (root is a split)
        assert_eq!(fitted.depth(), 1);
        assert_eq!(fitted.leaf_count(), 2);
    }

    #[test]
    fn test_decision_tree_default_depth() {
        let config = DecisionTreeConfig::new();
        assert_eq!(config.max_depth, 3);
    }

    #[test]
    fn test_decision_tree_depth_limit() {
        let config = DecisionTreeConfig::new().max_depth(5);
        assert_eq!(config.max_depth, 5);

        // Should clamp to max 10
        let config = DecisionTreeConfig::new().max_depth(20);
        assert_eq!(config.max_depth, 10);

        // Should clamp to min 1
        let config = DecisionTreeConfig::new().max_depth(0);
        assert_eq!(config.max_depth, 1);
    }

    #[test]
    fn test_decision_tree_predict_single() {
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let tree = DecisionTree::new().max_depth(2);
        let fitted = tree.fit(&features, &targets);

        // Test predictions
        let pred_0 = fitted.predict_one(&[0.0]);
        let pred_2 = fitted.predict_one(&[2.5]);

        assert!((pred_0 - 1.0).abs() < 0.5);
        assert!((pred_2 - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_decision_tree_predict_batch() {
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let tree = DecisionTree::new().max_depth(2);
        let fitted = tree.fit(&features, &targets);

        let predictions = fitted.predict_batch(&features);
        assert_eq!(predictions.len(), 4);

        for (pred, target) in predictions.iter().zip([1.0, 1.0, 2.0, 2.0].iter()) {
            assert!((pred - target).abs() < 0.5);
        }
    }

    #[test]
    fn test_decision_tree_min_samples_leaf() {
        // With min_samples_leaf=3, shouldn't split 4 samples
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 2.0, 2.0]);

        let tree = DecisionTree::new().min_samples_leaf(3);
        let fitted = tree.fit(&features, &targets);

        // Should be a single leaf (can't split with min 3 per leaf)
        assert_eq!(fitted.leaf_count(), 1);
    }

    #[test]
    fn test_decision_tree_constant_target() {
        let features = Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![5.0, 5.0, 5.0, 5.0]);

        let tree = DecisionTree::new().max_depth(3);
        let fitted = tree.fit(&features, &targets);

        // Should predict the mean
        let pred = fitted.predict_one(&[1.5]);
        assert!((pred - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_decision_tree_multi_feature() {
        let features =
            Tensor2D::<CpuBackend>::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], 4, 2);
        let targets = Tensor1D::<CpuBackend>::new(vec![0.0, 1.0, 1.0, 2.0]);

        let tree = DecisionTree::new().max_depth(2);
        let fitted = tree.fit(&features, &targets);

        // Verify tree structure
        assert!(fitted.depth() <= 2);
        assert!(fitted.node_count() > 1);
    }
}
