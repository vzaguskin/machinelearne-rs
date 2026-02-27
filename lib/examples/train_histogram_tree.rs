//! Example demonstrating histogram-based decision trees for gradient boosting.
//!
//! This example compares:
//! 1. Exact decision tree (traditional approach)
//! 2. Histogram-based decision tree (faster for large datasets)
//!
//! Histogram trees use quantile-based binning to discretize features,
//! then find splits by scanning bins instead of data points.
//! Complexity: O(#bins × #features) vs O(#data × #features) for exact trees.

use machinelearne_rs::{
    backend::CpuBackend,
    ensemble::{DecisionTree, GradientBoostingTrainer, HistogramTree, LeastSquaresLoss},
    Tensor1D, Tensor2D,
};
use std::time::Instant;

fn generate_data(n_samples: usize, n_features: usize) -> (Vec<f32>, Vec<f32>) {
    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut targets = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut sample = Vec::with_capacity(n_features);
        let mut sum = 0.0;
        for j in 0..n_features {
            let val = ((i * n_features + j) as f32 * 0.1).sin();
            sample.push(val);
            if j == 0 {
                sum += 2.0 * val as f64;
            } else {
                sum += 0.5 * val as f64;
            }
        }
        features.extend(sample);
        targets.push((sum + 0.1 * (i as f64).sin()) as f32);
    }

    (features, targets)
}

fn main() {
    println!("=== Histogram-Based Decision Tree Example ===\n");

    // Generate synthetic data
    let n_samples = 1000;
    let n_features = 10;
    println!(
        "Generating {} samples with {} features...",
        n_samples, n_features
    );
    let (features, targets) = generate_data(n_samples, n_features);

    let features = Tensor2D::<CpuBackend>::new(features, n_samples, n_features);
    let targets = Tensor1D::<CpuBackend>::new(targets);

    // Compare single tree performance
    println!("\n--- Single Tree Comparison ---");

    // Exact decision tree
    let exact_tree = DecisionTree::new().max_depth(6);
    let start = Instant::now();
    let exact_fitted = exact_tree.fit(&features, &targets);
    let exact_time = start.elapsed();
    println!(
        "Exact tree: depth={}, nodes={} ({:?})",
        exact_fitted.depth(),
        exact_fitted.node_count(),
        exact_time
    );

    // Histogram-based tree
    let hist_tree = HistogramTree::new().max_depth(6).num_bins(64);
    let start = Instant::now();
    let hist_fitted = hist_tree.fit(&features, &targets);
    let hist_time = start.elapsed();
    println!(
        "Histogram tree: depth={}, nodes={} ({:?})",
        hist_fitted.depth(),
        hist_fitted.node_count(),
        hist_time
    );

    // Compare predictions
    let test_sample: Vec<f64> = (0..n_features).map(|j| (j as f64 * 0.1).sin()).collect();
    let exact_pred = exact_fitted.predict_one(&test_sample);
    let hist_pred = hist_fitted.predict_one(&test_sample);
    println!(
        "\nTest sample predictions: exact={:.4}, histogram={:.4}",
        exact_pred, hist_pred
    );

    // Compare in gradient boosting
    println!("\n--- Gradient Boosting Comparison ---");

    // Gradient boosting with exact trees
    let exact_weak_learner = DecisionTree::new().max_depth(4);
    let trainer = GradientBoostingTrainer::new(LeastSquaresLoss)
        .n_estimators(100)
        .learning_rate(0.1);

    let start = Instant::now();
    let exact_model = trainer.fit_with_weak_learner(&features, &targets, &exact_weak_learner);
    let exact_train_time = start.elapsed();

    // Gradient boosting with histogram trees
    let hist_weak_learner = HistogramTree::new().max_depth(4).num_bins(64);

    let start = Instant::now();
    let hist_model = trainer.fit_with_weak_learner(&features, &targets, &hist_weak_learner);
    let hist_train_time = start.elapsed();

    // Evaluate on training data
    let exact_preds = exact_model.predict_batch(&features);
    let hist_preds = hist_model.predict_batch(&features);
    let target_vec = targets.to_vec();

    let exact_mse: f64 = exact_preds
        .to_vec()
        .iter()
        .zip(target_vec.iter())
        .map(|(p, t)| (p - t).powi(2) as f64)
        .sum::<f64>()
        / n_samples as f64;

    let hist_mse: f64 = hist_preds
        .to_vec()
        .iter()
        .zip(target_vec.iter())
        .map(|(p, t)| (p - t).powi(2) as f64)
        .sum::<f64>()
        / n_samples as f64;

    println!(
        "Exact trees (100): MSE={:.6}, time={:?}",
        exact_mse, exact_train_time
    );
    println!(
        "Histogram trees (100): MSE={:.6}, time={:?}",
        hist_mse, hist_train_time
    );

    // Test on new sample
    let test_input =
        Tensor1D::<CpuBackend>::new((0..n_features).map(|j| (j as f32 * 0.05).sin()).collect());
    let exact_pred = exact_model.predict(&test_input);
    let hist_pred = hist_model.predict(&test_input);
    println!(
        "\nNew sample predictions: exact={:.4}, histogram={:.4}",
        exact_pred.to_f64(),
        hist_pred.to_f64()
    );

    // Demonstrate configuration options
    println!("\n--- Configuration Options ---");
    let custom_tree = HistogramTree::new()
        .max_depth(8)
        .min_samples_leaf(5)
        .num_bins(128)
        .min_samples_split(10);

    let custom_fitted = custom_tree.fit(&features, &targets);
    println!(
        "Custom config: depth={}, leaves={}, bins=128",
        custom_fitted.depth(),
        custom_fitted.leaf_count()
    );

    println!("\n=== Summary ===");
    println!("Histogram trees provide:");
    println!("- Faster training on large datasets (O(bins × features) vs O(data × features))");
    println!("- GPU-friendly algorithm (bin aggregation parallelizes well)");
    println!("- Slightly different accuracy due to discretization");
    println!("- Better scalability for millions of samples");
}
