//! Example demonstrating histogram-based decision trees for gradient boosting.
//!
//! This example compares:
//! 1. Exact decision tree (traditional approach)
//! 2. Histogram-based decision tree (faster for large datasets)
//!
//! Histogram trees use quantile-based binning to discretize features,
//! then find splits by scanning bins instead of data points.
//! Complexity: O(#bins × #features) vs O(#data × #features) for exact trees.
//!
//! Optimizations implemented:
//! - Parallel histogram building using Rayon
//! - Parallel split finding across features
//! - Compact u8 bin indices (8x memory reduction)
//! - Packed BinData structs for better cache locality

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

fn compute_mse(predictions: &[f64], targets: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / predictions.len() as f64
}

fn benchmark_single_tree(
    n_samples: usize,
    n_features: usize,
    features: &Tensor2D<CpuBackend>,
    targets: &Tensor1D<CpuBackend>,
) {
    println!(
        "\n--- Single Tree Comparison ({} samples, {} features) ---",
        n_samples, n_features
    );

    // Exact decision tree
    let exact_tree = DecisionTree::new().max_depth(6);
    let start = Instant::now();
    let exact_fitted = exact_tree.fit(features, targets);
    let exact_time = start.elapsed();
    println!(
        "  Exact tree:      depth={:2}, nodes={:4}, time={:?}",
        exact_fitted.depth(),
        exact_fitted.node_count(),
        exact_time
    );

    // Histogram-based tree
    let hist_tree = HistogramTree::new().max_depth(6).num_bins(64);
    let start = Instant::now();
    let hist_fitted = hist_tree.fit(features, targets);
    let hist_time = start.elapsed();
    println!(
        "  Histogram tree:  depth={:2}, nodes={:4}, time={:?}",
        hist_fitted.depth(),
        hist_fitted.node_count(),
        hist_time
    );
}

fn benchmark_gradient_boosting(
    n_samples: usize,
    n_features: usize,
    features: &Tensor2D<CpuBackend>,
    targets: &Tensor1D<CpuBackend>,
    n_estimators: usize,
) {
    println!(
        "\n--- Gradient Boosting Comparison ({} samples, {} features, {} trees) ---",
        n_samples, n_features, n_estimators
    );

    let target_vec = targets.to_vec();

    // Gradient boosting with exact trees
    let exact_weak_learner = DecisionTree::new().max_depth(4);
    let trainer = GradientBoostingTrainer::new(LeastSquaresLoss)
        .n_estimators(n_estimators)
        .learning_rate(0.1);

    let start = Instant::now();
    let exact_model = trainer.fit_with_weak_learner(features, targets, &exact_weak_learner);
    let exact_train_time = start.elapsed();

    // Gradient boosting with histogram trees
    let hist_weak_learner = HistogramTree::new().max_depth(4).num_bins(64);

    let start = Instant::now();
    let hist_model = trainer.fit_with_weak_learner(features, targets, &hist_weak_learner);
    let hist_train_time = start.elapsed();

    // Evaluate on training data
    let exact_preds = exact_model.predict_batch(features);
    let hist_preds = hist_model.predict_batch(features);

    let exact_mse = compute_mse(&exact_preds.to_vec(), &target_vec);
    let hist_mse = compute_mse(&hist_preds.to_vec(), &target_vec);

    println!(
        "  Exact trees:     MSE={:.6}, time={:?}",
        exact_mse, exact_train_time
    );
    println!(
        "  Histogram trees: MSE={:.6}, time={:?}",
        hist_mse, hist_train_time
    );

    // Compute speedup
    let speedup = exact_train_time.as_secs_f64() / hist_train_time.as_secs_f64();
    println!("  Speedup:         {:.2}x", speedup);
}

fn main() {
    println!("=== Histogram-Based Decision Tree Example ===");
    println!("\nOptimizations enabled:");
    println!("  - Parallel histogram building (Rayon)");
    println!("  - Parallel split finding across features");
    println!("  - Compact u8 bin indices (8x memory reduction)");
    println!("  - Packed BinData structs for cache locality");

    // Benchmark with different dataset sizes
    let dataset_sizes = [
        (1_000, 10),  // Small dataset
        (10_000, 20), // Medium dataset
        (50_000, 30), // Large dataset
    ];

    for (n_samples, n_features) in dataset_sizes {
        println!("\n========================================");
        println!(
            "Generating {} samples with {} features...",
            n_samples, n_features
        );
        let (features, targets) = generate_data(n_samples, n_features);

        let features = Tensor2D::<CpuBackend>::new(features, n_samples, n_features);
        let targets = Tensor1D::<CpuBackend>::new(targets);

        // Single tree comparison
        benchmark_single_tree(n_samples, n_features, &features, &targets);

        // Gradient boosting comparison
        let n_estimators = if n_samples <= 10_000 { 100 } else { 50 };
        benchmark_gradient_boosting(n_samples, n_features, &features, &targets, n_estimators);
    }

    // Demonstrate configuration options
    println!("\n--- Configuration Options ---");
    let (features, targets) = generate_data(1000, 10);
    let features = Tensor2D::<CpuBackend>::new(features, 1000, 10);
    let targets = Tensor1D::<CpuBackend>::new(targets);

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
    println!("- Parallel implementation for multi-core CPUs");
    println!("- Better cache locality through packed data structures");
    println!("- Slightly different accuracy due to discretization");
    println!("- Better scalability for millions of samples");
}
