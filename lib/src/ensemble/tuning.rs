//! Hyperparameter tuning utilities for gradient boosting.
//!
//! Provides grid search for finding optimal hyperparameters
//! and utilities for analyzing tuning results.

use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::ensemble::GradientBoostingRegressor;
use serde::{Deserialize, Serialize};

/// A single hyperparameter configuration to test.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperParamConfig {
    /// Number of estimators (weak learners)
    pub n_estimators: usize,
    /// Learning rate (shrinkage)
    pub learning_rate: f64,
    /// Maximum depth of trees (None for stumps)
    pub max_depth: Option<usize>,
    /// Feature subsampling rate
    pub colsample_bytree: f64,
}

impl Default for HyperParamConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: None,
            colsample_bytree: 1.0,
        }
    }
}

/// Result of evaluating a single hyperparameter configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalResult {
    /// The hyperparameter configuration
    pub config: HyperParamConfig,
    /// Validation MSE
    pub mse: f64,
    /// Validation MAE
    pub mae: f64,
    /// Training time in milliseconds
    pub train_time_ms: u64,
}

impl EvalResult {
    /// Create a new evaluation result.
    pub fn new(config: HyperParamConfig, mse: f64, mae: f64, train_time_ms: u64) -> Self {
        Self {
            config,
            mse,
            mae,
            train_time_ms,
        }
    }
}

/// Configuration for grid search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GridSearchConfig {
    /// Fraction of data to use for validation
    pub validation_fraction: f64,
    /// Whether to print progress
    pub verbose: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for GridSearchConfig {
    fn default() -> Self {
        Self {
            validation_fraction: 0.2,
            verbose: false,
            random_seed: None,
        }
    }
}

/// Results from a grid search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GridSearchResults {
    /// All evaluation results
    pub results: Vec<EvalResult>,
    /// Configuration used
    pub config: GridSearchConfig,
}

impl GridSearchResults {
    /// Get the best result (lowest MSE).
    pub fn best(&self) -> Option<&EvalResult> {
        self.results
            .iter()
            .min_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap())
    }

    /// Get the top N results by MSE.
    pub fn top_n(&self, n: usize) -> Vec<&EvalResult> {
        let mut sorted: Vec<&EvalResult> = self.results.iter().collect();
        sorted.sort_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap());
        sorted.into_iter().take(n).collect()
    }

    /// Get results sorted by MSE.
    pub fn sorted_by_mse(&self) -> Vec<&EvalResult> {
        let mut sorted: Vec<&EvalResult> = self.results.iter().collect();
        sorted.sort_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap());
        sorted
    }
}

/// Grid search for hyperparameter tuning.
pub struct GridSearchGB {
    /// Values to try for n_estimators
    pub n_estimators: Vec<usize>,
    /// Values to try for learning_rate
    pub learning_rates: Vec<f64>,
    /// Values to try for max_depth
    pub max_depths: Vec<Option<usize>>,
    /// Values to try for colsample_bytree
    pub colsample_bytrees: Vec<f64>,
    /// Configuration
    config: GridSearchConfig,
}

impl Default for GridSearchGB {
    fn default() -> Self {
        Self {
            n_estimators: vec![50, 100, 200],
            learning_rates: vec![0.05, 0.1, 0.2],
            max_depths: vec![None, Some(3), Some(5)],
            colsample_bytrees: vec![0.8, 1.0],
            config: GridSearchConfig::default(),
        }
    }
}

impl GridSearchGB {
    /// Create a new grid search with default parameter ranges.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set n_estimators values to try.
    pub fn n_estimators(mut self, values: Vec<usize>) -> Self {
        self.n_estimators = values;
        self
    }

    /// Set learning rate values to try.
    pub fn learning_rates(mut self, values: Vec<f64>) -> Self {
        self.learning_rates = values;
        self
    }

    /// Set max_depth values to try.
    pub fn max_depths(mut self, values: Vec<Option<usize>>) -> Self {
        self.max_depths = values;
        self
    }

    /// Set colsample_bytree values to try.
    pub fn colsample_bytrees(mut self, values: Vec<f64>) -> Self {
        self.colsample_bytrees = values;
        self
    }

    /// Set the configuration.
    pub fn config(mut self, config: GridSearchConfig) -> Self {
        self.config = config;
        self
    }

    /// Set verbose output.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set the random seed.
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Get the total number of parameter combinations.
    pub fn total_combinations(&self) -> usize {
        self.n_estimators.len()
            * self.learning_rates.len()
            * self.max_depths.len()
            * self.colsample_bytrees.len()
    }

    /// Generate all parameter combinations.
    pub fn combinations(&self) -> Vec<HyperParamConfig> {
        let mut configs = Vec::new();
        for &n_est in &self.n_estimators {
            for &lr in &self.learning_rates {
                for &depth in &self.max_depths {
                    for &colsample in &self.colsample_bytrees {
                        configs.push(HyperParamConfig {
                            n_estimators: n_est,
                            learning_rate: lr,
                            max_depth: depth,
                            colsample_bytree: colsample,
                        });
                    }
                }
            }
        }
        configs
    }

    /// Run grid search on the given data.
    ///
    /// # Arguments
    /// * `features` - Training features
    /// * `targets` - Training targets
    ///
    /// # Returns
    /// Results for all parameter combinations.
    pub fn search<B: Backend>(
        &self,
        features: &Tensor2D<B>,
        targets: &Tensor1D<B>,
    ) -> GridSearchResults {
        let (n_samples, n_features) = features.shape();
        let feature_data: Vec<f64> = features.ravel().to_vec();
        let target_data: Vec<f64> = targets.to_vec();

        // Split into train/validation
        let n_val = ((n_samples as f64) * self.config.validation_fraction).floor() as usize;
        let n_val = n_val.max(1).min(n_samples / 2);
        let n_train = n_samples - n_val;

        let train_indices: Vec<usize> = (0..n_train).collect();
        let val_indices: Vec<usize> = (n_train..n_samples).collect();

        // Extract train and validation data
        let train_targets: Vec<f64> = train_indices.iter().map(|&i| target_data[i]).collect();
        let val_targets: Vec<f64> = val_indices.iter().map(|&i| target_data[i]).collect();

        // Create training tensor - build from feature_data by index
        let mut train_features_data = Vec::with_capacity(n_train * n_features);
        for &i in &train_indices {
            for j in 0..n_features {
                train_features_data.push(feature_data[i * n_features + j] as f32);
            }
        }
        let train_features = Tensor2D::<B>::new(train_features_data, n_train, n_features);
        let train_targets_tensor =
            Tensor1D::<B>::new(train_targets.iter().map(|&v| v as f32).collect());

        // Create validation tensor
        let mut val_features_data = Vec::with_capacity(n_val * n_features);
        for &i in &val_indices {
            for j in 0..n_features {
                val_features_data.push(feature_data[i * n_features + j] as f32);
            }
        }
        let val_features = Tensor2D::<B>::new(val_features_data, n_val, n_features);

        let combinations = self.combinations();
        let total = combinations.len();

        if self.config.verbose {
            println!("Grid Search: {} parameter combinations", total);
            println!("Train samples: {}, Validation samples: {}", n_train, n_val);
        }

        let mut results = Vec::with_capacity(total);

        for (i, config) in combinations.iter().enumerate() {
            if self.config.verbose {
                println!(
                    "  [{}/{}] n_estimators={}, lr={:.3}, max_depth={:?}, colsample={:.2}",
                    i + 1,
                    total,
                    config.n_estimators,
                    config.learning_rate,
                    config.max_depth,
                    config.colsample_bytree
                );
            }

            let start = std::time::Instant::now();

            // Build trainer with this config
            let mut trainer = GradientBoostingRegressor::default()
                .n_estimators(config.n_estimators)
                .learning_rate(config.learning_rate)
                .colsample_bytree(config.colsample_bytree);

            if let Some(seed) = self.config.random_seed {
                trainer = trainer.random_seed(seed);
            }

            // Train model
            let model = trainer.fit(&train_features, &train_targets_tensor);

            // Evaluate on validation set
            let predictions = model.predict_batch(&val_features).to_vec();

            let mse: f64 = val_targets
                .iter()
                .zip(predictions.iter())
                .map(|(&t, &p)| (t - p).powi(2))
                .sum::<f64>()
                / n_val as f64;

            let mae: f64 = val_targets
                .iter()
                .zip(predictions.iter())
                .map(|(&t, &p)| (t - p).abs())
                .sum::<f64>()
                / n_val as f64;

            let train_time_ms = start.elapsed().as_millis() as u64;

            results.push(EvalResult::new(config.clone(), mse, mae, train_time_ms));

            if self.config.verbose {
                println!(
                    "      MSE: {:.6}, MAE: {:.6}, Time: {}ms",
                    mse, mae, train_time_ms
                );
            }
        }

        if self.config.verbose {
            if let Some(best) = results
                .iter()
                .min_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap())
            {
                println!(
                    "\nBest config: n_estimators={}, lr={:.3}, max_depth={:?}, colsample={:.2}",
                    best.config.n_estimators,
                    best.config.learning_rate,
                    best.config.max_depth,
                    best.config.colsample_bytree
                );
                println!("Best MSE: {:.6}", best.mse);
            }
        }

        GridSearchResults {
            results,
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_grid_search_config() {
        let config = HyperParamConfig::default();
        assert_eq!(config.n_estimators, 100);
        assert!((config.learning_rate - 0.1).abs() < 1e-10);
        assert!(config.max_depth.is_none());
        assert!((config.colsample_bytree - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_search_config_custom() {
        let config = GridSearchConfig {
            validation_fraction: 0.3,
            verbose: true,
            random_seed: Some(42),
        };
        assert!((config.validation_fraction - 0.3).abs() < 1e-10);
        assert!(config.verbose);
        assert_eq!(config.random_seed, Some(42));
    }

    #[test]
    fn test_eval_result_new() {
        let config = HyperParamConfig {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            colsample_bytree: 0.8,
        };
        let result = EvalResult::new(config.clone(), 0.5, 0.6, 1000);
        assert_eq!(result.config.n_estimators, 100);
        assert!((result.mse - 0.5).abs() < 1e-10);
        assert!((result.mae - 0.6).abs() < 1e-10);
        assert_eq!(result.train_time_ms, 1000);
    }

    #[test]
    fn test_grid_search_combinations() {
        let search = GridSearchGB::new()
            .n_estimators(vec![50, 100])
            .learning_rates(vec![0.1])
            .max_depths(vec![None, Some(3)])
            .colsample_bytrees(vec![1.0]);

        assert_eq!(search.total_combinations(), 4);
        let combos = search.combinations();
        assert_eq!(combos.len(), 4);
    }

    #[test]
    fn test_grid_search_results() {
        let results = GridSearchResults {
            results: vec![
                EvalResult::new(
                    HyperParamConfig {
                        n_estimators: 100,
                        learning_rate: 0.1,
                        max_depth: None,
                        colsample_bytree: 1.0,
                    },
                    1.0,
                    0.8,
                    100,
                ),
                EvalResult::new(
                    HyperParamConfig {
                        n_estimators: 50,
                        learning_rate: 0.2,
                        max_depth: Some(3),
                        colsample_bytree: 0.8,
                    },
                    0.5,
                    0.5,
                    50,
                ),
            ],
            config: GridSearchConfig::default(),
        };

        let best = results.best().unwrap();
        assert_eq!(best.config.n_estimators, 50);
        assert!((best.mse - 0.5).abs() < 1e-10);

        let top = results.top_n(2);
        assert_eq!(top.len(), 2);

        let sorted = results.sorted_by_mse();
        assert_eq!(sorted.len(), 2);
        assert_eq!(sorted[0].config.n_estimators, 50); // Best first
    }

    #[test]
    fn test_grid_search_results_empty() {
        let results = GridSearchResults {
            results: vec![],
            config: GridSearchConfig::default(),
        };

        assert!(results.best().is_none());
        assert!(results.top_n(5).is_empty());
        assert!(results.sorted_by_mse().is_empty());
    }

    #[test]
    fn test_grid_search_results_top_n_more_than_available() {
        let results = GridSearchResults {
            results: vec![EvalResult::new(HyperParamConfig::default(), 0.5, 0.5, 100)],
            config: GridSearchConfig::default(),
        };

        let top = results.top_n(5);
        assert_eq!(top.len(), 1); // Only 1 result available
    }

    #[test]
    fn test_grid_search_small_data() {
        let features =
            Tensor2D::<CpuBackend>::new(vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 8, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let search = GridSearchGB::new()
            .n_estimators(vec![10, 20])
            .learning_rates(vec![0.1, 0.2])
            .max_depths(vec![None])
            .colsample_bytrees(vec![1.0])
            .verbose(false);

        let results = search.search(&features, &targets);
        assert_eq!(results.results.len(), 4);

        // All results should have finite MSE
        for result in &results.results {
            assert!(result.mse.is_finite());
        }
    }

    #[test]
    fn test_grid_search_with_random_seed() {
        let features =
            Tensor2D::<CpuBackend>::new(vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 8, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let search = GridSearchGB::new()
            .n_estimators(vec![10])
            .learning_rates(vec![0.1])
            .max_depths(vec![None])
            .colsample_bytrees(vec![0.8])
            .random_seed(42)
            .verbose(false);

        let results = search.search(&features, &targets);
        assert_eq!(results.results.len(), 1);
        assert_eq!(results.config.random_seed, Some(42));
    }

    #[test]
    fn test_grid_search_with_config() {
        let config = GridSearchConfig {
            validation_fraction: 0.25,
            verbose: false,
            random_seed: None,
        };

        let search = GridSearchGB::new()
            .n_estimators(vec![10])
            .learning_rates(vec![0.1])
            .max_depths(vec![None])
            .colsample_bytrees(vec![1.0])
            .config(config);

        let features =
            Tensor2D::<CpuBackend>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 8, 1);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let results = search.search(&features, &targets);
        assert!((results.config.validation_fraction - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_grid_search_default_parameters() {
        let search = GridSearchGB::default();
        assert_eq!(search.n_estimators, vec![50, 100, 200]);
        assert_eq!(search.learning_rates, vec![0.05, 0.1, 0.2]);
        assert_eq!(search.max_depths, vec![None, Some(3), Some(5)]);
        assert_eq!(search.colsample_bytrees, vec![0.8, 1.0]);
    }
}
