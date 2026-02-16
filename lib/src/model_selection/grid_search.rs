//! Grid search with cross-validation for hyperparameter tuning.
//!
//! This module provides [`GridSearchCV`] for systematically searching over
//! hyperparameter combinations with cross-validation to find the best model.
//!
//! # Example
//!
//! ```rust,ignore
//! use machinelearne_rs::model_selection::{GridSearchCV, KFold, LinearRegressionGrid};
//! use machinelearne_rs::metrics::RegressionMetric;
//! use machinelearne_rs::backend::CpuBackend;
//!
//! let param_grid = LinearRegressionGrid::new()
//!     .with_learning_rates(vec![0.001, 0.01, 0.1])
//!     .with_lambdas(vec![0.0, 0.01, 0.1]);
//!
//! let grid_search = GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2)
//!     .with_cv(KFold::new(5).with_random_state(42))
//!     .verbose(1);
//!
//! let result = grid_search.fit(&dataset, n_features)?;
//! println!("Best R² score: {:.4}", result.best_params.mean_score);
//! ```

use std::fmt::{Debug, Display};

use crate::backend::{Backend, Tensor1D};
use crate::dataset::memory::InMemoryDataset;
use crate::dataset::Dataset;
use crate::loss::MSELoss;
use crate::metrics::Scorer;
use crate::model::linear::{LinearModel, LinearRegression};
use crate::model::{Fitted, InferenceModel};
use crate::optimizer::SGD;
use crate::regularizers::{NoRegularizer, L2};
use crate::trainer::Trainer;

use super::{CVSplit, KFold, LinearRegressionGrid, ParamCombination};

/// Result for a single parameter combination.
#[derive(Clone, Debug)]
pub struct GridSearchResult {
    /// Learning rate used.
    pub learning_rate: f64,
    /// Regularization lambda used (0.0 if no regularization).
    pub lambda: f64,
    /// Batch size used.
    pub batch_size: usize,
    /// Max epochs used.
    pub max_epochs: usize,
    /// Polynomial degree used (1 if no polynomial features).
    pub poly_degree: usize,
    /// Mean cross-validation score (higher is better).
    pub mean_score: f64,
    /// Standard deviation of cross-validation scores.
    pub std_score: f64,
    /// Individual fold scores.
    pub fold_scores: Vec<f64>,
}

/// Result from GridSearchCV containing the best model and all results.
pub struct GridSearchCVResult<B: Backend> {
    /// Best fitted model trained on full dataset.
    pub best_model: LinearModel<B, Fitted>,
    /// Parameters of the best model.
    pub best_params: GridSearchResult,
    /// All parameter combinations and their scores, sorted by mean_score descending.
    pub all_results: Vec<GridSearchResult>,
    /// Name of the scoring metric used.
    pub scoring: String,
}

/// Grid search with cross-validation.
///
/// Exhaustively searches over specified parameter values for a linear regression
/// model, using cross-validation to evaluate each combination.
pub struct GridSearchCV<B: Backend, S: Scorer<B>, C: CVSplit = KFold>
where
    S: Scorer<B, Prediction = Tensor1D<B>, Target = Tensor1D<B>>,
{
    param_grid: LinearRegressionGrid,
    cv: C,
    scorer: S,
    verbose: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend, S: Scorer<B>> GridSearchCV<B, S, KFold>
where
    S: Scorer<B, Prediction = Tensor1D<B>, Target = Tensor1D<B>>,
    B::Scalar: Debug + Display,
{
    /// Create a new grid search with the given parameter grid and scorer.
    ///
    /// # Arguments
    /// * `param_grid` - The parameter grid defining search space
    /// * `scorer` - The scoring metric (higher is better)
    pub fn new(param_grid: LinearRegressionGrid, scorer: S) -> Self {
        Self {
            param_grid,
            cv: KFold::new(5),
            scorer,
            verbose: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend, S: Scorer<B>, C: CVSplit> GridSearchCV<B, S, C>
where
    S: Scorer<B, Prediction = Tensor1D<B>, Target = Tensor1D<B>>,
    B::Scalar: Debug + Display,
{
    /// Set the cross-validation splitter.
    pub fn with_cv<NewC: CVSplit>(self, cv: NewC) -> GridSearchCV<B, S, NewC> {
        GridSearchCV {
            param_grid: self.param_grid,
            cv,
            scorer: self.scorer,
            verbose: self.verbose,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set verbosity level (0 = silent, 1 = progress, 2 = detailed).
    pub fn verbose(mut self, level: usize) -> Self {
        self.verbose = level;
        self
    }

    /// Fit the grid search on a dataset.
    ///
    /// # Arguments
    /// * `dataset` - The full dataset to use for cross-validation
    /// * `n_features` - Number of features in the dataset
    ///
    /// # Returns
    /// The best model and all results on success, or an error message.
    pub fn fit(
        &self,
        dataset: &InMemoryDataset,
        n_features: usize,
    ) -> Result<GridSearchCVResult<B>, String> {
        let n_samples = dataset.len().ok_or("Dataset length unknown")?;
        let total_combinations = self.param_grid.n_combinations();

        if self.verbose >= 1 {
            println!(
                "Fitting {} folds for {} candidates, totalling {} fits",
                self.cv.n_splits(),
                total_combinations,
                total_combinations * self.cv.n_splits()
            );
        }

        let mut all_results = Vec::with_capacity(total_combinations);
        let splits = self.cv.split(n_samples);

        // Iterate over all parameter combinations
        for (comb_idx, params) in self.param_grid.iter().enumerate() {
            if self.verbose >= 1 {
                println!(
                    "[{}/{}] Evaluating lr={}, lambda={}, batch={}, epochs={}",
                    comb_idx + 1,
                    total_combinations,
                    params.learning_rate,
                    params.lambda,
                    params.batch_size,
                    params.max_epochs
                );
            }

            let fold_scores = self.evaluate_combination(dataset, &splits, &params, n_features)?;

            // Compute mean and std
            let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            let variance = fold_scores
                .iter()
                .map(|&s| (s - mean_score).powi(2))
                .sum::<f64>()
                / fold_scores.len() as f64;
            let std_score = variance.sqrt();

            all_results.push(GridSearchResult {
                learning_rate: params.learning_rate,
                lambda: params.lambda,
                batch_size: params.batch_size,
                max_epochs: params.max_epochs,
                poly_degree: params.poly_degree,
                mean_score,
                std_score,
                fold_scores,
            });
        }

        // Sort by mean_score descending (higher is better)
        all_results.sort_by(|a, b| {
            b.mean_score
                .partial_cmp(&a.mean_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best_params = all_results[0].clone();

        if self.verbose >= 1 {
            println!(
                "\nBest parameters: lr={}, lambda={}, batch={}, epochs={}",
                best_params.learning_rate,
                best_params.lambda,
                best_params.batch_size,
                best_params.max_epochs
            );
            println!(
                "Best {} score: {:.4} (+/- {:.4})",
                self.scorer.name(),
                best_params.mean_score,
                best_params.std_score
            );
        }

        // Retrain best model on full dataset
        let best_model = self.train_model(dataset, &best_params, n_features)?;

        Ok(GridSearchCVResult {
            best_model,
            best_params,
            all_results,
            scoring: self.scorer.name().to_string(),
        })
    }

    /// Evaluate a single parameter combination using cross-validation.
    fn evaluate_combination(
        &self,
        dataset: &InMemoryDataset,
        splits: &[(Vec<usize>, Vec<usize>)],
        params: &ParamCombination,
        n_features: usize,
    ) -> Result<Vec<f64>, String> {
        let mut fold_scores = Vec::with_capacity(splits.len());

        for (fold_idx, (train_indices, test_indices)) in splits.iter().enumerate() {
            let train_dataset = dataset.subset(train_indices)?;
            let test_dataset = dataset.subset(test_indices)?;

            // Train model
            let model = self.train_single_model(&train_dataset, params, n_features)?;

            // Evaluate on test fold
            let score = self.score_model(&model, &test_dataset)?;

            if self.verbose >= 2 {
                println!("  Fold {}: score = {:.4}", fold_idx, score);
            }

            fold_scores.push(score);
        }

        Ok(fold_scores)
    }

    /// Train a model with the given parameters.
    fn train_single_model(
        &self,
        dataset: &InMemoryDataset,
        params: &ParamCombination,
        n_features: usize,
    ) -> Result<LinearModel<B, Fitted>, String> {
        let model = LinearRegression::<B>::new(n_features);
        let loss = MSELoss;
        let optimizer = SGD::new(params.learning_rate);

        if params.lambda > 0.0 {
            let trainer = Trainer::builder(loss, optimizer, L2::new(params.lambda))
                .batch_size(params.batch_size)
                .max_epochs(params.max_epochs)
                .verbose(false)
                .build();
            trainer.fit(model, dataset)
        } else {
            let trainer = Trainer::builder(loss, optimizer, NoRegularizer)
                .batch_size(params.batch_size)
                .max_epochs(params.max_epochs)
                .verbose(false)
                .build();
            trainer.fit(model, dataset)
        }
    }

    /// Train the final model on the full dataset.
    fn train_model(
        &self,
        dataset: &InMemoryDataset,
        params: &GridSearchResult,
        n_features: usize,
    ) -> Result<LinearModel<B, Fitted>, String> {
        let model = LinearRegression::<B>::new(n_features);
        let loss = MSELoss;
        let optimizer = SGD::new(params.learning_rate);

        if params.lambda > 0.0 {
            let trainer = Trainer::builder(loss, optimizer, L2::new(params.lambda))
                .batch_size(params.batch_size)
                .max_epochs(params.max_epochs)
                .verbose(false)
                .build();
            trainer.fit(model, dataset)
        } else {
            let trainer = Trainer::builder(loss, optimizer, NoRegularizer)
                .batch_size(params.batch_size)
                .max_epochs(params.max_epochs)
                .verbose(false)
                .build();
            trainer.fit(model, dataset)
        }
    }

    /// Score a fitted model on a dataset.
    fn score_model(
        &self,
        model: &LinearModel<B, Fitted>,
        dataset: &InMemoryDataset,
    ) -> Result<f64, String> {
        let n_samples = dataset.len().ok_or("Dataset length unknown")?;

        // Get all data at once
        let (x_tensor, y_tensor) = dataset
            .get_batch::<B>(0..n_samples)
            .map_err(|e| format!("Batch error: {:?}", e))?;

        // Predict
        let predictions = model.predict_batch(&x_tensor);

        // Score
        Ok(self.scorer.score(&predictions, &y_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::metrics::RegressionMetric;
    use crate::model_selection::TrainerGrid;

    fn create_linear_dataset() -> InMemoryDataset {
        // y = 2*x1 + 3*x2 + 1
        let x = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 2.0],
            vec![3.0, 0.0],
        ];
        let y = vec![1.0, 3.0, 4.0, 6.0, 5.0, 7.0, 8.0, 10.0, 11.0, 7.0];
        InMemoryDataset::new(x, y).unwrap()
    }

    #[test]
    fn test_grid_search_basic() {
        let dataset = create_linear_dataset();

        let param_grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0]);

        let grid_search =
            GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2).verbose(0);

        let result = grid_search.fit(&dataset, 2).unwrap();

        assert!(!result.all_results.is_empty());
        assert!(result.best_params.mean_score > 0.0);
    }

    #[test]
    fn test_grid_search_multiple_params() {
        let dataset = create_linear_dataset();

        let param_grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.01, 0.1])
            .with_lambdas(vec![0.0, 0.1]);

        let grid_search = GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::NegMSE)
            .with_cv(KFold::new(3))
            .verbose(0);

        let result = grid_search.fit(&dataset, 2).unwrap();

        // Should have 4 combinations
        assert_eq!(result.all_results.len(), 4);

        // Results should be sorted by score (descending)
        for i in 1..result.all_results.len() {
            assert!(
                result.all_results[i - 1].mean_score >= result.all_results[i].mean_score,
                "Results not sorted correctly"
            );
        }
    }

    #[test]
    fn test_grid_search_best_model_can_predict() {
        let dataset = create_linear_dataset();

        let param_grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0]);

        let grid_search =
            GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2).verbose(0);

        let result = grid_search.fit(&dataset, 2).unwrap();

        // Best model should be able to predict
        use crate::backend::Tensor2D;
        let test_input = Tensor2D::<CpuBackend>::new(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let predictions = result.best_model.predict_batch(&test_input);

        // Predictions should be close to expected values
        let pred_vec = predictions.to_vec();
        assert!(
            (pred_vec[0] - 3.0).abs() < 0.5,
            "Expected ~3.0, got {}",
            pred_vec[0]
        );
        assert!(
            (pred_vec[1] - 4.0).abs() < 0.5,
            "Expected ~4.0, got {}",
            pred_vec[1]
        );
    }

    #[test]
    fn test_grid_search_fold_scores() {
        let dataset = create_linear_dataset();

        let param_grid = LinearRegressionGrid::new().with_learning_rates(vec![0.1]);

        let grid_search = GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2)
            .with_cv(KFold::new(5))
            .verbose(0);

        let result = grid_search.fit(&dataset, 2).unwrap();

        // Should have 5 fold scores
        assert_eq!(result.best_params.fold_scores.len(), 5);
    }

    #[test]
    fn test_grid_search_scoring_name() {
        let dataset = create_linear_dataset();

        let param_grid = LinearRegressionGrid::new();

        let grid_search = GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2);

        let result = grid_search.fit(&dataset, 2).unwrap();

        assert_eq!(result.scoring, "r2");
    }

    #[test]
    fn test_grid_search_returns_correct_best() {
        let dataset = create_linear_dataset();

        // Use parameters where higher learning rate should win
        let param_grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.001, 0.1]) // 0.1 should be better
            .with_trainer(TrainerGrid::new(vec![32], vec![500]));

        let grid_search = GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2)
            .with_cv(KFold::new(3))
            .verbose(0);

        let result = grid_search.fit(&dataset, 2).unwrap();

        // Best should have learning_rate = 0.1
        assert!(
            (result.best_params.learning_rate - 0.1).abs() < 1e-6,
            "Expected best lr=0.1, got {}",
            result.best_params.learning_rate
        );
    }

    #[test]
    fn test_grid_search_with_regularization() {
        let dataset = create_linear_dataset();

        // Test with L2 regularization
        let param_grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.5]);

        let grid_search = GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2)
            .with_cv(KFold::new(3))
            .verbose(0);

        let result = grid_search.fit(&dataset, 2).unwrap();

        // Should have found the best params with lambda = 0.5
        assert_eq!(result.all_results.len(), 1);
        assert!((result.best_params.lambda - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_grid_search_verbose_output() {
        let dataset = create_linear_dataset();

        let param_grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0]);

        let grid_search = GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2)
            .with_cv(KFold::new(3))
            .verbose(2); // Test verbose level 2

        let result = grid_search.fit(&dataset, 2).unwrap();
        assert!(result.best_params.mean_score > 0.0);
    }

    #[test]
    fn test_grid_search_neg_mse() {
        let dataset = create_linear_dataset();

        let param_grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0]);

        let grid_search = GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::NegMSE)
            .with_cv(KFold::new(3))
            .verbose(0);

        let result = grid_search.fit(&dataset, 2).unwrap();

        // NegMSE should be negative (or zero for perfect fit)
        assert!(result.best_params.mean_score <= 0.0);
        assert_eq!(result.scoring, "neg_mean_squared_error");
    }

    #[test]
    fn test_grid_search_neg_rmse() {
        let dataset = create_linear_dataset();

        let param_grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0]);

        let grid_search = GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::NegRMSE)
            .with_cv(KFold::new(3))
            .verbose(0);

        let result = grid_search.fit(&dataset, 2).unwrap();

        assert_eq!(result.scoring, "neg_root_mean_squared_error");
    }

    #[test]
    fn test_grid_search_neg_mae() {
        let dataset = create_linear_dataset();

        let param_grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0]);

        let grid_search = GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::NegMAE)
            .with_cv(KFold::new(3))
            .verbose(0);

        let result = grid_search.fit(&dataset, 2).unwrap();

        assert_eq!(result.scoring, "neg_mean_absolute_error");
    }
}
