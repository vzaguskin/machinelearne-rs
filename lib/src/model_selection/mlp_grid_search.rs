//! Grid search with cross-validation for MLP hyperparameter tuning.
//!
//! This module provides [`MLPGridSearchCV`] for systematically searching over
//! MLP hyperparameter combinations with cross-validation to find the best model.
//!
//! # Example
//!
//! ```rust,ignore
//! use machinelearne_rs::model_selection::{MLPGridSearchCV, KFold, MLPGrid, MLPArchitecture};
//! use machinelearne_rs::metrics::RegressionMetric;
//! use machinelearne_rs::backend::CpuBackend;
//! use machinelearne_rs::model::Activation;
//!
//! let param_grid = MLPGrid::new()
//!     .with_architectures(vec![
//!         MLPArchitecture::single(8),
//!         MLPArchitecture::double(16, 8),
//!     ])
//!     .with_hidden_activations(vec![Activation::ReLU, Activation::Tanh], 1)
//!     .with_learning_rates(vec![0.001, 0.01]);
//!
//! let grid_search = MLPGridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2)
//!     .with_cv(KFold::new(3).with_random_state(42));
//!
//! let result = grid_search.fit(&dataset, n_features, 1)?;
//! println!("Best R² score: {:.4}", result.best_params.mean_score);
//! ```

use std::fmt::{Debug, Display};

use crate::backend::{Backend, Tensor1D};
use crate::dataset::memory::InMemoryDataset;
use crate::dataset::Dataset;
use crate::loss::MSELoss;
use crate::metrics::Scorer;
use crate::model::mlp::MLPModel;
use crate::model::{Activation, Fitted, InferenceModel};
use crate::optimizer::SGD;
use crate::regularizers::{NoRegularizer, L2};
use crate::trainer::Trainer;

use super::{CVSplit, KFold, MLPGrid, MLPParamCombination};

/// Result for a single MLP parameter combination.
#[derive(Clone, Debug)]
pub struct MLPGridSearchResult {
    /// Hidden layer sizes.
    pub hidden_layers: Vec<usize>,
    /// Activation functions used.
    pub activations: Vec<Activation>,
    /// Learning rate used.
    pub learning_rate: f64,
    /// Regularization lambda used (0.0 if no regularization).
    pub lambda: f64,
    /// Batch size used.
    pub batch_size: usize,
    /// Max epochs used.
    pub max_epochs: usize,
    /// Mean cross-validation score (higher is better).
    pub mean_score: f64,
    /// Standard deviation of cross-validation scores.
    pub std_score: f64,
    /// Individual fold scores.
    pub fold_scores: Vec<f64>,
}

impl MLPGridSearchResult {
    /// Get a string representation of the architecture.
    pub fn architecture_string(&self) -> String {
        format!("{:?}", self.hidden_layers)
    }

    /// Get a string representation of the activations.
    pub fn activations_string(&self) -> String {
        let names: Vec<&str> = self
            .activations
            .iter()
            .map(|a| match a {
                Activation::ReLU => "ReLU",
                Activation::Sigmoid => "Sigmoid",
                Activation::Tanh => "Tanh",
                Activation::Identity => "Identity",
            })
            .collect();
        names.join(" -> ")
    }
}

/// Result from MLPGridSearchCV containing the best model and all results.
pub struct MLPGridSearchCVResult<B: Backend> {
    /// Best fitted model trained on full dataset.
    pub best_model: MLPModel<B, Fitted>,
    /// Parameters of the best model.
    pub best_params: MLPGridSearchResult,
    /// All parameter combinations and their scores, sorted by mean_score descending.
    pub all_results: Vec<MLPGridSearchResult>,
    /// Name of the scoring metric used.
    pub scoring: String,
    /// Number of input features.
    pub n_features: usize,
    /// Number of output features.
    pub n_outputs: usize,
}

/// Grid search with cross-validation for MLP models.
///
/// Exhaustively searches over specified parameter values for an MLP
/// model, using cross-validation to evaluate each combination.
pub struct MLPGridSearchCV<B: Backend, S: Scorer<B>, C: CVSplit = KFold>
where
    S: Scorer<B, Prediction = Tensor1D<B>, Target = Tensor1D<B>>,
{
    param_grid: MLPGrid,
    cv: C,
    scorer: S,
    n_outputs: usize,
    verbose: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend, S: Scorer<B>> MLPGridSearchCV<B, S, KFold>
where
    S: Scorer<B, Prediction = Tensor1D<B>, Target = Tensor1D<B>>,
    B::Scalar: Debug + Display,
{
    /// Create a new grid search with the given parameter grid and scorer.
    ///
    /// # Arguments
    /// * `param_grid` - The parameter grid defining search space
    /// * `scorer` - The scoring metric (higher is better)
    pub fn new(param_grid: MLPGrid, scorer: S) -> Self {
        Self {
            param_grid,
            cv: KFold::new(5),
            scorer,
            n_outputs: 1,
            verbose: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the number of output features (default: 1).
    pub fn with_n_outputs(mut self, n_outputs: usize) -> Self {
        self.n_outputs = n_outputs;
        self
    }
}

impl<B: Backend, S: Scorer<B>, C: CVSplit> MLPGridSearchCV<B, S, C>
where
    S: Scorer<B, Prediction = Tensor1D<B>, Target = Tensor1D<B>>,
    B::Scalar: Debug + Display,
{
    /// Set the cross-validation splitter.
    pub fn with_cv<NewC: CVSplit>(self, cv: NewC) -> MLPGridSearchCV<B, S, NewC> {
        MLPGridSearchCV {
            param_grid: self.param_grid,
            cv,
            scorer: self.scorer,
            n_outputs: self.n_outputs,
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
    /// * `n_features` - Number of input features in the dataset
    /// * `n_outputs` - Number of output features (default: 1)
    ///
    /// # Returns
    /// The best model and all results on success, or an error message.
    pub fn fit(
        &self,
        dataset: &InMemoryDataset,
        n_features: usize,
        n_outputs: usize,
    ) -> Result<MLPGridSearchCVResult<B>, String> {
        let n_samples = dataset.len().ok_or("Dataset length unknown")?;

        // Count valid combinations (compatible architecture/activations)
        let valid_combinations: Vec<_> = self
            .param_grid
            .iter()
            .filter(|params| {
                let num_layers = params.architecture.hidden_layers.len() + 1;
                params.activations.activations.len() == num_layers
            })
            .collect();

        let total_combinations = valid_combinations.len();

        if total_combinations == 0 {
            return Err("No valid parameter combinations found. Check architecture/activation compatibility.".to_string());
        }

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

        // Iterate over valid parameter combinations
        for (comb_idx, params) in valid_combinations.into_iter().enumerate() {
            if self.verbose >= 1 {
                println!(
                    "[{}/{}] Evaluating arch={:?}, act={}, lr={}, lambda={}, batch={}, epochs={}",
                    comb_idx + 1,
                    total_combinations,
                    params.architecture.hidden_layers,
                    format_activations(&params.activations.activations),
                    params.learning_rate,
                    params.lambda,
                    params.batch_size,
                    params.max_epochs
                );
            }

            let fold_scores =
                self.evaluate_combination(dataset, &splits, &params, n_features, n_outputs)?;

            // Compute mean and std
            let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            let variance = fold_scores
                .iter()
                .map(|&s| (s - mean_score).powi(2))
                .sum::<f64>()
                / fold_scores.len() as f64;
            let std_score = variance.sqrt();

            all_results.push(MLPGridSearchResult {
                hidden_layers: params.architecture.hidden_layers.clone(),
                activations: params.activations.activations.clone(),
                learning_rate: params.learning_rate,
                lambda: params.lambda,
                batch_size: params.batch_size,
                max_epochs: params.max_epochs,
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
                "\nBest parameters: arch={:?}, act={}, lr={}, lambda={}, batch={}, epochs={}",
                best_params.hidden_layers,
                format_activations(&best_params.activations),
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
        let best_model = self.train_model(dataset, &best_params, n_features, n_outputs)?;

        Ok(MLPGridSearchCVResult {
            best_model,
            best_params,
            all_results,
            scoring: self.scorer.name().to_string(),
            n_features,
            n_outputs,
        })
    }

    /// Evaluate a single parameter combination using cross-validation.
    fn evaluate_combination(
        &self,
        dataset: &InMemoryDataset,
        splits: &[(Vec<usize>, Vec<usize>)],
        params: &MLPParamCombination,
        n_features: usize,
        n_outputs: usize,
    ) -> Result<Vec<f64>, String> {
        let mut fold_scores = Vec::with_capacity(splits.len());

        for (fold_idx, (train_indices, test_indices)) in splits.iter().enumerate() {
            let train_dataset = dataset.subset(train_indices)?;
            let test_dataset = dataset.subset(test_indices)?;

            // Train model
            let model = self.train_single_model(&train_dataset, params, n_features, n_outputs)?;

            // Evaluate on test fold
            let score = self.score_model(&model, &test_dataset, n_outputs)?;

            if self.verbose >= 2 {
                println!("  Fold {}: score = {:.4}", fold_idx, score);
            }

            fold_scores.push(score);
        }

        Ok(fold_scores)
    }

    /// Build layer sizes from n_features, hidden_layers, and n_outputs.
    fn build_layer_sizes(
        &self,
        hidden_layers: &[usize],
        n_features: usize,
        n_outputs: usize,
    ) -> Vec<usize> {
        let mut sizes = vec![n_features];
        sizes.extend_from_slice(hidden_layers);
        sizes.push(n_outputs);
        sizes
    }

    /// Train a model with the given parameters.
    fn train_single_model(
        &self,
        dataset: &InMemoryDataset,
        params: &MLPParamCombination,
        n_features: usize,
        n_outputs: usize,
    ) -> Result<MLPModel<B, Fitted>, String> {
        let layer_sizes =
            self.build_layer_sizes(&params.architecture.hidden_layers, n_features, n_outputs);
        let model = crate::model::mlp::MLP::<B>::new(&layer_sizes, &params.activations.activations);
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
        params: &MLPGridSearchResult,
        n_features: usize,
        n_outputs: usize,
    ) -> Result<MLPModel<B, Fitted>, String> {
        let layer_sizes = self.build_layer_sizes(&params.hidden_layers, n_features, n_outputs);
        let model = crate::model::mlp::MLP::<B>::new(&layer_sizes, &params.activations);
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
        model: &MLPModel<B, Fitted>,
        dataset: &InMemoryDataset,
        _n_outputs: usize,
    ) -> Result<f64, String> {
        let n_samples = dataset.len().ok_or("Dataset length unknown")?;

        // Get all data at once
        let (x_tensor, y_tensor) = dataset
            .get_batch::<B>(0..n_samples)
            .map_err(|e| format!("Batch error: {:?}", e))?;

        // Predict using batch inference
        let predictions_2d = model.predict_batch(&x_tensor);

        // Flatten predictions for scoring
        let pred_data = predictions_2d.ravel().to_vec();
        let predictions = Tensor1D::new(pred_data.into_iter().map(|x| x as f32).collect());

        // Score
        Ok(self.scorer.score(&predictions, &y_tensor))
    }
}

/// Format activations for display.
fn format_activations(activations: &[Activation]) -> String {
    let names: Vec<&str> = activations
        .iter()
        .map(|a| match a {
            Activation::ReLU => "ReLU",
            Activation::Sigmoid => "Sigmoid",
            Activation::Tanh => "Tanh",
            Activation::Identity => "Id",
        })
        .collect();
    names.join("-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{CpuBackend, Tensor2D};
    use crate::metrics::RegressionMetric;
    use crate::model_selection::{MLPActivations, MLPArchitecture, TrainerGrid};

    fn create_nonlinear_dataset() -> InMemoryDataset {
        // XOR-like problem: y = x1 * x2
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in -5..=5 {
            for j in -5..=5 {
                x.push(vec![i as f32 / 5.0, j as f32 / 5.0]);
                y.push((i * j) as f32 / 25.0);
            }
        }
        InMemoryDataset::new(x, y).unwrap()
    }

    #[test]
    fn test_mlp_grid_search_basic() {
        let dataset = create_nonlinear_dataset();

        let param_grid = MLPGrid::new()
            .with_architectures(vec![MLPArchitecture::single(4)])
            .with_hidden_activations(vec![Activation::ReLU], 1)
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0])
            .with_trainer(TrainerGrid::new(vec![32], vec![100]));

        let grid_search =
            MLPGridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2).verbose(0);

        let result = grid_search.fit(&dataset, 2, 1).unwrap();

        assert!(!result.all_results.is_empty());
        // Just check that we got a result (score can be negative for underfit models)
        assert!(result.best_params.mean_score.is_finite());
    }

    #[test]
    fn test_mlp_grid_search_multiple_architectures() {
        let dataset = create_nonlinear_dataset();

        let param_grid = MLPGrid::new()
            .with_architectures(vec![MLPArchitecture::single(4), MLPArchitecture::single(8)])
            .with_hidden_activations(vec![Activation::ReLU], 1)
            .with_learning_rates(vec![0.1])
            .with_trainer(TrainerGrid::new(vec![32], vec![100]));

        let grid_search = MLPGridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2)
            .with_cv(KFold::new(2))
            .verbose(0);

        let result = grid_search.fit(&dataset, 2, 1).unwrap();

        // Should have 2 combinations
        assert_eq!(result.all_results.len(), 2);

        // Results should be sorted by score (descending)
        for i in 1..result.all_results.len() {
            assert!(
                result.all_results[i - 1].mean_score >= result.all_results[i].mean_score,
                "Results not sorted correctly"
            );
        }
    }

    #[test]
    fn test_mlp_grid_search_best_model_can_predict() {
        let dataset = create_nonlinear_dataset();

        let param_grid = MLPGrid::new()
            .with_architectures(vec![MLPArchitecture::single(8)])
            .with_hidden_activations(vec![Activation::Tanh], 1)
            .with_learning_rates(vec![0.5])
            .with_trainer(TrainerGrid::new(vec![32], vec![200]));

        let grid_search =
            MLPGridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2).verbose(0);

        let result = grid_search.fit(&dataset, 2, 1).unwrap();

        // Best model should be able to predict
        let test_input = Tensor2D::<CpuBackend>::new(vec![0.5, 0.5, -0.5, -0.5], 2, 2);
        let predictions = result.best_model.predict_batch(&test_input);

        // Predictions should have shape (2, 1)
        assert_eq!(predictions.shape(), (2, 1));
    }

    #[test]
    fn test_mlp_grid_search_fold_scores() {
        let dataset = create_nonlinear_dataset();

        let param_grid = MLPGrid::new()
            .with_architectures(vec![MLPArchitecture::single(4)])
            .with_hidden_activations(vec![Activation::ReLU], 1)
            .with_learning_rates(vec![0.1]);

        let grid_search = MLPGridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2)
            .with_cv(KFold::new(3))
            .verbose(0);

        let result = grid_search.fit(&dataset, 2, 1).unwrap();

        // Should have 3 fold scores
        assert_eq!(result.best_params.fold_scores.len(), 3);
    }

    #[test]
    fn test_mlp_grid_search_scoring_name() {
        let dataset = create_nonlinear_dataset();

        let param_grid = MLPGrid::new()
            .with_architectures(vec![MLPArchitecture::single(4)])
            .with_hidden_activations(vec![Activation::ReLU], 1);

        let grid_search =
            MLPGridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::NegMSE);

        let result = grid_search.fit(&dataset, 2, 1).unwrap();

        assert_eq!(result.scoring, "neg_mean_squared_error");
    }

    #[test]
    fn test_mlp_grid_search_result_helpers() {
        let result = MLPGridSearchResult {
            hidden_layers: vec![16, 8],
            activations: vec![Activation::ReLU, Activation::Tanh, Activation::Identity],
            learning_rate: 0.01,
            lambda: 0.1,
            batch_size: 32,
            max_epochs: 100,
            mean_score: 0.95,
            std_score: 0.02,
            fold_scores: vec![0.93, 0.95, 0.97],
        };

        assert_eq!(result.architecture_string(), "[16, 8]");
        assert!(result.activations_string().contains("ReLU"));
        assert!(result.activations_string().contains("Tanh"));
    }

    #[test]
    fn test_mlp_grid_search_with_regularization() {
        let dataset = create_nonlinear_dataset();

        let param_grid = MLPGrid::new()
            .with_architectures(vec![MLPArchitecture::single(8)])
            .with_hidden_activations(vec![Activation::ReLU], 1)
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0, 0.01]);

        let grid_search = MLPGridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2)
            .with_cv(KFold::new(2))
            .verbose(0);

        let result = grid_search.fit(&dataset, 2, 1).unwrap();

        // Should have 2 combinations
        assert_eq!(result.all_results.len(), 2);
    }

    #[test]
    fn test_mlp_grid_search_incompatible_activations_skipped() {
        let dataset = create_nonlinear_dataset();

        // Single hidden layer (2 activations needed) + double hidden layer (3 activations needed)
        let param_grid = MLPGrid::new()
            .with_architectures(vec![
                MLPArchitecture::single(4),    // needs 2 activations
                MLPArchitecture::double(8, 4), // needs 3 activations
            ])
            .with_activations(vec![
                MLPActivations::relu_all(1), // provides 2 activations
            ])
            .with_learning_rates(vec![0.1]);

        let grid_search = MLPGridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2)
            .with_cv(KFold::new(2))
            .verbose(0);

        let result = grid_search.fit(&dataset, 2, 1).unwrap();

        // Should only have 1 result (single hidden layer matches)
        assert_eq!(result.all_results.len(), 1);
        assert_eq!(result.all_results[0].hidden_layers, vec![4]);
    }
}
