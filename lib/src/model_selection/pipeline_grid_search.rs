//! Pipeline-level grid search with proper cross-validation.
//!
//! This module provides [`PipelineGridSearchCV`] for searching over preprocessing
//! options and model hyperparameters with proper cross-validation that avoids
//! data leakage by fitting preprocessors within each CV fold.
//!
//! # Key Design: No Data Leakage
//!
//! Unlike applying preprocessing to the full dataset before CV split,
//! this implementation fits the preprocessor on training data within each fold:
//!
//! ```text
//! raw_data -> CV split -> for each fold:
//!                          train_fold -> fit preprocessor -> transform -> train model
//!                          test_fold  -> transform (same preprocessor) -> evaluate
//! ```
//!
//! # Example
//!
//! ```rust
//! use machinelearne_rs::model_selection::{
//!     PipelineGridSearchCV, PipelineGrid, PreprocessingGrid, ScalerGrid, ScalerType,
//!     LinearRegressionGrid, KFold
//! };
//! use machinelearne_rs::metrics::RegressionMetric;
//! use machinelearne_rs::backend::{CpuBackend, Tensor2D, Tensor1D};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let x = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
//! let y = Tensor1D::<CpuBackend>::new(vec![3.0, 5.0, 7.0]);
//!
//! let preprocessing = PreprocessingGrid::new()
//!     .with_scaler(ScalerGrid::new(vec![ScalerType::Standard]));
//!
//! let model = LinearRegressionGrid::new()
//!     .with_learning_rates(vec![0.1]);
//!
//! let search = PipelineGridSearchCV::<CpuBackend, _>::new(
//!     PipelineGrid::new(preprocessing, model),
//!     RegressionMetric::R2,
//! )
//! .with_cv(KFold::new(2).with_random_state(42));
//!
//! let result = search.fit(&x, &y)?;
//! println!("Best R² score: {:.4}", result.best_params.mean_score);
//! # Ok(())
//! # }
//! ```

use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::dataset::memory::InMemoryDataset;
use crate::loss::MSELoss;
use crate::metrics::Scorer;
use crate::model::linear::{Fitted, LinearModel, LinearRegression};
use crate::model::InferenceModel;
use crate::model_selection::{CVSplit, KFold, PipelineGrid, PipelineParams, PreprocessingParams};
use crate::optimizer::SGD;
use crate::pipeline::FittedPipeline;
use crate::preprocessing::feature_engineering::PolynomialFeatures;
use crate::preprocessing::imputation::{ImputeStrategy, SimpleImputer};
use crate::preprocessing::pipeline::Pipeline;
use crate::preprocessing::scaling::{MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler};
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use crate::regularizers::{NoRegularizer, L2};
use crate::trainer::Trainer;

use super::param_grid::{ImputeStrategy as GridImputeStrategy, ModelParams, ScalerType};

/// Result for a single pipeline parameter combination.
#[derive(Clone, Debug)]
pub struct PipelineSearchResult {
    /// Preprocessing parameters.
    pub preprocessing: PreprocessingParams,
    /// Model parameters.
    pub model: ModelParams,
    /// Mean cross-validation score.
    pub mean_score: f64,
    /// Standard deviation of CV scores.
    pub std_score: f64,
    /// Individual fold scores.
    pub fold_scores: Vec<f64>,
}

/// Result from PipelineGridSearchCV containing the best pipeline and all results.
pub struct PipelineSearchResultB<B: Backend> {
    /// Best fitted pipeline trained on full dataset.
    pub best_pipeline: FittedPipeline<B>,
    /// Parameters of the best pipeline.
    pub best_params: PipelineSearchResult,
    /// All parameter combinations and their scores, sorted by mean_score descending.
    pub all_results: Vec<PipelineSearchResult>,
    /// Name of the scoring metric used.
    pub scoring: String,
}

/// Pipeline-level grid search with proper cross-validation.
///
/// This struct performs exhaustive search over preprocessing options
/// and model hyperparameters, ensuring no data leakage by fitting
/// preprocessors within each CV fold.
pub struct PipelineGridSearchCV<B: Backend, S: Scorer<B>, C: CVSplit = KFold>
where
    S: Scorer<B, Prediction = Tensor1D<B>, Target = Tensor1D<B>>,
{
    param_grid: PipelineGrid,
    cv: C,
    scorer: S,
    verbose: usize,
    _phantom: PhantomData<B>,
}

impl<B: Backend, S: Scorer<B>> PipelineGridSearchCV<B, S, KFold>
where
    S: Scorer<B, Prediction = Tensor1D<B>, Target = Tensor1D<B>>,
    B::Scalar: Debug + Display,
{
    /// Create a new pipeline grid search.
    ///
    /// # Arguments
    /// * `param_grid` - The parameter grid defining preprocessing and model search space
    /// * `scorer` - The scoring metric (higher is better)
    pub fn new(param_grid: PipelineGrid, scorer: S) -> Self {
        Self {
            param_grid,
            cv: KFold::new(5),
            scorer,
            verbose: 0,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, S: Scorer<B>, C: CVSplit> PipelineGridSearchCV<B, S, C>
where
    S: Scorer<B, Prediction = Tensor1D<B>, Target = Tensor1D<B>>,
    B::Scalar: Debug + Display,
{
    /// Set the cross-validation splitter.
    pub fn with_cv<NewC: CVSplit>(self, cv: NewC) -> PipelineGridSearchCV<B, S, NewC> {
        PipelineGridSearchCV {
            param_grid: self.param_grid,
            cv,
            scorer: self.scorer,
            verbose: self.verbose,
            _phantom: PhantomData,
        }
    }

    /// Set verbosity level (0 = silent, 1 = progress, 2 = detailed).
    pub fn verbose(mut self, level: usize) -> Self {
        self.verbose = level;
        self
    }

    /// Fit the grid search on raw data.
    ///
    /// This method takes RAW data (not preprocessed) and performs proper
    /// cross-validation by fitting preprocessors within each fold.
    ///
    /// # Arguments
    /// * `raw_x` - Raw feature matrix (n_samples x n_features)
    /// * `y` - Target values
    ///
    /// # Returns
    /// The best pipeline and all results on success, or an error message.
    pub fn fit(
        &self,
        raw_x: &Tensor2D<B>,
        y: &Tensor1D<B>,
    ) -> Result<PipelineSearchResultB<B>, String> {
        let (n_samples, n_features) = raw_x.shape();

        // Convert tensors to dataset format for CV
        let x_vec = raw_x.ravel().to_vec();
        let y_vec = y.to_vec();

        let x_2d: Vec<Vec<f32>> = (0..n_samples)
            .map(|i| {
                (0..n_features)
                    .map(|j| x_vec[i * n_features + j] as f32)
                    .collect()
            })
            .collect();

        let y_f32: Vec<f32> = y_vec.iter().map(|&v| v as f32).collect();
        let _dataset = InMemoryDataset::new(x_2d, y_f32).map_err(|e| format!("{:?}", e))?;

        let total_combinations = self.param_grid.n_combinations();
        let splits = self.cv.split(n_samples);

        if self.verbose >= 1 {
            println!(
                "Fitting {} folds for {} candidates, totalling {} fits",
                self.cv.n_splits(),
                total_combinations,
                total_combinations * self.cv.n_splits()
            );
        }

        let mut all_results = Vec::with_capacity(total_combinations);

        // Iterate over all parameter combinations
        for (comb_idx, params) in self.param_grid.iter().enumerate() {
            if self.verbose >= 1 {
                println!(
                    "[{}/{}] scaler={:?}, poly={}, lr={}, lambda={}",
                    comb_idx + 1,
                    total_combinations,
                    params.preprocessing.scaler,
                    params.preprocessing.poly_degree,
                    params.model.learning_rate,
                    params.model.lambda
                );
            }

            let fold_scores = self.evaluate_combination(raw_x, y, &splits, &params, n_features)?;

            // Compute mean and std
            let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            let variance = fold_scores
                .iter()
                .map(|&s| (s - mean_score).powi(2))
                .sum::<f64>()
                / fold_scores.len() as f64;
            let std_score = variance.sqrt();

            all_results.push(PipelineSearchResult {
                preprocessing: params.preprocessing.clone(),
                model: params.model.clone(),
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
                "\nBest parameters: scaler={:?}, poly={}, lr={}, lambda={}",
                best_params.preprocessing.scaler,
                best_params.preprocessing.poly_degree,
                best_params.model.learning_rate,
                best_params.model.lambda
            );
            println!(
                "Best {} score: {:.4} (+/- {:.4})",
                self.scorer.name(),
                best_params.mean_score,
                best_params.std_score
            );
        }

        // Train final pipeline on ALL data with best params
        let best_pipeline = self.fit_final_pipeline(raw_x, y, &best_params, n_features)?;

        Ok(PipelineSearchResultB {
            best_pipeline,
            best_params,
            all_results,
            scoring: self.scorer.name().to_string(),
        })
    }

    /// Evaluate a single parameter combination using cross-validation.
    ///
    /// This is the key method that ensures no data leakage by fitting
    /// the preprocessor on training data within each fold.
    fn evaluate_combination(
        &self,
        raw_x: &Tensor2D<B>,
        y: &Tensor1D<B>,
        splits: &[(Vec<usize>, Vec<usize>)],
        params: &PipelineParams,
        n_features: usize,
    ) -> Result<Vec<f64>, String> {
        let mut fold_scores = Vec::with_capacity(splits.len());

        for (fold_idx, (train_indices, test_indices)) in splits.iter().enumerate() {
            // Split raw data
            let x_train = self.subset_rows(raw_x, train_indices);
            let y_train = self.subset_1d(y, train_indices);
            let x_test = self.subset_rows(raw_x, test_indices);
            let y_test = self.subset_1d(y, test_indices);

            // Build and fit preprocessor on TRAINING data only
            let (preproc, x_train_proc) =
                self.build_and_fit_preprocessor(&x_train, &params.preprocessing, n_features)?;

            // Transform test data with TRAIN-FITTED preprocessor
            let x_test_proc = if let Some(ref p) = preproc {
                p.transform(&x_test).map_err(|e| format!("{:?}", e))?
            } else {
                x_test.clone()
            };

            // Apply polynomial features
            let (poly, x_train_poly) =
                self.build_and_fit_polynomial(&x_train_proc, params.preprocessing.poly_degree)?;
            let x_test_poly = if let Some(ref p) = poly {
                p.transform(&x_test_proc).map_err(|e| format!("{:?}", e))?
            } else {
                x_test_proc
            };

            // Train model on processed training data
            let model = self.train_model(&x_train_poly, &y_train, &params.model)?;

            // Evaluate on processed test data
            let predictions = model.predict_batch(&x_test_poly);
            let score = self.scorer.score(&predictions, &y_test);

            if self.verbose >= 2 {
                println!("  Fold {}: score = {:.4}", fold_idx, score);
            }

            fold_scores.push(score);
        }

        Ok(fold_scores)
    }

    /// Fit the final pipeline on all data with best parameters.
    fn fit_final_pipeline(
        &self,
        raw_x: &Tensor2D<B>,
        y: &Tensor1D<B>,
        best_params: &PipelineSearchResult,
        n_features: usize,
    ) -> Result<FittedPipeline<B>, String> {
        // Build and fit preprocessor on ALL data
        let (preproc, x_proc) =
            self.build_and_fit_preprocessor(raw_x, &best_params.preprocessing, n_features)?;

        // Apply polynomial features
        let (poly, x_poly) =
            self.build_and_fit_polynomial(&x_proc, best_params.preprocessing.poly_degree)?;

        // Train model on all processed data
        let model = self.train_model(&x_poly, y, &best_params.model)?;

        Ok(FittedPipeline::new(preproc, poly, model))
    }

    /// Build and fit the preprocessor for given preprocessing params.
    fn build_and_fit_preprocessor(
        &self,
        x: &Tensor2D<B>,
        params: &PreprocessingParams,
        _n_features: usize,
    ) -> Result<
        (
            Option<crate::preprocessing::pipeline::FittedPipeline<B>>,
            Tensor2D<B>,
        ),
        String,
    > {
        let mut pipeline = Pipeline::<B>::new();
        let mut has_steps = false;

        // Add imputer if specified
        if let Some(ref impute_strategy) = params.imputer {
            let strategy = match impute_strategy {
                GridImputeStrategy::Mean => ImputeStrategy::Mean,
                GridImputeStrategy::Median => ImputeStrategy::Median,
                GridImputeStrategy::MostFrequent => ImputeStrategy::MostFrequent,
                GridImputeStrategy::Constant(v) => ImputeStrategy::Constant((*v) as f64),
            };
            pipeline = pipeline.add_simple_imputer(SimpleImputer::new(strategy));
            has_steps = true;
        }

        // Add scaler if not None
        if params.scaler != ScalerType::None {
            match params.scaler {
                ScalerType::Standard => {
                    pipeline = pipeline.add_standard_scaler(StandardScaler::new());
                }
                ScalerType::MinMax => {
                    pipeline = pipeline.add_minmax_scaler(MinMaxScaler::new());
                }
                ScalerType::Robust => {
                    pipeline = pipeline.add_robust_scaler(RobustScaler::new());
                }
                ScalerType::MaxAbs => {
                    pipeline = pipeline.add_maxabs_scaler(MaxAbsScaler::new());
                }
                ScalerType::None => unreachable!(),
            }
            has_steps = true;
        }

        if has_steps {
            let fitted = pipeline
                .fit(x)
                .map_err(|e| format!("Preprocessor fit error: {:?}", e))?;
            let transformed = fitted
                .transform(x)
                .map_err(|e| format!("Preprocessor transform error: {:?}", e))?;
            Ok((Some(fitted), transformed))
        } else {
            Ok((None, x.clone()))
        }
    }

    /// Build and fit polynomial features transformer.
    fn build_and_fit_polynomial(
        &self,
        x: &Tensor2D<B>,
        degree: usize,
    ) -> Result<
        (
            Option<crate::preprocessing::feature_engineering::FittedPolynomialFeatures<B>>,
            Tensor2D<B>,
        ),
        String,
    > {
        if degree <= 1 {
            return Ok((None, x.clone()));
        }

        let poly = PolynomialFeatures::<B>::new()
            .with_degree(degree)
            .with_include_bias(false);

        let fitted = poly
            .fit(x)
            .map_err(|e| format!("Polynomial fit error: {:?}", e))?;
        let transformed = fitted
            .transform(x)
            .map_err(|e| format!("Polynomial transform error: {:?}", e))?;

        Ok((Some(fitted), transformed))
    }

    /// Train a model with given parameters.
    fn train_model(
        &self,
        x: &Tensor2D<B>,
        y: &Tensor1D<B>,
        params: &ModelParams,
    ) -> Result<LinearModel<B, Fitted>, String> {
        let (rows, n_features) = x.shape();

        // Create dataset from tensors
        let x_vec = x.ravel().to_vec();
        let y_vec = y.to_vec();
        let x_2d: Vec<Vec<f32>> = (0..rows)
            .map(|i| {
                (0..n_features)
                    .map(|j| x_vec[i * n_features + j] as f32)
                    .collect()
            })
            .collect();

        let y_f32: Vec<f32> = y_vec.iter().map(|&v| v as f32).collect();
        let dataset = InMemoryDataset::new(x_2d, y_f32).map_err(|e| format!("{:?}", e))?;

        let model = LinearRegression::<B>::new(n_features);
        let loss = MSELoss;
        let optimizer = SGD::new(params.learning_rate);

        let fitted = if params.lambda > 0.0 {
            let trainer = Trainer::builder(loss, optimizer, L2::new(params.lambda))
                .batch_size(params.batch_size)
                .max_epochs(params.max_epochs)
                .verbose(false)
                .build();
            trainer.fit(model, &dataset)?
        } else {
            let trainer = Trainer::builder(loss, optimizer, NoRegularizer)
                .batch_size(params.batch_size)
                .max_epochs(params.max_epochs)
                .verbose(false)
                .build();
            trainer.fit(model, &dataset)?
        };

        Ok(fitted)
    }

    /// Subset rows from a 2D tensor.
    fn subset_rows(&self, tensor: &Tensor2D<B>, indices: &[usize]) -> Tensor2D<B> {
        let (_rows, cols) = tensor.shape();
        let data = tensor.ravel().to_vec();

        let mut subset_data = Vec::with_capacity(indices.len() * cols);
        for &idx in indices {
            for j in 0..cols {
                subset_data.push(data[idx * cols + j] as f32);
            }
        }

        Tensor2D::new(subset_data, indices.len(), cols)
    }

    /// Subset a 1D tensor.
    fn subset_1d(&self, tensor: &Tensor1D<B>, indices: &[usize]) -> Tensor1D<B> {
        let data = tensor.to_vec();
        let subset: Vec<f32> = indices.iter().map(|&i| data[i] as f32).collect();
        Tensor1D::new(subset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::metrics::RegressionMetric;
    use crate::model_selection::{LinearRegressionGrid, PreprocessingGrid, ScalerGrid};

    fn create_raw_data() -> (Tensor2D<CpuBackend>, Tensor1D<CpuBackend>) {
        // y = 2*x1 + 3*x2 + 1
        let x = Tensor2D::<CpuBackend>::new(
            vec![
                0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 2.0,
                2.0, 2.0, 3.0, 0.0,
            ],
            10,
            2,
        );
        let y =
            Tensor1D::<CpuBackend>::new(vec![1.0, 3.0, 4.0, 6.0, 5.0, 7.0, 8.0, 10.0, 11.0, 7.0]);
        (x, y)
    }

    #[test]
    fn test_pipeline_grid_search_basic() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new(); // No preprocessing

        let model = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(3))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();

        assert!(!result.all_results.is_empty());
        assert!(result.best_params.mean_score > 0.0);
    }

    #[test]
    fn test_pipeline_grid_search_with_scalers() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new().with_scaler(ScalerGrid::new(vec![
            ScalerType::Standard,
            ScalerType::None,
        ]));

        let model = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(3))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();

        // Should have 2 combinations (2 scalers)
        assert_eq!(result.all_results.len(), 2);
    }

    #[test]
    fn test_pipeline_grid_search_best_pipeline_predicts() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new();

        let model = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(3))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();

        // Best pipeline should be able to predict
        let test_input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 0.0, 0.0, 1.0], 2, 2);
        let predictions = result.best_pipeline.predict(&test_input).unwrap();

        // Predictions should be close to expected values (3 and 4)
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
    fn test_pipeline_grid_search_serialization() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new();

        let model = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(3))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();

        // Save and load the pipeline
        let temp_file = std::env::temp_dir().join("test_pipeline_search.bin");
        result.best_pipeline.save_to_file(&temp_file).unwrap();

        let loaded = FittedPipeline::<CpuBackend>::load_from_file(&temp_file).unwrap();

        // Compare predictions
        let test_input = Tensor2D::<CpuBackend>::new(vec![1.0f32, 0.0], 1, 2);
        let pred1 = result.best_pipeline.predict(&test_input).unwrap();
        let pred2 = loaded.predict(&test_input).unwrap();

        assert!((pred1.to_vec()[0] - pred2.to_vec()[0]).abs() < 1e-6);

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_pipeline_grid_search_multiple_params() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new().with_scaler(ScalerGrid::new(vec![
            ScalerType::Standard,
            ScalerType::None,
        ]));

        let model = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.01, 0.1])
            .with_lambdas(vec![0.0, 0.1]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        // 2 * 2 * 2 = 8 combinations
        assert_eq!(pipeline_grid.n_combinations(), 8);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(3))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();

        assert_eq!(result.all_results.len(), 8);

        // Results should be sorted by score (descending)
        for i in 1..result.all_results.len() {
            assert!(
                result.all_results[i - 1].mean_score >= result.all_results[i].mean_score,
                "Results not sorted correctly"
            );
        }
    }

    #[test]
    fn test_pipeline_grid_search_scoring_name() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new();

        let model = LinearRegressionGrid::new();

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::NegMSE);

        let result = search.fit(&x, &y).unwrap();

        assert_eq!(result.scoring, "neg_mean_squared_error");
    }

    #[test]
    fn test_pipeline_grid_search_results_sorted() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new();

        let model = LinearRegressionGrid::new().with_learning_rates(vec![0.001, 0.1]); // 0.1 should be better

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(3))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();

        // Best should have learning_rate = 0.1
        assert!(
            (result.best_params.model.learning_rate - 0.1).abs() < 1e-6,
            "Expected best lr=0.1, got {}",
            result.best_params.model.learning_rate
        );
    }

    #[test]
    fn test_pipeline_grid_search_with_polynomial() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new()
            .with_polynomial(crate::model_selection::PolynomialGrid::new(vec![1, 2]));

        let model = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.0]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(3))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();

        // Should have 2 combinations (2 polynomial degrees)
        assert_eq!(result.all_results.len(), 2);
        assert!(!result.best_params.fold_scores.is_empty());
    }

    #[test]
    fn test_pipeline_grid_search_fold_scores() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new();

        let model = LinearRegressionGrid::new().with_learning_rates(vec![0.1]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(5))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();

        // Should have 5 fold scores
        assert_eq!(result.best_params.fold_scores.len(), 5);
    }

    #[test]
    fn test_pipeline_grid_search_neg_metrics() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new();

        let model = LinearRegressionGrid::new().with_learning_rates(vec![0.1]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        // Test NegMSE
        let search = PipelineGridSearchCV::<CpuBackend, _>::new(
            pipeline_grid.clone(),
            RegressionMetric::NegMSE,
        )
        .with_cv(KFold::new(3))
        .verbose(0);

        let result = search.fit(&x, &y).unwrap();
        assert_eq!(result.scoring, "neg_mean_squared_error");

        // Test NegRMSE
        let search = PipelineGridSearchCV::<CpuBackend, _>::new(
            pipeline_grid.clone(),
            RegressionMetric::NegRMSE,
        )
        .with_cv(KFold::new(3))
        .verbose(0);

        let result = search.fit(&x, &y).unwrap();
        assert_eq!(result.scoring, "neg_root_mean_squared_error");

        // Test NegMAE
        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::NegMAE)
                .with_cv(KFold::new(3))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();
        assert_eq!(result.scoring, "neg_mean_absolute_error");
    }

    #[test]
    fn test_pipeline_grid_search_with_regularization() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new();

        let model = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1])
            .with_lambdas(vec![0.5]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(3))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();

        // Should find params with lambda = 0.5
        assert_eq!(result.all_results.len(), 1);
        assert!((result.best_params.model.lambda - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_grid_search_all_scalers() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new().with_scaler(ScalerGrid::new(vec![
            ScalerType::Standard,
            ScalerType::MinMax,
            ScalerType::Robust,
            ScalerType::MaxAbs,
            ScalerType::None,
        ]));

        let model = LinearRegressionGrid::new().with_learning_rates(vec![0.1]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(3))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();

        // Should have 5 combinations (5 scalers)
        assert_eq!(result.all_results.len(), 5);
    }

    #[test]
    fn test_pipeline_grid_search_verbose_level_2() {
        let (x, y) = create_raw_data();

        let preprocessing = PreprocessingGrid::new();

        let model = LinearRegressionGrid::new().with_learning_rates(vec![0.1]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(3))
                .verbose(2); // Test verbose level 2

        let result = search.fit(&x, &y).unwrap();
        assert!(result.best_params.mean_score > 0.0);
    }

    #[test]
    fn test_pipeline_grid_search_preprocessing_params() {
        let (x, y) = create_raw_data();

        let preprocessing =
            PreprocessingGrid::new().with_scaler(ScalerGrid::new(vec![ScalerType::Standard]));

        let model = LinearRegressionGrid::new().with_learning_rates(vec![0.1]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let search =
            PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
                .with_cv(KFold::new(3))
                .verbose(0);

        let result = search.fit(&x, &y).unwrap();

        // Check preprocessing params are captured
        assert_eq!(
            result.best_params.preprocessing.scaler,
            ScalerType::Standard
        );
        assert_eq!(result.best_params.preprocessing.poly_degree, 1);
        assert!(result.best_params.preprocessing.imputer.is_none());
    }
}
