//! Model selection utilities for hyperparameter tuning and evaluation.
//!
//! This module provides tools for:
//! - Train/test splitting
//! - Cross-validation
//! - Hyperparameter grid search
//!
//! # Example
//!
//! ```rust,ignore
//! use machinelearne_rs::model_selection::{GridSearchCV, KFold, LinearRegressionGrid};
//! use machinelearne_rs::metrics::RegressionMetric;
//!
//! let param_grid = LinearRegressionGrid::new()
//!     .with_learning_rates(vec![0.001, 0.01, 0.1])
//!     .with_lambdas(vec![0.0, 0.01, 0.1]);
//!
//! let grid_search = GridSearchCV::new(param_grid, RegressionMetric::R2)
//!     .with_cv(KFold::new(5));
//!
//! let result = grid_search.fit(&dataset, n_features)?;
//! println!("Best score: {}", result.best_params.mean_score);
//! ```

mod cross_validation;
mod grid_search;
mod param_grid;
mod split;

pub use cross_validation::{CVSplit, KFold};
pub use grid_search::{GridSearchCV, GridSearchCVResult, GridSearchResult};
pub use param_grid::{
    LinearRegressionGrid, ParamCombination, PolynomialGrid, RegularizerGrid, SGDGrid, TrainerGrid,
};
pub use split::train_test_split;
