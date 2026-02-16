//! Metrics for evaluating model performance.
//!
//! This module provides scoring metrics for model evaluation and selection.
//! All scorers follow the convention that **higher is better** - metrics
//! like MSE and MAE are negated internally.
//!
//! # Available Metrics
//!
//! - [`RegressionMetric::NegMSE`] — Negative Mean Squared Error
//! - [`RegressionMetric::NegRMSE`] — Negative Root Mean Squared Error
//! - [`RegressionMetric::NegMAE`] — Negative Mean Absolute Error
//! - [`RegressionMetric::R2`] — R² (coefficient of determination)
//!
//! # Example
//!
//! ```rust
//! use machinelearne_rs::metrics::{Scorer, RegressionMetric};
//! use machinelearne_rs::backend::{CpuBackend, Tensor1D};
//!
//! let metric = RegressionMetric::R2;
//! let y_true = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]);
//! let y_pred = Tensor1D::<CpuBackend>::new(vec![1.1, 1.9, 3.2]);
//!
//! let score = metric.score(&y_pred, &y_true);
//! println!("R² score: {:.4}", score);
//! ```

mod regression;

pub use regression::{RegressionMetric, RegressionMetrics, Scorer};
