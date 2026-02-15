//! Imputation transformers for handling missing values.
//!
//! This module provides transformers for imputing (filling in) missing values
//! in datasets.
//!
//! # Available Transformers
//!
//! | Transformer | Description |
//! |-------------|-------------|
//! | [`SimpleImputer`] | Impute with mean, median, most_frequent, or constant |
//!
//! # Example
//!
//! ```ignore
//! use machinelearne_rs::preprocessing::imputation::SimpleImputer;
//! use machinelearne_rs::preprocessing::{Transformer, ImputeStrategy};
//! use machinelearne_rs::backend::CpuBackend;
//!
//! let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Mean);
//! let fitted = imputer.fit(&data)?;
//! let imputed = fitted.transform(&new_data)?;
//! ```

pub mod simple;

pub use simple::{FittedSimpleImputer, ImputeStrategy, SimpleImputer, SimpleImputerParams};
