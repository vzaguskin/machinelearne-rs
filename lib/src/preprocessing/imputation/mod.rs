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
//! ```
//! use machinelearne_rs::preprocessing::imputation::SimpleImputer;
//! use machinelearne_rs::preprocessing::{Transformer, FittedTransformer, ImputeStrategy};
//! use machinelearne_rs::backend::{CpuBackend, Tensor2D};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let data = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//!
//! let imputer = SimpleImputer::<CpuBackend>::new(ImputeStrategy::Mean);
//! let fitted = imputer.fit(&data)?;
//! let imputed = fitted.transform(&data)?;
//!
//! assert_eq!(imputed.shape(), (2, 2));
//! # Ok(())
//! # }
//! ```

pub mod simple;

pub use simple::{FittedSimpleImputer, ImputeStrategy, SimpleImputer, SimpleImputerParams};
