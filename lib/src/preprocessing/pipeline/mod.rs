//! Pipeline utilities for chaining transformers.
//!
//! This module provides tools for combining multiple transformers into
//! a single pipeline that can be fitted and used for inference.
//!
//! # Available Components
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`Pipeline`] | Chain transformers sequentially |
//!
//! # Example
//!
//! ```
//! use machinelearne_rs::preprocessing::pipeline::Pipeline;
//! use machinelearne_rs::preprocessing::{StandardScaler, MinMaxScaler, Transformer, FittedTransformer};
//! use machinelearne_rs::backend::{CpuBackend, Tensor2D};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let data = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//!
//! let pipeline = Pipeline::<CpuBackend>::new()
//!     .add_standard_scaler(StandardScaler::new())
//!     .add_minmax_scaler(MinMaxScaler::new());
//!
//! let fitted = pipeline.fit(&data)?;
//! let transformed = fitted.transform(&data)?;
//!
//! assert_eq!(transformed.shape(), (2, 2));
//! # Ok(())
//! # }
//! ```

#[allow(clippy::module_inception)]
pub mod pipeline;

pub use pipeline::{FittedPipeline, Pipeline, PipelineParams, PipelineStep, PipelineStepEnum};
