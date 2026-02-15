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
//! ```ignore
//! use machinelearne_rs::preprocessing::pipeline::Pipeline;
//! use machinelearne_rs::preprocessing::{StandardScaler, MinMaxScaler, Transformer};
//! use machinelearne_rs::backend::CpuBackend;
//!
//! let pipeline = Pipeline::<CpuBackend>::new()
//!     .add_standard_scaler(StandardScaler::new())
//!     .add_minmax_scaler(MinMaxScaler::new());
//!
//! let fitted = pipeline.fit(&data)?;
//! let transformed = fitted.transform(&new_data)?;
//! ```

#[allow(clippy::module_inception)]
pub mod pipeline;

pub use pipeline::{FittedPipeline, Pipeline, PipelineParams, PipelineStep, PipelineStepEnum};
