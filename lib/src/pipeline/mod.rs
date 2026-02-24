//! Pipeline module for end-to-end ML workflows.
//!
//! This module provides [`FittedPipeline`] and [`MLPFittedPipeline`], which combine
//! preprocessing transformers with trained models for unified inference and deployment.
//!
//! # Example (Linear Model)
//!
//! ```ignore
//! use machinelearne_rs::pipeline::FittedPipeline;
//! use machinelearne_rs::backend::CpuBackend;
//!
//! // Load a trained pipeline
//! let pipeline = FittedPipeline::<CpuBackend>::load_from_file("model.bin")?;
//!
//! // Predict on raw data
//! let predictions = pipeline.predict(&raw_data)?;
//! ```
//!
//! # Example (MLP Model)
//!
//! ```ignore
//! use machinelearne_rs::pipeline::MLPFittedPipeline;
//! use machinelearne_rs::backend::CpuBackend;
//!
//! // Load a trained MLP pipeline
//! let pipeline = MLPFittedPipeline::<CpuBackend>::load_from_file("mlp_model.bin")?;
//!
//! // Predict on raw data
//! let predictions = pipeline.predict(&raw_data)?;
//! ```

mod fitted_pipeline;
mod mlp_fitted_pipeline;

pub use fitted_pipeline::{FittedPipeline, PipelineError, PipelineMetadata};
pub use mlp_fitted_pipeline::{MLPFittedPipeline, MLPPipelineError, MLPPipelineMetadata};
