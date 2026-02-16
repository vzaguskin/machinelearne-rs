//! Pipeline module for end-to-end ML workflows.
//!
//! This module provides [`FittedPipeline`], which combines preprocessing
//! transformers with a trained model for unified inference and deployment.
//!
//! # Example
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

mod fitted_pipeline;

pub use fitted_pipeline::{FittedPipeline, PipelineError, PipelineMetadata};
