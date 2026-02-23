//! ONNX export and inference support.
//!
//! This module provides functionality to export trained models and pipelines
//! to ONNX format for portable deployment and optimized inference.
//!
//! # Features
//!
//! - `onnx`: Enable ONNX export functionality (stable)
//! - `onnx-inference`: Enable ONNX Runtime inference (includes `onnx`) - **EXPERIMENTAL**
//! - `onnx-server`: Enable HTTP inference server (includes `onnx-inference`)
//!
//! ## Experimental Status
//!
//! The `onnx-inference` feature is experimental and may require manual setup:
//! - Download ONNX Runtime 1.23.0 from GitHub releases
//! - Set `ORT_LIB_LOCATION` and `ORT_PREFER_DYNAMIC_LINK=1` environment variables
//!
//! The `onnx` export feature is stable and works out of the box.
//!
//! # Example
//!
//! ```rust,ignore
//! use machinelearne_rs::backend::CpuBackend;
//! use machinelearne_rs::model::linear::{LinearModel, Fitted};
//! use machinelearne_rs::onnx::OnnxExportable;
//!
//! // Train a model
//! let model = train_model();
//!
//! // Export to ONNX
//! model.save_onnx("model.onnx")?;
//!
//! // Load and run inference with ONNX Runtime (requires onnx-inference feature)
//! // Note: onnx-inference is experimental
//! let session = OnnxInferenceSession::load("model.onnx")?;
//! let predictions = session.predict(&input)?;
//! ```
//!
//! ## Inference Server (onnx-server feature)
//!
//! ```rust,ignore
//! use machinelearne_rs::onnx::server::{OnnxServer, ExecutionProvider};
//!
//! // Start an inference server
//! OnnxServer::new()
//!     .model("model.onnx")
//!     .host("0.0.0.0")
//!     .port(8080)
//!     .provider(ExecutionProvider::Cpu)
//!     .run()
//!     .await?;
//! ```

mod error;

pub use error::OnnxError;

#[cfg(feature = "onnx")]
mod export;
#[cfg(feature = "onnx")]
mod from_pipeline;
#[cfg(feature = "onnx")]
mod graph;
#[cfg(feature = "onnx")]
mod operators;
#[cfg(feature = "onnx")]
mod proto;

#[cfg(feature = "onnx")]
pub use export::OnnxExportable;
#[cfg(feature = "onnx")]
pub use from_pipeline::export_pipeline_to_onnx;
#[cfg(feature = "onnx")]
pub use from_pipeline::validate_pipeline_export;
#[cfg(feature = "onnx")]
pub use graph::OnnxGraphBuilder;

#[cfg(feature = "onnx-inference")]
mod inference;

#[cfg(feature = "onnx-inference")]
pub use inference::OnnxInferenceSession;

#[cfg(feature = "onnx-server")]
pub mod server;

/// Default ONNX opset version for main domain.
pub const DEFAULT_OPSET_VERSION: i64 = 17;

/// ONNX ML opset version for traditional ML operators.
pub const ML_OPSET_VERSION: i64 = 3;
