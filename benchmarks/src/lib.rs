//! Benchmark utilities and common modules for machinelearne-rs benchmarks.
//!
//! This library provides common functionality for benchmarking the machinelearne-rs
//! library against sklearn, including:
//!
//! - Data loading utilities
//! - Metrics calculation (MSE, MAE, R²)
//! - Timing and benchmarking utilities

pub mod data;
pub mod metrics;
pub mod utils;

pub use data::CaliforniaHousingDataset;
pub use metrics::{Metrics, RegressionMetrics};
pub use utils::{benchmark_fn, benchmark_with_warmup, time_fn, BenchmarkStats, Timer};
