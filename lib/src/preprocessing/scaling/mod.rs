//! Scaling transformers for feature normalization.
//!
//! This module provides transformers that scale features to a common range
//! or distribution, which is essential for many machine learning algorithms.
//!
//! # Available Transformers
//!
//! | Transformer | Description | Use Case |
//! |-------------|-------------|----------|
//! | [`StandardScaler`] | Z-score normalization (mean=0, std=1) | Default choice for most algorithms |
//! | [`MinMaxScaler`] | Scale to [0, 1] or custom range | When bounded output is needed |
//! | [`RobustScaler`] | Use median and IQR | Data with outliers |
//! | [`MaxAbsScaler`] | Scale by max absolute value | Sparse data |
//! | [`Normalizer`] | Scale individual samples to unit norm | Text classification, clustering |
//!
//! # Example
//!
//! ```ignore
//! use machinelearne_rs::preprocessing::scaling::StandardScaler;
//! use machinelearne_rs::preprocessing::Transformer;
//! use machinelearne_rs::backend::CpuBackend;
//!
//! let scaler = StandardScaler::<CpuBackend>::new();
//! let fitted = scaler.fit(&data)?;
//! let scaled = fitted.transform(&new_data)?;
//! ```

pub mod maxabs;
pub mod minmax;
pub mod normalizer;
pub mod robust;
pub mod standard;

pub use maxabs::{FittedMaxAbsScaler, MaxAbsScaler, MaxAbsScalerParams};
pub use minmax::{FittedMinMaxScaler, MinMaxScaler, MinMaxScalerConfig, MinMaxScalerParams};
pub use normalizer::{FittedNormalizer, NormType, Normalizer, NormalizerParams};
pub use robust::{FittedRobustScaler, RobustScaler, RobustScalerConfig, RobustScalerParams};
pub use standard::{
    FittedStandardScaler, StandardScaler, StandardScalerConfig, StandardScalerParams,
};
