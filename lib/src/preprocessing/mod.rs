//! Data preprocessing transformers for machine learning pipelines.
//!
//! This module provides a comprehensive set of data preprocessing transformers
//! following the same type-state pattern as models in this library.
//!
//! # Design Philosophy
//!
//! - **Type Safety**: Transformers use phantom types to distinguish fitted/unfitted states
//! - **Backend Agnostic**: All transformers work with any `Backend` implementation
//! - **Serializable**: Fitted transformers can be saved and loaded
//! - **sklearn-compatible**: API familiar to users of scikit-learn
//!
//! # Core Traits
//!
//! - [`Transformer`]: Unfitted transformer with hyperparameters
//! - [`FittedTransformer`]: Fitted transformer ready for inference
//!
//! # Available Transformers
//!
//! ## Scaling
//! - [`StandardScaler`]: Z-score normalization
//! - [`MinMaxScaler`]: Scale to [0, 1] or custom range
//! - [`RobustScaler`]: Use median and IQR (robust to outliers)
//! - [`MaxAbsScaler`]: Scale by maximum absolute value
//! - [`Normalizer`]: Scale individual samples to unit norm
//!
//! ## Imputation
//! - [`SimpleImputer`]: Fill missing values with mean, median, most_frequent, or constant
//!
//! ## Encoding
//! - [`OneHotEncoder`]: One-hot (dummy) encoding for categorical features
//! - [`OrdinalEncoder`]: Ordinal (integer) encoding for categorical features
//! - [`LabelEncoder`]: Label encoding for 1D classification targets
//!
//! ## Feature Engineering
//! - [`PolynomialFeatures`]: Generate polynomial and interaction features
//!
//! ## Pipeline
//! - [`Pipeline`]: Chain multiple transformers together
//!
//! ## Column Transformer
//! - [`ColumnTransformer`]: Apply different transformers to different columns
//!
//! # Example
//!
//! ```ignore
//! use machinelearne_rs::preprocessing::{Transformer, StandardScaler};
//! use machinelearne_rs::backend::CpuBackend;
//!
//! // Create and fit a scaler
//! let scaler = StandardScaler::<CpuBackend>::new()
//!     .with_mean(true)
//!     .with_std(true);
//!
//! let fitted = scaler.fit(&training_data)?;
//!
//! // Transform training data
//! let scaled_train = fitted.transform(&training_data)?;
//!
//! // Save for later use
//! fitted.save_to_file("scaler.bin")?;
//!
//! // Later, load and transform new data
//! let loaded = FittedStandardScaler::load_from_file("scaler.bin")?;
//! let scaled_test = loaded.transform(&test_data)?;
//! ```

pub mod column_transformer;
pub mod encoding;
pub mod error;
pub mod feature_engineering;
pub mod imputation;
pub mod pipeline;
pub mod predictive_pipeline;
pub mod scaling;
pub mod traits;

// Re-export main types
pub use column_transformer::{
    ColumnSpec, ColumnTransformer, ColumnTransformerParams, FittedColumnTransformer,
};
pub use encoding::{
    FittedLabelEncoder, FittedOneHotEncoder, FittedOrdinalEncoder, HandleUnknown, LabelEncoder,
    LabelEncoderParams, OneHotEncoder, OneHotEncoderParams, OneHotOutput, OrdinalEncoder,
    OrdinalEncoderParams,
};
pub use error::PreprocessingError;
pub use feature_engineering::{
    FittedPolynomialFeatures, PolynomialFeatures, PolynomialFeaturesParams,
};
pub use imputation::{FittedSimpleImputer, ImputeStrategy, SimpleImputer, SimpleImputerParams};
pub use pipeline::{FittedPipeline, Pipeline, PipelineParams, PipelineStep, PipelineStepEnum};
pub use predictive_pipeline::{PredictivePipeline, PredictivePipelineParams};
pub use scaling::{
    FittedMaxAbsScaler, FittedMinMaxScaler, FittedNormalizer, FittedRobustScaler,
    FittedStandardScaler, MaxAbsScaler, MaxAbsScalerParams, MinMaxScaler, MinMaxScalerConfig,
    MinMaxScalerParams, NormType, Normalizer, NormalizerParams, RobustScaler, RobustScalerConfig,
    RobustScalerParams, StandardScaler, StandardScalerConfig, StandardScalerParams,
};
pub use traits::{FittedTransformer, Transformer};
