//! Experimental Pipeline V2 with trait objects and type-state enforcement.
//!
//! This is an experimental implementation that uses:
//! - Trait objects to avoid enum match boilerplate
//! - Type-state pattern to enforce preprocessing dependencies
//!
//! # Design Goals
//! - No enum matching for transform calls (use trait objects + vtable)
//! - Compile-time enforcement of preprocessing order:
//!   - Imputer can be applied to Raw data
//!   - Scaler requires Imputed data (must come after imputer)
//!   - Encoder requires Scaled data (must come after scaler)
//!   - Model requires all preprocessing complete
//!
//! # Example
//! ```ignore
//! let raw: Data<CpuBackend, Raw> = Data::from(raw_tensor);
//!
//! // Imputer works on Raw data
//! let imputer = SimpleImputer::new(ImputeStrategy::Mean);
//! let fitted_imputer = imputer.fit(&raw)?;
//! let imputed: Data<CpuBackend, Imputed> = fitted_imputer.transform(raw)?;
//!
//! // Scaler requires Imputed data (won't compile with Raw)
//! let scaler = StandardScaler::new();
//! let fitted_scaler = scaler.fit(&imputed)?;
//! let scaled: Data<CpuBackend, Scaled> = fitted_scaler.transform(imputed)?;
//!
//! // Pipeline tracks state at compile time
//! let pipeline = Pipeline::new()
//!     .add_imputer(fitted_imputer)  // Returns Pipeline<Imputed>
//!     .add_scaler(fitted_scaler);   // Returns Pipeline<Scaled>
//!
//! // Model requires Scaled data
//! let model = LinearModel::new();
//! model.predict(&scaled)?;  // Works!
//! // model.predict(&raw)?;  // Compile error!
//! ```

mod data;
mod pipeline;
mod state;
mod step;

pub use data::Data;
pub use pipeline::{FittedPipelineV2, PipelineV2};
pub use state::{
    Encoded, Imputed, Raw, RequiresEncoding, RequiresImputation, RequiresScaling, Scaled, State,
};

// Re-export step trait for external use
pub use step::FittedStep;
