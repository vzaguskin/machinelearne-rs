//! Categorical feature encoding transformers.
//!
//! This module provides encoders for converting categorical features to numerical
//! representations that can be used by machine learning models.
//!
//! # Available Encoders
//!
//! ## OneHotEncoder
//! Converts categorical integer values to one-hot (dummy) encoding.
//!
//! ```ignore
//! // Input: [[0], [1], [2]]  (3 samples, 1 categorical feature)
//! // Output: [[1,0,0], [0,1,0], [0,0,1]]  (3 samples, 3 binary features)
//! ```
//!
//! ## OrdinalEncoder
//! Maps categorical values to integer ordinals (0, 1, 2, ...).
//!
//! ## LabelEncoder
//! Encodes 1D target labels to integers (for classification targets).
//!
//! # Design Notes
//!
//! All encoders work with `f32` tensor representations, where categorical values
//! are pre-mapped to integers by the user. This design:
//! - Maintains consistency with the tensor-based API
//! - Avoids string handling in the core library
//! - Allows users to handle their own category-to-integer mapping

mod label;
mod one_hot;
mod ordinal;

pub use label::{FittedLabelEncoder, LabelEncoder, LabelEncoderParams};
pub use one_hot::{FittedOneHotEncoder, OneHotEncoder, OneHotEncoderParams, OneHotOutput};
pub use ordinal::{FittedOrdinalEncoder, OrdinalEncoder, OrdinalEncoderParams};

/// Strategy for handling unknown categories during transform.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HandleUnknown {
    /// Raise an error when unknown categories are encountered.
    #[default]
    Error,
    /// Ignore unknown categories (output zeros for one-hot, NaN for ordinal).
    Ignore,
}
