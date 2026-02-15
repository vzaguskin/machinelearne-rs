//! Feature engineering transformers.
//!
//! This module provides transformers for generating new features from existing data.

mod polynomial;

pub use polynomial::{FittedPolynomialFeatures, PolynomialFeatures, PolynomialFeaturesParams};
