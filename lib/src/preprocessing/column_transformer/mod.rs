//! ColumnTransformer for applying different transformers to different columns.
//!
//! This module provides the `ColumnTransformer` which allows applying different
//! preprocessing steps to different subsets of columns in a dataset.

#[allow(clippy::module_inception)]
mod column_transformer;

pub use column_transformer::{
    ColumnSpec, ColumnTransformer, ColumnTransformerParams, FittedColumnTransformer,
};
