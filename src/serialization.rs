//! Serialization of fitted model parameters.
//!
//! This module provides a backend-agnostic way to serialize and deserialize
//! the numerical parameters of a fitted model, without coupling to specific
//! serialization formats or backend resources (e.g., GPU buffers).

use std::error::Error;

/// A trait for parameter representations that can be serialized to and from bytes.
///
/// Implementors should contain only plain numerical data (e.g., `Vec<f32>`, scalars),
/// not backend-specific tensors or handles.
pub trait SerializableParams: Sized {
    /// The error type returned during (de)serialization.
    type Error: Error + Send + Sync + 'static;

    /// Serialize the parameters into a byte buffer.
    fn to_bytes(&self) -> Result<Vec<u8>, Self::Error>;

    /// Deserialize the parameters from a byte buffer.
    fn from_bytes(bytes: &[u8]) -> Result<Self, Self::Error>;
}

// Optional serde integration
#[cfg(feature = "serde")]
impl<T> SerializableParams for T
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    type Error = bincode::Error;

    fn to_bytes(&self) -> Result<Vec<u8>, Self::Error> {
        bincode::serialize(self)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Self::Error> {
        bincode::deserialize(bytes)
    }
}