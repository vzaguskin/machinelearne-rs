//! Data wrapper that tracks preprocessing state at compile time.

#![allow(dead_code)] // Experimental module - some methods not yet used

use crate::backend::{Backend, Tensor2D};
use std::marker::PhantomData;

use super::state::State;

/// A data wrapper that tracks preprocessing state at the type level.
///
/// This ensures that transformers are only applied to data in the correct state.
/// For example, a Scaler can only be applied to `Data<B, Imputed>`, not `Data<B, Raw>`.
#[derive(Clone)]
pub struct Data<B: Backend, S: State> {
    /// The underlying tensor data.
    pub inner: Tensor2D<B>,
    /// Phantom data for state tracking.
    _state: PhantomData<S>,
}

impl<B: Backend, S: State> Data<B, S> {
    /// Create new data with the given state.
    pub fn new(inner: Tensor2D<B>) -> Self {
        Self {
            inner,
            _state: PhantomData,
        }
    }

    /// Get the shape of the data.
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Get the number of rows (samples).
    pub fn n_rows(&self) -> usize {
        self.inner.shape().0
    }

    /// Get the number of columns (features).
    pub fn n_features(&self) -> usize {
        self.inner.shape().1
    }

    /// Transition to a new state (internal use).
    /// This is only called by transformers that produce a new state.
    pub(crate) fn transition<NewState: State>(self) -> Data<B, NewState> {
        Data {
            inner: self.inner,
            _state: PhantomData,
        }
    }
}

impl<B: Backend, S: State> From<Tensor2D<B>> for Data<B, S> {
    fn from(inner: Tensor2D<B>) -> Self {
        Self::new(inner)
    }
}

impl<B: Backend, S: State> AsRef<Tensor2D<B>> for Data<B, S> {
    fn as_ref(&self) -> &Tensor2D<B> {
        &self.inner
    }
}

// Allow unwrapping to get the inner tensor
impl<B: Backend, S: State> Data<B, S> {
    /// Unwrap to get the inner tensor.
    pub fn into_inner(self) -> Tensor2D<B> {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::preprocessing::pipeline_v2::state::Raw;

    #[test]
    fn test_data_creation() {
        let tensor = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let data: Data<CpuBackend, Raw> = Data::new(tensor);

        assert_eq!(data.shape(), (2, 2));
        assert_eq!(data.n_rows(), 2);
        assert_eq!(data.n_features(), 2);
    }

    #[test]
    fn test_data_from_tensor() {
        let tensor = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);
        let data: Data<CpuBackend, Raw> = tensor.clone().into();

        assert_eq!(data.shape(), tensor.shape());
    }

    #[test]
    fn test_data_into_inner() {
        let tensor = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0], 1, 2);
        let data: Data<CpuBackend, Raw> = Data::new(tensor.clone());

        let recovered = data.into_inner();
        assert_eq!(recovered.shape(), tensor.shape());
    }
}
