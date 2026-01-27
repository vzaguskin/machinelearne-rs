//! Dataset abstractions for machine learning workloads.
//!
//! This module provides a generic [`Dataset`] trait for uniform access to training data
//! and a [`DatasetBatchIter`] iterator for efficient batch loading across different backends.
//!
//! # Core Concepts
//!
//! - **Dataset** — A source of `(X, y)` pairs where `X` is a feature matrix of shape `(n_samples, n_features)`
//!   and `y` is a target vector of shape `(n_samples,)`.
//! - **Backend** — Tensor implementation (`CPU`, `CUDA`, etc.) defined by the [`Backend`] trait.
//! - **Batch** — A contiguous subset of samples for mini-batch gradient descent.
//!
//! # Example
//!
//! ```rust
//! use machinelearne_rs::dataset::{Dataset, InMemoryDataset};
//! use machinelearne_rs::backend::CpuBackend;
//!
//! // Create an in-memory dataset: 2 samples × 2 features
//! let x = vec![vec![1.0], vec![2.0]];
//! let y = vec![0.0, 1.0];
//! let dataset = InMemoryDataset::new(x, y).unwrap();
//!
//! // Iterate over batches of size 1
//! for batch in dataset.batches::<CpuBackend>(1) {
//!     let (x_batch, y_batch) = batch.unwrap();
//!     // ... train your model
//! }
//! ```

use crate::backend::{Backend, Tensor1D, Tensor2D};
use std::{fmt::Debug, ops::Range};

pub mod memory;
pub use self::memory::InMemoryDataset;

/// Abstract interface for a machine learning dataset.
///
/// Defines a contract for loading data in `(X, y)` format where:
/// - `X` — Feature matrix with shape `(n_samples, n_features)`
/// - `y` — Target vector with shape `(n_samples,)`
///
/// # Associated Types
///
/// - `Error` — Error type returned when accessing data (must implement [`Debug`])
/// - `Item` — Type of a single dataset item (rarely used directly; often `()` as a placeholder)
///
/// # Example Implementation
///
/// ```rust
/// use machinelearne_rs::dataset::Dataset;
/// use machinelearne_rs::backend::{Backend, Tensor1D, Tensor2D};
/// use std::ops::Range;
///
/// struct MyDataset { /* ... */ }
///
/// impl Dataset for MyDataset {
///     type Error = String;
///     type Item = ();
///
///     fn len(&self) -> Option<usize> {
///         Some(1000) // or None if size is unknown (e.g., streaming data)
///     }
///
///     fn get_batch<B: Backend>(
///         &self,
///         range: Range<usize>,
///     ) -> Result<(Tensor2D<B>, Tensor1D<B>), Self::Error> {
///         // Implement range-based data loading
///         # unimplemented!()
///     }
/// }
/// ```
pub trait Dataset {
    /// Error type returned when accessing data.
    type Error: Debug + 'static;

    /// Type of a single dataset item (typically unused directly).
    type Item: ?Sized;

    /// Returns the total number of samples in the dataset, if known.
    ///
    /// # Returns
    ///
    /// - `Some(n)` — Exact number of samples
    /// - `None` — Size is unknown (e.g., infinite streams, lazy-loaded sources)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use machinelearne_rs::dataset::Dataset;
    /// # use machinelearne_rs::backend::{Backend, Tensor1D, Tensor2D};
    /// # struct MyDataset;
    /// # impl Dataset for MyDataset {
    /// #     type Error = ();
    /// #     type Item = ();
    /// #     fn len(&self) -> Option<usize> { Some(42) }
    /// #     fn get_batch<B: machinelearne_rs::backend::Backend>(&self, _: std::ops::Range<usize>) -> Result<(Tensor2D<B>, Tensor1D<B>), Self::Error> { unimplemented!() }
    /// # }
    /// let ds = MyDataset;
    /// assert_eq!(ds.len(), Some(42));
    /// ```
    fn len(&self) -> Option<usize>;

    /// Checks whether the dataset is empty.
    ///
    /// Default implementation checks if `len() == Some(0)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use machinelearne_rs::dataset::Dataset;
    /// # use machinelearne_rs::backend::{Backend, Tensor1D, Tensor2D};
    /// # struct EmptyDataset;
    /// # impl Dataset for EmptyDataset {
    /// #     type Error = ();
    /// #     type Item = ();
    /// #     fn len(&self) -> Option<usize> { Some(0) }
    /// #     fn get_batch<B: Backend>(&self, _: std::ops::Range<usize>) -> Result<(Tensor2D<B>, Tensor1D<B>), Self::Error> { unimplemented!() }
    /// # }
    /// let ds = EmptyDataset;
    /// assert!(ds.is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }

    /// Creates an iterator over fixed-size batches.
    ///
    /// # Parameters
    ///
    /// - `batch_size` — Desired batch size (last batch may be smaller)
    ///
    /// # Behavior
    ///
    /// - Returns batches of size `batch_size`, except possibly the last one
    /// - Iterator yields `Result` to propagate errors from [`get_batch`]
    /// - Requires `Sized` bound because the iterator holds a reference to `self`
    ///
    /// # Example
    ///
    /// ```rust
    /// # use machinelearne_rs::dataset::{Dataset, InMemoryDataset};
    /// # use machinelearne_rs::backend::CpuBackend;
    /// let x = vec![vec![1.0], vec![2.0]];
    /// let y = vec![0.0, 1.0];
    /// let ds = InMemoryDataset::new(x, y).unwrap();
    ///
    /// let batches: Vec<_> = ds.batches::<CpuBackend>(1).collect();
    /// assert_eq!(batches.len(), 2); // 2 batches of 1 sample each
    /// ```
    fn batches<'a, B: Backend>(&'a self, batch_size: usize) -> DatasetBatchIter<'a, B, Self>
    where
        Self: Sized,
    {
        DatasetBatchIter {
            dataset: self,
            batch_size,
            current: 0,
            _backend: std::marker::PhantomData,
        }
    }

    /// Loads a subset of data as tensors for the given index range.
    ///
    /// # Parameters
    ///
    /// - `range` — Sample index range `[start, end)`
    ///
    /// # Returns
    ///
    /// - `Ok((X, y))` — Feature matrix `(n, n_features)` and target vector `(n,)`
    /// - `Err(e)` — Data access error (e.g., out-of-bounds access)
    ///
    /// # Panics
    ///
    /// Does not panic if implemented correctly. Boundary checks are the implementor's responsibility.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use machinelearne_rs::dataset::Dataset;
    /// # use machinelearne_rs::backend::CpuBackend;
    /// # use machinelearne_rs::backend::{Backend, Tensor1D, Tensor2D};
    /// # struct MyDataset;
    /// # impl Dataset for MyDataset {
    /// #     type Error = String;
    /// #     type Item = ();
    /// #     fn len(&self) -> Option<usize> { Some(10) }
    /// #     fn get_batch<B: Backend>(&self, range: std::ops::Range<usize>) -> Result<(Tensor2D<B>, Tensor1D<B>), Self::Error> {
    /// #         Ok((Tensor2D::new(vec![0.0; (range.end - range.start) * 2], range.end - range.start, 2), Tensor1D::new(vec![0.0; range.end - range.start])))
    /// #     }
    /// # }
    /// let ds = MyDataset;
    /// let (x, y) = ds.get_batch::<CpuBackend>(0..5).unwrap();
    /// assert_eq!(x.shape().0, 5); // 5 samples
    /// ```
    fn get_batch<B: Backend>(
        &self,
        range: Range<usize>,
    ) -> Result<(Tensor2D<B>, Tensor1D<B>), Self::Error>;
}

/// Iterator over dataset batches.
///
/// Created by [`Dataset::batches`], yields consecutive batches of fixed size
/// (last batch may be smaller than requested).
///
/// # Type Parameters
///
/// - `'a` — Lifetime of the reference to the dataset
/// - `B` — Backend type for tensors (implements [`Backend`])
/// - `D` — Dataset type (implements [`Dataset`])
///
/// # Characteristics
///
/// - **Lazy loading**: Data is fetched only when `next()` is called
/// - **Error propagation**: Returns `Result` to forward errors from `get_batch`
/// - **Partial batches**: Last batch may contain fewer samples than `batch_size`
pub struct DatasetBatchIter<'a, B: Backend, D: ?Sized> {
    /// Reference to the source dataset.
    dataset: &'a D,
    /// Desired batch size (actual size may be smaller for the last batch).
    batch_size: usize,
    /// Current position in the dataset (index of next sample to yield).
    current: usize,
    /// Phantom marker for the backend type parameter.
    _backend: std::marker::PhantomData<B>,
}

impl<'a, B: Backend, D: Dataset> Iterator for DatasetBatchIter<'a, B, D> {
    /// Iterator item type: `Result` containing tensor pair or dataset error.
    type Item = Result<(Tensor2D<B>, Tensor1D<B>), D::Error>;

    /// Returns the next batch or `None` if all samples have been consumed.
    ///
    /// # Algorithm
    ///
    /// 1. Checks dataset size via `len()`; returns `None` if unknown
    /// 2. Returns `None` if `current >= total_samples`
    /// 3. Computes range `[current, min(current + batch_size, total))`
    /// 4. Calls `dataset.get_batch(range)` and returns the result wrapped in `Some`
    ///
    /// # Error Handling
    ///
    /// Errors from `get_batch` are propagated as `Some(Err(e))`. The iterator
    /// terminates only when all samples are consumed (`None`).
    fn next(&mut self) -> Option<Self::Item> {
        let total = self.dataset.len()?;
        if self.current >= total {
            return None;
        }

        let end = (self.current + self.batch_size).min(total);
        let range = self.current..end;
        self.current = end;

        // Fetch subset and convert to tensors
        match self.dataset.get_batch::<B>(range) {
            Ok((x, y)) => Some(Ok((x, y))),
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use std::ops::Range;

    // Mock dataset for iterator logic testing
    struct MockDataset {
        len: usize,
    }

    impl Dataset for MockDataset {
        type Error = &'static str;
        type Item = ();

        fn len(&self) -> Option<usize> {
            Some(self.len)
        }

        fn get_batch<B: Backend>(
            &self,
            range: Range<usize>,
        ) -> Result<(Tensor2D<B>, Tensor1D<B>), Self::Error> {
            if range.start >= self.len || range.end > self.len {
                return Err("range out of bounds");
            }

            let n = range.len();
            let start = range.start;

            // X: (n, 2) — unique values per sample: [start*2, start*2+1, ...]
            let x_data: Vec<f32> = (0..n * 2).map(|i| (start * 2 + i) as f32).collect();
            let x = Tensor2D::<B>::new(x_data, n, 2);

            // y: (n,) — sequential values starting from `start`
            let y_data: Vec<f32> = (start..range.end).map(|i| i as f32).collect();
            let y = Tensor1D::<B>::new(y_data);

            Ok((x, y))
        }
    }

    #[test]
    fn test_dataset_is_empty() {
        let empty = MockDataset { len: 0 };
        assert!(empty.is_empty());

        let non_empty = MockDataset { len: 1 };
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_batches_full() {
        let dataset = MockDataset { len: 6 };
        let mut iter = dataset.batches::<CpuBackend>(2);

        // Should yield 3 full batches
        for i in 0..3 {
            let batch = iter.next().unwrap().unwrap();
            let (x, y) = batch;
            assert_eq!(x.shape(), (2, 2));
            assert_eq!(y.to_vec(), vec![i as f64 * 2.0, i as f64 * 2.0 + 1.0]);
        }
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_batches_partial_last() {
        let dataset = MockDataset { len: 5 };
        let mut iter = dataset.batches::<CpuBackend>(2);

        // 2 full batches + 1 partial batch
        assert_eq!(iter.next().unwrap().unwrap().0.shape(), (2, 2));
        assert_eq!(iter.next().unwrap().unwrap().0.shape(), (2, 2));
        assert_eq!(iter.next().unwrap().unwrap().0.shape(), (1, 2));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_batches_larger_than_dataset() {
        let dataset = MockDataset { len: 3 };
        let mut iter = dataset.batches::<CpuBackend>(10);

        // Single batch covering entire dataset
        let batch = iter.next().unwrap().unwrap();
        assert_eq!(batch.0.shape(), (3, 2));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_batches_empty_dataset() {
        let dataset = MockDataset { len: 0 };
        let mut iter = dataset.batches::<CpuBackend>(2);
        assert!(iter.next().is_none());
    }
}
