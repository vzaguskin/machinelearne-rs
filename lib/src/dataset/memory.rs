//! In-memory dataset implementation for tabular data.
//!
//! Provides [`InMemoryDataset`], a simple dataset backed by `Vec<Vec<f32>>` for features
//! and `Vec<f32>` for targets. Suitable for small-to-medium datasets that fit entirely in RAM.
//!
//! # Design Philosophy
//!
//! Following the separation of concerns principle (see ADR: *separate-trainer-losses*):
//! - Datasets are **pure data containers** without training logic or hyperparameters
//! - Training loops, optimizers, and loss functions live in separate components
//! - Fitted models contain only inference parameters (`predict` method), not training state
//!
//! # Example
//!
//! ```rust
//! # use machinelearne_rs::dataset::memory::InMemoryDataset;
//! # use machinelearne_rs::backend::CpuBackend;
//! # use machinelearne_rs::dataset::Dataset;
//! # fn main() {
//! // Create dataset: 3 samples × 2 features
//! let x = vec![
//!     vec![1.0, 0.0],
//!     vec![0.0, 1.0],
//!     vec![1.0, 1.0],
//! ];
//! let y = vec![1.0, 0.0, 1.0];
//!
//! let dataset = InMemoryDataset::new(x, y).unwrap();
//!
//! // Iterate over batches
//! for batch in dataset.batches::<CpuBackend>(2) {
//!     let (x_batch, y_batch) = batch.unwrap();
//!     // ... use tensors for training/inference
//! }
//! # }
//! ```

use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::dataset::Dataset;
use std::ops::Range;

/// An in-memory dataset storing tabular data as nested vectors.
///
/// Holds features `X` as `Vec<Vec<f32>>` (rows × features) and targets `y` as `Vec<f32>`.
/// Validates structural invariants at construction time:
/// - Equal length of `X` and `y`
/// - Non-empty dataset
/// - Uniform feature dimensionality across all samples
///
/// # Memory Layout
///
/// Data is stored in row-major order (each inner `Vec` is a sample). When converted
/// to tensors via [`get_batch`], data is flattened into contiguous column-major layout
/// required by most ML backends.
///
/// # Thread Safety
///
/// `InMemoryDataset` is `Send + Sync` (via inherent `Vec` properties) and can be
/// safely shared across threads for read-only access during training.
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    /// Feature matrix: outer vector = samples, inner vector = features per sample.
    x: Vec<Vec<f32>>,
    /// Target vector: one value per sample.
    y: Vec<f32>,
}

impl InMemoryDataset {
    /// Constructs a validated in-memory dataset from feature and target vectors.
    ///
    /// # Validation
    ///
    /// Returns `Err` if any of the following conditions are violated:
    /// - `x.len() != y.len()` — mismatched sample counts
    /// - `x.is_empty()` — empty dataset (no samples)
    /// - Non-uniform feature dimensions — rows in `x` have different lengths
    ///
    /// # Parameters
    ///
    /// - `x`: Feature matrix where each inner vector represents one sample's features
    /// - `y`: Target values corresponding to each sample in `x`
    ///
    /// # Returns
    ///
    /// - `Ok(InMemoryDataset)` — validated dataset ready for training/inference
    /// - `Err(String)` — descriptive validation error
    ///
    /// # Example
    ///
    /// ```rust
    /// use machinelearne_rs::dataset::memory::InMemoryDataset;
    ///
    /// // Valid dataset: 2 samples × 2 features
    /// let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    /// let y = vec![0.0, 1.0];
    /// let dataset = InMemoryDataset::new(x, y).unwrap();
    ///
    /// // Invalid: mismatched lengths
    /// let x = vec![vec![1.0]];
    /// let y = vec![0.0, 1.0];
    /// assert!(InMemoryDataset::new(x, y).is_err());
    /// ```
    pub fn new(x: Vec<Vec<f32>>, y: Vec<f32>) -> Result<Self, String> {
        if x.len() != y.len() {
            return Err("x and y must have same length".into());
        }
        if x.is_empty() {
            return Err("Dataset is empty".into());
        }
        let n_features = x[0].len();
        if !x.iter().all(|row| row.len() == n_features) {
            return Err("All rows must have the same number of features".into());
        }
        Ok(Self { x, y })
    }
}

impl Dataset for InMemoryDataset {
    /// Error type for data access operations.
    ///
    /// Uses [`Infallible`](std::convert::Infallible) because:
    /// - Structural validation happens at construction time
    /// - Range checks are handled by slice indexing (panics on OOB, not recoverable errors)
    /// - No I/O operations that could fail at runtime
    type Error = std::convert::Infallible;

    /// Type of a single dataset item: `(feature_vector, target)`.
    type Item = (Vec<f32>, f32);

    /// Returns the exact number of samples in the dataset.
    ///
    /// Always `Some(n)` since in-memory datasets have known size.
    fn len(&self) -> Option<usize> {
        Some(self.x.len())
    }

    /// Loads a contiguous range of samples as backend-specific tensors.
    ///
    /// # Parameters
    ///
    /// - `range`: Sample index range `[start, end)` (half-open interval)
    ///
    /// # Returns
    ///
    /// - `Ok((X, y))` where:
    ///   - `X` is a `(batch_size, n_features)` tensor in column-major layout
    ///   - `y` is a `(batch_size,)` tensor
    ///
    /// # Panics
    ///
    /// Panics if `range` is out of bounds (caller should ensure valid ranges via `len()`).
    /// This is intentional: boundary checks belong to the iterator layer ([`DatasetBatchIter`]),
    /// not the dataset implementation itself.
    ///
    /// # Implementation Notes
    ///
    /// - Flattens row-major `Vec<Vec<f32>>` into contiguous column-major buffer
    /// - Preserves backend abstraction: tensors are constructed generically for any `B: Backend`
    fn get_batch<B: Backend>(
        &self,
        range: Range<usize>,
    ) -> Result<(Tensor2D<B>, Tensor1D<B>), Self::Error> {
        let batch_x = &self.x[range.clone()];
        let batch_y = &self.y[range];

        let batch_size = batch_x.len();
        let n_features = batch_x[0].len();

        let data = batch_x.iter().flat_map(|row| row.iter()).copied().collect();
        let x_tensor = Tensor2D::<B>::new(data, batch_size, n_features);

        let y_tensor = Tensor1D::<B>::new(batch_y.to_vec());

        Ok((x_tensor, y_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_in_memory_dataset_new_success() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y = vec![0.0, 1.0];
        let dataset = InMemoryDataset::new(x, y);
        assert!(dataset.is_ok());
    }

    #[test]
    fn test_in_memory_dataset_new_mismatched_lengths() {
        let x = vec![vec![1.0, 2.0]];
        let y = vec![0.0, 1.0];
        let dataset = InMemoryDataset::new(x, y);
        assert!(dataset.is_err());
        assert_eq!(dataset.unwrap_err(), "x and y must have same length");
    }

    #[test]
    fn test_in_memory_dataset_new_empty() {
        let x = vec![];
        let y = vec![];
        let dataset = InMemoryDataset::new(x, y);
        assert!(dataset.is_err());
        assert_eq!(dataset.unwrap_err(), "Dataset is empty");
    }

    #[test]
    fn test_in_memory_dataset_new_uneven_rows() {
        let x = vec![vec![1.0, 2.0], vec![3.0]]; // second row shorter
        let y = vec![0.0, 1.0];
        let dataset = InMemoryDataset::new(x, y);
        assert!(dataset.is_err());
        assert_eq!(
            dataset.unwrap_err(),
            "All rows must have the same number of features"
        );
    }

    #[test]
    fn test_in_memory_dataset_len() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();
        assert_eq!(dataset.len(), Some(2));
    }

    #[test]
    fn test_in_memory_dataset_batches_integration() {
        let x = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let y = vec![1.0, 0.0, 1.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let mut batches = dataset.batches::<CpuBackend>(2);
        let batch1 = batches.next().unwrap().unwrap();
        assert_eq!(batch1.0.shape(), (2, 2));
        assert_eq!(batch1.1.to_vec(), vec![1.0, 0.0]);

        let batch2 = batches.next().unwrap().unwrap();
        assert_eq!(batch2.0.shape(), (1, 2));
        assert_eq!(batch2.1.to_vec(), vec![1.0]);

        assert!(batches.next().is_none());
    }
}
