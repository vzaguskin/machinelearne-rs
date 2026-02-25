//! Validation callback for evaluating model during training.
//!
//! Runs validation on a separate dataset at specified intervals.

use crate::backend::scalar::ScalarOps;
use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::callbacks::{Callback, TrainingState};
use crate::dataset::memory::InMemoryDataset;
use crate::dataset::Dataset;
use crate::loss::Loss;
use crate::model::TrainableModel;
use std::marker::PhantomData;

/// Callback that runs validation on a separate dataset.
///
/// Validation runs every `frequency` epochs and stores results
/// in the training state's metrics hashmap.
///
/// # Type Parameters
/// - `B`: Backend type
/// - `L`: Loss function type
/// - `M`: Model type (must implement TrainableModel)
///
/// # Example
///
/// ```rust,ignore
/// use machinelearne_rs::callbacks::ValidationCallback;
/// use machinelearne_rs::loss::MSELoss;
///
/// let val_data = InMemoryDataset::new(val_x, val_y).unwrap();
/// let val_callback = ValidationCallback::new(val_data, MSELoss, 5); // Every 5 epochs
/// ```
pub struct ValidationCallback<B, L, M>
where
    B: Backend,
    L: Loss<B>,
    M: TrainableModel<B>,
{
    /// Validation dataset.
    val_dataset: InMemoryDataset,
    /// Loss function for validation.
    loss_fn: L,
    /// Run validation every N epochs.
    frequency: usize,
    /// Batch size for validation.
    batch_size: usize,
    _phantom: PhantomData<(B, M)>,
}

impl<B, L, M> ValidationCallback<B, L, M>
where
    B: Backend,
    L: Loss<B, Target = Tensor1D<B>, Prediction = M::Prediction>,
    M: TrainableModel<B, Input = Tensor2D<B>>,
{
    /// Creates a new validation callback.
    ///
    /// # Arguments
    /// * `val_dataset` - Dataset to use for validation
    /// * `loss_fn` - Loss function to compute validation loss
    /// * `frequency` - Run validation every N epochs
    pub fn new(val_dataset: InMemoryDataset, loss_fn: L, frequency: usize) -> Self {
        Self {
            val_dataset,
            loss_fn,
            frequency,
            batch_size: 32,
            _phantom: PhantomData,
        }
    }

    /// Sets the batch size for validation.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Computes the validation loss.
    fn compute_validation_loss(&self, model: &M) -> Option<f64> {
        let n_total = self.val_dataset.len()?;
        if n_total == 0 {
            return None;
        }

        let mut total_loss = 0.0;
        let mut n_samples = 0;

        for (batch_x, batch_y) in self.val_dataset.batches::<B>(self.batch_size).flatten() {
            let batch_size = batch_x.shape().0;
            let preds = model.forward(&batch_x);
            let batch_loss = self.loss_fn.loss(&preds, &batch_y);
            total_loss += batch_loss.data.to_f64() * batch_size as f64;
            n_samples += batch_size;
        }

        if n_samples > 0 {
            Some(total_loss / n_samples as f64)
        } else {
            None
        }
    }
}

impl<B, L, M> Callback<B, M> for ValidationCallback<B, L, M>
where
    B: Backend,
    L: Loss<B, Target = Tensor1D<B>, Prediction = M::Prediction>,
    M: TrainableModel<B, Input = Tensor2D<B>>,
{
    fn on_epoch_end(&mut self, state: &mut TrainingState<B, M>) {
        // Only run at specified frequency
        if self.frequency == 0 || !(state.epoch + 1).is_multiple_of(self.frequency) {
            return;
        }

        if let Some(val_loss) = self.compute_validation_loss(state.model) {
            state.set_metric("val_loss", val_loss);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::loss::MSELoss;
    use crate::model::linear::LinearRegression;

    #[test]
    fn test_validation_callback_creation() {
        let val_x = vec![vec![1.0], vec![2.0]];
        let val_y = vec![2.0, 4.0];
        let val_dataset = InMemoryDataset::new(val_x, val_y).unwrap();

        let callback: ValidationCallback<CpuBackend, MSELoss, LinearRegression<CpuBackend>> =
            ValidationCallback::new(val_dataset, MSELoss, 5);

        assert_eq!(callback.frequency, 5);
        assert_eq!(callback.batch_size, 32);
    }

    #[test]
    fn test_validation_callback_batch_size() {
        let val_x = vec![vec![1.0], vec![2.0]];
        let val_y = vec![2.0, 4.0];
        let val_dataset = InMemoryDataset::new(val_x, val_y).unwrap();

        let callback: ValidationCallback<CpuBackend, MSELoss, LinearRegression<CpuBackend>> =
            ValidationCallback::new(val_dataset, MSELoss, 5).with_batch_size(16);

        assert_eq!(callback.batch_size, 16);
    }
}
