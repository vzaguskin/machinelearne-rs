//! Training callbacks for monitoring and controlling the training loop.
//!
//! This module provides a callback-based architecture for hooking into
//! training events. Callbacks can be used for:
//! - Validation during training
//! - Checkpoint saving
//! - Logging and metrics tracking
//! - Learning rate scheduling
//! - Early stopping (custom implementations)
//!
//! # Example
//!
//! ```rust,ignore
//! use machinelearne_rs::callbacks::{Callback, TrainingState};
//!
//! struct MyCallback;
//!
//! impl<B, M> Callback<B, M> for MyCallback
//! where
//!     B: Backend,
//!     M: TrainableModel<B>,
//! {
//!     fn on_epoch_end(&mut self, state: &mut TrainingState<B, M>) {
//!         println!("Epoch {} loss: {}", state.epoch, state.loss);
//!     }
//! }
//! ```

use crate::backend::Backend;
use crate::model::TrainableModel;
use std::collections::HashMap;
use std::marker::PhantomData;

pub mod checkpoint;
pub mod logging;
pub mod validation;

pub use checkpoint::{
    CheckpointCallback, CheckpointMetadata, Checkpointable, MetricMode, SaveStrategy,
};
pub use logging::LoggingCallback;
pub use validation::ValidationCallback;

/// State information passed to callbacks during training.
///
/// Contains all relevant training information at the current point in time.
/// Some fields are mutable (via `mut` access in `on_epoch_end`) allowing
/// callbacks to modify training behavior.
pub struct TrainingState<'a, B: Backend, M: TrainableModel<B>> {
    /// Current epoch number (0-indexed).
    pub epoch: usize,
    /// Current batch number within the epoch (0-indexed).
    pub batch: usize,
    /// Total number of epochs to train.
    pub total_epochs: usize,
    /// Total number of batches per epoch.
    pub total_batches: usize,
    /// Current loss value.
    pub loss: f64,
    /// Reference to the model being trained.
    pub model: &'a M,
    /// Current learning rate.
    pub learning_rate: f64,
    /// Custom metrics collected during training (e.g., validation loss).
    pub metrics: HashMap<String, f64>,
    /// Flag to request training termination.
    pub stop_requested: bool,
    _phantom: PhantomData<B>,
}

impl<'a, B: Backend, M: TrainableModel<B>> TrainingState<'a, B, M> {
    /// Creates a new training state.
    pub fn new(
        epoch: usize,
        batch: usize,
        total_epochs: usize,
        total_batches: usize,
        loss: f64,
        model: &'a M,
        learning_rate: f64,
    ) -> Self {
        Self {
            epoch,
            batch,
            total_epochs,
            total_batches,
            loss,
            model,
            learning_rate,
            metrics: HashMap::new(),
            stop_requested: false,
            _phantom: PhantomData,
        }
    }

    /// Requests training to stop after the current epoch.
    pub fn request_stop(&mut self) {
        self.stop_requested = true;
    }

    /// Sets a metric value.
    pub fn set_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.insert(name.into(), value);
    }

    /// Gets a metric value, if present.
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }
}

/// Trait for training callbacks.
///
/// Callbacks hook into training events to monitor or modify training behavior.
/// All methods have empty default implementations, so you only need to implement
/// the ones you care about.
///
/// # Lifecycle
///
/// Callbacks are invoked in this order:
/// 1. `on_train_start` - Before training begins
/// 2. For each epoch:
///    a. `on_epoch_start` - Before epoch begins
///    b. For each batch:
///       - `on_batch_start` - Before batch processing
///       - `on_batch_end` - After batch processing
///         c. `on_epoch_end` - After epoch completes (mutable state access)
/// 3. `on_train_end` - After training completes
///
/// # Example
///
/// ```rust,ignore
/// use machinelearne_rs::callbacks::{Callback, TrainingState};
/// use machinelearne_rs::backend::CpuBackend;
/// use machinelearne_rs::model::LinearRegression;
///
/// struct LossLogger;
///
/// impl Callback<CpuBackend, LinearRegression<CpuBackend>> for LossLogger {
///     fn on_epoch_end(&mut self, state: &mut TrainingState<CpuBackend, LinearRegression<CpuBackend>>) {
///         println!("Epoch {}: loss = {:.6}", state.epoch + 1, state.loss);
///     }
/// }
/// ```
pub trait Callback<B: Backend, M: TrainableModel<B>> {
    /// Called once before training begins.
    ///
    /// Use this for setup operations like creating log files or
    /// initializing internal state.
    fn on_train_start(&mut self, _state: &TrainingState<B, M>) {}

    /// Called once after training completes.
    ///
    /// Use this for cleanup operations like flushing buffers or
    /// writing final summaries.
    fn on_train_end(&mut self, _state: &TrainingState<B, M>) {}

    /// Called at the start of each epoch.
    fn on_epoch_start(&mut self, _state: &TrainingState<B, M>) {}

    /// Called at the end of each epoch.
    ///
    /// This is the primary hook for most callbacks. The mutable state
    /// allows callbacks to request training termination via `state.request_stop()`.
    fn on_epoch_end(&mut self, _state: &mut TrainingState<B, M>) {}

    /// Called at the start of each batch.
    fn on_batch_start(&mut self, _state: &TrainingState<B, M>) {}

    /// Called at the end of each batch.
    fn on_batch_end(&mut self, _state: &mut TrainingState<B, M>) {}
}

/// A no-op callback that does nothing.
///
/// Useful as a placeholder or for testing.
pub struct NoopCallback;

impl<B: Backend, M: TrainableModel<B>> Callback<B, M> for NoopCallback {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::model::linear::LinearRegression;

    #[test]
    fn test_training_state_creation() {
        let model = LinearRegression::<CpuBackend>::new(2);
        let state = TrainingState::new(0, 5, 100, 10, 0.5, &model, 0.01);

        assert_eq!(state.epoch, 0);
        assert_eq!(state.batch, 5);
        assert_eq!(state.total_epochs, 100);
        assert_eq!(state.total_batches, 10);
        assert!((state.loss - 0.5).abs() < 1e-10);
        assert!((state.learning_rate - 0.01).abs() < 1e-10);
        assert!(!state.stop_requested);
    }

    #[test]
    fn test_training_state_metrics() {
        let model = LinearRegression::<CpuBackend>::new(2);
        let mut state = TrainingState::new(0, 0, 100, 10, 0.5, &model, 0.01);

        state.set_metric("val_loss", 0.3);
        state.set_metric("val_accuracy", 0.95);

        assert!((state.get_metric("val_loss").unwrap() - 0.3).abs() < 1e-10);
        assert!((state.get_metric("val_accuracy").unwrap() - 0.95).abs() < 1e-10);
        assert!(state.get_metric("nonexistent").is_none());
    }

    #[test]
    fn test_training_state_stop_request() {
        let model = LinearRegression::<CpuBackend>::new(2);
        let mut state = TrainingState::new(0, 0, 100, 10, 0.5, &model, 0.01);

        assert!(!state.stop_requested);
        state.request_stop();
        assert!(state.stop_requested);
    }

    #[test]
    fn test_noop_callback() {
        let model = LinearRegression::<CpuBackend>::new(2);
        let mut state = TrainingState::new(0, 0, 100, 10, 0.5, &model, 0.01);

        let mut callback = NoopCallback;
        // All methods should do nothing and not panic
        callback.on_train_start(&state);
        callback.on_epoch_start(&state);
        callback.on_batch_start(&mut state);
        callback.on_batch_end(&mut state);
        callback.on_epoch_end(&mut state);
        callback.on_train_end(&state);
    }
}
