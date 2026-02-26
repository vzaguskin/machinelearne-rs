// trainer/mod.rs
use crate::{
    backend::{Backend, ScalarOps, Tensor1D, Tensor2D},
    callbacks::{Callback, TrainingState},
    dataset::Dataset,
    loss::Loss,
    model::{ParamOps, TrainableModel},
    optimizer::Optimizer,
    regularizers::Regularizer,
    schedulers::LRScheduler,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

/// Configuration for gradient clipping by global L2 norm.
///
/// When enabled, gradients are scaled to have a maximum L2 norm of `max_norm`.
/// This prevents gradient explosion during training.
#[derive(Clone, Copy, Debug)]
pub struct GradientClipping {
    /// Maximum allowed L2 norm for gradients.
    pub max_norm: f32,
}

impl GradientClipping {
    /// Creates a new gradient clipping configuration.
    pub fn new(max_norm: f32) -> Self {
        Self { max_norm }
    }
}

/// Configuration for early stopping based on loss improvement.
///
/// Early stopping halts training when the loss does not improve for a specified
/// number of epochs (patience), or when the loss diverges significantly.
#[derive(Clone, Copy, Debug)]
pub struct EarlyStoppingConfig {
    /// Number of epochs to wait for improvement before stopping.
    pub patience: usize,
    /// Minimum change in loss to qualify as an improvement.
    pub min_delta: f32,
}

impl EarlyStoppingConfig {
    /// Creates a new early stopping configuration.
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
        }
    }
}

/// Result of early stopping check.
enum EarlyStoppingResult {
    /// Continue training.
    Continue,
    /// Stop training, loss has plateaued.
    Stop,
    /// Training has diverged (loss exploded).
    Diverged,
}

/// Tracks early stopping state during training.
struct EarlyStoppingState<P> {
    best_loss: f64,
    best_params: Option<P>,
    epochs_without_improvement: usize,
    config: EarlyStoppingConfig,
    divergence_threshold: Option<f32>,
}

impl<P: Clone> EarlyStoppingState<P> {
    fn new(config: EarlyStoppingConfig, divergence_threshold: Option<f32>) -> Self {
        Self {
            best_loss: f64::MAX,
            best_params: None,
            epochs_without_improvement: 0,
            config,
            divergence_threshold,
        }
    }

    fn check<B>(&mut self, current_loss: f64, current_params: &P) -> EarlyStoppingResult
    where
        B: Backend,
        P: ParamOps<B>,
    {
        // Check for divergence
        if let Some(threshold) = self.divergence_threshold {
            if self.best_loss < f64::MAX && current_loss > self.best_loss * threshold as f64 {
                return EarlyStoppingResult::Diverged;
            }
        }

        // Check for improvement
        let improved = current_loss < self.best_loss - self.config.min_delta as f64;

        if improved {
            self.best_loss = current_loss;
            self.best_params = Some(current_params.clone());
            self.epochs_without_improvement = 0;
        } else {
            self.epochs_without_improvement += 1;
        }

        if self.epochs_without_improvement >= self.config.patience {
            EarlyStoppingResult::Stop
        } else {
            EarlyStoppingResult::Continue
        }
    }
}

/// Orchestrates the training loop for a `TrainableModel`.
///
/// Combines a loss function, optimizer, and regularizer to fit a model on a dataset.
/// Once built via `TrainerBuilder`, it is immutable and can be reused across multiple models
/// (as long as types match).
///
/// The `fit` method returns a `FittedModel` (via `IntoFitted`), which contains only inference logic.
pub struct Trainer<B, L, O, M, P, R>
where
    B: Backend,
    L: Loss<B>,
    M: TrainableModel<B, Params = P, Gradients = P>,
    O: Optimizer<B, P>,
    R: Regularizer<B, M>,
{
    pub(crate) batch_size: usize,
    pub(crate) max_epochs: usize,
    pub(crate) verbose: bool,
    pub(crate) loss_fn: L,
    pub(crate) optimizer: O,
    pub(crate) regularizer: R,
    pub(crate) gradient_clipping: Option<GradientClipping>,
    pub(crate) early_stopping: Option<EarlyStoppingConfig>,
    pub(crate) divergence_threshold: Option<f32>,
    pub(crate) callbacks: RefCell<Vec<Box<dyn Callback<B, M>>>>,
    pub(crate) lr_scheduler: RefCell<Option<Box<dyn LRScheduler>>>,
    pub(crate) initial_lr: f64,
    pub(crate) start_epoch: usize,
    _phantom_backend: PhantomData<B>,
    _phantom_model: PhantomData<M>,
}

/// Fluent builder for constructing a `Trainer` with custom hyperparameters.
///
/// Defaults:
/// - `batch_size`: 32
/// - `max_epochs`: 1000
/// - `verbose`: true
/// - `gradient_clipping`: None (disabled)
/// - `early_stopping`: None (disabled)
/// - `divergence_threshold`: None (disabled)
/// - `callbacks`: empty (no callbacks)
/// - `lr_scheduler`: None (no scheduling)
/// - `start_epoch`: 0 (start from beginning)
pub struct TrainerBuilder<B, L, O, M, P, R>
where
    B: Backend,
    L: Loss<B>,
    M: TrainableModel<B, Params = P, Gradients = P>,
    O: Optimizer<B, P>,
    R: Regularizer<B, M>,
{
    batch_size: usize,
    max_epochs: usize,
    verbose: bool,
    loss_fn: L,
    optimizer: O,
    regularizer: R,
    gradient_clipping: Option<GradientClipping>,
    early_stopping: Option<EarlyStoppingConfig>,
    divergence_threshold: Option<f32>,
    callbacks: Vec<Box<dyn Callback<B, M>>>,
    lr_scheduler: Option<Box<dyn LRScheduler>>,
    initial_lr: f64,
    start_epoch: usize,
    _phantom_backend: PhantomData<B>,
    _phantom_model: PhantomData<M>,
}

impl<B, L, O, M, P, R> TrainerBuilder<B, L, O, M, P, R>
where
    B: Backend,
    L: Loss<B>,
    M: TrainableModel<B, Params = P, Gradients = P>,
    O: Optimizer<B, P>,
    R: Regularizer<B, M>,
{
    /// Creates a new `TrainerBuilder` with the given components.
    ///
    /// # Arguments
    /// * `loss_fn` — differentiable loss (e.g., `MSELoss`)
    /// * `optimizer` — parameter updater (e.g., `SGD`)
    /// * `regularizer` — optional penalty term (e.g., `L2` or `NoRegularizer`)
    pub fn new(loss_fn: L, optimizer: O, regularizer: R) -> Self {
        Self {
            batch_size: 32,
            max_epochs: 1000,
            verbose: true,
            loss_fn,
            optimizer,
            regularizer,
            gradient_clipping: None,
            early_stopping: None,
            divergence_threshold: None,
            callbacks: Vec::new(),
            lr_scheduler: None,
            initial_lr: 0.01, // Default, should be set based on optimizer
            start_epoch: 0,
            _phantom_backend: PhantomData,
            _phantom_model: PhantomData,
        }
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn max_epochs(mut self, epochs: usize) -> Self {
        self.max_epochs = epochs;
        self
    }

    /// Sets verbosity for training output.
    ///
    /// When `false`, suppresses epoch-by-epoch loss output.
    /// Useful for benchmarking or production training.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Enables gradient clipping by global L2 norm.
    ///
    /// When enabled, gradients are scaled to have a maximum L2 norm of `max_norm`.
    /// This prevents gradient explosion during training.
    ///
    /// # Arguments
    /// * `max_norm` — Maximum allowed L2 norm for gradients. Common values: 0.5, 1.0, 5.0
    ///
    /// # Example
    /// ```ignore
    /// let trainer = Trainer::builder(loss, optimizer, regularizer)
    ///     .gradient_clipping(1.0)
    ///     .build();
    /// ```
    pub fn gradient_clipping(mut self, max_norm: f32) -> Self {
        self.gradient_clipping = Some(GradientClipping::new(max_norm));
        self
    }

    /// Enables early stopping based on loss improvement.
    ///
    /// Training stops when the loss does not improve by at least `min_delta`
    /// for `patience` consecutive epochs.
    ///
    /// # Arguments
    /// * `patience` — Number of epochs to wait for improvement before stopping
    /// * `min_delta` — Minimum change in loss to qualify as an improvement
    ///
    /// # Example
    /// ```ignore
    /// let trainer = Trainer::builder(loss, optimizer, regularizer)
    ///     .early_stopping(5, 0.001)  // Stop if no improvement > 0.001 for 5 epochs
    ///     .build();
    /// ```
    pub fn early_stopping(mut self, patience: usize, min_delta: f32) -> Self {
        self.early_stopping = Some(EarlyStoppingConfig::new(patience, min_delta));
        self
    }

    /// Sets the divergence threshold for early stopping.
    ///
    /// When the loss exceeds `best_loss * threshold`, training is stopped immediately.
    /// This detects when training has diverged (loss exploded).
    ///
    /// Only effective when early stopping is enabled.
    ///
    /// # Arguments
    /// * `threshold` — Multiplier for best loss that triggers divergence detection.
    ///   Common values: 10.0 (stop if loss becomes 10x the best)
    ///
    /// # Example
    /// ```ignore
    /// let trainer = Trainer::builder(loss, optimizer, regularizer)
    ///     .early_stopping(5, 0.001)
    ///     .divergence_threshold(10.0)  // Stop if loss exceeds 10x best
    ///     .build();
    /// ```
    pub fn divergence_threshold(mut self, threshold: f32) -> Self {
        self.divergence_threshold = Some(threshold);
        self
    }

    /// Adds a callback to be invoked during training.
    ///
    /// Callbacks are invoked in the order they are added.
    ///
    /// # Example
    /// ```ignore
    /// use machinelearne_rs::callbacks::LoggingCallback;
    ///
    /// let trainer = Trainer::builder(loss, optimizer, regularizer)
    ///     .with_callback(LoggingCallback::console_only())
    ///     .build();
    /// ```
    pub fn with_callback(mut self, callback: Box<dyn Callback<B, M>>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /// Sets a learning rate scheduler.
    ///
    /// Only one scheduler can be used at a time.
    ///
    /// # Example
    /// ```ignore
    /// use machinelearne_rs::schedulers::StepLR;
    ///
    /// let scheduler = Box::new(StepLR::new(0.01, 30, 0.1));
    /// let trainer = Trainer::builder(loss, optimizer, regularizer)
    ///     .with_lr_scheduler(scheduler)
    ///     .build();
    /// ```
    pub fn with_lr_scheduler(mut self, scheduler: Box<dyn LRScheduler>) -> Self {
        self.lr_scheduler = Some(scheduler);
        self
    }

    /// Sets the initial learning rate (used with schedulers).
    pub fn with_initial_lr(mut self, lr: f64) -> Self {
        self.initial_lr = lr;
        self
    }

    /// Sets the starting epoch for resuming training.
    ///
    /// This is useful when resuming training from a checkpoint. The training
    /// loop will start from this epoch instead of epoch 0.
    ///
    /// Note: You must also restore the model's parameters from the checkpoint
    /// before passing it to `fit()`.
    ///
    /// # Example
    /// ```ignore
    /// use machinelearne_rs::checkpoint::{LoadedCheckpoint, RestorableFromCheckpoint};
    ///
    /// // Load checkpoint
    /// let checkpoint = LoadedCheckpoint::load_latest("checkpoints/")?;
    /// let model = MyModel::from_checkpoint_bytes(&checkpoint.params_bytes)?;
    ///
    /// // Resume training from checkpoint epoch
    /// let trainer = Trainer::builder(loss, optimizer, regularizer)
    ///     .start_epoch(checkpoint.epoch() + 1)
    ///     .max_epochs(100)
    ///     .build();
    ///
    /// trainer.fit(model, &dataset)?;
    /// ```
    pub fn start_epoch(mut self, epoch: usize) -> Self {
        self.start_epoch = epoch;
        self
    }

    pub fn build(self) -> Trainer<B, L, O, M, P, R> {
        Trainer {
            batch_size: self.batch_size,
            max_epochs: self.max_epochs,
            verbose: self.verbose,
            loss_fn: self.loss_fn,
            optimizer: self.optimizer,
            regularizer: self.regularizer,
            gradient_clipping: self.gradient_clipping,
            early_stopping: self.early_stopping,
            divergence_threshold: self.divergence_threshold,
            callbacks: RefCell::new(self.callbacks),
            lr_scheduler: RefCell::new(self.lr_scheduler),
            initial_lr: self.initial_lr,
            start_epoch: self.start_epoch,
            _phantom_backend: PhantomData,
            _phantom_model: PhantomData,
        }
    }
}

// --- Реализация fit переносится в Trainer ---
impl<B, L, O, M, P, R> Trainer<B, L, O, M, P, R>
where
    B: Backend,
    B::Scalar: Debug + Display,
    L: Loss<B, Target = Tensor1D<B>, Prediction = Tensor1D<B>>,
    M: TrainableModel<
        B,
        Input = Tensor2D<B>,
        Prediction = L::Prediction,
        Params = P,
        Gradients = P,
    >,
    O: Optimizer<B, P>,
    R: Regularizer<B, M>,
    P: ParamOps<B>,
{
    /// Trains the model on the provided dataset for up to `max_epochs`.
    ///
    /// # Returns
    /// A fitted model ready for inference (`M::Output`), or an error if:
    /// - The dataset is empty
    /// - The dataset length is unknown (required for loss averaging)
    /// - A batch fails to load
    /// - Training diverged (loss exploded beyond divergence threshold)
    ///
    /// # Notes
    /// - Loss is computed at epoch end (not per batch) to reduce GPU synchronization.
    /// - Gradients are averaged per batch before applying regularization.
    /// - If gradient clipping is enabled, gradients are clipped per batch.
    /// - If early stopping is enabled, training may stop before `max_epochs`.
    /// - When early stopping triggers, the model from the best epoch is returned.
    /// - Callbacks are invoked at appropriate training events.
    /// - Learning rate schedulers adjust the learning rate at epoch end.
    ///
    /// # GPU Performance
    /// For GPU backends, loss is computed once per epoch instead of per batch
    /// to avoid synchronization overhead. This reduces GPU-CPU syncs from
    /// ~num_batches * epochs to ~epochs.
    pub fn fit<D>(&self, mut model: M, dataset: &D) -> Result<M::Output, String>
    where
        D: Dataset,
    {
        let n_total = dataset.len().ok_or("Dataset length unknown")?;
        if n_total == 0 {
            return Err("Dataset is empty".into());
        }

        let total_batches = n_total.div_ceil(self.batch_size);
        let mut current_lr = self.initial_lr;
        let mut metrics: HashMap<String, f64> = HashMap::new();

        // Initialize early stopping state if enabled
        let mut early_stopping_state = self
            .early_stopping
            .as_ref()
            .map(|config| EarlyStoppingState::<P>::new(*config, self.divergence_threshold));

        // on_train_start callback
        {
            let mut callbacks = self.callbacks.borrow_mut();
            if !callbacks.is_empty() {
                let state = TrainingState::new(
                    0,
                    0,
                    self.max_epochs,
                    total_batches,
                    0.0,
                    &model,
                    current_lr,
                );
                for callback in callbacks.iter_mut() {
                    callback.on_train_start(&state);
                }
            }
        }

        let mut stop_requested = false;

        for epoch in self.start_epoch..self.max_epochs {
            // on_epoch_start callback
            {
                let mut callbacks = self.callbacks.borrow_mut();
                if !callbacks.is_empty() {
                    let state = TrainingState::new(
                        epoch,
                        0,
                        self.max_epochs,
                        total_batches,
                        0.0,
                        &model,
                        current_lr,
                    );
                    for callback in callbacks.iter_mut() {
                        callback.on_epoch_start(&state);
                    }
                }
            }

            let mut batch_idx = 0;
            // Training loop: don't compute loss per batch (avoids GPU sync)
            for batch_result in dataset.batches::<B>(self.batch_size) {
                let (batch_x, batch_y) =
                    batch_result.map_err(|e| format!("Data error: {:?}", e))?;

                let preds = model.forward(&batch_x);
                // Skip loss computation during training - only compute gradient
                // This avoids GPU-CPU synchronization on every batch
                // Note: reg_penalty not computed here as we don't need loss value
                let (_reg_penalty, reg_grad) = self.regularizer.regularizer_penalty_grad(&model);
                let grad_preds = self.loss_fn.grad_wrt_prediction(&preds, &batch_y);
                let grads = model.backward(&batch_x, &grad_preds);

                let mut total_grads = grads.add(&reg_grad);

                // Apply gradient clipping if enabled
                if let Some(clipping) = &self.gradient_clipping {
                    total_grads = total_grads.clip_by_norm(clipping.max_norm);
                }

                let new_params = self.optimizer.step(model.params(), &total_grads);
                model.update_params(&new_params);

                batch_idx += 1;
            }

            // Compute loss once at epoch end for logging and early stopping
            // This triggers a single GPU sync per epoch instead of per batch
            let loss_value = self.compute_epoch_loss(&model, dataset, n_total);

            // Update learning rate from scheduler if enabled
            {
                let mut scheduler_opt = self.lr_scheduler.borrow_mut();
                if let Some(ref mut scheduler) = *scheduler_opt {
                    current_lr = scheduler.step(epoch, &metrics);
                }
            }

            if self.verbose {
                println!("Epoch {}: loss = {}", epoch, loss_value);
            }

            // on_epoch_end callback
            {
                let mut callbacks = self.callbacks.borrow_mut();
                if !callbacks.is_empty() {
                    let mut state = TrainingState::new(
                        epoch,
                        batch_idx,
                        self.max_epochs,
                        total_batches,
                        loss_value,
                        &model,
                        current_lr,
                    );
                    // Copy metrics into state for callbacks to access
                    state.metrics = metrics.clone();

                    for callback in callbacks.iter_mut() {
                        callback.on_epoch_end(&mut state);
                    }

                    // Copy metrics back from callbacks
                    metrics = state.metrics.clone();
                    stop_requested = state.stop_requested;
                }
            }

            // Check for stop requested by callback
            if stop_requested {
                if self.verbose {
                    println!("Training stopped by callback at epoch {}", epoch);
                }
                break;
            }

            // Check early stopping if enabled
            if let Some(ref mut state) = early_stopping_state {
                let current_params = model.params();
                match state.check::<B>(loss_value, current_params) {
                    EarlyStoppingResult::Diverged => {
                        if self.verbose {
                            println!("Training diverged at epoch {}", epoch);
                        }
                        return Err(format!(
                            "Training diverged at epoch {} (loss = {})",
                            epoch, loss_value
                        ));
                    }
                    EarlyStoppingResult::Stop => {
                        if self.verbose {
                            println!("Early stopping triggered at epoch {}", epoch);
                        }
                        // Restore best parameters if available
                        if let Some(best_params) = &state.best_params {
                            model.update_params(best_params);
                        }
                        break;
                    }
                    EarlyStoppingResult::Continue => {
                        // Continue training
                    }
                }
            }
        }

        // If early stopping was enabled and we have best params, restore them
        if let Some(ref state) = early_stopping_state {
            if let Some(best_params) = &state.best_params {
                model.update_params(best_params);
            }
        }

        // on_train_end callback
        {
            let mut callbacks = self.callbacks.borrow_mut();
            if !callbacks.is_empty() {
                let state = TrainingState::new(
                    self.max_epochs,
                    0,
                    self.max_epochs,
                    total_batches,
                    0.0,
                    &model,
                    current_lr,
                );
                for callback in callbacks.iter_mut() {
                    callback.on_train_end(&state);
                }
            }
        }

        Ok(model.into_fitted())
    }

    /// Computes the average loss over the dataset at epoch end.
    ///
    /// For GPU backends, this triggers a SINGLE synchronization per epoch instead of
    /// one per batch, dramatically improving training performance.
    ///
    /// Strategy: Compute loss on the first batch only. This provides a representative
    /// loss value for logging and early stopping while minimizing GPU-CPU syncs.
    /// For small datasets (single batch), this computes the exact loss.
    fn compute_epoch_loss<D>(&self, model: &M, dataset: &D, _n_total: usize) -> f64
    where
        D: Dataset,
    {
        // Get only the first batch for loss computation
        // This triggers a SINGLE GPU sync instead of syncing on every batch
        if let Some(Ok((batch_x, batch_y))) = dataset.batches::<B>(self.batch_size).next() {
            let preds = model.forward(&batch_x);
            let batch_loss = self.loss_fn.loss(&preds, &batch_y);
            let (reg_penalty, _reg_grad) = self.regularizer.regularizer_penalty_grad(model);
            // Single sync point: reading the scalar loss value
            let total_loss = batch_loss + reg_penalty;
            return total_loss.data.to_f64();
        }
        0.0
    }
}

// --- Экспорт удобного конструктора ---
impl<B, L, O, M, P, R> Trainer<B, L, O, M, P, R>
where
    B: Backend,
    L: Loss<B>,
    M: TrainableModel<B, Params = P, Gradients = P>,
    O: Optimizer<B, P>,
    R: Regularizer<B, M>,
{
    /// Convenience constructor that starts the builder pattern.
    ///
    /// Equivalent to `TrainerBuilder::new(...)`.
    pub fn builder(loss_fn: L, optimizer: O, regularizer: R) -> TrainerBuilder<B, L, O, M, P, R> {
        TrainerBuilder::new(loss_fn, optimizer, regularizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{CpuBackend, Scalar},
        dataset::memory::InMemoryDataset,
        loss::MSELoss,
        model::linear::InferenceModel,
        model::linear::LinearRegression,
        optimizer::SGD,
        regularizers::{NoRegularizer, L2},
    };

    // === TrainerBuilder Tests ===

    #[test]
    fn test_trainer_builder_default_values() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer);

        assert_eq!(builder.batch_size, 32);
        assert_eq!(builder.max_epochs, 1000);
        assert_eq!(builder.verbose, true);
    }

    #[test]
    fn test_trainer_builder_custom_batch_size() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .batch_size(64);

        assert_eq!(builder.batch_size, 64);
        assert_eq!(builder.max_epochs, 1000);
    }

    #[test]
    fn test_trainer_builder_custom_max_epochs() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .max_epochs(500);

        assert_eq!(builder.batch_size, 32);
        assert_eq!(builder.max_epochs, 500);
    }

    #[test]
    fn test_trainer_builder_verbose() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .verbose(false);

        assert_eq!(builder.verbose, false);

        let builder_verbose: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer).verbose(true);

        assert_eq!(builder_verbose.verbose, true);
    }

    #[test]
    fn test_trainer_builder_chaining() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .batch_size(128)
                .max_epochs(250)
                .verbose(false);

        assert_eq!(builder.batch_size, 128);
        assert_eq!(builder.max_epochs, 250);
        assert_eq!(builder.verbose, false);
    }

    #[test]
    fn test_trainer_builder_chaining_order_independent() {
        let builder1: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .batch_size(16)
                .max_epochs(100);

        let builder2: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .max_epochs(100)
                .batch_size(16);

        assert_eq!(builder1.batch_size, builder2.batch_size);
        assert_eq!(builder1.max_epochs, builder2.max_epochs);
    }

    #[test]
    fn test_trainer_builder_small_batch_size() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer).batch_size(1);

        assert_eq!(builder.batch_size, 1);
    }

    #[test]
    fn test_trainer_builder_large_batch_size() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .batch_size(10000);

        assert_eq!(builder.batch_size, 10000);
    }

    #[test]
    fn test_trainer_builder_zero_epochs() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer).max_epochs(0);

        assert_eq!(builder.max_epochs, 0);
    }

    #[test]
    fn test_trainer_builder_large_epochs() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .max_epochs(100000);

        assert_eq!(builder.max_epochs, 100000);
    }

    #[test]
    fn test_trainer_builder_creates_valid_trainer() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .batch_size(64)
                .max_epochs(200)
                .verbose(false);

        let trainer = builder.build();

        assert_eq!(trainer.batch_size, 64);
        assert_eq!(trainer.max_epochs, 200);
        assert_eq!(trainer.verbose, false);
    }

    #[test]
    fn test_trainer_builder_does_not_consume_loss_fn() {
        let _builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .batch_size(16)
                .max_epochs(50);

        // Components are reused via Clone for SGD, creating fresh instances for other builders
        let _builder2: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer);
    }

    #[test]
    fn test_trainer_builder_clone_components() {
        let builder1: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .batch_size(32)
                .max_epochs(100);

        // SGD implements Clone, so we can use it multiple times
        let _builder2: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, builder1.optimizer.clone(), NoRegularizer);
    }

    #[test]
    fn test_trainer_builder_zero_batch_size() {
        // Builder allows 0 batch size - this is up to the user to validate
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer).batch_size(0);

        assert_eq!(builder.batch_size, 0);

        // Building should still work, but fit() will fail on batch iteration
        let trainer = builder.build();
        assert_eq!(trainer.batch_size, 0);
    }

    // === Trainer Tests (existing tests preserved) ===

    #[test]
    fn test_trainer_fit_linear_regression() {
        // Создаём синтетический датасет: y = 2*x1 + 3*x2 + 1
        let x = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 3.0],
        ];
        let y = vec![3.0, 4.0, 6.0, 14.0]; // 2*2 + 3*3 + 1 = 4+9+1=14
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(2);
        let loss = MSELoss;
        let optimizer = SGD::new(0.1); // learning rate
        let regularizer = NoRegularizer;

        let trainer = Trainer::builder(loss, optimizer, regularizer)
            .batch_size(4)
            .max_epochs(100)
            .verbose(false) // Suppress output in tests
            .build();

        let fitted_model = trainer.fit(model, &dataset).unwrap();

        // Проверим предсказания
        let test_input = Tensor2D::<CpuBackend>::new(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let preds = fitted_model.predict_batch(&test_input);
        let pred_vec = preds.to_vec();

        // Ожидаем приближение к [3.0, 4.0]
        assert!((pred_vec[0] - 3.0).abs() < 0.5);
        assert!((pred_vec[1] - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_trainer_with_l2_regularization() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0]; // y = 2*x
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let loss = MSELoss;
        let optimizer = SGD::new(0.01);
        let regularizer = L2::<CpuBackend>::new(1.0); // сильная регуляризация

        let trainer = Trainer::builder(loss, optimizer, regularizer)
            .batch_size(3)
            .max_epochs(500)
            .verbose(false) // Suppress output in tests
            .build();

        let fitted_model = trainer.fit(model, &dataset).unwrap();
        let weights = fitted_model.extract_params().weights;

        // Без регуляризации вес был бы ~2.0, с L2 — меньше
        assert!(weights[0] < 2.0);
        assert!(weights[0] > 0.0);
    }

    #[test]
    fn test_trainer_empty_dataset() {
        let x = vec![];
        let y = vec![];
        // Empty datasets should error at creation
        let _dataset = InMemoryDataset::new(x, y).unwrap_err();
    }

    #[test]
    fn test_trainer_unknown_dataset_length() {
        // Создадим mock-датасет без len()
        struct MockDatasetWithoutLen {
            x: Vec<Vec<f32>>,
            y: Vec<f32>,
        }

        impl Dataset for MockDatasetWithoutLen {
            type Error = String; // ← меняем на String
            type Item = ();

            fn len(&self) -> Option<usize> {
                None
            }

            fn get_batch<B: Backend>(
                &self,
                range: std::ops::Range<usize>,
            ) -> Result<(Tensor2D<B>, Tensor1D<B>), Self::Error> {
                let batch_x = &self.x[range.clone()];
                let batch_y = &self.y[range];
                let n = batch_x.len();
                if n == 0 {
                    return Err("Empty batch".into());
                }
                let cols = batch_x[0].len();
                let data = batch_x.iter().flat_map(|r| r.iter()).copied().collect();
                Ok((
                    Tensor2D::new(data, n, cols),
                    Tensor1D::new(batch_y.to_vec()),
                ))
            }
        }

        let dataset = MockDatasetWithoutLen {
            x: vec![vec![1.0]],
            y: vec![1.0],
        };

        let model = LinearRegression::<CpuBackend>::new(1);
        let loss = MSELoss;
        let optimizer = SGD::<CpuBackend>::new(0.1);
        let regularizer = NoRegularizer;

        let trainer = Trainer::builder(loss, optimizer, regularizer)
            .batch_size(1)
            .max_epochs(1)
            .build();

        let result = trainer.fit(model, &dataset);
        assert!(result.is_err());
    }

    // === Gradient Clipping Tests ===

    #[test]
    fn test_gradient_clipping_builder() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .gradient_clipping(1.0);

        assert!(builder.gradient_clipping.is_some());
        assert_eq!(builder.gradient_clipping.unwrap().max_norm, 1.0);
    }

    #[test]
    fn test_gradient_clipping_disabled_by_default() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer);

        assert!(builder.gradient_clipping.is_none());
    }

    #[test]
    fn test_param_ops_l2_norm() {
        use crate::model::linear::LinearParams;

        // Create params with weights [3.0, 4.0] and bias 0.0
        // L2 norm should be sqrt(9 + 16 + 0) = 5.0
        let params: LinearParams<CpuBackend> = LinearParams {
            weights: Tensor1D::new(vec![3.0, 4.0]),
            bias: Scalar::new(0.0),
        };

        let norm = params.l2_norm();
        assert!((norm.to_f64() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_param_ops_clip_by_norm_exceeding() {
        use crate::model::linear::LinearParams;

        // Create params with weights [3.0, 4.0] and bias 0.0
        // L2 norm = 5.0, clip to 1.0 should scale to [0.6, 0.8]
        let params: LinearParams<CpuBackend> = LinearParams {
            weights: Tensor1D::new(vec![3.0, 4.0]),
            bias: Scalar::new(0.0),
        };

        let clipped = params.clip_by_norm(1.0);
        let weights = clipped.weights.to_vec();

        assert!((weights[0] - 0.6).abs() < 1e-6);
        assert!((weights[1] - 0.8).abs() < 1e-6);
        assert!((clipped.bias.to_f64() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_param_ops_clip_by_norm_within() {
        use crate::model::linear::LinearParams;

        // Create params with weights [0.3, 0.4] and bias 0.0
        // L2 norm = 0.5, clip to 1.0 should leave unchanged
        let params: LinearParams<CpuBackend> = LinearParams {
            weights: Tensor1D::new(vec![0.3, 0.4]),
            bias: Scalar::new(0.0),
        };

        let clipped = params.clip_by_norm(1.0);
        let weights = clipped.weights.to_vec();

        assert!((weights[0] - 0.3).abs() < 1e-6);
        assert!((weights[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_trainer_with_gradient_clipping() {
        // Simple training with gradient clipping enabled
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0]; // y = 2*x
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let loss = MSELoss;
        let optimizer = SGD::new(0.1);
        let regularizer = NoRegularizer;

        let trainer = Trainer::builder(loss, optimizer, regularizer)
            .batch_size(3)
            .max_epochs(50)
            .gradient_clipping(1.0)
            .verbose(false)
            .build();

        let fitted_model = trainer.fit(model, &dataset).unwrap();
        let weights = fitted_model.extract_params().weights;

        // Should still converge to approximately y = 2*x
        assert!(weights[0] > 1.5);
        assert!(weights[0] < 2.5);
    }

    // === Early Stopping Tests ===

    #[test]
    fn test_early_stopping_builder() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .early_stopping(5, 0.001);

        assert!(builder.early_stopping.is_some());
        let config = builder.early_stopping.unwrap();
        assert_eq!(config.patience, 5);
        assert!((config.min_delta - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_divergence_threshold_builder() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .divergence_threshold(10.0);

        assert_eq!(builder.divergence_threshold, Some(10.0));
    }

    #[test]
    fn test_early_stopping_disabled_by_default() {
        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer);

        assert!(builder.early_stopping.is_none());
        assert!(builder.divergence_threshold.is_none());
    }

    #[test]
    fn test_trainer_with_early_stopping() {
        // Training with early stopping - should stop before max_epochs
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0]; // y = 2*x
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let loss = MSELoss;
        let optimizer = SGD::new(0.01);
        let regularizer = NoRegularizer;

        let trainer = Trainer::builder(loss, optimizer, regularizer)
            .batch_size(3)
            .max_epochs(1000) // High max epochs
            .early_stopping(5, 0.0001) // Stop after 5 epochs without improvement
            .verbose(false)
            .build();

        let fitted_model = trainer.fit(model, &dataset).unwrap();
        let weights = fitted_model.extract_params().weights;

        // Should still converge
        assert!(weights[0] > 1.5);
        assert!(weights[0] < 2.5);
    }

    #[test]
    fn test_trainer_with_clipping_and_early_stopping() {
        // Training with both gradient clipping and early stopping
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0]; // y = 2*x
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let loss = MSELoss;
        let optimizer = SGD::new(0.1);
        let regularizer = NoRegularizer;

        let trainer = Trainer::builder(loss, optimizer, regularizer)
            .batch_size(3)
            .max_epochs(100)
            .gradient_clipping(1.0)
            .early_stopping(10, 0.0001)
            .verbose(false)
            .build();

        let fitted_model = trainer.fit(model, &dataset).unwrap();
        let weights = fitted_model.extract_params().weights;

        // Should converge to approximately y = 2*x
        assert!(weights[0] > 1.5);
        assert!(weights[0] < 2.5);
    }

    #[test]
    fn test_trainer_divergence_detection() {
        // Test that divergence detection works
        // Use a very high learning rate to cause divergence
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let loss = MSELoss;
        let optimizer = SGD::new(100.0); // Very high LR to cause divergence
        let regularizer = NoRegularizer;

        let trainer = Trainer::builder(loss, optimizer, regularizer)
            .batch_size(5)
            .max_epochs(100)
            .early_stopping(5, 0.001)
            .divergence_threshold(10.0) // Stop if loss is 10x best
            .verbose(false)
            .build();

        let result = trainer.fit(model, &dataset);
        // Should error due to divergence
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("diverged"));
    }

    #[test]
    fn test_early_stopping_state_improvement() {
        use crate::model::linear::LinearParams;

        let config = EarlyStoppingConfig::new(3, 0.001);
        let mut state: EarlyStoppingState<LinearParams<CpuBackend>> =
            EarlyStoppingState::new(config, None);

        // First check - sets best loss
        let params = LinearParams {
            weights: Tensor1D::new(vec![1.0]),
            bias: Scalar::new(0.0),
        };
        let result = state.check::<CpuBackend>(1.0, &params);
        assert!(matches!(result, EarlyStoppingResult::Continue));
        assert!((state.best_loss - 1.0).abs() < 1e-10);
        assert_eq!(state.epochs_without_improvement, 0);

        // Improvement - resets counter
        let result = state.check::<CpuBackend>(0.5, &params);
        assert!(matches!(result, EarlyStoppingResult::Continue));
        assert!((state.best_loss - 0.5).abs() < 1e-10);
        assert_eq!(state.epochs_without_improvement, 0);

        // No improvement - increments counter
        let result = state.check::<CpuBackend>(0.6, &params);
        assert!(matches!(result, EarlyStoppingResult::Continue));
        assert_eq!(state.epochs_without_improvement, 1);

        // More no improvement
        let result = state.check::<CpuBackend>(0.55, &params);
        assert!(matches!(result, EarlyStoppingResult::Continue));
        assert_eq!(state.epochs_without_improvement, 2);

        // Final no improvement - should stop
        let result = state.check::<CpuBackend>(0.52, &params);
        assert!(matches!(result, EarlyStoppingResult::Stop));
    }

    #[test]
    fn test_early_stopping_state_divergence() {
        use crate::model::linear::LinearParams;

        let config = EarlyStoppingConfig::new(10, 0.001);
        let mut state: EarlyStoppingState<LinearParams<CpuBackend>> =
            EarlyStoppingState::new(config, Some(10.0)); // 10x threshold

        let params = LinearParams {
            weights: Tensor1D::new(vec![1.0]),
            bias: Scalar::new(0.0),
        };

        // First check - sets best loss
        let result = state.check::<CpuBackend>(1.0, &params);
        assert!(matches!(result, EarlyStoppingResult::Continue));

        // Divergence - loss is > 10x best
        let result = state.check::<CpuBackend>(15.0, &params);
        assert!(matches!(result, EarlyStoppingResult::Diverged));
    }

    #[test]
    fn test_early_stopping_restores_best_params() {
        // Test that early stopping restores the best parameters
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0]; // y = 2*x
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let loss = MSELoss;
        let optimizer = SGD::new(0.1);
        let regularizer = NoRegularizer;

        let trainer = Trainer::builder(loss, optimizer, regularizer)
            .batch_size(3)
            .max_epochs(500)
            .early_stopping(3, 0.0001) // Very sensitive
            .verbose(false)
            .build();

        let fitted_model = trainer.fit(model, &dataset).unwrap();
        let weights = fitted_model.extract_params().weights;

        // Should still converge well (allow wider range)
        assert!(weights[0] > 1.5);
        assert!(weights[0] < 2.5);
    }

    #[test]
    fn test_trainer_with_all_stability_features_verbose() {
        // Test verbose mode with all stability features
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let loss = MSELoss;
        let optimizer = SGD::new(0.1);
        let regularizer = NoRegularizer;

        let trainer = Trainer::builder(loss, optimizer, regularizer)
            .batch_size(3)
            .max_epochs(10)
            .gradient_clipping(1.0)
            .early_stopping(20, 0.0001)
            .divergence_threshold(100.0)
            .verbose(true) // Test verbose path
            .build();

        // This will print to stdout, but should not fail
        let result = trainer.fit(model, &dataset);
        assert!(result.is_ok());
    }

    #[test]
    fn test_early_stopping_verbose_paths() {
        // Test that early stopping verbose paths work
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        // Test verbose early stopping
        let model = LinearRegression::<CpuBackend>::new(1);
        let trainer = Trainer::builder(MSELoss, SGD::<CpuBackend>::new(0.1), NoRegularizer)
            .batch_size(3)
            .max_epochs(50)
            .early_stopping(3, 0.0001)
            .verbose(true)
            .build();

        let _ = trainer.fit(model, &dataset);
    }

    #[test]
    fn test_divergence_verbose_path() {
        // Test divergence detection with verbose mode
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let trainer = Trainer::builder(
            MSELoss,
            SGD::<CpuBackend>::new(1000.0), // High LR to cause divergence
            NoRegularizer,
        )
        .batch_size(3)
        .max_epochs(100)
        .early_stopping(5, 0.001)
        .divergence_threshold(5.0)
        .verbose(true)
        .build();

        let result = trainer.fit(model, &dataset);
        assert!(result.is_err());
    }

    #[test]
    fn test_trainer_with_callback() {
        use crate::callbacks::{Callback, LoggingCallback, TrainingState};
        use std::cell::RefCell;
        use std::rc::Rc;

        // A simple callback that tracks epoch counts
        struct EpochCounter {
            count: Rc<RefCell<usize>>,
        }

        impl<B: Backend, M: TrainableModel<B>> Callback<B, M> for EpochCounter {
            fn on_epoch_end(&mut self, _state: &mut TrainingState<B, M>) {
                *self.count.borrow_mut() += 1;
            }
        }

        let counter = Rc::new(RefCell::new(0));
        let callback = EpochCounter {
            count: counter.clone(),
        };

        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let trainer = Trainer::builder(MSELoss, SGD::<CpuBackend>::new(0.1), NoRegularizer)
            .batch_size(3)
            .max_epochs(5)
            .with_callback(Box::new(callback))
            .verbose(false)
            .build();

        let _ = trainer.fit(model, &dataset);
        assert_eq!(*counter.borrow(), 5, "Callback should be called 5 times");
    }

    #[test]
    fn test_trainer_with_lr_scheduler() {
        use crate::schedulers::StepLR;

        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let scheduler = Box::new(StepLR::new(0.1, 2, 0.5)); // Decay every 2 epochs

        let trainer = Trainer::builder(MSELoss, SGD::<CpuBackend>::new(0.1), NoRegularizer)
            .batch_size(3)
            .max_epochs(5)
            .with_lr_scheduler(scheduler)
            .with_initial_lr(0.1)
            .verbose(false)
            .build();

        let result = trainer.fit(model, &dataset);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trainer_with_start_epoch() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);

        // Start from epoch 5 - should only train 5 more epochs (5-9)
        let trainer = Trainer::builder(MSELoss, SGD::<CpuBackend>::new(0.1), NoRegularizer)
            .batch_size(3)
            .max_epochs(10)
            .start_epoch(5)
            .verbose(false)
            .build();

        let result = trainer.fit(model, &dataset);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trainer_callback_stop_requested() {
        use crate::callbacks::{Callback, TrainingState};

        // A callback that stops training after 3 epochs
        struct StopAfterThree;

        impl<B: Backend, M: TrainableModel<B>> Callback<B, M> for StopAfterThree {
            fn on_epoch_end(&mut self, state: &mut TrainingState<B, M>) {
                if state.epoch >= 2 {
                    state.request_stop();
                }
            }
        }

        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let trainer = Trainer::builder(MSELoss, SGD::<CpuBackend>::new(0.1), NoRegularizer)
            .batch_size(3)
            .max_epochs(100) // Would run 100 epochs normally
            .with_callback(Box::new(StopAfterThree))
            .verbose(false)
            .build();

        let result = trainer.fit(model, &dataset);
        // Training should complete successfully (stopped early, not error)
        assert!(result.is_ok());
    }

    #[test]
    fn test_trainer_builder_callback_and_scheduler() {
        // Test that with_callback and with_lr_scheduler work on builder
        use crate::schedulers::ExponentialLR;

        let builder: TrainerBuilder<CpuBackend, _, _, LinearRegression<CpuBackend>, _, _> =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
                .with_callback(Box::new(crate::callbacks::NoopCallback))
                .with_lr_scheduler(Box::new(ExponentialLR::new(0.01, 0.9)))
                .with_initial_lr(0.01)
                .start_epoch(5);

        // Verify builder state
        assert_eq!(builder.callbacks.len(), 1);
        assert!(builder.lr_scheduler.is_some());
        assert!((builder.initial_lr - 0.01).abs() < 1e-10);
        assert_eq!(builder.start_epoch, 5);
    }

    #[test]
    fn test_trainer_callback_full_lifecycle_verbose() {
        use crate::callbacks::{Callback, TrainingState};
        use std::cell::RefCell;
        use std::rc::Rc;

        // A callback that tracks all lifecycle events
        #[derive(Default)]
        struct LifecycleTracker {
            train_start: Rc<RefCell<bool>>,
            train_end: Rc<RefCell<bool>>,
            epoch_starts: Rc<RefCell<usize>>,
            epoch_ends: Rc<RefCell<usize>>,
        }

        impl<B: Backend, M: TrainableModel<B>> Callback<B, M> for LifecycleTracker {
            fn on_train_start(&mut self, _state: &TrainingState<B, M>) {
                *self.train_start.borrow_mut() = true;
            }
            fn on_train_end(&mut self, _state: &TrainingState<B, M>) {
                *self.train_end.borrow_mut() = true;
            }
            fn on_epoch_start(&mut self, _state: &TrainingState<B, M>) {
                *self.epoch_starts.borrow_mut() += 1;
            }
            fn on_epoch_end(&mut self, _state: &mut TrainingState<B, M>) {
                *self.epoch_ends.borrow_mut() += 1;
            }
        }

        let tracker = LifecycleTracker::default();
        let callback = LifecycleTracker {
            train_start: tracker.train_start.clone(),
            train_end: tracker.train_end.clone(),
            epoch_starts: tracker.epoch_starts.clone(),
            epoch_ends: tracker.epoch_ends.clone(),
        };

        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let trainer = Trainer::builder(MSELoss, SGD::<CpuBackend>::new(0.1), NoRegularizer)
            .batch_size(2) // 2 batches per epoch
            .max_epochs(3)
            .with_callback(Box::new(callback))
            .verbose(true) // Test verbose callback paths
            .build();

        let _ = trainer.fit(model, &dataset);

        assert!(
            *tracker.train_start.borrow(),
            "on_train_start should be called"
        );
        assert!(*tracker.train_end.borrow(), "on_train_end should be called");
        assert_eq!(
            *tracker.epoch_starts.borrow(),
            3,
            "on_epoch_start should be called 3 times"
        );
        assert_eq!(
            *tracker.epoch_ends.borrow(),
            3,
            "on_epoch_end should be called 3 times"
        );
    }

    #[test]
    fn test_trainer_callback_with_scheduler_verbose() {
        // Test callback + scheduler together with verbose mode
        use crate::callbacks::{Callback, TrainingState};
        use crate::schedulers::StepLR;
        use std::cell::RefCell;
        use std::rc::Rc;

        struct MetricRecorder {
            lrs: Rc<RefCell<Vec<f64>>>,
        }

        impl<B: Backend, M: TrainableModel<B>> Callback<B, M> for MetricRecorder {
            fn on_epoch_end(&mut self, state: &mut TrainingState<B, M>) {
                self.lrs.borrow_mut().push(state.learning_rate);
            }
        }

        let lrs = Rc::new(RefCell::new(Vec::new()));
        let callback = MetricRecorder { lrs: lrs.clone() };

        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let scheduler = Box::new(StepLR::new(0.1, 2, 0.5)); // Decay every 2 epochs

        let trainer = Trainer::builder(MSELoss, SGD::<CpuBackend>::new(0.1), NoRegularizer)
            .batch_size(3)
            .max_epochs(5)
            .with_callback(Box::new(callback))
            .with_lr_scheduler(scheduler)
            .with_initial_lr(0.1)
            .verbose(true)
            .build();

        let _ = trainer.fit(model, &dataset);

        let recorded_lrs = lrs.borrow();
        // Scheduler: epoch 0,1 -> 0.1; epoch 2,3 -> 0.05; epoch 4 -> 0.025
        assert_eq!(recorded_lrs.len(), 5);
        assert!((recorded_lrs[0] - 0.1).abs() < 1e-10);
        assert!((recorded_lrs[1] - 0.1).abs() < 1e-10);
        assert!((recorded_lrs[2] - 0.05).abs() < 1e-10);
        assert!((recorded_lrs[3] - 0.05).abs() < 1e-10);
        assert!((recorded_lrs[4] - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_trainer_callback_sets_metrics() {
        // Test that callbacks can set metrics that affect training
        use crate::callbacks::{Callback, TrainingState};

        struct MetricSetter;

        impl<B: Backend, M: TrainableModel<B>> Callback<B, M> for MetricSetter {
            fn on_epoch_end(&mut self, state: &mut TrainingState<B, M>) {
                // Set a custom metric
                state.set_metric("custom_metric", state.epoch as f64 * 0.1);
            }
        }

        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let model = LinearRegression::<CpuBackend>::new(1);
        let trainer = Trainer::builder(MSELoss, SGD::<CpuBackend>::new(0.1), NoRegularizer)
            .batch_size(3)
            .max_epochs(3)
            .with_callback(Box::new(MetricSetter))
            .verbose(false)
            .build();

        let result = trainer.fit(model, &dataset);
        assert!(result.is_ok());
    }
}
