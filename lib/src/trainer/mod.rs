// trainer/mod.rs
use crate::{
    backend::{Backend, Scalar, ScalarOps, Tensor1D, Tensor2D},
    dataset::Dataset,
    loss::Loss,
    model::{ParamOps, TrainableModel},
    optimizer::Optimizer,
    regularizers::Regularizer,
};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

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
    _phantom_backend: PhantomData<B>,
    _phantom_model: PhantomData<M>,
}

/// Fluent builder for constructing a `Trainer` with custom hyperparameters.
///
/// Defaults:
/// - `batch_size`: 32
/// - `max_epochs`: 1000
/// - `verbose`: true
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

    pub fn build(self) -> Trainer<B, L, O, M, P, R> {
        Trainer {
            batch_size: self.batch_size,
            max_epochs: self.max_epochs,
            verbose: self.verbose,
            loss_fn: self.loss_fn,
            optimizer: self.optimizer,
            regularizer: self.regularizer,
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
    ///
    /// # Notes
    /// - Loss is averaged over the entire dataset per epoch.
    /// - Gradients are averaged per batch before applying regularization.
    pub fn fit<D>(&self, mut model: M, dataset: &D) -> Result<M::Output, String>
    where
        D: Dataset,
    {
        let n_total = dataset.len().ok_or("Dataset length unknown")?;
        if n_total == 0 {
            return Err("Dataset is empty".into());
        }

        for epoch in 0..self.max_epochs {
            let mut total_loss = Scalar::<B>::new(0.);
            //let mut total_samples = 0;
            for batch_result in dataset.batches::<B>(self.batch_size) {
                let (batch_x, batch_y) =
                    batch_result.map_err(|e| format!("Data error: {:?}", e))?;
                //let mut total_loss = Scalar::<B>::new(0.);
                let preds = model.forward(&batch_x);
                total_loss = total_loss + self.loss_fn.loss(&preds, &batch_y);
                let (reg_penalty, reg_grad) = self.regularizer.regularizer_penalty_grad(&model);
                total_loss = total_loss + reg_penalty;
                let grad_preds = self.loss_fn.grad_wrt_prediction(&preds, &batch_y);
                let grads = model.backward(&batch_x, &grad_preds);

                let total_grads = grads.add(&reg_grad);
                let new_params = self.optimizer.step(model.params(), &total_grads);
                model.update_params(&new_params);
            }

            let avg_loss = total_loss / Scalar::<B>::new(n_total as f64);
            if self.verbose {
                println!("Epoch {}: loss = {}", epoch, avg_loss.data.to_f64());
            }
        }

        Ok(model.into_fitted())
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
        backend::CpuBackend,
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
        let builder = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer);

        assert_eq!(builder.batch_size, 32);
        assert_eq!(builder.max_epochs, 1000);
        assert_eq!(builder.verbose, true);
    }

    #[test]
    fn test_trainer_builder_custom_batch_size() {
        let builder = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
            .batch_size(64);

        assert_eq!(builder.batch_size, 64);
        assert_eq!(builder.max_epochs, 1000);
    }

    #[test]
    fn test_trainer_builder_custom_max_epochs() {
        let builder = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
            .max_epochs(500);

        assert_eq!(builder.batch_size, 32);
        assert_eq!(builder.max_epochs, 500);
    }

    #[test]
    fn test_trainer_builder_verbose() {
        let builder = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
            .verbose(false);

        assert_eq!(builder.verbose, false);

        let builder_verbose =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer).verbose(true);

        assert_eq!(builder_verbose.verbose, true);
    }

    #[test]
    fn test_trainer_builder_chaining() {
        let builder = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
            .batch_size(128)
            .max_epochs(250)
            .verbose(false);

        assert_eq!(builder.batch_size, 128);
        assert_eq!(builder.max_epochs, 250);
        assert_eq!(builder.verbose, false);
    }

    #[test]
    fn test_trainer_builder_chaining_order_independent() {
        let builder1 = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
            .batch_size(16)
            .max_epochs(100);

        let builder2 = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
            .max_epochs(100)
            .batch_size(16);

        assert_eq!(builder1.batch_size, builder2.batch_size);
        assert_eq!(builder1.max_epochs, builder2.max_epochs);
    }

    #[test]
    fn test_trainer_builder_small_batch_size() {
        let builder =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer).batch_size(1);

        assert_eq!(builder.batch_size, 1);
    }

    #[test]
    fn test_trainer_builder_large_batch_size() {
        let builder = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
            .batch_size(10000);

        assert_eq!(builder.batch_size, 10000);
    }

    #[test]
    fn test_trainer_builder_zero_epochs() {
        let builder =
            TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer).max_epochs(0);

        assert_eq!(builder.max_epochs, 0);
    }

    #[test]
    fn test_trainer_builder_large_epochs() {
        let builder = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
            .max_epochs(100000);

        assert_eq!(builder.max_epochs, 100000);
    }

    #[test]
    fn test_trainer_builder_creates_valid_trainer() {
        let builder = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
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
        let _builder = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
            .batch_size(16)
            .max_epochs(50);

        // Components are reused via Clone for SGD, creating fresh instances for other builders
        let _builder2 = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer);
    }

    #[test]
    fn test_trainer_builder_clone_components() {
        let builder1 = TrainerBuilder::new(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
            .batch_size(32)
            .max_epochs(100);

        // SGD implements Clone, so we can use it multiple times
        let _builder2 = TrainerBuilder::new(MSELoss, builder1.optimizer.clone(), NoRegularizer);
    }

    #[test]
    fn test_trainer_builder_zero_batch_size() {
        // Builder allows 0 batch size - this is up to the user to validate
        let builder =
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
}
