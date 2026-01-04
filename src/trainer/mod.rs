// trainer/mod.rs
use std::marker::PhantomData;
use std::fmt::{Debug, Display};
use crate::{
    backend::{Backend, ScalarOps},
    loss::Loss,
    model::{TrainableModel, ParamOps},
    optimizer::Optimizer,
    regularizers::Regularizer,
};

// --- Основная структура (immutable после build) ---
pub struct Trainer<B, L, O, M, P, R>
where
    B: Backend,
    L: Loss<B>,
    M: TrainableModel<B, Params = P, Gradients = P>,
    O: Optimizer<B, P>,
    R: Regularizer<B, M>
{
    pub(crate) batch_size: usize,
    pub(crate) max_epochs: usize,
    pub(crate) loss_fn: L,
    pub(crate) optimizer: O,
    pub(crate) regularizer: R,
    _phantom_backend: PhantomData<B>,
    _phantom_model: PhantomData<M>,
}

// --- Билдер ---
pub struct TrainerBuilder<B, L, O, M, P, R>
where
    B: Backend,
    L: Loss<B>,
    M: TrainableModel<B, Params = P, Gradients = P>,
    O: Optimizer<B, P>,
    R: Regularizer<B, M>
{
    batch_size: usize,
    max_epochs: usize,
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
    R: Regularizer<B, M>
{
    pub fn new(loss_fn: L, optimizer: O, regularizer: R) -> Self {
        Self {
            batch_size: 32,
            max_epochs: 1000,
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

    pub fn build(self) -> Trainer<B, L, O, M, P, R> {
        Trainer {
            batch_size: self.batch_size,
            max_epochs: self.max_epochs,
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
    L: Loss<B, Target = B::Tensor1D, Prediction = B::Tensor1D>,
    M: TrainableModel<B, Input = B::Tensor2D, Prediction = L::Prediction, Params = P, Gradients = P>,
    O: Optimizer<B, P>,
    R: Regularizer<B, M>,
    P: ParamOps<B>
{
    pub fn fit(
        &self,
        mut model: M,
        x: &[Vec<f64>],
        y: &[f64],
    ) -> Result<M::Output, String> {
        if x.len() != y.len() {
            return Err("x and y must have same length".into());
        }
        if x.is_empty() {
            return Err("Dataset is empty".into());
        }
        let n_features = x[0].len();
        let n = x.len();
        let device = B::default_device();

        for epoch in 0..self.max_epochs {
            let mut total_loss = B::Scalar::zero();

            for i in (0..n).step_by(self.batch_size) {
                let end = (i + self.batch_size).min(n);
                let batch_x_raw = &x[i..end];
                let batch_y_raw = &y[i..end];
                let batch_size = batch_x_raw.len();

                let mut x_tensor = B::zeros_2d(batch_size, n_features, &device);
                for (row, features) in batch_x_raw.iter().enumerate() {
                    for (col, &val) in features.iter().enumerate() {
                        B::set_2d(&mut x_tensor, row, col, B::Scalar::from_f64(val));
                    }
                }

                let mut y_tensor = B::zeros_1d(batch_size, &device);
                for (idx, &val) in batch_y_raw.iter().enumerate() {
                    B::set_1d(&mut y_tensor, idx, B::Scalar::from_f64(val));
                }

                let preds = model.forward(&x_tensor);
                total_loss = total_loss + self.loss_fn.loss(&preds, &y_tensor);
                let (reg_penalty, reg_grad) = self.regularizer.regularizer_penalty_grad(&model);
                total_loss = total_loss + reg_penalty;
                let grad_preds = self.loss_fn.grad_wrt_prediction(&preds, &y_tensor);
                let grads = model.backward(&x_tensor, &grad_preds);

                let total_grads = grads.add(&reg_grad);
                let new_params = self.optimizer.step(model.params(), &total_grads);
                model.update_params(&new_params);
            }

            let avg_loss = total_loss / B::Scalar::from_f64(n as f64);
            println!("Epoch {}: loss = {}", epoch, avg_loss); // ← Display вместо Debug
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
    R: Regularizer<B, M>
{
    pub fn builder(loss_fn: L, optimizer: O, regularizer: R) -> TrainerBuilder<B, L, O, M, P, R> {
        TrainerBuilder::new(loss_fn, optimizer, regularizer)
    }
}