// trainer/mod.rs
use std::marker::PhantomData;
use std::fmt::{Debug, Display};
use crate::{
    backend::backend::{Backend},
    backend::scalar::{ScalarOps, Scalar},
    loss::Loss,
    model::{TrainableModel, ParamOps},
    optimizer::Optimizer,
    regularizers::Regularizer,
    backend::tensor1d::Tensor1D,
    backend::tensor2d::Tensor2D,
    dataset::Dataset,
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
    L: Loss<B, Target = Tensor1D<B>, Prediction = Tensor1D<B>>,
    M: TrainableModel<B, Input = Tensor2D<B>, Prediction = L::Prediction, Params = P, Gradients = P>,
    O: Optimizer<B, P>,
    R: Regularizer<B, M>,
    P: ParamOps<B>
{
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
                let (batch_x, batch_y) = batch_result.map_err(|e| format!("Data error: {:?}", e))?;
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
            println!("Epoch {}: loss = {}", epoch, avg_loss.data.to_f64()); // ← Display вместо Debug
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