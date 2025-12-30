// trainer/mod.rs
use std::marker::PhantomData;
use std::fmt::{Debug, Display};
use crate::{
    backend::{Backend, ScalarOps},
    loss::Loss,
    model::TrainableModel,
    optimizer::Optimizer,
};

pub struct Trainer<B, L, O, M, Output, P>
where
    B: Backend,
    L: Loss<B>,
    M: TrainableModel<B, Output, Params = P, Gradients = P>,
    O: Optimizer<B, P>,

{
    pub batch_size: usize,
    pub max_epochs: usize,
    pub loss_fn: L,
    pub optimizer: O,
    _phantom_backend: PhantomData<B>,
    _phantom_model: PhantomData<M>,
    _phantom_output: PhantomData<Output>,
}

impl<B, L, O, M, Output, P> Trainer<B, L, O, M, Output, P>
where
    B: Backend,
    B::Scalar: Debug + Display,
    L: Loss<B, Target = B::Tensor1D, Prediction = B::Tensor1D>,
    M: TrainableModel<B, Output, Input = B::Tensor2D, Prediction = L::Prediction,
    Params = P, Gradients = P>,
    
    O: Optimizer<B, M::Params>,
{
    pub fn new(loss_fn: L, optimizer: O) -> Self {
        Self {
            batch_size: 32,
            max_epochs: 1000,
            loss_fn,
            optimizer,
            _phantom_backend: PhantomData,
            _phantom_model: PhantomData,
            _phantom_output: PhantomData,
        }
    }

    /// Train the model on batches of data.
    /// 
    /// - `x`: list of samples, each sample is `Vec<f64>` (features)
    /// - `y`: list of targets, each is `f64` (or whatever `L::Target` is)
    pub fn fit(
        &self,
        mut model: M,
        x: &[Vec<f64>],
        y: &[f64], // ← assume L::Target = f64-compatible
    ) -> Result<Output, String> {
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

                // --- Convert to backend tensors ---
                // Input: [batch_size, n_features]
                let mut x_tensor = B::zeros_2d(batch_size, n_features, &device);
                for (row, features) in batch_x_raw.iter().enumerate() {
                    for (col, &val) in features.iter().enumerate() {
                        B::set_2d(&mut x_tensor, row, col, B::Scalar::from_f64(val));
                    }
                }

                // Target: [batch_size] — assume L::Target = B::Scalar or convertible
                let mut y_tensor = B::zeros_1d(batch_size, &device);
                for (idx, &val) in batch_y_raw.iter().enumerate() {
                    B::set_1d(&mut y_tensor, idx, B::Scalar::from_f64(val));
                }

                // --- Forward ---
                let preds = model.forward(&x_tensor);

                // --- Loss and grad ---
                total_loss = total_loss + self.loss_fn.loss(&preds, &y_tensor);
                let grad_preds = self.loss_fn.grad_wrt_prediction(&preds, &y_tensor);

                // --- Backward ---
                let grads = model.backward(&x_tensor, &grad_preds);

                // --- Optimizer step ---
                let new_params = self.optimizer.step(model.params(), &grads);
                model.update_params(&new_params);
            }

            let avg_loss = total_loss / B::Scalar::from_f64(n as f64);
            println!("Epoch {}: loss = {:?}", epoch, avg_loss);
        }

        Ok(model.into_fitted())
    }
}