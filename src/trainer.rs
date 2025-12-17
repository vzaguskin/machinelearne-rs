// src/trainer.rs
use crate::{
    model::linear::LinearModel,
    loss::MSELoss,
    optimizer::SGD,
    backend::Backend,
    Unfitted, Fitted,
    PhantomData,
};

pub struct Trainer<
    B: Backend,
    const EPOCHS: usize,
    const LR_NUM: u32,
    const LR_DEN: u32,
> {
    _backend: PhantomData<B>,
}

impl<B: Backend, const EPOCHS: usize, const LR_NUM: u32, const LR_DEN: u32>
    Trainer<B, EPOCHS, LR_NUM, LR_DEN>
where
    // Требуем, чтобы все операции были реализованы
    B::Tensor1D: Clone
        + std::ops::Sub<Output = B::Tensor1D>
        + std::ops::Mul<B::Scalar, Output = B::Tensor1D>,
    B::Tensor2D: Clone,
    for<'a> &'a B::Tensor2D: std::ops::Mul<&'a B::Tensor1D, Output = B::Tensor1D>,
    B::Scalar: From<f32> + Clone + std::ops::Sub<Output = B::Scalar>,
{
    pub const fn new() -> Self {
        Self { _backend: PhantomData }
    }

    pub fn fit(
        &self,
        model: LinearModel<B, Unfitted>,
        x: &B::Tensor2D,
        y: &B::Tensor1D,
    ) -> LinearModel<B, Fitted> {
        let mut model = model;
        let _ = x; // заглушка
        let _ = y;

        let mut i = 0;
        while i < EPOCHS {
            let (_, grads) = MSELoss::grad(&model, x, y);
            model = SGD::<LR_NUM, LR_DEN>::step(model, grads);
            i += 1;
        }

        LinearModel::into_fitted(model)
    }
}