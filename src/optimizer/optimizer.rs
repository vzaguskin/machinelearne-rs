// src/optimizer.rs
use crate::{model::linear::LinearModel, backend::Backend, Unfitted};

pub struct SGD<const LR_NUM: u32, const LR_DEN: u32>;

impl<const LR_NUM: u32, const LR_DEN: u32, B: Backend> SGD<LR_NUM, LR_DEN>
where
    B::Tensor1D: std::ops::Sub<Output = B::Tensor1D>
        + std::ops::Mul<B::Scalar, Output = B::Tensor1D>,
    B::Scalar: Clone + std::ops::Sub<Output = B::Scalar> + std::ops::Mul<Output = B::Scalar>,
{
    pub fn step(
        model: LinearModel<B, Unfitted>,
        (grad_w, grad_b): (B::Tensor1D, B::Scalar),
    ) -> LinearModel<B, Unfitted> {
        let lr = B::Scalar::from(LR_NUM as f32 / LR_DEN as f32);
        model.apply_gradients(grad_w, grad_b, lr)
    }
}