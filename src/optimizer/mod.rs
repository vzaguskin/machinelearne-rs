use crate::backend::backend::Backend;
use crate::backend::scalar::Scalar;
use crate::backend::tensor1d::Tensor1D;
use crate::loss::TensorLike;
use crate::model::linear::LinearParams;

pub trait Optimizer<B: Backend, P> {
    /// Выполняет шаг оптимизации: `params = params - lr * gradients`.
    /// Возвращает обновлённые параметры.
    fn step(&self, params: &P, gradients: &P) -> P;
}

#[derive(Clone)]
pub struct SGD<B: Backend> {
    lr: Scalar<B>,
}

impl<B: Backend> SGD<B> {
    pub fn new(lr: f64) -> Self {
        Self {
            lr: Scalar::<B>::new(lr),
        }
    }
}

impl<B: Backend> Optimizer<B, LinearParams<B>> for SGD<B>
where
    Tensor1D<B>: Clone,
    Scalar<B>: Clone,
{
    fn step(&self, params: &LinearParams<B>, grads: &LinearParams<B>) -> LinearParams<B> {
        // weights_new = weights - lr * grad_weights
        let neg_lr = Scalar::<B>::new(0.) - self.lr.clone();
        let scaled_grad = grads.weights.scale(&neg_lr);
        let weights_update = params.weights.add(&scaled_grad);
        // bias_new = bias - lr * grad_bias
        let bias_update = params.bias - (self.lr * grads.bias);

        LinearParams {
            weights: weights_update,
            bias: bias_update,
        }
    }
}
