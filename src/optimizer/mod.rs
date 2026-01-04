use crate::model::linear::{LinearParams,};
use crate::backend::{Backend, ScalarOps};

pub trait Optimizer<B: Backend, P> {
    /// Выполняет шаг оптимизации: `params = params - lr * gradients`.
    /// Возвращает обновлённые параметры.
    fn step(&self, params: &P, gradients: &P) -> P;
}

#[derive(Clone, Copy, Debug)]
pub struct SGD<B: Backend>{
    lr: B::Scalar,
}


impl <B: Backend> SGD<B>{
    pub fn new(lr: f64) -> Self{
        Self{lr: B::Scalar::from_f64(lr)}
    }
}

impl<B: Backend> Optimizer<B, LinearParams<B>> for SGD<B>
where
    B::Tensor1D: Clone,
    B::Scalar: Clone,
{
    fn step(&self, params: &LinearParams<B>, grads: &LinearParams<B>) -> LinearParams<B> {
        // weights_new = weights - lr * grad_weights
        let neg_lr = B::Scalar::from_f64(0.) - self.lr;
        let scaled_grad = B::scale_1d(neg_lr, &grads.weights);
        let weights_update = B::add_1d(&params.weights, &scaled_grad);
        // bias_new = bias - lr * grad_bias
        let bias_update = params.bias - (self.lr * grads.bias);

        LinearParams {
            weights: weights_update,
            bias: bias_update,
        }
    }
}