pub use crate::backend::{Backend, ScalarOps, Tensor};
pub use crate::model::{TrainableModel};
pub use crate::model::linear::{LinearModel, Unfitted, LinearParams};

pub trait Loss<B: Backend> {
    type Prediction;
    type Target;

    /// Computes the scalar loss value (for logging/metrics).
    fn loss(&self, prediction: &Self::Prediction, target: &Self::Target) -> B::Scalar;

    /// Computes the gradient of the loss w.r.t. the prediction: ∂L/∂pred.
    /// This is what gets passed to `model.backward()`.
    fn grad_wrt_prediction(
        &self,
        prediction: &Self::Prediction,
        target: &Self::Target,
    ) -> Self::Prediction;
}

pub struct MSELoss;

impl<B: Backend> Loss<B> for MSELoss
where
    B::Tensor1D: Clone,
{
    type Prediction = B::Tensor1D;
    type Target = B::Tensor1D;

    fn loss(&self, pred: &B::Tensor1D, target: &B::Tensor1D) -> B::Scalar {
        let diff = B::sub_1d(pred, target);
        let sq = B::dot(&diff, &diff);
        let mean = sq / B::Scalar::from_f64(B::len_1d(&diff) as f64);
        mean
    }

    fn grad_wrt_prediction(
        &self,
        pred: &B::Tensor1D,
        target: &B::Tensor1D,
    ) -> B::Tensor1D {
        // d/dp ( (p - y)^2 ) = 2(p - y)
        // But commonly: MSE = (1/n) * sum(...), so grad = (2/n)(p - y)
        // However, in practice, we often omit 2 and let LR absorb it.
        // We'll return (pred - target) — standard in many frameworks.
        let diff: B::Tensor1D = B::sub_1d(pred, target);
        let n = B::Scalar::from_f64(1. / (B::len_1d(&pred) as f64)); // or use backend method
        B::scale_1d(n, &diff)
    }
}