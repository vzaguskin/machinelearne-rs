pub use crate::backend::backend::{Backend};
pub use crate::backend::scalar::{Scalar, ScalarOps};
pub use crate::backend::tensorlike::TensorLike;
pub use crate::backend::tensor1d::Tensor1D;

pub use crate::model::{TrainableModel};
pub use crate::model::linear::{LinearModel, Unfitted, LinearParams};

pub trait Loss<B: Backend> {
    type Prediction: TensorLike<B>;
    type Target: TensorLike<B>;

    /// Computes the scalar loss value (for logging/metrics).
    fn loss(&self, prediction: &Self::Prediction, target: &Self::Target) -> Scalar<B>;

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
    type Prediction = Tensor1D<B>;
    type Target = Tensor1D<B>;

    fn loss(&self, pred: &Self::Prediction, target: &Self::Target) -> Scalar<B> {
        let diff = pred.sub(&target);
        diff.dot(&diff) / Scalar::<B>::new(B::len_1d(&diff.data) as f64)
    }

    fn grad_wrt_prediction(
        &self,
        pred: &Self::Prediction,
        target: &Self::Target,
    ) -> Tensor1D<B> {
        // d/dp ( (p - y)^2 ) = 2(p - y)
        // But commonly: MSE = (1/n) * sum(...), so grad = (2/n)(p - y)
        // However, in practice, we often omit 2 and let LR absorb it.
        // We'll return (pred - target) — standard in many frameworks.
        let diff= pred.sub(&target);
        let n = Scalar::<B>::new(1. / (B::len_1d(&pred.data) as f64)); // or use backend method
        diff.scale(&n)
    }
}

pub struct MAELoss;

impl<B: Backend> Loss<B> for MAELoss
where
    B::Tensor1D: Clone,
{
    type Prediction = Tensor1D<B>;
    type Target = Tensor1D<B>;

    fn loss(&self, pred: &Self::Prediction, target: &Self::Target) -> Scalar<B>{
        pred.sub(&target).abs().mean()
    }

    fn grad_wrt_prediction(
        &self,
        pred: &Self::Prediction,
        target: &Self::Target,
    ) -> Tensor1D<B> {
        let diff = pred.sub(&target);
        let sign = diff.sign();
        let n = Scalar::<B>::new(1.0) / pred.len();
        sign.scale(&n)
    }
}

pub struct BCEWithLogitsLoss;

impl<B: Backend> Loss<B> for BCEWithLogitsLoss
where
    B::Tensor1D: Clone,
{
    type Prediction = Tensor1D<B>;
    type Target = Tensor1D<B>;

    fn loss(&self, logits: &Self::Prediction, targets: &Self::Target) -> Scalar<B>{
        // Numerically stable BCE: -(t * log(s(z)) + (1-t) * log(1 - s(z)))
        // = max(z, 0) - z * t + log(1 + exp(-|z|))
        let max_logits = logits.maximum(Self::Prediction::zeros(logits.len().data.to_f64() as usize));
        let term2 = logits
                                            .abs()
                                            .scale(&Scalar::<B>::new(-1.))
                                            .exp()
                                            .add_scalar(&Scalar::<B>::new(1.))
                                            .log();
        
        
        let term1 = max_logits.sub(&logits.mul(&targets));
        let total = term1.add(&term2);
        total.mean()
    }

    fn grad_wrt_prediction(&self, logits: &Self::Prediction, targets: &Self::Target) -> Self::Prediction {
        // d/dz BCE = sigmoid(z) - t
        let n = Scalar::<B>::new(1.0) / logits.len();
        logits.sigmoid().sub(&targets).scale(&n)
    }
}