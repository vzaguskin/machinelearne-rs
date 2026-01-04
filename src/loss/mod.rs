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

pub struct MAELoss;

impl<B: Backend> Loss<B> for MAELoss
where
    B::Tensor1D: Clone,
{
    type Prediction = B::Tensor1D;
    type Target = B::Tensor1D;

    fn loss(&self, pred: &B::Tensor1D, target: &B::Tensor1D) -> B::Scalar {
        let diff = B::sub_1d(pred, target);
        let abs_diff_sum = B::sum_1d(&B::abs_1d(&diff));
        let mean = abs_diff_sum / B::Scalar::from_f64(B::len_1d(&diff) as f64);
        mean
    }

    fn grad_wrt_prediction(
        &self,
        pred: &B::Tensor1D,
        target: &B::Tensor1D,
    ) -> B::Tensor1D {
        let diff = B::sub_1d(pred, target);
        let sign = B::sign_1d(&diff);
        let n = B::Scalar::from_f64(1.0 / (B::len_1d(pred) as f64));
        B::scale_1d(n, &sign)
    }
}

pub struct BCEWithLogitsLoss;

impl<B: Backend> Loss<B> for BCEWithLogitsLoss
where
    B::Tensor1D: Clone,
{
    type Prediction = B::Tensor1D; // logits
    type Target = B::Tensor1D;     // labels in {0.0, 1.0}

    fn loss(&self, logits: &B::Tensor1D, targets: &B::Tensor1D) -> B::Scalar {
        // Numerically stable BCE: -(t * log(s(z)) + (1-t) * log(1 - s(z)))
        // = max(z, 0) - z * t + log(1 + exp(-|z|))
        let max_logits = B::maximum_1d(logits, &B::zeros_1d(B::len_1d(logits), &B::default_device()));
        let neg_abs_logits = B::neg_1d(&B::abs_1d(logits));
        let log1p_exp_negabs = B::log_1d(&B::add_scalar_1d(&B::exp_1d(&neg_abs_logits), B::Scalar::from_f64(1.0)));

        let term1 = B::sub_1d(&max_logits, &B::mul_1d(logits, targets));
        let term2 = log1p_exp_negabs;
        let total = B::add_1d(&term1, &term2);
        let mean = B::sum_1d(&total) / B::Scalar::from_f64(B::len_1d(logits) as f64);
        mean
    }

    fn grad_wrt_prediction(&self, logits: &B::Tensor1D, targets: &B::Tensor1D) -> B::Tensor1D {
        // d/dz BCE = sigmoid(z) - t
        let sigmoid = B::sigmoid_1d(logits);
        let diff = B::sub_1d(&sigmoid, targets);
        let n = B::Scalar::from_f64(1.0 / (B::len_1d(logits) as f64));
        B::scale_1d(n, &diff)
    }
}