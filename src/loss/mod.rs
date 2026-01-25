pub use crate::backend::backend::{Backend};
pub use crate::backend::scalar::{Scalar, ScalarOps};
pub use crate::backend::tensorlike::TensorLike;
pub use crate::backend::tensor1d::Tensor1D;

pub use crate::model::{TrainableModel};
pub use crate::model::linear::{LinearModel, Unfitted, LinearParams};

/// A trait for differentiable loss functions used during model training.
///
/// Implementors must define:
/// - How to compute the scalar loss value (for logging/metrics).
/// - How to compute the gradient of the loss w.r.t. the model's predictions.
///
/// This gradient is passed to the model's `backward()` method to update parameters.
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
/// Mean Squared Error (MSE) loss: `L = (1/n) * Σ(pred_i - target_i)^2`
///
/// Gradient w.r.t. prediction: `∂L/∂pred = (pred - target) / n`
///
/// Note: The factor of 2 is omitted, as it can be absorbed into the learning rate.
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
        let n = Scalar::<B>::new(1. / pred.len() as f64); // or use backend method
        diff.scale(&n)
    }
}

/// Mean Absolute Error (MAE) loss: `L = (1/n) * Σ|pred_i - target_i|`
///
/// Gradient w.r.t. prediction: `∂L/∂pred = sign(pred - target) / n`
/// (subgradient is used at zero).
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
        let n = Scalar::<B>::new(1.0 / pred.len() as f64);
        sign.scale(&n)
    }
}
/// Binary Cross-Entropy loss with logits input (numerically stable).
///
/// Computes: `L = -(t * log(σ(z)) + (1-t) * log(1 - σ(z)))`
/// using the stable formulation: `max(z,0) - z*t + log(1 + exp(-|z|))`
///
/// Gradient w.r.t. logits: `∂L/∂z = (σ(z) - t) / n`
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
        let max_logits = logits.maximum(Self::Prediction::zeros(logits.len()));
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
        let n = Scalar::<B>::new(1.0 / logits.len() as f64);
        logits.sigmoid().sub(&targets).scale(&n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_mse_loss() {
        let pred = Tensor1D::<CpuBackend>::new(vec![3.0f32, 5.0]);
        let target = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0]);

        let mse = MSELoss;
        let loss_val = mse.loss(&pred, &target);
        // ((3-1)^2 + (5-2)^2) / 2 = (4 + 9) / 2 = 6.5
        assert!((loss_val.data - 6.5).abs() < 1e-12);

        let grad = mse.grad_wrt_prediction(&pred, &target);
        // grad = (pred - target) / n = [2.0, 3.0] / 2 = [1.0, 1.5]
        let expected_grad = vec![1.0, 1.5];
        assert_eq!(grad.to_vec(), expected_grad);
    }

    #[test]
    fn test_mae_loss() {
        let pred = Tensor1D::<CpuBackend>::new(vec![3.0f32, -1.0]);
        let target = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0]);

        let mae = MAELoss;
        let loss_val = mae.loss(&pred, &target);
        // (|3-1| + |-1-2|) / 2 = (2 + 3) / 2 = 2.5
        assert!((loss_val.data - 2.5).abs() < 1e-12);

        let grad = mae.grad_wrt_prediction(&pred, &target);
        // sign(pred - target) / n = [1.0, -1.0] / 2 = [0.5, -0.5]
        let expected_grad = vec![0.5, -0.5];
        assert_eq!(grad.to_vec(), expected_grad);
    }

    #[test]
    fn test_bce_with_logits_loss() {
        let logits = Tensor1D::<CpuBackend>::new(vec![0.0f32, 2.0, -2.0]); // sigmoid: [0.5, ~0.88, ~0.12]
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0f32, 1.0, 0.0]);

        let bce = BCEWithLogitsLoss;
        let loss_val = bce.loss(&logits, &targets);

        // Expected (computed manually or via PyTorch):
        // For z=0, t=1: -(0 + log(1+1)) = -log(2) ≈ -0.6931 → but formula gives: max(0,0) - 0*1 + log(1+1) = 0 + 0 + log(2) ≈ 0.6931
        // For z=2, t=1: max(2,0) - 2*1 + log(1+exp(-2)) = 2 - 2 + log(1+0.135) ≈ 0.127
        // For z=-2, t=0: max(-2,0)=0 - (-2)*0 + log(1+exp(-2)) = 0 + 0 + log(1.135) ≈ 0.127
        // Mean ≈ (0.6931 + 0.127 + 0.127) / 3 ≈ 0.3156
        let expected_loss = 0.3156;
        assert!((loss_val.data - expected_loss).abs() < 1e-3);

        let grad = bce.grad_wrt_prediction(&logits, &targets);
        // grad = (sigmoid(z) - t) / n
        let sig = vec![0.5, 1.0 / (1.0 + (-2.0f64).exp()), 1.0 / (1.0 + (2.0f64).exp())];
        let expected_grad: Vec<f64> = sig
            .iter()
            .zip(targets.to_vec().iter())
            .map(|(s, t)| (s - t) / 3.0)
            .collect();
        for (g, e) in grad.to_vec().iter().zip(expected_grad.iter()) {
            assert!((g - e).abs() < 1e-5);
        }
    }

    #[test]
    fn test_bce_numerical_stability() {
        // Large positive and negative logits
        let logits = Tensor1D::<CpuBackend>::new(vec![100.0f32, -100.0]);
        let targets = Tensor1D::<CpuBackend>::new(vec![1.0f32, 0.0]);

        let bce = BCEWithLogitsLoss;
        let loss_val = bce.loss(&logits, &targets);
        // Should not overflow or produce NaN
        assert!(loss_val.data.is_finite());

        let grad = bce.grad_wrt_prediction(&logits, &targets);
        assert!(grad.to_vec().iter().all(|&x| x.is_finite()));
    }
}