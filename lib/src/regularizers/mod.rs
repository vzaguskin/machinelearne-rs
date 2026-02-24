use crate::backend::scalar::Scalar;
use crate::backend::tensorlike::TensorLike;
use crate::backend::Backend;
use crate::loss::{LinearParams, Tensor1D};
use crate::model::linear::LinearRegression;
use crate::model::mlp::{LayerParams, MLPParams, MLP};
use crate::model::TrainableModel;
/// Computes the regularization penalty and its gradient w.r.t. model parameters.
///
/// This trait enables pluggable regularization strategies (e.g., L2, L1) without
/// modifying the model or trainer logic.
///
/// # Returns
/// A tuple `(penalty, grad)` where:
/// - `penalty` is a scalar value added to the total loss (for logging/metrics),
/// - `grad` is the gradient of the penalty w.r.t. model parameters, to be combined
///   with the data-driven gradient during backpropagation.
pub trait Regularizer<B: Backend, M: TrainableModel<B>> {
    fn regularizer_penalty_grad(&self, model: &M) -> (Scalar<B>, M::Gradients);
}

/// L2 (ridge) regularization: penalty = λ * ||w||².
///
/// Only applies to weights; bias is not regularized (standard practice).
///
/// Gradient w.r.t. weights: ∂/∂w (λ * wᵀw) = 2λw.
pub struct L2<B: Backend> {
    lambda: Scalar<B>,
}

impl<B: Backend> L2<B> {
    /// Creates an L2 regularizer with strength `lambda`.
    ///
    /// # Arguments
    /// * `lambda` — non-negative regularization coefficient (λ ≥ 0).
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda: Scalar::<B>::new(lambda),
        }
    }
}

impl<B> Regularizer<B, LinearRegression<B>> for L2<B>
where
    B: Backend,
{
    fn regularizer_penalty_grad(
        &self,
        model: &LinearRegression<B>,
    ) -> (
        Scalar<B>,
        <LinearRegression<B> as TrainableModel<B>>::Gradients,
    ) {
        let params = model.params();
        let weight_grad = params.weights.scale(&(self.lambda * Scalar::<B>::new(2.)));

        let loss = params.weights.dot(&params.weights);

        let loss = self.lambda * loss;

        (
            loss,
            LinearParams::<B> {
                weights: weight_grad,
                bias: Scalar::<B>::new(0.),
            },
        )
    }
}

/// L1 (Lasso) regularization: penalty = λ * ||w||₁ = λ * Σ|w_i|.
///
/// Only applies to weights; bias is not regularized (standard practice).
///
/// Gradient w.r.t. weights: ∂/∂w (λ * |w|) = λ * sign(w).
///
/// L1 encourages sparsity in model weights, making it useful for feature selection
/// and producing more interpretable models compared to L2.
pub struct L1<B: Backend> {
    lambda: Scalar<B>,
}

impl<B: Backend> L1<B> {
    /// Creates an L1 regularizer with strength `lambda`.
    ///
    /// # Arguments
    /// * `lambda` — non-negative regularization coefficient (λ ≥ 0).
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda: Scalar::<B>::new(lambda),
        }
    }
}

impl<B> Regularizer<B, LinearRegression<B>> for L1<B>
where
    B: Backend,
{
    fn regularizer_penalty_grad(
        &self,
        model: &LinearRegression<B>,
    ) -> (
        Scalar<B>,
        <LinearRegression<B> as TrainableModel<B>>::Gradients,
    ) {
        let params = model.params();

        // Penalty: λ * ||w||₁ = λ * Σ|w_i|
        let abs_weights = params.weights.abs();
        let l1_norm = abs_weights.sum();
        let loss = self.lambda * l1_norm;

        // Gradient: λ * sign(w)
        let weight_grad = params.weights.sign().scale(&self.lambda);

        (
            loss,
            LinearParams::<B> {
                weights: weight_grad,
                bias: Scalar::<B>::new(0.),
            },
        )
    }
}

/// A no-op regularizer that adds zero penalty and zero gradient.
///
/// Useful as a default when no regularization is desired.
pub struct NoRegularizer;

impl<B> Regularizer<B, LinearRegression<B>> for NoRegularizer
where
    B: Backend,
{
    fn regularizer_penalty_grad(
        &self,
        model: &LinearRegression<B>,
    ) -> (
        Scalar<B>,
        <LinearRegression<B> as TrainableModel<B>>::Gradients,
    ) {
        let params = model.params();
        let weight_grad = Tensor1D::<B>::zeros(params.weights.len());

        (
            Scalar::<B>::new(0.),
            LinearParams::<B> {
                weights: weight_grad,
                bias: Scalar::<B>::new(0.),
            },
        )
    }
}

// =============================================================================
// MLP Regularizers
// =============================================================================

impl<B: Backend> Regularizer<B, MLP<B>> for L2<B> {
    fn regularizer_penalty_grad(
        &self,
        model: &MLP<B>,
    ) -> (Scalar<B>, <MLP<B> as TrainableModel<B>>::Gradients) {
        let params = model.params();
        let mut total_penalty = Scalar::<B>::new(0.0);
        let two_lambda = self.lambda * Scalar::<B>::new(2.0);

        let layer_grads: Vec<LayerParams<B>> = params
            .layers
            .iter()
            .map(|layer| {
                // L2 penalty on weights: λ * ||W||²
                let weights_flat = layer.weights.ravel();
                let weight_penalty = weights_flat.dot(&weights_flat);
                total_penalty = total_penalty + self.lambda * weight_penalty;

                // Gradient: 2λW for weights, 0 for bias
                let weight_grad = layer.weights.scale(two_lambda);
                let bias_grad = Tensor1D::<B>::zeros(layer.bias.len());

                LayerParams {
                    weights: weight_grad,
                    bias: bias_grad,
                }
            })
            .collect();

        (
            total_penalty,
            MLPParams {
                layers: layer_grads,
            },
        )
    }
}

impl<B: Backend> Regularizer<B, MLP<B>> for L1<B> {
    fn regularizer_penalty_grad(
        &self,
        model: &MLP<B>,
    ) -> (Scalar<B>, <MLP<B> as TrainableModel<B>>::Gradients) {
        let params = model.params();
        let mut total_penalty = Scalar::<B>::new(0.0);

        let layer_grads: Vec<LayerParams<B>> = params
            .layers
            .iter()
            .map(|layer| {
                // L1 penalty on weights: λ * ||W||₁
                let weights_flat = layer.weights.ravel();
                let abs_weights = weights_flat.abs();
                let l1_norm = abs_weights.sum();
                total_penalty = total_penalty + self.lambda * l1_norm;

                // Gradient: λ * sign(W) for weights, 0 for bias
                let weight_grad = layer.weights.sign().scale(self.lambda);
                let bias_grad = Tensor1D::<B>::zeros(layer.bias.len());

                LayerParams {
                    weights: weight_grad,
                    bias: bias_grad,
                }
            })
            .collect();

        (
            total_penalty,
            MLPParams {
                layers: layer_grads,
            },
        )
    }
}

impl<B: Backend> Regularizer<B, MLP<B>> for NoRegularizer {
    fn regularizer_penalty_grad(
        &self,
        model: &MLP<B>,
    ) -> (Scalar<B>, <MLP<B> as TrainableModel<B>>::Gradients) {
        let params = model.params();

        let layer_grads: Vec<LayerParams<B>> = params
            .layers
            .iter()
            .map(|layer| LayerParams {
                weights: Tensor2D::<B>::zeros(layer.weights.shape().0, layer.weights.shape().1),
                bias: Tensor1D::<B>::zeros(layer.bias.len()),
            })
            .collect();

        (
            Scalar::<B>::new(0.0),
            MLPParams {
                layers: layer_grads,
            },
        )
    }
}

use crate::backend::Tensor2D;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::model::linear::{LinearParams, LinearRegression};

    #[test]
    fn test_l2_regularizer() {
        // Создаём параметры напрямую
        let weights = Tensor1D::<CpuBackend>::new(vec![3.0f32, 4.0]);
        let bias = Scalar::<CpuBackend>::new(1.0);
        let params = LinearParams { weights, bias };

        let model = LinearRegression::<CpuBackend>::from_params(params);

        let lambda = 0.5;
        let l2 = L2::<CpuBackend>::new(lambda);

        let (penalty, grad) = l2.regularizer_penalty_grad(&model);

        // ||w||² = 3² + 4² = 25
        // penalty = λ * ||w||² = 0.5 * 25 = 12.5
        assert!((penalty.data - 12.5).abs() < 1e-12);

        // grad_w = 2 * λ * w = 2 * 0.5 * [3, 4] = [3, 4]
        assert_eq!(grad.weights.to_vec(), vec![3.0, 4.0]);
        // grad_b = 0
        assert_eq!(grad.bias.data, 0.0);
    }

    #[test]
    fn test_no_regularizer() {
        let weights = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0]);
        let bias = Scalar::<CpuBackend>::new(5.0);
        let params = LinearParams { weights, bias };
        let model = LinearRegression::<CpuBackend>::from_params(params);

        let no_reg = NoRegularizer;
        let (penalty, grad) = no_reg.regularizer_penalty_grad(&model);

        assert_eq!(penalty.data, 0.0);
        assert_eq!(grad.weights.to_vec(), vec![0.0, 0.0, 0.0]);
        assert_eq!(grad.bias.data, 0.0);
    }

    #[test]
    fn test_l2_zero_weights() {
        let weights = Tensor1D::<CpuBackend>::zeros(2);
        let bias = Scalar::<CpuBackend>::new(0.0);
        let params = LinearParams { weights, bias };
        let model = LinearRegression::<CpuBackend>::from_params(params);

        let l2 = L2::<CpuBackend>::new(1.0);
        let (penalty, grad) = l2.regularizer_penalty_grad(&model);

        assert_eq!(penalty.data, 0.0);
        assert_eq!(grad.weights.to_vec(), vec![0.0, 0.0]);
        assert_eq!(grad.bias.data, 0.0);
    }

    #[test]
    fn test_l1_regularizer_positive_weights() {
        let weights = Tensor1D::<CpuBackend>::new(vec![3.0f32, 4.0]);
        let bias = Scalar::<CpuBackend>::new(1.0);
        let params = LinearParams { weights, bias };
        let model = LinearRegression::<CpuBackend>::from_params(params);

        let lambda = 0.5;
        let l1 = L1::<CpuBackend>::new(lambda);

        let (penalty, grad) = l1.regularizer_penalty_grad(&model);

        // ||w||₁ = |3| + |4| = 7
        // penalty = λ * ||w||₁ = 0.5 * 7 = 3.5
        assert!((penalty.data - 3.5).abs() < 1e-12);

        // grad_w = λ * sign(w) = 0.5 * [1, 1] = [0.5, 0.5]
        assert_eq!(grad.weights.to_vec(), vec![0.5, 0.5]);
        // grad_b = 0
        assert_eq!(grad.bias.data, 0.0);
    }

    #[test]
    fn test_l1_regularizer_mixed_sign_weights() {
        let weights = Tensor1D::<CpuBackend>::new(vec![-2.0f32, 3.0]);
        let bias = Scalar::<CpuBackend>::new(1.0);
        let params = LinearParams { weights, bias };
        let model = LinearRegression::<CpuBackend>::from_params(params);

        let lambda = 1.0;
        let l1 = L1::<CpuBackend>::new(lambda);

        let (penalty, grad) = l1.regularizer_penalty_grad(&model);

        // ||w||₁ = |-2| + |3| = 5
        // penalty = λ * ||w||₁ = 1.0 * 5 = 5.0
        assert!((penalty.data - 5.0).abs() < 1e-12);

        // grad_w = λ * sign(w) = 1.0 * [-1, 1] = [-1.0, 1.0]
        assert_eq!(grad.weights.to_vec(), vec![-1.0, 1.0]);
        // grad_b = 0
        assert_eq!(grad.bias.data, 0.0);
    }

    #[test]
    fn test_l1_regularizer_with_zero_weights() {
        let weights = Tensor1D::<CpuBackend>::new(vec![0.0f32, 2.0]);
        let bias = Scalar::<CpuBackend>::new(0.5);
        let params = LinearParams { weights, bias };
        let model = LinearRegression::<CpuBackend>::from_params(params);

        let lambda = 1.0;
        let l1 = L1::<CpuBackend>::new(lambda);

        let (penalty, grad) = l1.regularizer_penalty_grad(&model);

        // ||w||₁ = |0| + |2| = 2
        // penalty = λ * ||w||₁ = 1.0 * 2 = 2.0
        assert!((penalty.data - 2.0).abs() < 1e-12);

        // grad_w = λ * sign(w) = 1.0 * [0, 1] = [0.0, 1.0]
        // sign(0) = 0 (subgradient at zero)
        assert_eq!(grad.weights.to_vec(), vec![0.0, 1.0]);
        // grad_b = 0
        assert_eq!(grad.bias.data, 0.0);
    }

    #[test]
    fn test_l1_regularizer_zero_lambda() {
        let weights = Tensor1D::<CpuBackend>::new(vec![3.0f32, 4.0]);
        let bias = Scalar::<CpuBackend>::new(1.0);
        let params = LinearParams { weights, bias };
        let model = LinearRegression::<CpuBackend>::from_params(params);

        let l1 = L1::<CpuBackend>::new(0.0);

        let (penalty, grad) = l1.regularizer_penalty_grad(&model);

        // With λ = 0, both penalty and gradient should be zero
        assert_eq!(penalty.data, 0.0);
        assert_eq!(grad.weights.to_vec(), vec![0.0, 0.0]);
        assert_eq!(grad.bias.data, 0.0);
    }
}
