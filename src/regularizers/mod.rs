use crate::backend::backend::Backend;
use crate::backend::scalar::{Scalar, ScalarOps};
use crate::loss::{LinearParams, Tensor1D};
use crate::model::linear::LinearRegression;
use crate::model::TrainableModel;
pub trait Regularizer<B: Backend, M: TrainableModel<B>> {
    fn regularizer_penalty_grad(&self, model: &M) -> (Scalar<B>, M::Gradients);
}

pub struct L2<B: Backend> {
    lambda: Scalar<B>,
}

impl<B: Backend> L2<B> {
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
}
