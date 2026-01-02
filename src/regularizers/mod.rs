use crate::backend::{Backend, ScalarOps};
use crate::loss::LinearParams;
use crate::model::{TrainableModel};
use crate::model::linear::LinearRegression;
pub trait Regularizer<B: Backend, M: TrainableModel<B>>{
    fn regularizer_penalty_grad(&self, model: &M) -> (B::Scalar, M::Gradients);
}

struct L2<B: Backend>{
    lambda: B::Scalar,
}

impl <B> Regularizer<B, LinearRegression::<B>> for L2<B>
    where 
    B: Backend
{

    fn regularizer_penalty_grad(&self, model: &LinearRegression::<B>) -> (B::Scalar, <LinearRegression<B> as TrainableModel<B>>::Gradients){
        let params = model.params();
        let weight_grad = B::scale_1d(self.lambda * B::Scalar::from_f64(2.), &params.weights);

        let loss = B::dot(&params.weights, &params.weights);

        let loss = self.lambda * loss;

        (loss, LinearParams::<B>{weights: weight_grad, bias: B::Scalar::from_f64(0.)})
        
    }
}

pub struct NoRegularizer;

impl<B> Regularizer<B, LinearRegression::<B>> for NoRegularizer
where
    B: Backend,
{
    fn regularizer_penalty_grad(&self, model: &LinearRegression::<B>) -> (B::Scalar, <LinearRegression<B> as TrainableModel<B>>::Gradients){
        let params = model.params();
        let weight_grad = B::zeros_1d(B::len_1d(&params.weights), &B::default_device());

        (B::Scalar::from_f64(0.), LinearParams::<B>{weights: weight_grad, bias: B::Scalar::from_f64(0.)})
        
    }
}