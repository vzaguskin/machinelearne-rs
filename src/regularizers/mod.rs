use crate::backend::backend::Backend;
use crate::loss::{LinearParams, Tensor1D};
use crate::model::{TrainableModel};
use crate::model::linear::LinearRegression;
use crate::backend::scalar::{Scalar, ScalarOps};
pub trait Regularizer<B: Backend, M: TrainableModel<B>>{
    fn regularizer_penalty_grad(&self, model: &M) -> (Scalar<B>, M::Gradients);
}

pub struct L2<B: Backend>{
    lambda: Scalar<B>,
}

impl <B: Backend> L2<B>{
    pub fn new(lambda: f64) -> Self{
        Self{lambda: Scalar::<B>::new(lambda)}
    }
}

impl <B> Regularizer<B, LinearRegression::<B>> for L2<B>
    where 
    B: Backend
{

    fn regularizer_penalty_grad(&self, model: &LinearRegression::<B>) -> (Scalar<B>, <LinearRegression<B> as TrainableModel<B>>::Gradients){
        let params = model.params();
        let weight_grad = params.weights.scale(&(self.lambda * Scalar::<B>::new(2.)));

        let loss = params.weights.dot(&params.weights);

        let loss = self.lambda * loss;

        (loss, LinearParams::<B>{weights: weight_grad, bias: Scalar::<B>::new(0.)})
        
    }
}

pub struct NoRegularizer;

impl<B> Regularizer<B, LinearRegression::<B>> for NoRegularizer
where
    B: Backend,
{
    fn regularizer_penalty_grad(&self, model: &LinearRegression::<B>) -> (Scalar<B>, <LinearRegression<B> as TrainableModel<B>>::Gradients){
        let params = model.params();
        let weight_grad = Tensor1D::<B>::zeros(params.weights.len().data.to_f64() as usize);

        (Scalar::<B>::new(0.), LinearParams::<B>{weights: weight_grad, bias: Scalar::<B>::new(0.)})
        
    }
}