
pub use crate::backend::Backend;
pub use crate::model::{TrainableModel, ForwardResult};

pub struct Unfitted;
pub struct Fitted;

//#[derive(Clone)]
pub struct LinearParams<B: Backend>
where
    B::Tensor1D: Clone,
    B::Scalar: Clone,
{
    weights: B::Tensor1D,
    bias: B::Scalar,
}

impl<B: Backend> Clone for LinearParams<B>
where
    B::Tensor1D: Clone,
    B::Scalar: Clone,
{
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
        }
    }
}

pub struct LinearModel<B: Backend, S> {
    params: LinearParams<B>,
    _state: std::marker::PhantomData<S>,
}

pub struct LinearForwardContext<B:Backend>{
    params: LinearParams<B>,
    grads: LinearParams<B>,
}

impl <B: Backend> TrainableModel<B, B::Tensor2D> for LinearModel<B, Unfitted>{
   type ForwardContext = LinearForwardContext<B>;
   type Params = LinearParams<B>;
   type Prediction = B::Tensor1D;

    fn forward(&self, x: B::Tensor2D) -> ForwardResult<Self, B, B::Tensor2D>{
        let mut preds = B::matvec(&x, &self.params.weights);
        B::add_scalar_1d_inplace(&mut preds, self.params.bias);
        let cw = self.params.clone();
        let cg = self.params.clone();
        ForwardResult{output: preds, context: Self::ForwardContext{params: cw, grads: cg}}
    }
    fn update_params(&mut self, params: &Self::Params){

    }

}