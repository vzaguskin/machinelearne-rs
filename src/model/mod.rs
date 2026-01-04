pub mod state;
pub use state::{Unfitted, Fitted};

pub mod linear;
pub use crate::backend::{Backend, ScalarOps, Tensor};



pub trait TrainableModel<B: Backend> {
    type Input;
    type Prediction;
    type Params;
    type Gradients;
    type Output;

    
    fn forward(&self, input: &Self::Input) -> Self::Prediction;
    fn backward(&self, input: &Self::Input, grad_output: &Self::Prediction) -> Self::Gradients;
    fn params(&self) -> &Self::Params;
    fn update_params(&mut self, new_params: &Self::Params);

    fn into_fitted(self) -> Self::Output;

}

pub trait ParamOps<B: Backend>: Clone {
    fn add(&self, other: &Self) -> Self;
    fn scale(&self, scalar: B::Scalar) -> Self;
}