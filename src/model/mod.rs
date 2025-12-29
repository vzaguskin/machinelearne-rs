pub mod linear;
pub use crate::backend::{Backend, ScalarOps, Tensor};



pub trait TrainableModel<B: Backend, Output> {
    type Input;
    type Prediction;
    type Params;
    type Gradients;

    
    fn forward(&self, input: &Self::Input) -> Self::Prediction;
    fn backward(&self, input: &Self::Input, grad_output: &Self::Prediction) -> Self::Gradients;
    fn params(&self) -> &Self::Params;
    fn update_params(&mut self, new_params: &Self::Params);

    fn into_fitted(self) -> Output;

}