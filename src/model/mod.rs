pub mod state;
pub use state::{Unfitted, Fitted};

pub mod linear;
pub use crate::backend::backend::{Backend};
pub use crate::backend::scalar::{ScalarOps, Scalar};



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
    fn scale(&self, scalar: Scalar<B>) -> Self;
}

pub trait InferenceModel<B: Backend>{
    type InputSingle;
    type OutputSingle;
    type InputBatch;
    type OutputBatch;
    fn predict(&self, input: &Self::InputSingle) -> Self::OutputSingle;
    fn predict_batch(&self, input: &Self::InputBatch) -> Self::OutputBatch;

}