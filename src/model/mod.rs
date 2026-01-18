pub mod state;
pub use state::{Unfitted, Fitted};

pub mod linear;
pub use crate::backend::backend::{Backend};
pub use crate::backend::scalar::{ScalarOps, Scalar};
use crate::serialization::SerializableParams;



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
    type ParamsRepr: SerializableParams;
    fn predict(&self, input: &Self::InputSingle) -> Self::OutputSingle;
    fn predict_batch(&self, input: &Self::InputBatch) -> Self::OutputBatch;
    fn extract_params(&self) -> Self::ParamsRepr;
    fn from_params(params: Self::ParamsRepr) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;

    fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let bytes = self.extract_params().to_bytes()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, bytes)
    }

    fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        let bytes = std::fs::read(path)?;
        let params = Self::ParamsRepr::from_bytes(&bytes)
            .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
        Self::from_params(params)
    }
}