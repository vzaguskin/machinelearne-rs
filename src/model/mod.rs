mod linear;
pub use crate::backend::{Backend, ScalarOps, Tensor};

pub struct ForwardResult<M: TrainableModel<B, Input>, B: Backend, Input> {
    pub output: M::Prediction,
    pub(crate) context: M::ForwardContext, // ← выносим контекст в ассоциированный тип модели
}


pub trait TrainableModel<B: Backend, Input> {
    type Params;
    type ForwardContext;
    type Prediction;

    fn forward(&self, x: Input) -> ForwardResult<Self, B, Input>
    where
        Self: Sized;
    fn update_params(&mut self, params: &Self::Params);
}