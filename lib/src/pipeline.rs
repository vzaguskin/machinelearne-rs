// src/pipeline.rs
pub trait TrainablePipeline {
    type Model;
    type Loss;
    type Optimizer;
    type Backend;

    fn fit(
        trainer: &Trainer<Self::Model, Self::Loss, Self::Optimizer>,
        model: Self::Model,
        x: &<Self::Backend as Backend>::Tensor2D,
        y: &<Self::Backend as Backend>::Tensor1D,
    ) -> <Self::Model as Model>::Fitted;
}
