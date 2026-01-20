
pub use crate::backend::backend::{Backend};
pub use crate::backend::scalar::{ScalarOps, Scalar};
pub use crate::backend::tensor1d::Tensor1D;
pub use crate::backend::tensor2d::Tensor2D;
use crate::loss::TensorLike;
pub use crate::model::{TrainableModel, InferenceModel, Unfitted, Fitted, ParamOps};
use std::marker::PhantomData;



#[derive(Clone)]
pub struct LinearParams<B: Backend>
where
    Tensor1D<B>: Clone,
    Scalar<B>: Clone,
{
    pub weights: Tensor1D<B>,
    pub bias: Scalar<B>,
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SerializableLinearParams {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl<B: Backend> From<&LinearParams<B>> for SerializableLinearParams {
    fn from(params: &LinearParams<B>) -> Self {
        // Предполагаем, что эти методы существуют в вашем бэкенде
        let weights = params.weights.to_vec().into_iter().map(|x| {x as f32}).collect(); // Vec<f32>
        let bias = params.bias.data.to_f64() as f32;        // f32
        Self { weights, bias }
    }
}

impl<B: Backend> TryFrom<SerializableLinearParams> for LinearParams<B> {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: SerializableLinearParams) -> Result<Self, Self::Error> {
        let weights = Tensor1D::<B>::new(value.weights);
        let bias = Scalar::<B>::new(value.bias as f64);
        Ok(Self { weights, bias })
    }
}

impl <B> ParamOps<B> for LinearParams<B>
where B: Backend
{
    fn add(&self, other: &Self) -> Self{
        let w = self.weights.add(&other.weights);
        let b = self.bias.clone() + other.bias.clone();
        Self{weights: w, bias: b}
    }
    fn scale(&self, scalar: Scalar<B>) -> Self{
        let w = self.weights.scale(&scalar); 
        let b = self.bias.clone() * scalar.clone();
        Self{weights: w, bias: b}

    }
}
pub struct LinearModel<B: Backend, S> {
    params: LinearParams<B>,
    _state: std::marker::PhantomData<S>,
}

impl<B: Backend> LinearModel<B, Fitted> {
    /// Create a new fitted linear model from params.
    pub fn new(params: LinearParams<B>) -> Self {
        Self{params, _state: std::marker::PhantomData::<Fitted>}
    }
}


impl<B: Backend> InferenceModel<B> for LinearModel<B, Fitted> {
    type InputSingle = Tensor1D<B>;
    type InputBatch = Tensor2D<B>;
    type OutputSingle = Scalar<B>;
    type OutputBatch = Tensor1D<B>;
    type ParamsRepr = SerializableLinearParams;

    /// Predict on a single sample (feature vector).
    fn predict(&self, input: &Self::InputSingle) -> Self::OutputSingle
    {
        self.params.weights.dot(input) + self.params.bias.clone()
    }

    fn predict_batch(&self, input: &Self::InputBatch) -> Self::OutputBatch{
        input.dot(&self.params.weights).add_scalar(&self.params.bias)

    }

    fn extract_params(&self) -> Self::ParamsRepr {
        (&self.params).into()
    }

    fn from_params(params: Self::ParamsRepr) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        let internal_params = LinearParams::<B>::try_from(params)?;
        Ok(Self::new(internal_params))
    }
}

impl <B: Backend> TrainableModel<B> for LinearModel<B, Unfitted>{
   type Params = LinearParams<B>;
   type Gradients = LinearParams<B>;
   type Prediction = Tensor1D<B>;
   type Input = Tensor2D<B>;
   type Output = LinearModel<B, Fitted>;

    fn forward(&self, x: &Self::Input) -> Self::Prediction{
        x.dot(&self.params.weights).add_scalar(&self.params.bias)
    }

    fn params(&self) -> &Self::Params{
        &self.params
    }
    
    fn update_params(&mut self, params: &Self::Params){
        self.params = params.clone();

    }

    fn into_fitted(self) -> LinearModel<B, Fitted>
    {
        LinearModel::<B, Fitted>::new(self.params)
    }

    fn backward(&self, x: &Self::Input, grad_output: &Self::Prediction) -> Self::Gradients {
        let grad_weights = x.tdot(grad_output);
        let grad_bias = grad_output.sum();
        LinearParams {
            weights: grad_weights,
            bias: grad_bias,
        }
    }

}

pub type LinearRegression<B> = LinearModel<B, Unfitted>;

impl<B: Backend> LinearRegression<B> {
    pub fn new(n_features: usize) -> Self {
        let params = LinearParams {
            weights: Tensor1D::<B>::zeros(n_features),
            bias: Scalar::<B>::new(0.),
        };
        Self { params, _state: PhantomData }
    }

    pub fn from_params(params: LinearParams<B>) -> Self {
        Self { params, _state: PhantomData }
    }
}

// Удобный алиас для CPU (в lib.rs или linear/mod.rs)
pub type LinearRegressor = LinearRegression<crate::backend::CpuBackend>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_linear_model_save_load() -> Result<(), Box<dyn std::error::Error>> {
        let weights = vec![1.0, 2.0, 3.0];
        let bias = 0.5;
        let serial = SerializableLinearParams { weights, bias };
        let params = LinearParams::<CpuBackend>::try_from(serial)?;
        let model = LinearModel::<CpuBackend, Fitted>::new(params);

        // Save
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("model.bin");
        model.save_to_file(&path)?;

        // Load
        let loaded = LinearModel::<CpuBackend, Fitted>::load_from_file(&path)?;

        // Compare
        let orig_repr = model.extract_params();
        let loaded_repr = loaded.extract_params();
        assert_eq!(orig_repr.weights, loaded_repr.weights);
        assert_eq!(orig_repr.bias, loaded_repr.bias);

        Ok(())
    }
}