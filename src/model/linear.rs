
pub use crate::backend::backend::{Backend};
pub use crate::backend::scalar::{ScalarOps, Scalar};
pub use crate::backend::tensor1d::Tensor1D;
use crate::loss::TensorLike;
pub use crate::model::{TrainableModel, Unfitted, Fitted, ParamOps};
use std::marker::PhantomData;



//#[derive(Clone)]
pub struct LinearParams<B: Backend>
where
    Tensor1D<B>: Clone,
    B::Scalar: Clone,
{
    pub weights: Tensor1D<B>,
    pub bias: Scalar<B>,
}

impl<B: Backend> Clone for LinearParams<B>
where
    Tensor1D<B>: Clone,
    B::Scalar: Clone,
{
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
        }
    }
}

impl <B> ParamOps<B> for LinearParams<B>
where B: Backend
{
    fn add(&self, other: &Self) -> Self{
        let w = self.weights.add(&other.weights);
        let b = self.bias + other.bias;
        Self{weights: w, bias: b}
    }
    fn scale(&self, scalar: Scalar<B>) -> Self{
        let w = self.weights.scale(scalar); 
        let b = self.bias * scalar;
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

    /// Predict on a single sample (feature vector).
    pub fn predict(&self, x: &[f64]) -> f64
    where
        // We need to construct a Tensor1D from &[f64]
        // For now, assume Device::Cpu is always used in predict for simplicity
    {
        let device = B::default_device();

        // Build input tensor
        let mut x_tensor = B::zeros_1d(x.len(), &device);
        for (i, &val) in x.iter().enumerate() {
            B::set_1d(&mut x_tensor, i, B::Scalar::from_f64(val));
        }

        // Compute w·x
        let mut output = B::dot(&self.params.weights, &x_tensor);
        // Add bias
        output = output + self.params.bias;

        B::Scalar::to_f64(output)
    }
}

impl <B: Backend> TrainableModel<B> for LinearModel<B, Unfitted>{
   type Params = LinearParams<B>;
   type Gradients = LinearParams<B>;
   type Prediction = B::Tensor1D;
   type Input = B::Tensor2D;
   type Output = LinearModel<B, Fitted>;

    fn forward(&self, x: &Self::Input) -> Self::Prediction{
        let dp = B::matvec(&x, &self.params.weights);
        B::add_scalar_1d(&dp, self.params.bias)
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
        let grad_weights = B::matvec_transpose(x, grad_output);
        let grad_bias = B::sum_1d(grad_output);
        LinearParams {
            weights: grad_weights,
            bias: grad_bias,
        }
    }

}

pub type LinearRegression<B> = LinearModel<B, Unfitted>;

impl<B: Backend> LinearRegression<B> {
    pub fn new(n_features: usize) -> Self {
        let device = B::default_device();
        let params = LinearParams {
            weights: B::zeros_1d(n_features, &device),
            bias: B::Scalar::zero(),
        };
        Self { params, _state: PhantomData }
    }
}

// Удобный алиас для CPU (в lib.rs или linear/mod.rs)
pub type LinearRegressor = LinearRegression<crate::backend::CpuBackend>;