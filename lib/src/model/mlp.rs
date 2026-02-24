//! Multi-Layer Perceptron (MLP) neural network.
//!
//! This module implements a configurable feedforward neural network with:
//! - Variable number of hidden layers
//! - Configurable activation functions per layer
//! - Full gradient computation for backpropagation
//!
//! # Architecture
//!
//! An MLP consists of multiple dense layers:
//! ```text
//! Input -> [Linear -> Activation] x N -> Output
//! ```
//!
//! # Example
//!
//! ```rust
//! use machinelearne_rs::model::{MLP, Activation, InferenceModel};
//! use machinelearne_rs::backend::CpuBackend;
//!
//! // Create: 2 inputs -> 4 hidden (ReLU) -> 1 output (Identity)
//! let model = MLP::<CpuBackend>::new(
//!     &[2, 4, 1],
//!     &[Activation::ReLU, Activation::Identity]
//! );
//! ```

use crate::backend::tensorlike::TensorLike;
use crate::backend::{Backend, CpuBackend, Scalar, Tensor1D, Tensor2D};
use crate::model::state::{Fitted, Unfitted};
use crate::model::{Activation, InferenceModel, ParamOps, TrainableModel};
use std::marker::PhantomData;

// =============================================================================
// Layer Parameters
// =============================================================================

/// Parameters for a single dense layer: weights and bias.
///
/// For a layer with `in_features` inputs and `out_features` outputs:
/// - `weights`: Shape (out_features, in_features)
/// - `bias`: Shape (out_features,)
#[derive(Clone)]
pub struct LayerParams<B: Backend> {
    pub weights: Tensor2D<B>,
    pub bias: Tensor1D<B>,
}

impl<B: Backend> LayerParams<B> {
    /// Creates layer parameters initialized to zeros.
    ///
    /// For production use, prefer `random_init()` for proper weight initialization.
    pub fn zeros(in_features: usize, out_features: usize) -> Self {
        Self {
            weights: Tensor2D::zeros(out_features, in_features),
            bias: Tensor1D::zeros(out_features),
        }
    }

    /// Creates layer parameters with Xavier/Glorot initialization.
    ///
    /// Weights are initialized from a uniform distribution:
    /// `U(-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out)))`
    ///
    /// Biases are initialized to zero.
    pub fn xavier_init(in_features: usize, out_features: usize) -> Self {
        let scale = (6.0 / (in_features + out_features) as f64).sqrt() as f32;

        // Use a simple deterministic pseudo-random sequence for initialization
        // This ensures reproducibility while breaking symmetry
        let n = out_features * in_features;
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            // Simple pseudo-random based on position
            let x = ((i * 1103515245 + 12345) % 2147483648) as f32 / 2147483647.0;
            let rand_val = x * 2.0 - 1.0; // Map to [-1, 1]
            weights.push(rand_val * scale);
        }

        let weights = Tensor2D::new(weights, out_features, in_features);
        let bias = Tensor1D::zeros(out_features);
        Self { weights, bias }
    }
}

/// Serializable representation of layer parameters.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SerializableLayerParams {
    pub weights: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
    pub bias: Vec<f32>,
}

impl<B: Backend> From<&LayerParams<B>> for SerializableLayerParams {
    fn from(params: &LayerParams<B>) -> Self {
        let (rows, cols) = params.weights.shape();
        Self {
            weights: params
                .weights
                .ravel()
                .to_vec()
                .into_iter()
                .map(|x| x as f32)
                .collect(),
            in_features: cols,
            out_features: rows,
            bias: params.bias.to_vec().into_iter().map(|x| x as f32).collect(),
        }
    }
}

impl<B: Backend> TryFrom<SerializableLayerParams> for LayerParams<B> {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: SerializableLayerParams) -> Result<Self, Self::Error> {
        let weights = Tensor2D::<B>::new(value.weights, value.out_features, value.in_features);
        let bias = Tensor1D::<B>::new(value.bias);
        Ok(Self { weights, bias })
    }
}

// =============================================================================
// MLP Parameters
// =============================================================================

/// Container for all MLP layer parameters.
#[derive(Clone)]
pub struct MLPParams<B: Backend> {
    pub layers: Vec<LayerParams<B>>,
}

impl<B: Backend> MLPParams<B> {
    /// Creates parameters for an MLP with the given layer sizes.
    ///
    /// Uses Xavier initialization for weights and zero initialization for biases.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "MLP must have at least 2 layers (input -> output)"
        );

        let layers: Vec<LayerParams<B>> = layer_sizes
            .windows(2)
            .map(|w| LayerParams::xavier_init(w[0], w[1]))
            .collect();

        Self { layers }
    }
}

/// Serializable representation of MLP parameters.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SerializableMLPParams {
    pub layers: Vec<SerializableLayerParams>,
}

impl<B: Backend> From<&MLPParams<B>> for SerializableMLPParams {
    fn from(params: &MLPParams<B>) -> Self {
        Self {
            layers: params.layers.iter().map(|l| l.into()).collect(),
        }
    }
}

impl<B: Backend> TryFrom<SerializableMLPParams> for MLPParams<B> {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: SerializableMLPParams) -> Result<Self, Self::Error> {
        let layers: Vec<LayerParams<B>> = value
            .layers
            .into_iter()
            .map(|l| l.try_into())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }
}

// =============================================================================
// ParamOps Implementation
// =============================================================================

impl<B: Backend> ParamOps<B> for MLPParams<B> {
    fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.layers.len(),
            other.layers.len(),
            "Cannot add MLPParams with different number of layers"
        );

        let layers: Vec<LayerParams<B>> = self
            .layers
            .iter()
            .zip(other.layers.iter())
            .map(|(a, b)| LayerParams {
                weights: a.weights.add(&b.weights),
                bias: a.bias.add(&b.bias),
            })
            .collect();

        Self { layers }
    }

    fn scale(&self, scalar: Scalar<B>) -> Self {
        let layers: Vec<LayerParams<B>> = self
            .layers
            .iter()
            .map(|l| LayerParams {
                weights: l.weights.scale(scalar),
                bias: l.bias.scale(&scalar),
            })
            .collect();

        Self { layers }
    }

    fn l2_norm(&self) -> Scalar<B> {
        let mut total_sq = Scalar::<B>::new(0.0);

        for layer in &self.layers {
            // L2 norm of weights
            let w_flat = layer.weights.ravel();
            let w_sq = w_flat.dot(&w_flat);
            // L2 norm of bias
            let b_sq = layer.bias.dot(&layer.bias);
            total_sq = total_sq + w_sq + b_sq;
        }

        total_sq.sqrt()
    }
}

// =============================================================================
// MLP Model
// =============================================================================

/// Multi-Layer Perceptron model with type-state for training/inference separation.
///
/// # Type Parameters
/// - `B`: Backend for tensor operations
/// - `S`: State marker (`Unfitted` for training, `Fitted` for inference)
#[derive(Clone)]
pub struct MLPModel<B: Backend, S> {
    params: MLPParams<B>,
    activations: Vec<Activation>,
    layer_sizes: Vec<usize>,
    _state: PhantomData<S>,
}

/// Type alias for unfitted MLP (training mode).
pub type MLP<B> = MLPModel<B, Unfitted>;

/// Type alias for CPU-based MLP regressor.
pub type MLPRegressor = MLP<CpuBackend>;

impl<B: Backend> MLPModel<B, Unfitted> {
    /// Creates a new MLP model with the given architecture.
    ///
    /// # Arguments
    /// * `layer_sizes` - Slice of layer sizes, e.g., `[input, hidden1, hidden2, output]`
    /// * `activations` - Slice of activation functions for each layer transition
    ///
    /// # Panics
    /// Panics if:
    /// - `layer_sizes` has fewer than 2 elements
    /// - `activations` length doesn't match `layer_sizes.len() - 1`
    ///
    /// # Example
    /// ```rust
    /// use machinelearne_rs::model::{MLP, Activation};
    /// use machinelearne_rs::backend::CpuBackend;
    ///
    /// // 2 inputs -> 4 hidden (ReLU) -> 1 output (Identity)
    /// let model = MLP::<CpuBackend>::new(
    ///     &[2, 4, 1],
    ///     &[Activation::ReLU, Activation::Identity]
    /// );
    /// ```
    pub fn new(layer_sizes: &[usize], activations: &[Activation]) -> Self {
        assert!(layer_sizes.len() >= 2, "MLP must have at least 2 layers");
        assert_eq!(
            activations.len(),
            layer_sizes.len() - 1,
            "Number of activations must match number of layer transitions"
        );

        Self {
            params: MLPParams::new(layer_sizes),
            activations: activations.to_vec(),
            layer_sizes: layer_sizes.to_vec(),
            _state: PhantomData,
        }
    }
}

// =============================================================================
// TrainableModel Implementation
// =============================================================================

/// Cache for storing intermediate values during forward pass.
pub struct ForwardCache<B: Backend> {
    /// Pre-activation values (z = Wx + b) for each layer - shape: (batch_size, out_features)
    pub pre_activations: Vec<Tensor2D<B>>,
    /// Post-activation values (a = activation(z)) for each layer - shape: (batch_size, out_features)
    pub post_activations: Vec<Tensor2D<B>>,
}

impl<B: Backend> TrainableModel<B> for MLPModel<B, Unfitted> {
    type Input = Tensor2D<B>;
    type Prediction = Tensor1D<B>; // For Trainer compatibility with single-output models
    type Params = MLPParams<B>;
    type Gradients = MLPParams<B>;
    type Output = MLPModel<B, Fitted>;

    fn forward(&self, input: &Self::Input) -> Self::Prediction {
        let (cache, _) = self.forward_with_cache(input);
        // Get the final output (batch_size, output_features)
        let output_2d = cache.post_activations.last().unwrap();

        // Flatten to 1D: for single-output models, this is (batch_size,)
        // For multi-output models, this flattens to (batch_size * output_features,)
        let output_data = output_2d.ravel().to_vec();
        Tensor1D::new(output_data.into_iter().map(|x| x as f32).collect())
    }

    fn backward(&self, input: &Self::Input, grad_output: &Self::Prediction) -> Self::Gradients {
        let (cache, batch_size) = self.forward_with_cache(input);

        let num_layers = self.params.layers.len();
        let mut layer_grads = Vec::with_capacity(num_layers);

        // Get output features from the last layer
        let output_features = *self.layer_sizes.last().unwrap();

        // Reshape 1D gradient back to 2D: (batch_size, output_features)
        let grad_data = grad_output.to_vec();
        let mut delta = Tensor2D::new(
            grad_data.into_iter().map(|x| x as f32).collect(),
            batch_size,
            output_features,
        );

        // Backpropagate through layers (reverse order)
        for i in (0..num_layers).rev() {
            // Apply activation derivative (element-wise, preserves batch dimension)
            delta = self.activations[i].backward_2d(&cache.pre_activations[i], &delta);

            // Get input to this layer - shape: (batch_size, input_features)
            let layer_input = if i == 0 {
                input.clone()
            } else {
                cache.post_activations[i - 1].clone()
            };

            // Compute weight gradients: grad_W = delta^T @ layer_input / batch_size
            // delta shape: (batch_size, out_features) -> transposed: (out_features, batch_size)
            // layer_input shape: (batch_size, in_features)
            // Result: (out_features, in_features)
            let delta_t = delta.transpose();
            let grad_weights = delta_t.matmul(&layer_input);
            let grad_weights = grad_weights.scale(Scalar::new(1.0 / batch_size as f64));

            // Compute bias gradients: grad_b = mean(delta, axis=0) = sum(delta, axis=0) / batch_size
            // delta shape: (batch_size, out_features)
            // We need to sum across the batch dimension
            let grad_bias = sum_rows(&delta);

            layer_grads.insert(
                0,
                LayerParams {
                    weights: grad_weights,
                    bias: grad_bias,
                },
            );

            // Propagate delta to previous layer: delta = delta @ W
            // delta shape: (batch_size, out_features), W shape: (out_features, in_features)
            // Result: (batch_size, in_features)
            if i > 0 {
                delta = delta.matmul(&self.params.layers[i].weights);
            }
        }

        MLPParams {
            layers: layer_grads,
        }
    }

    fn params(&self) -> &Self::Params {
        &self.params
    }

    fn update_params(&mut self, new_params: &Self::Params) {
        self.params = new_params.clone();
    }

    fn into_fitted(self) -> Self::Output {
        MLPModel {
            params: self.params,
            activations: self.activations,
            layer_sizes: self.layer_sizes,
            _state: PhantomData,
        }
    }
}

/// Sum rows of a 2D tensor, returning a 1D tensor.
fn sum_rows<B: Backend>(tensor: &Tensor2D<B>) -> Tensor1D<B> {
    let (rows, cols) = tensor.shape();
    let data = tensor.ravel().to_vec();

    let mut result = vec![0.0f64; cols];
    for r in 0..rows {
        for c in 0..cols {
            result[c] += data[r * cols + c];
        }
    }

    Tensor1D::new(result.into_iter().map(|x| x as f32).collect())
}

impl<B: Backend> MLPModel<B, Unfitted> {
    /// Forward pass that also returns cached values for backpropagation.
    ///
    /// Returns (cache, batch_size) where cache contains pre/post activation values.
    fn forward_with_cache(&self, input: &Tensor2D<B>) -> (ForwardCache<B>, usize) {
        let (batch_size, _) = input.shape();

        let mut cache = ForwardCache {
            pre_activations: Vec::with_capacity(self.params.layers.len()),
            post_activations: Vec::with_capacity(self.params.layers.len()),
        };

        // Process the entire batch using matrix operations
        // current shape: (batch_size, in_features)
        let mut current = input.clone();

        for (i, layer) in self.params.layers.iter().enumerate() {
            // z = current @ W^T + b
            // current: (batch_size, in_features), W: (out_features, in_features)
            // We need to compute: (batch_size, in_features) @ (in_features, out_features) + b
            // So we use W.transpose() to get (in_features, out_features)
            let w_t = layer.weights.transpose();

            // z = current @ W^T -> shape: (batch_size, out_features)
            let z = current.matmul(&w_t);

            // Add bias to each row: z + b (broadcasting)
            let z = add_bias_to_rows(&z, &layer.bias);

            // a = activation(z) - element-wise, preserves shape
            let a = self.activations[i].forward_2d(&z);

            cache.pre_activations.push(z);
            cache.post_activations.push(a.clone());

            current = a;
        }

        (cache, batch_size)
    }
}

/// Add bias to each row of a 2D tensor.
fn add_bias_to_rows<B: Backend>(tensor: &Tensor2D<B>, bias: &Tensor1D<B>) -> Tensor2D<B> {
    let (rows, cols) = tensor.shape();
    let tensor_data = tensor.ravel().to_vec();
    let bias_data = bias.to_vec();

    let mut result = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            result.push((tensor_data[r * cols + c] + bias_data[c]) as f32);
        }
    }

    Tensor2D::new(result, rows, cols)
}

// =============================================================================
// InferenceModel Implementation
// =============================================================================

impl<B: Backend> InferenceModel<B> for MLPModel<B, Fitted> {
    type InputSingle = Tensor1D<B>;
    type OutputSingle = Tensor1D<B>;
    type InputBatch = Tensor2D<B>;
    type OutputBatch = Tensor2D<B>;
    type ParamsRepr = SerializableMLPParams;

    fn predict(&self, input: &Self::InputSingle) -> Self::OutputSingle {
        let mut current = input.clone();

        for (i, layer) in self.params.layers.iter().enumerate() {
            // z = W @ x + b
            let z = layer.weights.matvec(&current).add(&layer.bias);
            // a = activation(z)
            current = self.activations[i].forward_1d(&z);
        }

        current
    }

    fn predict_batch(&self, input: &Self::InputBatch) -> Self::OutputBatch {
        // Optimized batch inference using matrix operations
        // Process the entire batch at once for GPU efficiency
        let mut current = input.clone();

        for (i, layer) in self.params.layers.iter().enumerate() {
            // z = current @ W^T + b
            // current: (batch_size, in_features), W: (out_features, in_features)
            // W^T: (in_features, out_features)
            let w_t = layer.weights.transpose();

            // Matrix multiplication: (batch_size, in_features) @ (in_features, out_features)
            let z = current.matmul(&w_t);

            // Add bias to each row
            let z = add_bias_to_rows(&z, &layer.bias);

            // Apply activation (element-wise, preserves batch dimension)
            current = self.activations[i].forward_2d(&z);
        }

        current
    }

    fn extract_params(&self) -> Self::ParamsRepr {
        (&self.params).into()
    }

    fn from_params(params: Self::ParamsRepr) -> Result<Self, Box<dyn std::error::Error>> {
        let mlp_params: MLPParams<B> = params.try_into()?;

        // Reconstruct layer sizes from parameters
        let layer_sizes: Vec<usize> = std::iter::once(
            mlp_params
                .layers
                .first()
                .map(|l| l.weights.shape().1)
                .unwrap_or(0),
        )
        .chain(mlp_params.layers.iter().map(|l| l.weights.shape().0))
        .collect();

        // Default activations (ReLU for hidden, Identity for output)
        let activations: Vec<Activation> = (0..mlp_params.layers.len())
            .map(|i| {
                if i == mlp_params.layers.len() - 1 {
                    Activation::Identity
                } else {
                    Activation::ReLU
                }
            })
            .collect();

        Ok(Self {
            params: mlp_params,
            activations,
            layer_sizes,
            _state: PhantomData,
        })
    }
}

// =============================================================================
// Accessor Methods (for ONNX export and introspection)
// =============================================================================

impl<B: Backend> MLPModel<B, Fitted> {
    /// Returns the layer sizes: [input, hidden1, ..., output]
    pub fn layer_sizes(&self) -> &[usize] {
        &self.layer_sizes
    }

    /// Returns the activation functions for each layer.
    pub fn activations(&self) -> &[Activation] {
        &self.activations
    }

    /// Returns the parameters for each layer.
    pub fn layers(&self) -> &[LayerParams<B>] {
        &self.params.layers
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::scalar::ScalarOps;
    use crate::backend::CpuBackend;

    #[test]
    fn test_mlp_creation() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        assert_eq!(model.layer_sizes, vec![2, 4, 1]);
        assert_eq!(model.activations.len(), 2);
    }

    #[test]
    #[should_panic(expected = "MLP must have at least 2 layers")]
    fn test_mlp_creation_too_few_layers() {
        let _ = MLP::<CpuBackend>::new(&[2], &[]);
    }

    #[test]
    #[should_panic(expected = "Number of activations must match")]
    fn test_mlp_creation_wrong_activations() {
        let _ = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU]);
    }

    #[test]
    fn test_mlp_forward() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0], 1, 2);
        let output = model.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_param_ops_add() {
        let p1 = MLPParams::<CpuBackend>::new(&[2, 2]);
        let p2 = MLPParams::<CpuBackend>::new(&[2, 2]);
        let sum = p1.add(&p2);
        assert_eq!(sum.layers.len(), 1);
    }

    #[test]
    fn test_param_ops_scale() {
        let params = MLPParams::<CpuBackend>::new(&[2, 2]);
        let scaled = params.scale(Scalar::new(2.0));
        assert_eq!(scaled.layers.len(), 1);
    }

    #[test]
    fn test_param_ops_l2_norm() {
        let params = MLPParams::<CpuBackend>::new(&[2, 2]);
        let norm = params.l2_norm();
        // With Xavier initialization, norm should be positive
        assert!(norm.data.to_f64() > 0.0);
    }

    #[test]
    fn test_mlp_backward() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0], 1, 2);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![0.5]);

        let gradients = model.backward(&input, &grad_output);
        assert_eq!(gradients.layers.len(), 2);
    }

    #[test]
    fn test_mlp_forward_with_cache() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0], 1, 2);

        let (cache, batch_size) = model.forward_with_cache(&input);
        assert_eq!(batch_size, 1);
        assert_eq!(cache.pre_activations.len(), 2);
        assert_eq!(cache.post_activations.len(), 2);
    }

    #[test]
    fn test_mlp_batch_forward() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        // Batch of 3 samples
        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let output = model.forward(&input);
        assert_eq!(output.len(), 3); // 3 samples * 1 output
    }

    #[test]
    fn test_mlp_batch_backward() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        // Batch of 3 samples
        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![0.5, 0.3, 0.1]);

        let gradients = model.backward(&input, &grad_output);
        assert_eq!(gradients.layers.len(), 2);
    }

    #[test]
    fn test_mlp_update_params() {
        let mut model =
            MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        let new_params = MLPParams::<CpuBackend>::new(&[2, 4, 1]);
        model.update_params(&new_params);
    }

    #[test]
    fn test_mlp_into_fitted() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        let fitted = model.into_fitted();
        assert_eq!(fitted.layer_sizes(), &[2, 4, 1]);
    }

    #[test]
    fn test_mlp_predict_single() {
        use crate::model::InferenceModel;

        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        let fitted = model.into_fitted();

        let input = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]);
        let output = fitted.predict(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_mlp_predict_batch() {
        use crate::model::InferenceModel;

        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        let fitted = model.into_fitted();

        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let output = fitted.predict_batch(&input);
        assert_eq!(output.shape(), (2, 1));
    }

    #[test]
    fn test_mlp_various_activations() {
        // Test with different activation combinations
        let model = MLP::<CpuBackend>::new(
            &[2, 4, 4, 1],
            &[Activation::Sigmoid, Activation::Tanh, Activation::Identity],
        );
        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0], 1, 2);
        let output = model.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_layer_params_accessors() {
        let params = MLPParams::<CpuBackend>::new(&[2, 4, 1]);
        assert_eq!(params.layers.len(), 2);

        // Check first layer
        let layer0 = &params.layers[0];
        assert_eq!(layer0.weights.shape(), (4, 2)); // out x in
        assert_eq!(layer0.bias.len(), 4);
    }

    #[test]
    fn test_mlp_model_accessors() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        let fitted = model.into_fitted();

        assert_eq!(fitted.layer_sizes(), &[2, 4, 1]);
        assert_eq!(fitted.activations().len(), 2);
        assert_eq!(fitted.layers().len(), 2);
    }

    #[test]
    fn test_mlp_batch_backward_3_samples() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        // Batch of 3 samples
        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![0.5, 0.3, 0.1]);

        let gradients = model.backward(&input, &grad_output);
        assert_eq!(gradients.layers.len(), 2);

        // Verify gradient shapes
        assert_eq!(gradients.layers[0].weights.shape(), (4, 2));
        assert_eq!(gradients.layers[0].bias.len(), 4);
        assert_eq!(gradients.layers[1].weights.shape(), (1, 4));
        assert_eq!(gradients.layers[1].bias.len(), 1);
    }

    #[test]
    fn test_mlp_sigmoid_activation() {
        let model =
            MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::Sigmoid, Activation::Identity]);
        let input = Tensor2D::<CpuBackend>::new(vec![0.5, -0.5], 1, 2);
        let output = model.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_mlp_tanh_activation() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::Tanh, Activation::Identity]);
        let input = Tensor2D::<CpuBackend>::new(vec![0.5, -0.5], 1, 2);
        let output = model.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_mlp_backward_sigmoid() {
        let model =
            MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::Sigmoid, Activation::Identity]);
        let input = Tensor2D::<CpuBackend>::new(vec![0.5, -0.5], 1, 2);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![1.0]);

        let gradients = model.backward(&input, &grad_output);
        assert_eq!(gradients.layers.len(), 2);
    }

    #[test]
    fn test_mlp_backward_tanh() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::Tanh, Activation::Identity]);
        let input = Tensor2D::<CpuBackend>::new(vec![0.5, -0.5], 1, 2);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![1.0]);

        let gradients = model.backward(&input, &grad_output);
        assert_eq!(gradients.layers.len(), 2);
    }

    #[test]
    fn test_mlp_3_layer_backward() {
        let model = MLP::<CpuBackend>::new(
            &[2, 4, 4, 1],
            &[Activation::ReLU, Activation::Tanh, Activation::Identity],
        );
        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0], 1, 2);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![0.5]);

        let gradients = model.backward(&input, &grad_output);
        assert_eq!(gradients.layers.len(), 3);
    }

    #[test]
    fn test_mlp_predict_batch_multiple_outputs() {
        use crate::model::InferenceModel;

        // Model with 2 outputs
        let model = MLP::<CpuBackend>::new(&[3, 4, 2], &[Activation::ReLU, Activation::Identity]);
        let fitted = model.into_fitted();

        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let output = fitted.predict_batch(&input);
        assert_eq!(output.shape(), (2, 2)); // 2 samples, 2 outputs each
    }

    #[test]
    fn test_sum_rows_helper() {
        // Test the sum_rows helper function indirectly through backward pass
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0]);

        let gradients = model.backward(&input, &grad_output);
        // Verify bias gradient is computed (sum_rows is used for this)
        assert_eq!(gradients.layers[1].bias.len(), 1);
    }

    #[test]
    fn test_layer_params_zeros() {
        let params = LayerParams::<CpuBackend>::zeros(3, 2);
        assert_eq!(params.weights.shape(), (2, 3));
        assert_eq!(params.bias.len(), 2);
        // All values should be zero
        for w in params.weights.ravel().to_vec() {
            assert_eq!(w, 0.0);
        }
        for b in params.bias.to_vec() {
            assert_eq!(b, 0.0);
        }
    }

    #[test]
    fn test_layer_params_xavier_init() {
        let params = LayerParams::<CpuBackend>::xavier_init(4, 2);
        assert_eq!(params.weights.shape(), (2, 4));
        assert_eq!(params.bias.len(), 2);
        // Biases should be zero
        for b in params.bias.to_vec() {
            assert_eq!(b, 0.0);
        }
        // Weights should not all be zero (xavier init)
        let weights = params.weights.ravel().to_vec();
        let non_zero: Vec<_> = weights.iter().filter(|&&x| x != 0.0).collect();
        assert!(!non_zero.is_empty());
    }

    #[test]
    fn test_serializable_layer_params_roundtrip() {
        let original = LayerParams::<CpuBackend>::xavier_init(3, 2);
        let serializable: SerializableLayerParams = (&original).into();
        let restored: LayerParams<CpuBackend> = serializable.try_into().unwrap();

        assert_eq!(restored.weights.shape(), original.weights.shape());
        assert_eq!(restored.bias.len(), original.bias.len());
    }

    #[test]
    fn test_serializable_mlp_params_roundtrip() {
        let params = MLPParams::<CpuBackend>::new(&[2, 4, 3, 1]);
        let serializable: SerializableMLPParams = (&params).into();
        let restored: MLPParams<CpuBackend> = serializable.try_into().unwrap();

        assert_eq!(restored.layers.len(), params.layers.len());
    }

    #[test]
    fn test_mlp_extract_and_from_params() {
        use crate::model::InferenceModel;

        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        let fitted = model.into_fitted();

        // Extract params
        let params_repr = fitted.extract_params();

        // Reconstruct from params
        let reconstructed = MLPModel::<CpuBackend, Fitted>::from_params(params_repr).unwrap();

        assert_eq!(reconstructed.layer_sizes(), &[2, 4, 1]);
        // Default activations should be ReLU for hidden, Identity for output
        assert_eq!(reconstructed.activations()[0], Activation::ReLU);
        assert_eq!(reconstructed.activations()[1], Activation::Identity);

        // Should give same predictions
        let input = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]);
        let original_pred = fitted.predict(&input);
        let reconstructed_pred = reconstructed.predict(&input);

        // Predictions should be identical
        for (a, b) in original_pred
            .to_vec()
            .iter()
            .zip(reconstructed_pred.to_vec().iter())
        {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mlp_from_params_multi_layer() {
        use crate::model::InferenceModel;

        // Create a 3-hidden-layer model
        let model = MLP::<CpuBackend>::new(
            &[3, 4, 4, 2, 1],
            &[
                Activation::ReLU,
                Activation::Tanh,
                Activation::Sigmoid,
                Activation::Identity,
            ],
        );
        let fitted = model.into_fitted();
        let params_repr = fitted.extract_params();

        // Reconstruct
        let reconstructed = MLPModel::<CpuBackend, Fitted>::from_params(params_repr).unwrap();

        // Should have 4 layers (3->4, 4->4, 4->2, 2->1)
        assert_eq!(reconstructed.layers().len(), 4);
        // Default activations: first 3 are ReLU, last is Identity
        assert_eq!(reconstructed.activations()[0], Activation::ReLU);
        assert_eq!(reconstructed.activations()[1], Activation::ReLU);
        assert_eq!(reconstructed.activations()[2], Activation::ReLU);
        assert_eq!(reconstructed.activations()[3], Activation::Identity);
    }

    #[test]
    fn test_add_bias_to_rows() {
        // Test the add_bias_to_rows helper indirectly through predict
        use crate::model::InferenceModel;

        let model =
            MLP::<CpuBackend>::new(&[2, 3, 1], &[Activation::Identity, Activation::Identity]);

        // Set specific weights and biases
        let layer0 = &model.params.layers[0];
        assert_eq!(layer0.weights.shape(), (3, 2));

        let fitted = model.into_fitted();
        let input = Tensor2D::<CpuBackend>::new(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let output = fitted.predict_batch(&input);

        // Output shape should be (2, 1)
        assert_eq!(output.shape(), (2, 1));
    }

    #[test]
    fn test_mlp_predict_single_sigmoid() {
        use crate::model::InferenceModel;

        let model =
            MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::Sigmoid, Activation::Identity]);
        let fitted = model.into_fitted();

        let input = Tensor1D::<CpuBackend>::new(vec![0.0, 0.0]);
        let output = fitted.predict(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_mlp_predict_single_tanh() {
        use crate::model::InferenceModel;

        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::Tanh, Activation::Identity]);
        let fitted = model.into_fitted();

        let input = Tensor1D::<CpuBackend>::new(vec![0.0, 0.0]);
        let output = fitted.predict(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_mlp_batch_backward_with_sigmoid() {
        let model =
            MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::Sigmoid, Activation::Identity]);
        let input = Tensor2D::<CpuBackend>::new(vec![0.5, -0.5, 1.0, 1.0], 2, 2);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![1.0, 0.5]);

        let gradients = model.backward(&input, &grad_output);
        assert_eq!(gradients.layers.len(), 2);
        // Verify gradient shapes
        assert_eq!(gradients.layers[0].weights.shape(), (4, 2));
        assert_eq!(gradients.layers[1].weights.shape(), (1, 4));
    }

    #[test]
    fn test_mlp_batch_backward_with_tanh() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::Tanh, Activation::Identity]);
        let input = Tensor2D::<CpuBackend>::new(vec![0.5, -0.5, 1.0, 1.0], 2, 2);
        let grad_output = Tensor1D::<CpuBackend>::new(vec![1.0, 0.5]);

        let gradients = model.backward(&input, &grad_output);
        assert_eq!(gradients.layers.len(), 2);
    }

    #[test]
    fn test_mlp_large_batch() {
        let model = MLP::<CpuBackend>::new(&[4, 8, 2], &[Activation::ReLU, Activation::Identity]);
        // Batch of 10 samples
        let input_data: Vec<f32> = (0..40).map(|x| x as f32 * 0.1).collect();
        let input = Tensor2D::<CpuBackend>::new(input_data, 10, 4);
        let output = model.forward(&input);
        assert_eq!(output.len(), 20); // 10 samples * 2 outputs
    }

    #[test]
    fn test_mlp_clone() {
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        let cloned = model.clone();
        assert_eq!(cloned.layer_sizes, model.layer_sizes);
    }

    #[test]
    fn test_mlp_params_clone() {
        let params = MLPParams::<CpuBackend>::new(&[2, 4, 1]);
        let cloned = params.clone();
        assert_eq!(cloned.layers.len(), params.layers.len());
    }
}
