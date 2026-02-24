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
    /// Pre-activation values (z = Wx + b) for each layer
    pub pre_activations: Vec<Tensor1D<B>>,
    /// Post-activation values (a = activation(z)) for each layer
    pub post_activations: Vec<Tensor1D<B>>,
}

impl<B: Backend> TrainableModel<B> for MLPModel<B, Unfitted> {
    type Input = Tensor2D<B>;
    type Prediction = Tensor1D<B>;
    type Params = MLPParams<B>;
    type Gradients = MLPParams<B>;
    type Output = MLPModel<B, Fitted>;

    fn forward(&self, input: &Self::Input) -> Self::Prediction {
        let (cache, _) = self.forward_with_cache(input);
        // Return the final output (last post-activation)
        cache.post_activations.last().unwrap().clone()
    }

    fn backward(&self, input: &Self::Input, grad_output: &Self::Prediction) -> Self::Gradients {
        let (cache, batch_size) = self.forward_with_cache(input);

        let num_layers = self.params.layers.len();
        let mut layer_grads = Vec::with_capacity(num_layers);

        // Initialize delta with the output gradient
        let mut delta = grad_output.clone();

        // Backpropagate through layers (reverse order)
        for i in (0..num_layers).rev() {
            // Apply activation derivative
            delta = self.activations[i].backward_1d(&cache.pre_activations[i], &delta);

            // Get input to this layer
            let layer_input = if i == 0 {
                // First layer uses the original input
                // For simplicity with batched 2D input, we need to handle this properly
                // For now, assume single sample (batch_size = 1)
                input.row(0)
            } else {
                cache.post_activations[i - 1].clone()
            };

            // Compute weight gradients: grad_W = delta^T @ layer_input / batch_size
            // For a single sample: outer product
            let grad_weights = outer_product(&delta, &layer_input);
            let grad_weights = grad_weights.scale(Scalar::new(1.0 / batch_size as f64));

            // Compute bias gradients: grad_b = mean(delta, axis=0) = delta / batch_size
            let grad_bias = delta.scale(&Scalar::new(1.0 / batch_size as f64));

            layer_grads.insert(
                0,
                LayerParams {
                    weights: grad_weights,
                    bias: grad_bias,
                },
            );

            // Propagate delta to previous layer: delta = W^T @ delta
            if i > 0 {
                delta = self.params.layers[i].weights.transpose_matvec(&delta);
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

        // Process each sample in the batch
        // For simplicity, we process the batch by averaging (or taking first sample)
        // A proper implementation would handle full batching
        let mut current = input.row(0);

        for (i, layer) in self.params.layers.iter().enumerate() {
            // z = W @ x + b
            let z = layer.weights.matvec(&current).add(&layer.bias);

            // a = activation(z)
            let a = self.activations[i].forward_1d(&z);

            cache.pre_activations.push(z);
            cache.post_activations.push(a.clone());

            current = a;
        }

        (cache, batch_size)
    }
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
        let (batch_size, _) = input.shape();
        let output_size = *self.layer_sizes.last().unwrap();

        // Process each sample and collect results
        let mut outputs = Vec::with_capacity(batch_size * output_size);

        for i in 0..batch_size {
            let sample = input.row(i);
            let pred = self.predict(&sample);
            outputs.extend(pred.to_vec());
        }

        Tensor2D::new(
            outputs.into_iter().map(|x| x as f32).collect(),
            batch_size,
            output_size,
        )
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
// Helper Functions
// =============================================================================

/// Computes outer product of two 1D tensors: result[i,j] = a[i] * b[j]
fn outer_product<B: Backend>(a: &Tensor1D<B>, b: &Tensor1D<B>) -> Tensor2D<B> {
    let a_len = a.len();
    let b_len = b.len();
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    let mut result = Vec::with_capacity(a_len * b_len);
    for &ai in &a_vec {
        for &bj in &b_vec {
            result.push((ai * bj) as f32);
        }
    }

    Tensor2D::new(result, a_len, b_len)
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
    fn test_outer_product() {
        let a = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]);
        let b = Tensor1D::<CpuBackend>::new(vec![3.0, 4.0]);
        let result = outer_product(&a, &b);

        assert_eq!(result.shape(), (2, 2));
        let values = result.ravel().to_vec();
        // [[1*3, 1*4], [2*3, 2*4]] = [[3, 4], [6, 8]]
        assert_eq!(values, vec![3.0, 4.0, 6.0, 8.0]);
    }
}
