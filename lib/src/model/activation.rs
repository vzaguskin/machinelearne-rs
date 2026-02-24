//! Activation functions for neural network layers.
//!
//! This module provides common activation functions used in neural networks,
//! with both forward and backward (derivative) implementations for training.
//!
//! # Available Activations
//!
//! | Activation | Range | Common Use |
//! |------------|-------|------------|
//! | ReLU | [0, ∞) | Hidden layers (default choice) |
//! | Sigmoid | (0, 1) | Binary classification output |
//! | Tanh | (-1, 1) | Hidden layers (zero-centered) |
//! | Identity | (-∞, ∞) | Regression output |
//!
//! # Example
//!
//! ```rust
//! use machinelearne_rs::model::Activation;
//! use machinelearne_rs::backend::{CpuBackend, Tensor1D};
//!
//! let x = Tensor1D::<CpuBackend>::new(vec![-1.0, 0.0, 1.0]);
//!
//! // Forward pass
//! let activated = Activation::ReLU.forward_1d::<CpuBackend>(&x);
//! assert_eq!(activated.to_vec(), vec![0.0, 0.0, 1.0]);
//!
//! // Backward pass (for backpropagation)
//! let grad = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 1.0]);
//! let backward = Activation::ReLU.backward_1d::<CpuBackend>(&x, &grad);
//! // grad * (x > 0) = [0, 0, 1]
//! ```

use crate::backend::tensorlike::TensorLike;
use crate::backend::{Backend, Scalar, Tensor1D};

/// Activation functions for neural network layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    ///
    /// Most commonly used activation for hidden layers.
    /// Fast computation, helps with vanishing gradient problem.
    #[default]
    ReLU,

    /// Sigmoid: 1 / (1 + exp(-x))
    ///
    /// Output range: (0, 1). Commonly used for binary classification output.
    Sigmoid,

    /// Hyperbolic tangent: tanh(x)
    ///
    /// Output range: (-1, 1). Zero-centered, often works better than sigmoid
    /// for hidden layers.
    Tanh,

    /// Identity: f(x) = x
    ///
    /// No transformation. Used for regression output layers.
    Identity,
}

impl Activation {
    /// Apply activation function to a 1D tensor (forward pass).
    pub fn forward_1d<B: Backend>(&self, x: &Tensor1D<B>) -> Tensor1D<B> {
        match self {
            Activation::ReLU => x.relu(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::Tanh => x.tanh(),
            Activation::Identity => x.clone(),
        }
    }

    /// Compute derivative of activation function (backward pass).
    ///
    /// # Arguments
    /// * `pre_activation` - The input to the activation (z = Wx + b)
    /// * `grad_output` - Gradient from the next layer
    ///
    /// # Returns
    /// Gradient with respect to the input of this layer.
    ///
    /// # Note
    /// For ReLU, Sigmoid, and Tanh, we can use the output (post-activation)
    /// for a more numerically stable computation. For Identity, the derivative
    /// is always 1.
    pub fn backward_1d<B: Backend>(
        &self,
        pre_activation: &Tensor1D<B>,
        grad_output: &Tensor1D<B>,
    ) -> Tensor1D<B> {
        match self {
            Activation::ReLU => {
                // ReLU'(z) = 1 if z > 0, else 0
                // Create mask: 1 where z > 0, 0 elsewhere
                let len = pre_activation.len();
                let zeros = Tensor1D::<B>::zeros(len);
                // sign gives -1, 0, or 1; max with 0 gives 0 or 1
                let sign = pre_activation.sign();
                let mask = zeros.maximum(sign);
                grad_output.mul(&mask)
            }
            Activation::Sigmoid => {
                // sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
                let output = self.forward_1d(pre_activation);
                // 1 - output
                let one = Scalar::<B>::new(1.0);
                let one_minus = Tensor1D::<B>::zeros(output.len())
                    .add_scalar(&one)
                    .sub(&output);
                // output * (1 - output)
                let sigmoid_deriv = output.mul(&one_minus);
                grad_output.mul(&sigmoid_deriv)
            }
            Activation::Tanh => {
                // tanh'(z) = 1 - tanh(z)^2 = 1 - output^2
                let output = self.forward_1d(pre_activation);
                // output^2
                let output_squared = output.mul(&output);
                // 1 - output^2
                let one = Scalar::<B>::new(1.0);
                let tanh_deriv = Tensor1D::<B>::zeros(output.len())
                    .add_scalar(&one)
                    .sub(&output_squared);
                grad_output.mul(&tanh_deriv)
            }
            Activation::Identity => {
                // Identity'(z) = 1, so just pass through the gradient
                grad_output.clone()
            }
        }
    }

    /// Returns the ONNX op type name for this activation.
    pub fn onnx_op_type(&self) -> &'static str {
        match self {
            Activation::ReLU => "Relu",
            Activation::Sigmoid => "Sigmoid",
            Activation::Tanh => "Tanh",
            Activation::Identity => "Identity",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_relu_forward() {
        let x = Tensor1D::<CpuBackend>::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = Activation::ReLU.forward_1d(&x);
        assert_eq!(y.to_vec(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_backward() {
        let x = Tensor1D::<CpuBackend>::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let grad = Tensor1D::<CpuBackend>::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let backward = Activation::ReLU.backward_1d(&x, &grad);
        let result = backward.to_vec();
        // ReLU derivative: 0 for x <= 0, 1 for x > 0
        assert_eq!(result[0], 0.0); // -2.0 <= 0
        assert_eq!(result[1], 0.0); // -1.0 <= 0
        assert_eq!(result[2], 0.0); // 0 <= 0 (subgradient at 0)
        assert_eq!(result[3], 1.0); // 1.0 > 0
        assert_eq!(result[4], 1.0); // 2.0 > 0
    }

    #[test]
    fn test_sigmoid_forward() {
        let x = Tensor1D::<CpuBackend>::new(vec![0.0]);
        let y = Activation::Sigmoid.forward_1d(&x);
        assert!((y.to_vec()[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_sigmoid_backward() {
        // At x=0, sigmoid(0) = 0.5, derivative = 0.5 * 0.5 = 0.25
        let x = Tensor1D::<CpuBackend>::new(vec![0.0]);
        let grad = Tensor1D::<CpuBackend>::new(vec![1.0]);
        let backward = Activation::Sigmoid.backward_1d(&x, &grad);
        assert!((backward.to_vec()[0] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_tanh_forward() {
        let x = Tensor1D::<CpuBackend>::new(vec![0.0]);
        let y = Activation::Tanh.forward_1d(&x);
        assert!(y.to_vec()[0].abs() < 1e-12); // tanh(0) = 0
    }

    #[test]
    fn test_tanh_backward() {
        // At x=0, tanh(0) = 0, derivative = 1 - 0^2 = 1
        let x = Tensor1D::<CpuBackend>::new(vec![0.0]);
        let grad = Tensor1D::<CpuBackend>::new(vec![1.0]);
        let backward = Activation::Tanh.backward_1d(&x, &grad);
        assert!((backward.to_vec()[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_identity_forward() {
        let x = Tensor1D::<CpuBackend>::new(vec![-2.0, 0.0, 2.0]);
        let y = Activation::Identity.forward_1d(&x);
        assert_eq!(y.to_vec(), vec![-2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_identity_backward() {
        let x = Tensor1D::<CpuBackend>::new(vec![-2.0, 0.0, 2.0]);
        let grad = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]);
        let backward = Activation::Identity.backward_1d(&x, &grad);
        assert_eq!(backward.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_onnx_op_type() {
        assert_eq!(Activation::ReLU.onnx_op_type(), "Relu");
        assert_eq!(Activation::Sigmoid.onnx_op_type(), "Sigmoid");
        assert_eq!(Activation::Tanh.onnx_op_type(), "Tanh");
        assert_eq!(Activation::Identity.onnx_op_type(), "Identity");
    }

    #[test]
    fn test_default_activation() {
        assert_eq!(Activation::default(), Activation::ReLU);
    }
}
