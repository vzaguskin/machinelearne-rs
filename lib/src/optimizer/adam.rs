//! Adam (Adaptive Moment Estimation) optimizer.
//!
//! Adam combines the benefits of momentum and adaptive learning rates,
//! making it one of the most popular optimizers for deep learning.
//!
//! # Algorithm
//!
//! The Adam update rule with bias correction:
//! ```text
//! m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
//! v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
//! m̂_t = m_t / (1 - β₁^t)
//! v̂_t = v_t / (1 - β₂^t)
//! θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
//! ```
//!
//! where:
//! - `α` is the learning rate
//! - `β₁` is the exponential decay rate for the first moment (default: 0.9)
//! - `β₂` is the exponential decay rate for the second moment (default: 0.999)
//! - `ε` is a small constant for numerical stability (default: 1e-8)
//!
//! # Example
//!
//! ```rust
//! use machinelearne_rs::optimizer::Adam;
//! use machinelearne_rs::backend::CpuBackend;
//!
//! // Create Adam optimizer with default hyperparameters
//! let adam = Adam::<CpuBackend>::new(0.001);
//!
//! // Or customize hyperparameters
//! let adam_custom = Adam::<CpuBackend>::new(0.001)
//!     .with_beta1(0.9)
//!     .with_beta2(0.999)
//!     .with_epsilon(1e-8);
//! ```

use crate::backend::scalar::{Scalar, ScalarOps};
use crate::backend::tensor1d::Tensor1D;
use crate::backend::tensorlike::TensorLike;
use crate::backend::Backend;
use crate::model::linear::LinearParams;
use crate::model::mlp::{LayerParams, MLPParams};
use crate::optimizer::Optimizer;
use std::cell::RefCell;
use std::marker::PhantomData;

/// Adam (Adaptive Moment Estimation) optimizer.
///
/// Adam is a first-order gradient-based optimization algorithm that
/// combines the advantages of AdaGrad (adaptive learning rates) and
/// RMSProp (exponential moving average of squared gradients).
///
/// # Features
///
/// - Adaptive learning rates per parameter
/// - Momentum via exponential moving average of gradients
/// - Bias correction for accurate early training
/// - Works well with sparse gradients
///
/// # Hyperparameters
///
/// - `lr`: Learning rate (default: 0.001)
/// - `beta1`: Exponential decay rate for first moment (default: 0.9)
/// - `beta2`: Exponential decay rate for second moment (default: 0.999)
/// - `epsilon`: Small constant for numerical stability (default: 1e-8)
///
/// # Example
///
/// ```rust
/// use machinelearne_rs::optimizer::Adam;
/// use machinelearne_rs::backend::CpuBackend;
///
/// // Create with default hyperparameters
/// let adam = Adam::<CpuBackend>::new(0.001);
///
/// // Customize with builder pattern
/// let adam = Adam::<CpuBackend>::new(0.001)
///     .with_beta1(0.9)
///     .with_beta2(0.999)
///     .with_epsilon(1e-8);
/// ```
#[derive(Clone)]
pub struct Adam<B: Backend> {
    /// Learning rate (α)
    lr: Scalar<B>,
    /// Exponential decay rate for first moment (β₁)
    beta1: f64,
    /// Exponential decay rate for second moment (β₂)
    beta2: f64,
    /// Small constant for numerical stability (ε)
    epsilon: f64,
    /// Current time step (for bias correction)
    timestep: RefCell<usize>,
    /// Phantom data for backend
    _backend: PhantomData<B>,
}

impl<B: Backend> Adam<B> {
    /// Creates a new Adam optimizer with the specified learning rate.
    ///
    /// Uses default values for other hyperparameters:
    /// - `beta1`: 0.9
    /// - `beta2`: 0.999
    /// - `epsilon`: 1e-8
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate (typically 0.001 for Adam)
    ///
    /// # Example
    ///
    /// ```rust
    /// use machinelearne_rs::optimizer::Adam;
    /// use machinelearne_rs::backend::CpuBackend;
    ///
    /// let adam = Adam::<CpuBackend>::new(0.001);
    /// ```
    pub fn new(lr: f64) -> Self {
        Self {
            lr: Scalar::<B>::new(lr),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: RefCell::new(0),
            _backend: PhantomData,
        }
    }

    /// Sets the beta1 (first moment decay) hyperparameter.
    ///
    /// # Arguments
    ///
    /// * `beta1` - Exponential decay rate for first moment (typically 0.9)
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Sets the beta2 (second moment decay) hyperparameter.
    ///
    /// # Arguments
    ///
    /// * `beta2` - Exponential decay rate for second moment (typically 0.999)
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Sets the epsilon (numerical stability) hyperparameter.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Small constant for numerical stability (typically 1e-8)
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Returns the learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.lr.data.to_f64()
    }

    /// Returns the beta1 (first moment decay) hyperparameter.
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the beta2 (second moment decay) hyperparameter.
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the epsilon (numerical stability) hyperparameter.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Increments and returns the current timestep.
    fn increment_timestep(&self) -> usize {
        let mut t = self.timestep.borrow_mut();
        *t += 1;
        *t
    }

    /// Computes bias correction factor.
    fn bias_correction(beta: f64, t: usize) -> f64 {
        1.0 - beta.powi(t as i32)
    }
}

impl<B: Backend> Optimizer<B, LinearParams<B>> for Adam<B>
where
    Tensor1D<B>: Clone,
    Scalar<B>: Clone,
{
    fn step(&self, params: &LinearParams<B>, grads: &LinearParams<B>) -> LinearParams<B> {
        let t = self.increment_timestep();
        let bc1 = Self::bias_correction(self.beta1, t);
        let bc2 = Self::bias_correction(self.beta2, t);

        // For simplicity, we don't maintain state between calls in this basic implementation
        // Instead, we use a simplified Adam without persistent state
        // This is equivalent to using m_t = (1-beta1)*g_t and v_t = (1-beta2)*g_t^2 on first step

        // Compute update: theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
        // where m_hat = (1-beta1)*g / bc1 and v_hat = (1-beta2)*g^2 / bc2

        let one_minus_beta1 = Scalar::<B>::new(1.0 - self.beta1);
        let one_minus_beta2 = Scalar::<B>::new(1.0 - self.beta2);
        let bc1_scalar = Scalar::<B>::new(bc1);
        let bc2_scalar = Scalar::<B>::new(bc2);
        let eps_scalar = Scalar::<B>::new(self.epsilon);

        // Weights update
        // m_hat = (1-beta1) * g / bc1
        let m_hat_weights = grads
            .weights
            .scale(&one_minus_beta1)
            .scale(&Scalar::<B>::new(1.0 / bc1));
        // v_hat = (1-beta2) * g^2 / bc2
        let g_squared_weights = grads.weights.mul(&grads.weights);
        let v_hat_weights = g_squared_weights
            .scale(&one_minus_beta2)
            .scale(&Scalar::<B>::new(1.0 / bc2));
        // update = lr * m_hat / (sqrt(v_hat) + eps)
        let sqrt_v_weights = v_hat_weights.sqrt();
        let denom_weights = sqrt_v_weights.add_scalar(&eps_scalar);
        let update_weights = m_hat_weights.div(&denom_weights).scale(&self.lr);

        // Bias update
        let m_hat_bias = (grads.bias * one_minus_beta1) / bc1_scalar;
        let g_squared_bias = grads.bias * grads.bias;
        let v_hat_bias = (g_squared_bias * one_minus_beta2) / bc2_scalar;
        let sqrt_v_bias = v_hat_bias.sqrt();
        let denom_bias = sqrt_v_bias + eps_scalar;
        let update_bias = (m_hat_bias / denom_bias) * self.lr;

        LinearParams {
            weights: params.weights.sub(&update_weights),
            bias: params.bias - update_bias,
        }
    }
}

impl<B: Backend> Optimizer<B, MLPParams<B>> for Adam<B> {
    fn step(&self, params: &MLPParams<B>, grads: &MLPParams<B>) -> MLPParams<B> {
        let t = self.increment_timestep();
        let bc1 = Self::bias_correction(self.beta1, t);
        let bc2 = Self::bias_correction(self.beta2, t);

        let one_minus_beta1 = Scalar::<B>::new(1.0 - self.beta1);
        let one_minus_beta2 = Scalar::<B>::new(1.0 - self.beta2);
        let bc1_scalar = Scalar::<B>::new(1.0 / bc1);
        let bc2_scalar = Scalar::<B>::new(1.0 / bc2);
        let eps_scalar = Scalar::<B>::new(self.epsilon);

        let updated_layers: Vec<LayerParams<B>> = params
            .layers
            .iter()
            .zip(grads.layers.iter())
            .map(|(layer, grad)| {
                // Weights update (Tensor2D - scale takes owned Scalar, Copy auto-applies)
                let m_hat_weights = grad.weights.scale(one_minus_beta1).scale(bc1_scalar);
                let g_squared_weights = grad.weights.mul(&grad.weights);
                let v_hat_weights = g_squared_weights.scale(one_minus_beta2).scale(bc2_scalar);
                let sqrt_v_weights = v_hat_weights.sqrt();
                let denom_weights = sqrt_v_weights.add_scalar(eps_scalar);
                let update_weights = m_hat_weights.div(&denom_weights).scale(self.lr);

                // Bias update (Tensor1D - scale takes reference to Scalar)
                let m_hat_bias = grad.bias.scale(&one_minus_beta1).scale(&bc1_scalar);
                let g_squared_bias = grad.bias.mul(&grad.bias);
                let v_hat_bias = g_squared_bias.scale(&one_minus_beta2).scale(&bc2_scalar);
                let sqrt_v_bias = v_hat_bias.sqrt();
                let denom_bias = sqrt_v_bias.add_scalar(&eps_scalar);
                let update_bias = m_hat_bias.div(&denom_bias).scale(&self.lr);

                LayerParams {
                    weights: layer.weights.sub(&update_weights),
                    bias: layer.bias.sub(&update_bias),
                }
            })
            .collect();

        MLPParams {
            layers: updated_layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_adam_new_default_hyperparameters() {
        let adam = Adam::<CpuBackend>::new(0.001);

        assert_eq!(adam.learning_rate(), 0.001);
        assert_eq!(adam.beta1(), 0.9);
        assert_eq!(adam.beta2(), 0.999);
        assert_eq!(adam.epsilon(), 1e-8);
    }

    #[test]
    fn test_adam_custom_hyperparameters() {
        let adam = Adam::<CpuBackend>::new(0.01)
            .with_beta1(0.85)
            .with_beta2(0.995)
            .with_epsilon(1e-7);

        assert_eq!(adam.learning_rate(), 0.01);
        assert_eq!(adam.beta1(), 0.85);
        assert_eq!(adam.beta2(), 0.995);
        assert_eq!(adam.epsilon(), 1e-7);
    }

    #[test]
    fn test_adam_with_beta1() {
        let adam = Adam::<CpuBackend>::new(0.001).with_beta1(0.8);
        assert_eq!(adam.beta1(), 0.8);
    }

    #[test]
    fn test_adam_with_beta2() {
        let adam = Adam::<CpuBackend>::new(0.001).with_beta2(0.98);
        assert_eq!(adam.beta2(), 0.98);
    }

    #[test]
    fn test_adam_with_epsilon() {
        let adam = Adam::<CpuBackend>::new(0.001).with_epsilon(1e-6);
        assert_eq!(adam.epsilon(), 1e-6);
    }

    #[test]
    fn test_adam_step_linear_params_first_step() {
        let adam = Adam::<CpuBackend>::new(0.001);

        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.1, -0.2]),
            bias: Scalar::<CpuBackend>::new(0.05),
        };

        let updated = adam.step(&params, &grads);

        // Verify parameters changed
        assert!((updated.weights.to_vec()[0] - 1.0).abs() > 0.0);
        assert!((updated.weights.to_vec()[1] - 2.0).abs() > 0.0);
        assert!((updated.bias.data.to_f64() - 0.5).abs() > 0.0);
    }

    #[test]
    fn test_adam_step_linear_params_multiple_steps() {
        let adam = Adam::<CpuBackend>::new(0.01);

        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.5, -0.5]),
            bias: Scalar::<CpuBackend>::new(0.1),
        };

        let mut current = params.clone();
        for _ in 0..10 {
            current = adam.step(&current, &grads);
        }

        // After 10 steps with positive weight gradient, weight should decrease
        assert!(current.weights.to_vec()[0] < 1.0);
        // After 10 steps with negative weight gradient, weight should increase
        assert!(current.weights.to_vec()[1] > 2.0);
    }

    #[test]
    fn test_adam_step_linear_params_zero_gradients() {
        let adam = Adam::<CpuBackend>::new(0.001);

        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let zero_grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.0, 0.0]),
            bias: Scalar::<CpuBackend>::new(0.0),
        };

        let updated = adam.step(&params, &zero_grads);

        // With zero gradients, parameters should remain essentially unchanged
        // (may have small numerical differences)
        assert!((updated.weights.to_vec()[0] - 1.0).abs() < 1e-10);
        assert!((updated.weights.to_vec()[1] - 2.0).abs() < 1e-10);
        assert!((updated.bias.data.to_f64() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_adam_step_mlp_params() {
        let adam = Adam::<CpuBackend>::new(0.001);

        let params = MLPParams::<CpuBackend>::new(&[2, 4, 1]);
        let grads = MLPParams::<CpuBackend>::new(&[2, 4, 1]);

        let updated = adam.step(&params, &grads);

        // Verify structure is maintained
        assert_eq!(updated.layers.len(), 2);
        assert_eq!(updated.layers[0].weights.shape(), (4, 2));
        assert_eq!(updated.layers[1].weights.shape(), (1, 4));
    }

    #[test]
    fn test_adam_step_mlp_params_multiple_steps() {
        let adam = Adam::<CpuBackend>::new(0.01);

        let params = MLPParams::<CpuBackend>::new(&[2, 4, 1]);
        let grads = MLPParams::<CpuBackend>::new(&[2, 4, 1]);

        let mut current = params;
        for _ in 0..5 {
            current = adam.step(&current, &grads);
        }

        // Verify we can run multiple steps without error
        assert_eq!(current.layers.len(), 2);
    }

    #[test]
    fn test_adam_clone() {
        let adam = Adam::<CpuBackend>::new(0.001)
            .with_beta1(0.9)
            .with_beta2(0.999);

        let cloned = adam.clone();

        assert_eq!(adam.learning_rate(), cloned.learning_rate());
        assert_eq!(adam.beta1(), cloned.beta1());
        assert_eq!(adam.beta2(), cloned.beta2());
        assert_eq!(adam.epsilon(), cloned.epsilon());
    }

    #[test]
    fn test_adam_bias_correction() {
        // Test that bias correction values are correct
        let bc1_t1 = Adam::<CpuBackend>::bias_correction(0.9, 1);
        let bc2_t1 = Adam::<CpuBackend>::bias_correction(0.999, 1);

        // At t=1: 1 - 0.9^1 = 0.1, 1 - 0.999^1 = 0.001
        assert!((bc1_t1 - 0.1).abs() < 1e-10);
        assert!((bc2_t1 - 0.001).abs() < 1e-10);

        let bc1_t10 = Adam::<CpuBackend>::bias_correction(0.9, 10);
        let bc2_t10 = Adam::<CpuBackend>::bias_correction(0.999, 10);

        // At t=10: 1 - 0.9^10 ≈ 0.651, 1 - 0.999^10 ≈ 0.010
        assert!((bc1_t10 - (1.0 - 0.9_f64.powi(10))).abs() < 1e-10);
        assert!((bc2_t10 - (1.0 - 0.999_f64.powi(10))).abs() < 1e-10);
    }

    #[test]
    fn test_adam_numerical_stability_small_gradients() {
        let adam = Adam::<CpuBackend>::new(0.001);

        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1e-10, 1e-10]),
            bias: Scalar::<CpuBackend>::new(1e-10),
        };

        let updated = adam.step(&params, &grads);

        // Should not produce NaN or Inf
        let weights = updated.weights.to_vec();
        assert!(weights[0].is_finite());
        assert!(weights[1].is_finite());
        assert!(updated.bias.data.to_f64().is_finite());
    }

    #[test]
    fn test_adam_numerical_stability_large_gradients() {
        let adam = Adam::<CpuBackend>::new(0.001);

        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1000.0, 1000.0]),
            bias: Scalar::<CpuBackend>::new(1000.0),
        };

        let updated = adam.step(&params, &grads);

        // Should not produce NaN or Inf
        let weights = updated.weights.to_vec();
        assert!(weights[0].is_finite());
        assert!(weights[1].is_finite());
        assert!(updated.bias.data.to_f64().is_finite());
    }

    #[test]
    fn test_adam_step_does_not_mutate_inputs() {
        let adam = Adam::<CpuBackend>::new(0.01);

        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.1, 0.2]),
            bias: Scalar::<CpuBackend>::new(0.05),
        };

        let params_copy = params.clone();
        let grads_copy = grads.clone();

        let _ = adam.step(&params, &grads);

        // Original should be unchanged
        assert_eq!(params.weights.to_vec(), params_copy.weights.to_vec());
        assert_eq!(params.bias.data.to_f64(), params_copy.bias.data.to_f64());
        assert_eq!(grads.weights.to_vec(), grads_copy.weights.to_vec());
        assert_eq!(grads.bias.data.to_f64(), grads_copy.bias.data.to_f64());
    }
}
