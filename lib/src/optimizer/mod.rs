use crate::backend::scalar::{Scalar, ScalarOps};
use crate::backend::tensor1d::Tensor1D;
use crate::backend::Backend;
use crate::loss::TensorLike;
use crate::model::linear::LinearParams;

/// Trait for gradient-based optimizers.
///
/// Optimizers are responsible for updating model parameters based on computed gradients.
/// The architecture follows separation of concerns principle: training logic (`Trainer`)
/// is decoupled from parameter update logic. This enables composable design where any
/// model can be paired with any optimizer while maintaining full type safety without
/// dynamic dispatch.
///
/// # Type Parameters
/// * `B` — computation backend implementing [`Backend`]
/// * `P` — model parameters type (e.g., [`LinearParams`])
///
/// # Example
/// ```rust
/// # use machinelearne_rs::optimizer::{SGD, Optimizer};
/// # use machinelearne_rs::backend::CpuBackend;
/// # use machinelearne_rs::model::linear::LinearParams;
/// # use machinelearne_rs::backend::tensor1d::Tensor1D;
/// # use machinelearne_rs::backend::scalar::Scalar;
/// #
/// # // Mock parameters and gradients for demonstration
/// # let params = LinearParams {
/// #     weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]),
/// #     bias: Scalar::<CpuBackend>::new(0.5),
/// # };
/// # let gradients = LinearParams {
/// #     weights: Tensor1D::<CpuBackend>::new(vec![0.1, -0.2, 0.05]),
/// #     bias: Scalar::<CpuBackend>::new(-0.01),
/// # };
/// let sgd = SGD::<CpuBackend>::new(0.01);
/// let updated_params = sgd.step(&params, &gradients);
/// ```
pub trait Optimizer<B: Backend, P> {
    /// Performs an optimization step using the update rule:
    /// ```text
    /// params_new = params - learning_rate * gradients
    /// ```
    ///
    /// # Arguments
    /// * `params` — current model parameters
    /// * `gradients` — loss gradients w.r.t. parameters (typically computed via backpropagation)
    ///
    /// # Returns
    /// A new owned instance of updated parameters.
    ///
    /// # Note
    /// This method does not mutate inputs — it returns a new value. This functional
    /// approach simplifies state management in training loops and enables easier
    /// composition with immutable data structures.
    fn step(&self, params: &P, gradients: &P) -> P;
}

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// The simplest first-order optimizer that updates parameters according to:
/// ```text
/// θ ← θ - η · ∇L(θ)
/// ```
/// where `η` is the learning rate and `∇L(θ)` is the loss gradient.
///
/// # Design Notes
/// * Backend-agnostic via [`Backend`] trait bounds
/// * Type-safe binding to specific parameter structures through trait implementation
/// * Stateless by design (no momentum, adaptive learning rates) — specialized variants
///   like Adam or RMSProp should be implemented as separate optimizers
/// * Immutable update semantics: returns new parameters instead of mutating in place
///
/// # Example
/// ```rust
/// use machinelearne_rs::optimizer::SGD;
/// use machinelearne_rs::backend::CpuBackend;
///
/// // Create SGD optimizer with learning rate 0.01
/// let optimizer = SGD::<CpuBackend>::new(0.01);
/// ```
#[derive(Clone)]
pub struct SGD<B: Backend> {
    /// Learning rate (η). Stored as a backend scalar to enable type-safe
    /// arithmetic operations with tensors.
    lr: Scalar<B>,
}

impl<B: Backend> SGD<B> {
    /// Creates a new SGD optimizer with the specified learning rate.
    ///
    /// # Arguments
    /// * `lr` — learning rate (positive value, typically in range 1e-4 .. 1e-1)
    ///
    /// # Panics
    /// Does not panic directly, but backend implementations may validate `lr`
    /// during scalar construction (e.g., rejecting NaN or negative values).
    pub fn new(lr: f64) -> Self {
        Self {
            lr: Scalar::<B>::new(lr),
        }
    }

    /// Returns the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.lr.data.to_f64()
    }
}

impl<B: Backend> Optimizer<B, LinearParams<B>> for SGD<B>
where
    Tensor1D<B>: Clone,
    Scalar<B>: Clone,
{
    fn step(&self, params: &LinearParams<B>, grads: &LinearParams<B>) -> LinearParams<B> {
        // weights_new = weights - lr * grad_weights
        // Using (-lr) enables single scaling operation instead of scale + subtract
        let neg_lr = Scalar::<B>::new(0.0) - self.lr;
        let scaled_grad = grads.weights.scale(&neg_lr);
        let weights_update = params.weights.add(&scaled_grad);

        // bias_new = bias - lr * grad_bias
        // Using explicit backend methods for consistency with tensor operations
        let scaled_bias_grad = grads.bias * self.lr;
        let bias_update = params.bias - scaled_bias_grad;

        LinearParams {
            weights: weights_update,
            bias: bias_update,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_sgd_new_initialization() {
        let lr = 0.01;
        let sgd = SGD::<CpuBackend>::new(lr);

        assert_eq!(sgd.learning_rate(), lr);
    }

    #[test]
    fn test_sgd_learning_rate_accessor() {
        let sgd = SGD::<CpuBackend>::new(0.001);
        assert_eq!(sgd.learning_rate(), 0.001);

        let sgd_large = SGD::<CpuBackend>::new(1.0);
        assert_eq!(sgd_large.learning_rate(), 1.0);
    }

    #[test]
    fn test_sgd_step_correctness_weights() {
        // params_new = params_old - lr * grads
        let lr = 0.1;
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![2.0, 3.0]),
            bias: Scalar::<CpuBackend>::new(1.0),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, -1.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };

        let sgd = SGD::<CpuBackend>::new(lr);
        let updated = sgd.step(&params, &grads);

        // weights: [2.0 - 0.1*1.0, 3.0 - 0.1*(-1.0)] = [1.9, 3.1]
        assert_eq!(updated.weights.to_vec(), vec![1.9, 3.1]);
    }

    #[test]
    fn test_sgd_step_correctness_bias() {
        let lr = 0.05;
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0]),
            bias: Scalar::<CpuBackend>::new(2.0),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.5]),
            bias: Scalar::<CpuBackend>::new(-1.0),
        };

        let sgd = SGD::<CpuBackend>::new(lr);
        let updated = sgd.step(&params, &grads);

        // bias: 2.0 - 0.05*(-1.0) = 2.0 + 0.05 = 2.05
        assert_eq!(updated.bias.data.to_f64(), 2.05);
    }

    #[test]
    fn test_sgd_step_single_parameter() {
        let sgd = SGD::<CpuBackend>::new(0.1);
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![5.0]),
            bias: Scalar::<CpuBackend>::new(0.0),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![2.0]),
            bias: Scalar::<CpuBackend>::new(0.1),
        };

        let updated = sgd.step(&params, &grads);

        // weight: 5.0 - 0.1*2.0 = 4.8
        // bias: 0.0 - 0.1*0.1 = -0.01
        // Use approximate comparison for floating point
        assert!((updated.weights.to_vec()[0] - 4.8).abs() < 1e-10);
        assert!((updated.bias.data.to_f64() - (-0.01)).abs() < 1e-10);
    }

    #[test]
    fn test_sgd_step_multiple_parameters() {
        let sgd = SGD::<CpuBackend>::new(0.01);
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.5, -0.25, 0.1, -0.2]),
            bias: Scalar::<CpuBackend>::new(0.01),
        };

        let updated = sgd.step(&params, &grads);

        // weights: [1.0-0.01*0.5, 2.0-0.01*(-0.25), 3.0-0.01*0.1, 4.0-0.01*(-0.2)]
        //         = [0.995, 2.0025, 2.999, 4.002]
        let expected_weights = vec![0.995, 2.0025, 2.999, 4.002];
        assert!(updated
            .weights
            .to_vec()
            .iter()
            .zip(expected_weights.iter())
            .all(|(a, b)| (a - b).abs() < 1e-10));

        // bias: 0.5 - 0.01*0.01 = 0.4999
        assert!((updated.bias.data.to_f64() - 0.4999).abs() < 1e-5);
    }

    #[test]
    fn test_sgd_step_zero_gradients() {
        // When gradients are zero, parameters should not change
        let sgd = SGD::<CpuBackend>::new(0.1);
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let zero_grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.0, 0.0]),
            bias: Scalar::<CpuBackend>::new(0.0),
        };

        let updated = sgd.step(&params, &zero_grads);

        assert_eq!(updated.weights.to_vec(), vec![1.0, 2.0]);
        assert_eq!(updated.bias.data.to_f64(), 0.5);
    }

    #[test]
    fn test_sgd_step_zero_learning_rate() {
        // Zero learning rate should not change parameters
        let sgd = SGD::<CpuBackend>::new(0.0);
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 1.0]),
            bias: Scalar::<CpuBackend>::new(1.0),
        };

        let updated = sgd.step(&params, &grads);

        // Parameters should remain unchanged
        assert_eq!(updated.weights.to_vec(), vec![1.0, 2.0]);
        assert_eq!(updated.bias.data.to_f64(), 0.5);
    }

    #[test]
    fn test_sgd_step_negative_learning_rate() {
        // Negative learning rate should move in opposite direction
        let sgd = SGD::<CpuBackend>::new(-0.1);
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0]),
            bias: Scalar::<CpuBackend>::new(0.0),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };

        let updated = sgd.step(&params, &grads);

        // With negative LR: params - (-0.1)*grad = params + 0.1*grad
        // weight: 1.0 + 0.1*1.0 = 1.1
        // bias: 0.0 + 0.1*0.5 = 0.05
        assert_eq!(updated.weights.to_vec(), vec![1.1]);
        assert_eq!(updated.bias.data.to_f64(), 0.05);
    }

    #[test]
    fn test_sgd_step_very_large_learning_rate() {
        let sgd = SGD::<CpuBackend>::new(100.0);
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0]),
            bias: Scalar::<CpuBackend>::new(0.0),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.01]),
            bias: Scalar::<CpuBackend>::new(0.01),
        };

        let updated = sgd.step(&params, &grads);

        // weight: 1.0 - 100.0*0.01 = 0.0
        // bias: 0.0 - 100.0*0.01 = -1.0
        assert!((updated.weights.to_vec()[0] - 0.0).abs() < 1e-6);
        assert!((updated.bias.data.to_f64() - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_step_very_large_gradients() {
        let sgd = SGD::<CpuBackend>::new(0.01);
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.0]),
            bias: Scalar::<CpuBackend>::new(0.0),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1000.0]),
            bias: Scalar::<CpuBackend>::new(1000.0),
        };

        let updated = sgd.step(&params, &grads);

        // weight: 0.0 - 0.01*1000.0 = -10.0
        // bias: 0.0 - 0.01*1000.0 = -10.0
        assert_eq!(updated.weights.to_vec(), vec![-10.0]);
        assert_eq!(updated.bias.data.to_f64(), -10.0);
    }

    #[test]
    fn test_sgd_step_negative_gradients() {
        let sgd = SGD::<CpuBackend>::new(0.1);
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0]),
            bias: Scalar::<CpuBackend>::new(0.0),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![-1.0]),
            bias: Scalar::<CpuBackend>::new(-0.5),
        };

        let updated = sgd.step(&params, &grads);

        // weight: 1.0 - 0.1*(-1.0) = 1.1
        // bias: 0.0 - 0.1*(-0.5) = 0.05
        assert_eq!(updated.weights.to_vec(), vec![1.1]);
        assert_eq!(updated.bias.data.to_f64(), 0.05);
    }

    #[test]
    fn test_sgd_step_fractional_values() {
        let sgd = SGD::<CpuBackend>::new(0.125);
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.5]),
            bias: Scalar::<CpuBackend>::new(0.25),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.8]),
            bias: Scalar::<CpuBackend>::new(0.4),
        };

        let updated = sgd.step(&params, &grads);

        // weight: 0.5 - 0.125*0.8 = 0.5 - 0.1 = 0.4
        // bias: 0.25 - 0.125*0.4 = 0.25 - 0.05 = 0.2
        // Use approximate comparison for floating point
        assert!((updated.weights.to_vec()[0] - 0.4).abs() < 1e-6);
        assert!((updated.bias.data.to_f64() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_step_does_not_mutate_inputs() {
        // Verify that step() does not mutate the original params or grads
        let sgd = SGD::<CpuBackend>::new(0.1);
        let original_params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let original_grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.5, 0.3]),
            bias: Scalar::<CpuBackend>::new(0.1),
        };

        // Clone to compare after step
        let params_copy = original_params.clone();
        let grads_copy = original_grads.clone();

        let _ = sgd.step(&original_params, &original_grads);

        // Original should be unchanged
        assert_eq!(
            original_params.weights.to_vec(),
            params_copy.weights.to_vec()
        );
        assert_eq!(
            original_params.bias.data.to_f64(),
            params_copy.bias.data.to_f64()
        );
        assert_eq!(original_grads.weights.to_vec(), grads_copy.weights.to_vec());
        assert_eq!(
            original_grads.bias.data.to_f64(),
            grads_copy.bias.data.to_f64()
        );
    }

    #[test]
    fn test_sgd_clone() {
        let sgd = SGD::<CpuBackend>::new(0.01);
        let sgd_clone = sgd.clone();

        assert_eq!(sgd.learning_rate(), sgd_clone.learning_rate());

        // Both should produce same updates
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0]),
            bias: Scalar::<CpuBackend>::new(0.0),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![0.5]),
            bias: Scalar::<CpuBackend>::new(0.2),
        };

        let updated1 = sgd.step(&params, &grads);
        let updated2 = sgd_clone.step(&params, &grads);

        assert_eq!(updated1.weights.to_vec(), updated2.weights.to_vec());
        assert_eq!(updated1.bias.data.to_f64(), updated2.bias.data.to_f64());
    }

    #[test]
    fn test_sgd_very_small_learning_rate() {
        let sgd = SGD::<CpuBackend>::new(1e-10);
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0]),
            bias: Scalar::<CpuBackend>::new(0.5),
        };
        let grads = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1000.0]),
            bias: Scalar::<CpuBackend>::new(1000.0),
        };

        let updated = sgd.step(&params, &grads);

        // With very small LR, parameters should barely change
        // weight: 1.0 - 1e-10 * 1000.0 = 0.9999999
        // bias: 0.5 - 1e-10 * 1000.0 = 0.4999999
        assert!((updated.weights.to_vec()[0] - 0.9999999).abs() < 1e-6);
        assert!((updated.bias.data.to_f64() - 0.4999999).abs() < 1e-6);
    }
}
