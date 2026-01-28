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
