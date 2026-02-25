## Context

The library currently only supports SGD (Stochastic Gradient Descent) for optimization. While SGD is simple and effective, it has limitations:
- Single learning rate for all parameters
- No momentum (unless manually added)
- Requires careful learning rate tuning
- Slow convergence on sparse gradients

Adam (Adaptive Moment Estimation) combines the benefits of momentum and adaptive learning rates, making it the de-facto standard optimizer for deep learning.

## Goals / Non-Goals

**Goals:**
- Implement `Adam<B>` optimizer following the existing `Optimizer<B, P>` trait pattern
- Support configurable hyperparameters (lr, beta1, beta2, epsilon)
- Maintain per-parameter state (first and second moments)
- Implement bias correction for early training steps
- Support both `LinearParams<B>` and `MLPParams<B>` parameter types
- Match the code quality and testing standards of existing SGD optimizer

**Non-Goals:**
- AdamW (weight decay variant) - can be added later
- AmsGrad variant - can be added later
- Learning rate scheduling - separate concern
- Serialization of optimizer state - can be added later

## Decisions

### 1. Stateful Optimizer Design
**Decision**: Adam will be stateful, maintaining momentum and velocity for each parameter.

**Rationale**: Unlike SGD which is stateless, Adam requires tracking:
- First moment (mean of gradients): `m`
- Second moment (uncentered variance of gradients): `v`
- Time step counter for bias correction: `t`

**Alternatives considered**:
- External state struct: More complex API, breaks the `Optimizer` trait
- State per call: Would lose momentum between steps, defeating the purpose

### 2. State Structure
**Decision**: Use `RefCell` for interior mutability to allow state updates within the immutable `step()` method.

**Rationale**: The `Optimizer::step()` signature takes `&self` (immutable), but Adam needs to update internal state. Using `RefCell<T>` allows mutation while maintaining the trait interface.

```rust
pub struct Adam<B: Backend> {
    lr: Scalar<B>,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    // State wrapped in RefCell for interior mutability
    state: RefCell<AdamState<B>>,
}
```

### 3. Hyperparameter Defaults
**Decision**: Use the paper's recommended defaults: lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8.

**Rationale**: These values work well for most deep learning tasks and match PyTorch/TensorFlow defaults.

### 4. Bias Correction
**Decision**: Implement bias correction as described in the original Adam paper.

**Rationale**: Without bias correction, early steps would have biased moment estimates, especially for `beta1` and `beta2` close to 1.

Update rule with bias correction:
```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
```

## Risks / Trade-offs

### Risk: Memory Usage
Adam stores 2x additional state per parameter (momentum + velocity).
**Mitigation**: Document memory implications; state is only created when Adam is instantiated.

### Risk: Interior Mutability Complexity
Using `RefCell` adds runtime borrow checking.
**Mitigation**: The pattern is well-established in Rust; panic on double-borrow indicates a logic error that should be caught in testing.

### Trade-off: No AdamW
Weight decay is handled separately in our architecture (via `Regularizer`).
**Mitigation**: Users can combine Adam with L2 regularizer for similar effect, though not mathematically identical to AdamW.
