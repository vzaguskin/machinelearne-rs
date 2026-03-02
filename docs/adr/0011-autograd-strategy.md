# ADR-0011: Autograd Implementation Strategy

## Status

**Proposed** - 2026-03-02

## Context

The machinelearne-rs library aims to evolve into a "real DNN library" capable of supporting complex architectures like CNNs, RNNs, and Transformers. This raises the question of whether automatic differentiation (autograd) is needed and how it fits with the current architectural principles.

### Current State

From ADR-0001, the library explicitly rejected autograd in favor of **manual gradient implementation**:

> "Implementing real autograd - postpone for next stages"

Current gradient flow:
- Models implement `backward(x, grad_output)` with hardcoded formulas
- Losses implement `grad_wrt_prediction(prediction, target)`
- MLP uses explicit backpropagation through cached activations

### Limitations of Manual Gradients

| Limitation | Impact |
|------------|--------|
| **Verbosity** | Every new operation requires manual derivative implementation |
| **Complexity** | Multi-branch graphs (skip connections, attention) require careful cache management |
| **Error-prone** | Hand-written derivatives can contain subtle bugs |
| **Limited composability** | Cannot arbitrarily compose operations |
| **New model development** | CNNs, RNNs, Transformers require substantial gradient code |

### Architectural Principles to Preserve

From ADR-0001 and subsequent decisions:

1. **Training/Inference separation**: Fitted models contain only inference parameters
2. **Backend agnosticism**: Operations must work across CPU, WGPU, ndarray backends
3. **Type safety**: `Unfitted`/`Fitted` state encoding at compile time
4. **Zero-cost abstractions**: Features should not impose overhead when not used

## Decision

Implement a **hybrid autograd system** with tape-based automatic differentiation as an opt-in feature.

### 1. Tape-Based Autograd as Opt-In Feature

Implement a Wengert-list (tape) style autograd as a separate module, feature-gated behind `autograd` feature flag. This approach:

- Records operations on a linear tape during forward pass
- Replays tape in reverse during backward pass
- Does not modify existing `Backend` trait
- Can be entirely avoided by not enabling the feature

```rust
// Feature gate in Cargo.toml
[features]
autograd = []

// Module structure
lib/src/autograd/
    mod.rs           - Public API and GradTensor wrapper
    tape.rs          - Tape implementation (Wengert list)
    ops.rs           - Differentiable operations
    backward.rs      - Backward pass execution
```

### 2. Preserve Manual Gradient Path

The existing manual gradient system remains the default and recommended approach for simple models:

- Zero overhead when autograd is not used
- Compile-time type safety preserved
- No breaking changes to existing code

```rust
// Manual gradients (current, default)
impl<B: Backend> TrainableModel<B> for LinearModel<B, Unfitted> {
    fn backward(&self, x: &Tensor2D<B>, grad_output: &Tensor1D<B>) -> LinearParams<B> {
        // Hand-written gradient: ∇w = x^T @ grad_pred, ∇b = sum(grad_pred)
        let grad_w = self.backend.matvec_transposed(x, grad_output);
        let grad_b = self.backend.sum_all_1d(grad_output);
        LinearParams { weights: grad_w, bias: grad_b }
    }
}

// Autograd (opt-in, feature-gated)
#[cfg(feature = "autograd")]
impl<B: Backend> TrainableModel<B> for LinearModel<B, Unfitted> {
    fn backward_autograd(&self, tape: &Tape, grad_output: &GradTensor<B>) -> GradTensor<B> {
        // Automatic gradient computation via tape replay
        tape.backward(grad_output)
    }
}
```

### 3. GradTensor as Tensor Wrapper

`GradTensor<B>` wraps existing tensors with gradient tracking, working on top of any Backend implementation:

```rust
/// Tensor wrapper with gradient tracking
pub struct GradTensor<B: Backend> {
    data: Tensor2D<B>,           // Forward value
    grad: Option<Tensor2D<B>>,   // Accumulated gradient
    tape: Rc<RefCell<Tape<B>>>,  // Shared operation tape
    node_id: NodeId,             // Reference to tape node
}

impl<B: Backend> GradTensor<B> {
    /// Create a tensor with gradient tracking
    pub fn with_grad(data: Tensor2D<B>, tape: &Rc<RefCell<Tape<B>>>) -> Self {
        // Records input node on tape
    }

    /// Operations are recorded on tape
    pub fn matmul(&self, other: &GradTensor<B>) -> GradTensor<B> {
        // Forward: compute result
        // Record backward function on tape
    }
}
```

### 4. Integration with Existing Trainer

The `Trainer` can optionally use autograd when the feature is enabled:

```rust
impl<L, O, R> Trainer<L, O, R> {
    #[cfg(feature = "autograd")]
    pub fn fit_autograd<B, M>(&self, model: M, dataset: &dyn Dataset)
        -> Result<M::Fitted, Error>
    where
        M: TrainableModel<B> + AutogradModel<B>
    {
        // Training loop using tape-based gradients
    }
}
```

## Consequences

### Positive

- **Progressive enhancement**: Simple models stay simple, complex models get autograd
- **Enables complex architectures**: Attention mechanisms, skip connections, custom layers
- **Backend agnostic**: Works with any Backend implementation
- **Opt-in overhead**: Zero cost when feature not used
- **Type safety preserved**: Unfitted/Fitted state machine unchanged
- **Easier experimentation**: New layer types without manual gradients

### Negative

- **Additional complexity**: New module to maintain and test
- **Memory overhead**: Tape stores intermediate values for backward pass
- **Performance gap**: Tape-based gradients slower than hand-optimized implementations
- **API surface**: Two ways to implement gradients may confuse users
- **Debugging difficulty**: Autograd bugs harder to trace than manual gradients

### Neutral

- Existing models unaffected unless explicitly migrated
- Documentation needs clear guidance on when to use each approach
- Test coverage requirements increase for autograd module

## Alternatives Considered

### Alternative 1: Full Graph-Based Autograd (PyTorch Style)

Build a computation graph with `Rc<RefCell<Node>>` where each node tracks its inputs and backward function.

```rust
struct Node {
    data: Tensor,
    grad: RefCell<Option<Tensor>>,
    backward_fn: Box<dyn Fn()>,
    parents: Vec<Rc<Node>>,
}
```

**Rejected**: Reference counting and interior mutability (`Rc<RefCell<>>`) add runtime overhead. Graph traversal is slower than linear tape replay. Memory leaks possible with circular references.

### Alternative 2: Source-to-Source Transformation

Use procedural macros to transform forward code into forward+backward code at compile time.

```rust
#[autograd]
fn forward(x: &Tensor, w: &Tensor) -> Tensor {
    x.matmul(w).relu()
}
// Macro generates backward function automatically
```

**Rejected**: Macro complexity is high. Debugging generated code is difficult. Limited IDE support. Error messages become cryptic.

### Alternative 3: Replace Manual Gradients Entirely

Remove manual gradient implementations and require autograd for all models.

**Rejected**: Breaking change to all existing code. Performance regression for simple models. Violates zero-cost abstraction principle.

### Alternative 4: dfdx-style Static Shapes

Use const generics for compile-time shape checking with integrated autograd.

```rust
type Linear = Linear<2, 8>;  // 2 inputs, 8 outputs
type Model = Seq<Linear, ReLU, Linear<8, 1>>;
```

**Rejected**: Major refactoring required. Changes public API significantly. Learning curve for type-level programming.

### Alternative 5: Candle-style Tensor-Centric Design

Make all tensors track gradients by default, like Candle or PyTorch.

**Rejected**: Overhead on every tensor operation. Violates zero-cost principle. Existing code would need refactoring.

## Implementation Phases

### Phase 1: Core Infrastructure (MVP)

- [ ] Create `autograd/` module with feature gate
- [ ] Implement `Tape` (Wengert list) data structure
- [ ] Define `GradTensor<B>` wrapper
- [ ] Basic operations: add, mul, matmul

### Phase 2: Integration

- [ ] `AutogradModel` trait for models using autograd
- [ ] Integration with existing `Loss` trait
- [ ] Trainer support for autograd models
- [ ] Unit tests for gradient correctness

### Phase 3: Extended Operations

- [ ] Activation functions (ReLU, Sigmoid, Tanh)
- [ ] Reductions (sum, mean)
- [ ] Broadcasting operations
- [ ] Softmax and LogSoftmax

### Phase 4: Complex Models

- [ ] Multi-layer networks with autograd
- [ ] Skip connections (ResNet-style)
- [ ] Attention mechanisms
- [ ] Convolution operations

## References

- ADR-0001: Separate trainer, losses, optimizers (manual gradients decision)
- ADR-0009: WGPU backend limitations (sync overhead considerations)
- ADR-0010: Gradient boosting architecture (alternative training paradigm)
- [PyTorch Autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [JAX Autodidax](https://jax.readthedocs.io/en/latest/autodidax.html)
- [dfdx](https://github.com/coreylowman/dfdx) - Rust ML with const generics
- [Candle](https://github.com/huggingface/candle) - Minimalist ML framework
