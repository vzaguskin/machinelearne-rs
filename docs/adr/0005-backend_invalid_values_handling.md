# ADR-0002: Backend Handling of Invalid Values

**Status**: Accepted  
**Date**: 2026-02-09  
**Author**: ML Library Development Team  

## Context

A machine learning library must define a consistent strategy for handling invalid inputs at the backend (tensor operation) level. Invalid values arise in two distinct categories:

1. **Mathematically undefined operations** (per IEEE 754 standard):
   - `log(x)` for `x ≤ 0.0`
   - `sqrt(x)` for `x < 0.0`
   - Division by zero

2. **Contract violations** (structural invariants):
   - Empty tensors in reductions (`mean`, `sum`)
   - Shape mismatches in binary operations (`add`, `sub`, `mul`, `div`)
   - Dimension mismatches in matrix operations (`dot`, `matmul`)

Different handling strategies (panic, `NaN`, `Result`) impact:
- Debuggability (early vs. late error detection)
- Performance (runtime checks in hot paths)
- User expectations (alignment with PyTorch/NumPy/TensorFlow)
- Composability (chaining operations in computational graphs)

## Decision

Apply a **differentiated strategy** based on error category:

### Category 1: IEEE 754-compliant elementary operations
For mathematically defined edge cases (`log`, `sqrt`, `exp`, division) — **return `NaN`/`±inf` without panicking**:

```rust
// Example: logarithm implementation
fn log_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
    x.iter().map(|&v| v.ln()).collect() // 0.0 → -inf, -1.0 → NaN
}
```

**Rationale**:
- Follows IEEE 754 standard and industry practice (PyTorch/NumPy/TensorFlow)
- Allows batch processing to continue when a single element is invalid
- Enables post-hoc validation via utility methods (`tensor.has_nan()`, `tensor.has_inf()`)
- Avoids performance overhead of pre-checks in hot paths

### Category 2: Structural contract violations
For operations with explicit preconditions (shapes, non-emptiness) — **panic with descriptive messages**:

```rust
// Example: shape-checked subtraction
fn sub_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
    if a.shape() != b.shape() {
        panic!(
            "sub_2d: shape mismatch — cannot subtract {:?} from {:?}. \
             Shapes must be identical for element-wise operations.",
            b.shape(),
            a.shape()
        );
    }
    // ... perform computation
}
```

**Rationale**:
- Adheres to "fail fast" principle for critical programming errors
- Empty tensors in reductions almost always indicate data pipeline bugs (empty batches, incorrect slicing)
- Human-readable error messages with context accelerate debugging
- Prevents silent corruption of numerical results (e.g., `.zip()` truncation)

### Explicitly rejected approaches
- ❌ `Result<Tensor, Error>` for primitive operations — breaks composability (`a.log()?.add(&b)?`)
- ❌ Returning `0.0`/magic values for empty reductions — masks critical bugs
- ❌ Silent truncation on shape mismatch (`.zip()` without validation) — creates hidden numerical errors

## Consequences

### Positive
- ✅ **Industry alignment**: Matches expectations of users migrating from PyTorch/NumPy
- ✅ **Rapid debugging**: Critical structural errors (empty batches, shape mismatches) surface immediately
- ✅ **Performance**: No runtime checks in hot paths for IEEE 754 operations
- ✅ **Predictability**: Clear semantic boundary between "mathematical edge case" vs. "programming error"

### Negative
- ⚠️ Requires user discipline: downstream validation of `NaN`/`inf` in loss functions and metrics
- ⚠️ Panics may be undesirable in some production scenarios (mitigated at higher API layers)

### Practical recommendations for users
```rust
// In loss functions — apply numerical stabilization
fn binary_cross_entropy<B: Backend>(
    pred: &Tensor1D<B>, 
    target: &Tensor1D<B>
) -> Scalar<B> {
    let eps = Scalar::<B>::new(1e-12);
    let pred_clipped = pred
        .maximum(&eps.into())
        .minimum(&Scalar::<B>::new(1.0 - 1e-12).into());
    // ... compute loss with -inf prevention
}

// During development — validate tensors
#[cfg(debug_assertions)]
fn validate_tensor<B: Backend>(t: &Tensor1D<B>, name: &str) {
    let v = t.to_vec();
    assert!(!v.iter().any(|x| x.is_nan()), 
            "NaN detected in {}: {:?}", name, v);
    assert!(!v.iter().any(|x| x.is_infinite()), 
            "Inf detected in {}: {:?}", name, v);
}
```

## Alternatives Considered

| Approach | Rejection Reason |
|----------|------------------|
| **Always panic** (including `log(0)`) | Breaks batch processing; violates ML ecosystem expectations; requires excessive pre-validation |
| **Always return `NaN`** (including empty tensors) | Masks critical pipeline bugs (empty batch → silent `NaN` propagation); harder root-cause analysis |
| **`Result` for all operations** | Breaks composability (`a.log()?.add(&b)?.mul(&c)?`); excessive boilerplate; performance overhead |
| **Automatic clipping** (`log(max(x, ε))`) | Hides architectural issues (e.g., missing `softmax` before `log`); violates mathematical semantics |

## References
- IEEE 754-2008 Floating-Point Standard  
- PyTorch `torch.log` semantics: https://pytorch.org/docs/stable/generated/torch.log.html  
- NumPy `np.log` behavior: https://numpy.org/doc/stable/reference/generated/numpy.log.html  
- Rust `f64::ln()` documentation: https://doc.rust-lang.org/std/primitive.f64.html#method.ln  
- Related ADR-0001: Separation of trainer, losses, and optimizers