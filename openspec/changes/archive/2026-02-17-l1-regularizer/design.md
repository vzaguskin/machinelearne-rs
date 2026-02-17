## Context

The regularizers module follows a trait-based pattern with `Regularizer<B, M>` trait. L2 already demonstrates the full pattern: struct with lambda parameter, `new()` constructor, and trait implementation returning `(penalty, gradients)`.

## Goals / Non-Goals

**Goals:**
- Add `L1<B>` struct matching L2's API pattern
- Implement L1 penalty: `λ * Σ|w_i|`
- Implement L1 gradient: `λ * sign(w)`
- Add comprehensive unit tests

**Non-Goals:**
- Elastic Net (L1+L2 combination) — can be added later
- Differentiable regularization weight — fixed lambda is sufficient

## Decisions

### Decision 1: Use sign() for gradient computation

The subgradient of |x| is sign(x), where sign(0) = 0. The `Tensor1D` type already has a `sign()` method via the backend's `sign_1d` operation, so we can reuse it directly.

**Rationale:** Consistent with how L2 uses existing tensor operations, and follows the library's design of delegating to backend operations.

### Decision 2: Match L2 structure exactly

L1 will have identical structure to L2:
- Generic over `B: Backend`
- Store `lambda` as `Scalar<B>`
- Implement `Regularizer<B, LinearRegression<B>>`
- Return `LinearParams<B>` with zero bias gradient

**Rationale:** Consistency aids discoverability and maintenance. Users familiar with L2 can immediately understand L1.
