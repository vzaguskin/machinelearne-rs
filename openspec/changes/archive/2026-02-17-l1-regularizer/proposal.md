## Why

The library currently only supports L2 (ridge) regularization. L1 (Lasso) regularization is a fundamental ML technique that encourages sparsity in model weights, enabling feature selection and producing more interpretable models.

## What Changes

- Add `L1<B>` struct implementing the `Regularizer` trait
- L1 penalty: `λ * Σ|w_i|` (sum of absolute weights)
- L1 gradient: `λ * sign(w)` (subgradient at zero)

## Capabilities

### New Capabilities
- `l1-regularization`: Apply L1 (Lasso) penalty to model weights during training

### Modified Capabilities
<!-- No existing capabilities are modified -->

## Impact

- `lib/src/regularizers/mod.rs`: Add `L1` struct and `Regularizer` implementation with tests
