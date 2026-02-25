## Why

SGD is the only optimizer currently available in the library, limiting users' ability to train neural networks efficiently. Adam (Adaptive Moment Estimation) is the most widely-used optimizer for deep learning due to its adaptive learning rates, momentum, and fast convergence. Adding Adam will significantly improve the library's usability for MLP training and other gradient-based optimization tasks.

## What Changes

- Add `Adam<B>` optimizer implementing the `Optimizer<B, P>` trait
- Support configurable hyperparameters: learning rate, beta1, beta2, epsilon
- Maintain per-parameter momentum (first moment) and velocity (second moment) state
- Implement bias correction for early training steps
- Add comprehensive unit tests for the optimizer
- Update examples to demonstrate Adam usage

## Capabilities

### New Capabilities

- `adam-optimizer`: Adam optimizer with adaptive learning rates, momentum tracking, and bias correction

### Modified Capabilities

- None (Adam is a new optimizer that doesn't change existing behavior)

## Impact

- **New file**: `lib/src/optimizer/adam.rs` - Adam optimizer implementation
- **Modified**: `lib/src/optimizer/mod.rs` - Export Adam optimizer
- **Examples**: Update MLP examples to optionally use Adam
- **API surface**: New public type `Adam<B>` following the same pattern as `SGD<B>`
