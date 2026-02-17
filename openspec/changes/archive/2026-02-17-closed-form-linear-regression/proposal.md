## Why

SGD-based training for linear regression is iterative and requires hyperparameter tuning (learning rate, epochs). A closed-form solution using the normal equation provides exact optimal parameters in a single step, which is faster for small-to-medium datasets and eliminates hyperparameter tuning for linear regression.

## What Changes

- Add `ClosedFormSolver<B>` trait in `lib/src/optimizer/mod.rs`
- Add `NormalEquation<B>` struct implementing `ClosedFormSolver` for linear regression
- Add `fit_closed_form()` method to `LinearRegression<B>`
- Add benchmark comparing SGD vs closed-form performance and accuracy
- Add example demonstrating closed-form usage

## Capabilities

### New Capabilities
- `closed-form-solver`: Trait and implementation for one-shot parameter estimation without gradients

### Modified Capabilities
- `model-training`: Add `fit_closed_form()` method to LinearRegression as alternative to Trainer

## Impact

- `lib/src/optimizer/mod.rs`: Add `ClosedFormSolver` trait and `NormalEquation` struct
- `lib/src/model/linear.rs`: Add `fit_closed_form()` method
- `lib/examples/`: Add `train_linear_closed_form.rs` example
- `benchmarks/`: Add benchmark comparing SGD vs closed-form
- `CHANGELOG.md`: Document new capability
