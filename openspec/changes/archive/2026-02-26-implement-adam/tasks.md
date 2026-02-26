## 1. Core Implementation

- [x] 1.1 Create `lib/src/optimizer/adam.rs` module with `AdamState<B>` struct for storing momentum and velocity
- [x] 1.2 Implement `Adam<B>` struct with configurable hyperparameters (lr, beta1, beta2, epsilon)
- [x] 1.3 Implement `Adam::new()` constructor with default hyperparameters
- [x] 1.4 Implement builder methods for custom hyperparameters (`with_beta1()`, `with_beta2()`, `with_epsilon()`)
- [x] 1.5 Implement hyperparameter accessors (`learning_rate()`, `beta1()`, `beta2()`, `epsilon()`)

## 2. Optimizer Trait Implementation

- [x] 2.1 Implement `Optimizer<B, LinearParams<B>>` for `Adam<B>` with bias-corrected updates
- [x] 2.2 Implement `Optimizer<B, MLPParams<B>>` for `Adam<B>` with bias-corrected updates
- [x] 2.3 Handle first-time state initialization when step is called
- [x] 2.4 Implement time step increment and bias correction formulas

## 3. Module Integration

- [x] 3.1 Add `mod adam` and `pub use adam::Adam` to `lib/src/optimizer/mod.rs`
- [x] 3.2 Re-export `Adam` from `lib/src/lib.rs` if needed
- [x] 3.3 Ensure `cargo build` passes without errors
- [x] 3.4 Ensure `cargo fmt` and `cargo clippy` pass

## 4. Unit Tests

- [x] 4.1 Test Adam construction with default hyperparameters
- [x] 4.2 Test Adam construction with custom hyperparameters
- [x] 4.3 Test first step produces correct bias-corrected updates
- [x] 4.4 Test multiple steps accumulate momentum correctly
- [x] 4.5 Test zero gradients don't cause numerical issues
- [x] 4.6 Test with LinearParams
- [x] 4.7 Test with MLPParams
- [x] 4.8 Test clone functionality
- [x] 4.9 Test hyperparameter accessors
- [x] 4.10 Test numerical stability with small/large gradients

## 5. Documentation and Examples

- [x] 5.1 Add rustdoc comments to `Adam` struct and public methods
- [x] 5.2 Add code example in rustdoc showing basic usage
- [x] 5.3 Update or create example using Adam optimizer for MLP training

## 6. Verification

- [x] 6.1 Run all tests: `cargo test -p machinelearne-rs --all-features`
- [x] 6.2 Verify coverage threshold is met
- [x] 6.3 Run benchmarks to verify no performance regression in SGD
