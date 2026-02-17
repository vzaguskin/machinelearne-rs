## 1. Core Implementation

- [x] 1.1 Add matrix inverse helper function to Backend or as standalone utility
- [x] 1.2 Add `ClosedFormSolver<B, M, P>` trait to `lib/src/optimizer/mod.rs` (simplified: direct method on LinearRegression)
- [x] 1.3 Implement `NormalEquation<B>` struct with `solve(X, y) -> LinearParams<B>` (in linalg module)
- [x] 1.4 Add `fit_closed_form(&self, X, y)` method to `LinearRegression<B>`

## 2. Error Handling

- [x] 2.1 Add error type for closed-form solver failures (singular matrix, dimension mismatch)
- [x] 2.2 Implement input validation (empty data, dimension mismatch)
- [x] 2.3 Handle numerical instability gracefully

## 3. Testing

- [x] 3.1 Test closed-form solution correctness (simple linear, multi-feature)
- [x] 3.2 Test error handling (empty input, dimension mismatch)
- [x] 3.3 Test numerical stability with ill-conditioned matrices
- [x] 3.4 Compare accuracy with SGD-trained models

## 4. Examples and Benchmarks

- [x] 4.1 Add `train_linear_closed_form.rs` example
- [x] 4.2 Add benchmark comparing SGD vs closed-form performance (in example)
- [x] 4.3 Add benchmark comparing SGD vs closed-form accuracy (in example)

## 5. Documentation

- [x] 5.1 Add doc comments to new trait and struct
- [x] 5.2 Update CHANGELOG.md
- [x] 5.3 Update README.md with closed-form usage example (skipped - no README.md in repo)
