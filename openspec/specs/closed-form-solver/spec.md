# closed-form-solver Specification

## Purpose
TBD - created by archiving change closed-form-linear-regression. Update Purpose after archive.
## Requirements
### Requirement: Normal Equation Solver

The NormalEquation solver SHALL compute optimal linear regression parameters using the closed-form solution w = (X^T X)^(-1) X^T y.

#### Scenario: Solve for simple linear regression

- **WHEN** X = [[1], [2], [3]] and y = [2, 4, 6] (y = 2x)
- **THEN** weights = [2.0] and bias ≈ 0.0

#### Scenario: Solve for multiple features

- **WHEN** X has 3 features and 100 samples
- **THEN** solution returns weights of shape (3,) and a scalar bias

### Requirement: Numerical Stability

The solver SHALL handle ill-conditioned matrices gracefully.

#### Scenario: Near-singular matrix

- **WHEN** X^T X is nearly singular (highly correlated features)
- **THEN** solver returns a valid solution using pseudo-inverse or regularization
- **AND** does not panic or return NaN

### Requirement: Error Handling

The solver SHALL return errors for invalid inputs.

#### Scenario: Empty dataset

- **WHEN** X or y is empty
- **THEN** solver returns an error (not panic)

#### Scenario: Dimension mismatch

- **WHEN** X has n rows but y has m != n elements
- **THEN** solver returns an error

### Requirement: Accuracy

The closed-form solution SHALL produce parameters that minimize MSE within numerical precision.

#### Scenario: MSE is minimal

- **WHEN** parameters are computed via closed-form
- **THEN** MSE on training data is at or very near the global minimum
- **AND** MSE is <= MSE from SGD-trained model (given sufficient SGD epochs)

