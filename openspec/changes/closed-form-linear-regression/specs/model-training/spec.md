## ADDED Requirements

### Requirement: Closed-Form Fitting Method

LinearRegression SHALL provide a `fit_closed_form()` method that computes optimal parameters in one step without gradients.

#### Scenario: Fit using closed-form

- **WHEN** `fit_closed_form(X, y)` is called on an Unfitted LinearRegression
- **THEN** a Fitted model is returned with optimal parameters
- **AND** no iteration or learning rate is required

#### Scenario: Closed-form returns same type as Trainer

- **WHEN** `fit_closed_form()` succeeds
- **THEN** return type is `LinearModel<B, Fitted>` (same as `Trainer.fit()`)

### Requirement: Closed-Form Input Validation

`fit_closed_form()` SHALL validate input dimensions and return errors for invalid inputs.

#### Scenario: Dimension mismatch error

- **WHEN** X has shape (n, d) but y has length m != n
- **THEN** method returns an error

#### Scenario: Empty input error

- **WHEN** X or y is empty
- **THEN** method returns an error

### Requirement: Closed-Form Accuracy

`fit_closed_form()` SHALL produce parameters that achieve minimal MSE on the training data.

#### Scenario: Exact solution for linear data

- **WHEN** data follows exact linear relationship y = X @ w_true + b_true
- **THEN** `fit_closed_form()` recovers w_true and b_true within numerical precision

#### Scenario: Better or equal to SGD

- **WHEN** comparing closed-form to SGD with sufficient epochs
- **THEN** closed-form MSE <= SGD MSE (closed-form is optimal)
