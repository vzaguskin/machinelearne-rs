## Context

The library currently only supports iterative training via SGD through the `Optimizer` trait. For linear regression, there exists a closed-form solution (normal equation) that computes optimal parameters in one step:

```
w = (X^T X)^(-1) X^T y
```

This is faster for small-to-medium datasets and requires no hyperparameter tuning.

## Goals / Non-Goals

**Goals:**
- Add a `ClosedFormSolver` trait separate from `Optimizer`
- Implement `NormalEquation` solver for linear regression
- Add `fit_closed_form()` method to `LinearRegression`
- Benchmark closed-form vs SGD performance and accuracy
- Provide example demonstrating usage

**Non-Goals:**
- Closed-form solutions for non-linear models
- Regularization support in closed-form (ridge regression could be added later)
- Replacing SGD as the default training method

## Decisions

### Decision 1: Separate trait from Optimizer

Create `ClosedFormSolver<B, M, P>` trait instead of extending `Optimizer`.

**Rationale:** The `Optimizer` trait assumes iterative gradient-based updates. Closed-form solutions have fundamentally different semantics:
- Take full dataset as input (not params + gradients)
- Return final parameters directly (not incremental updates)
- Don't fit into the Trainer's epoch loop

A separate trait maintains clean separation of concerns.

### Decision 2: Method on LinearRegression

Add `fit_closed_form(&self, X, y) -> Result<FittedModel, Error>` directly on `LinearRegression<B>`.

**Rationale:** More ergonomic than requiring users to instantiate a solver. The solver logic is internal - users just call the method.

### Decision 3: Augmented matrix approach for bias

Use the augmented matrix technique: append a column of 1s to X to solve for both weights and bias simultaneously.

**Rationale:** Simplifies implementation - single matrix operation instead of separate weight and bias computations.

```
X_aug = [X | 1]  (n x (d+1))
Solve: w_aug = (X_aug^T X_aug)^(-1) X_aug^T y
Extract: weights = w_aug[0:d], bias = w_aug[d]
```

### Decision 4: Backend requirements

The normal equation requires matrix operations not currently in the Backend trait:
- Matrix multiplication (2D x 2D)
- Matrix inverse (or pseudo-inverse for numerical stability)

**Options:**
1. Add to Backend trait
2. Implement directly in NormalEquation using existing primitives
3. Use nalgebra for the inverse (new dependency)

**Chosen:** Option 2 - implement using existing tensor operations. For matrix inverse, use Gauss-Jordan elimination or Cholesky decomposition on (X^T X).

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Numerical instability for ill-conditioned X^T X | Use pseudo-inverse or add small regularization (ridge) |
| Memory usage for large datasets | Document that closed-form is for small-to-medium datasets |
| Backend doesn't support matrix inverse | Implement using existing tensor operations; fail gracefully |

## Open Questions

- Should we add ridge regularization option to the closed-form solver? (Can be added later)
- What's the practical dataset size limit before closed-form becomes slower than SGD? (Benchmark will reveal this)
