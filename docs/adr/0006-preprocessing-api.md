# ADR-0006: Preprocessing API Design

## Status

Proposed

## Context

The machinelearne-rs library currently lacks data preprocessing capabilities, forcing users to handle scaling, imputation, encoding, and feature transformation outside the library. This issue adds a comprehensive preprocessing module that:

1. **Follows library patterns**: Type-state (Unfitted/Fitted), backend-agnostic, serializable
2. **Supports grid search**: Hyperparameters easily parameterized
3. **Enables pipelines**: Chain transformers, integrate with models
4. **Matches sklearn API**: Familiar interface for users

## Decision

### Two-Trait Design

We adopt a two-trait design pattern similar to the existing model system:

```rust
/// Trait for unfitted transformers with hyperparameters.
pub trait Transformer<B: Backend>: Clone {
    type Input;
    type Output;
    type Params: SerializableParams;
    type Fitted: FittedTransformer<B, Input=Self::Input, Output=Self::Output, Params=Self::Params>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError>;
    fn fit_transform(&self, data: &mut Self::Input) -> Result<Self::Output, PreprocessingError>;
}

/// Trait for fitted transformers ready for inference.
pub trait FittedTransformer<B: Backend>: Clone {
    type Input;
    type Output;
    type Params: SerializableParams;

    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError>;
    fn inverse_transform(&self, data: &Self::Output) -> Result<Self::Input, PreprocessingError>;
    fn extract_params(&self) -> Self::Params;
    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError>;
}
```

This separates:
- **Configuration** (hyperparameters like strategy, feature_range) in `Transformer`
- **State** (learned parameters like mean_, scale_) in `FittedTransformer`

### Phantom Type State Pattern

Like models, transformers use phantom type state:

```rust
pub struct StandardScaler<B: Backend, S = Unfitted> {
    // Configuration (present in both states)
    with_mean: bool,
    with_std: bool,

    // Learned parameters (only in Fitted state)
    mean_: Option<Tensor1D<B>>,
    std_: Option<Tensor1D<B>>,

    _state: PhantomData<S>,
}
```

### Backend-Agnostic via Generics

All transformers are generic over `B: Backend`, enabling CPU and GPU backends without code duplication:

```rust
impl<B: Backend> Transformer<B> for StandardScaler<B, Unfitted> {
    // ...
}
```

### Serializable Parameters

Each transformer defines a serializable `Params` struct:

```rust
#[derive(Serialize, Deserialize)]
pub struct StandardScalerParams {
    pub with_mean: bool,
    pub with_std: bool,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}
```

This enables model persistence and deployment.

### Data State Markers (Optional Extension)

For compile-time safety with missing values, data can carry type-state:

```rust
pub struct HasMissing;  // May contain NaN
pub struct NoMissing;   // Guaranteed no NaN

pub struct Data<B: Backend, S> {
    features: Tensor2D<B>,
    targets: Option<Tensor1D<B>>,
    _state: PhantomData<S>,
}
```

Transformers that clean data transition the state:
- `SimpleImputer`: `Data<B, HasMissing>` → `Data<B, NoMissing>`
- `StandardScaler`: Requires `Data<B, NoMissing>`

This prevents training on dirty data at compile time.

## Module Structure

```
lib/src/preprocessing/
├── mod.rs              # Public API, re-exports
├── traits.rs           # Transformer, FittedTransformer traits
├── error.rs            # PreprocessingError type
│
├── scaling/
│   ├── mod.rs
│   ├── standard.rs     # StandardScaler (z-score)
│   ├── minmax.rs       # MinMaxScaler ([0,1] or custom)
│   ├── robust.rs       # RobustScaler (median/IQR)
│   ├── maxabs.rs       # MaxAbsScaler
│   └── normalizer.rs   # Normalizer (L1/L2/Max per row)
│
├── imputation/
│   ├── mod.rs
│   └── simple.rs       # SimpleImputer (mean/median/mode/constant)
│
├── encoding/
│   ├── mod.rs
│   ├── one_hot.rs      # OneHotEncoder
│   ├── ordinal.rs      # OrdinalEncoder
│   └── label.rs        # LabelEncoder
│
├── transformation/
│   ├── mod.rs
│   └── power.rs        # PowerTransformer (Box-Cox, Yeo-Johnson)
│
├── discretization/
│   ├── mod.rs
│   └── kbins.rs        # KBinsDiscretizer
│
├── feature_extraction/
│   ├── mod.rs
│   ├── pca.rs          # PCA
│   └── polynomial.rs   # PolynomialFeatures
│
└── pipeline/
    ├── mod.rs
    ├── pipeline.rs     # Pipeline (chain transformers)
    └── column.rs       # ColumnTransformer (per-column)
```

## Backend Extensions Required

Column-wise and broadcasting operations needed in `Backend` trait:

```rust
// Column-wise statistics
fn col_mean_2d(t: &Self::Tensor2D) -> Self::Tensor1D;
fn col_std_2d(t: &Self::Tensor2D, ddof: usize) -> Self::Tensor1D;
fn col_min_2d(t: &Self::Tensor2D) -> Self::Tensor1D;
fn col_max_2d(t: &Self::Tensor2D) -> Self::Tensor1D;
fn col_sum_2d(t: &Self::Tensor2D) -> Self::Tensor1D;

// Row-wise operations
fn row_sum_2d(t: &Self::Tensor2D) -> Self::Tensor1D;

// Broadcasting
fn broadcast_sub_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D;
fn broadcast_div_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D;
fn broadcast_mul_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D;
```

## Consequences

### Positive

- **Type safety**: Compile-time guarantees about fitted/unfitted state
- **Consistency**: Same patterns as existing model system
- **Testability**: Separation of concerns enables unit testing
- **Serialization**: All transformers are persistable
- **Backend flexibility**: Works with any Backend implementation

### Negative

- **Verbosity**: Generic parameters add complexity
- **Learning curve**: Users must understand type-state pattern
- **Binary size**: More generics may increase compile times

### Neutral

- Requires extending the Backend trait with column-wise operations
- Pipeline implementation will need macro or trait magic for type safety

## Alternatives Considered

1. **Single trait with Option for fitted state**: Rejected because it loses compile-time safety.
2. **Runtime state checking**: Rejected because errors would only appear at runtime.
3. **Separate preprocessing crate**: Rejected to maintain library cohesion and avoid circular dependencies.

## References

- ADR-0001: Separate trainer from losses (sets precedent for type-state)
- ADR-0004: Model serialization (establishes SerializableParams pattern)
- scikit-learn preprocessing module: API inspiration
