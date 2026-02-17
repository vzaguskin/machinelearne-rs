## EXISTING Requirements

### Requirement: StandardScaler

StandardScaler SHALL transform features to zero mean and unit variance using `z = (x - mean) / std`.

#### Scenario: StandardScaler fit

- **WHEN** StandardScaler is fit on data with columns having different means and stds
- **THEN** learned mean_ and std_ match column-wise statistics

#### Scenario: StandardScaler transform

- **WHEN** data is transformed by fitted StandardScaler
- **THEN** output columns have approximately zero mean and unit variance

#### Scenario: StandardScaler with_mean=false

- **WHEN** StandardScaler is configured with `with_mean=false`
- **THEN** data is scaled but not centered

### Requirement: MinMaxScaler

MinMaxScaler SHALL transform features to a given range (default [0, 1]).

#### Scenario: MinMaxScaler default range

- **WHEN** MinMaxScaler is fit and transform is applied
- **THEN** all values are in range [0, 1]

#### Scenario: MinMaxScaler custom range

- **WHEN** MinMaxScaler is configured with `feature_range=(a, b)`
- **THEN** all transformed values are in range [a, b]

### Requirement: RobustScaler

RobustScaler SHALL use median and IQR for scaling, robust to outliers.

#### Scenario: RobustScaler ignores outliers

- **WHEN** data contains extreme outliers
- **THEN** RobustScaler centering/scaling is less affected than StandardScaler

### Requirement: MaxAbsScaler

MaxAbsScaler SHALL scale by maximum absolute value, preserving sparsity.

#### Scenario: MaxAbsScaler preserves zeros

- **WHEN** input data contains zeros
- **THEN** transformed data still contains zeros

### Requirement: Normalizer

Normalizer SHALL scale individual samples to unit norm (L1, L2, or Max).

#### Scenario: L2 normalization

- **WHEN** Normalizer with L2 norm is applied to sample `[3.0, 4.0]`
- **THEN** result is `[0.6, 0.8]` (unit L2 norm)

### Requirement: Transformer Trait Interface

All scalers SHALL implement `Transformer` and `FittedTransformer` traits.

#### Scenario: Fit-transform separation

- **WHEN** a scaler is fit on training data
- **THEN** the same fitted scaler can transform multiple datasets

#### Scenario: Inverse transform

- **WHEN** `inverse_transform` is called on transformed data
- **THEN** result approximates original data
