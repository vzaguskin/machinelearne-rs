## EXISTING Requirements

### Requirement: OneHotEncoder

OneHotEncoder SHALL convert categorical integer values to one-hot encoded vectors.

#### Scenario: One-hot encoding basic

- **WHEN** input column contains categories `[0, 1, 2]`
- **THEN** output is 3 columns with one-hot encoding

#### Scenario: One-hot encoding multi-column

- **WHEN** input has 2 columns with 3 and 2 categories respectively
- **THEN** output has `3 + 2 = 5` columns

#### Scenario: One-hot handle unknown

- **WHEN** unknown category is encountered during transform and `handle_unknown='ignore'`
- **THEN** all output columns for that sample are zero

### Requirement: OrdinalEncoder

OrdinalEncoder SHALL convert categorical values to integer ordinals.

#### Scenario: Ordinal encoding

- **WHEN** categories are `["low", "medium", "high"]`
- **THEN** they are mapped to `[0, 1, 2]`

### Requirement: LabelEncoder

LabelEncoder SHALL encode target labels with values between 0 and n_classes-1.

#### Scenario: Label encoding targets

- **WHEN** target values are `["cat", "dog", "cat", "bird"]`
- **THEN** encoded values are `[0, 1, 0, 2]` (or similar mapping)

#### Scenario: Label encoder inverse transform

- **WHEN** inverse_transform is called on encoded labels
- **THEN** original string labels are returned

### Requirement: Encoder Serialization

All encoders SHALL be serializable via their params representation.

#### Scenario: Encoder save/load

- **WHEN** a fitted encoder is saved and loaded
- **THEN** it produces identical encodings
