## EXISTING Requirements

### Requirement: Train-Test Split

train_test_split SHALL divide a dataset into training and testing subsets.

#### Scenario: Default split ratio

- **WHEN** train_test_split is called with default parameters
- **THEN** approximately 75% of data is in train set
- **AND** approximately 25% is in test set

#### Scenario: Custom test size

- **WHEN** train_test_split is called with `test_size=0.2`
- **THEN** 20% of data is in test set

#### Scenario: Stratified split

- **WHEN** stratified split is requested
- **THEN** class distribution is preserved in both train and test sets

#### Scenario: Random state reproducibility

- **WHEN** same random_state is used
- **THEN** identical splits are produced

### Requirement: K-Fold Cross-Validation

KFold SHALL split data into k consecutive folds for cross-validation.

#### Scenario: K-fold splits

- **WHEN** KFold with k=5 is created
- **THEN** 5 different train/test split pairs are generated
- **AND** each sample appears in test set exactly once

#### Scenario: K-fold no overlap

- **WHEN** KFold generates splits
- **THEN** train and test sets are disjoint

### Requirement: Grid Search

GridSearchCV SHALL exhaustively search over specified parameter values.

#### Scenario: Grid search finds best params

- **WHEN** GridSearchCV is run with parameter grid
- **THEN** best_params_ contains the combination with highest cross-validation score

#### Scenario: Grid search with cross-validation

- **WHEN** GridSearchCV with cv=5 is run
- **THEN** each parameter combination is evaluated with 5-fold cross-validation

### Requirement: Pipeline Grid Search

Grid search SHALL support pipelines with parameter references using `__` notation.

#### Scenario: Pipeline parameter grid

- **WHEN** parameter grid contains `"scaler__with_mean": [True, False]`
- **THEN** grid search varies the scaler's with_mean parameter

### Requirement: Cross-Validation Scoring

Cross-validation SHALL support multiple scoring metrics.

#### Scenario: Multiple metrics

- **WHEN** scoring is set to `['r2', 'neg_mean_squared_error']`
- **THEN** results contain scores for both metrics
