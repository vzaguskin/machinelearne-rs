//! Type-safe parameter grids for hyperparameter search.
//!
//! This module provides builder-pattern types for defining parameter grids
//! to search over during hyperparameter tuning.

// ============================================================================
// Preprocessing Types
// ============================================================================

/// Types of scalers for preprocessing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ScalerType {
    /// StandardScaler (zero mean, unit variance).
    Standard,
    /// MinMaxScaler (scale to [0, 1] or custom range).
    MinMax,
    /// RobustScaler (uses median and IQR, robust to outliers).
    Robust,
    /// MaxAbsScaler (scale by max absolute value).
    MaxAbs,
    /// No scaling.
    #[default]
    None,
}

/// Strategies for imputing missing values.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ImputeStrategy {
    /// Replace with mean of the feature.
    #[default]
    Mean,
    /// Replace with median of the feature.
    Median,
    /// Replace with most frequent value.
    MostFrequent,
    /// Replace with a constant value.
    Constant(f32),
}

/// Parameter grid for scaler selection.
#[derive(Clone, Debug, Default)]
pub struct ScalerGrid {
    /// Scaler types to search.
    pub types: Vec<ScalerType>,
}

impl ScalerGrid {
    /// Create a new scaler grid.
    pub fn new(types: Vec<ScalerType>) -> Self {
        Self { types }
    }

    /// Returns the number of scaler types.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Returns true if there are no scaler types.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
}

/// Parameter grid for imputer selection.
#[derive(Clone, Debug, Default)]
pub struct ImputerGrid {
    /// Impute strategies to search.
    pub strategies: Vec<ImputeStrategy>,
}

impl ImputerGrid {
    /// Create a new imputer grid.
    pub fn new(strategies: Vec<ImputeStrategy>) -> Self {
        Self { strategies }
    }

    /// Returns the number of impute strategies.
    pub fn len(&self) -> usize {
        self.strategies.len()
    }

    /// Returns true if there are no strategies.
    pub fn is_empty(&self) -> bool {
        self.strategies.is_empty()
    }
}

/// Concrete preprocessing parameters for a single configuration.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PreprocessingParams {
    /// Selected scaler type.
    pub scaler: ScalerType,
    /// Selected impute strategy (None means no imputation).
    pub imputer: Option<ImputeStrategy>,
    /// Polynomial degree (1 = no expansion).
    pub poly_degree: usize,
}

/// Parameter grid for preprocessing options.
#[derive(Clone, Debug, Default)]
pub struct PreprocessingGrid {
    /// Scaler options.
    pub scaler: Option<ScalerGrid>,
    /// Imputer options.
    pub imputer: Option<ImputerGrid>,
    /// Polynomial feature options.
    pub polynomial: Option<PolynomialGrid>,
}

impl PreprocessingGrid {
    /// Create a new empty preprocessing grid.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add scaler options.
    pub fn with_scaler(mut self, scaler: ScalerGrid) -> Self {
        self.scaler = Some(scaler);
        self
    }

    /// Add imputer options.
    pub fn with_imputer(mut self, imputer: ImputerGrid) -> Self {
        self.imputer = Some(imputer);
        self
    }

    /// Add polynomial options.
    pub fn with_polynomial(mut self, polynomial: PolynomialGrid) -> Self {
        self.polynomial = Some(polynomial);
        self
    }

    /// Count total preprocessing combinations.
    pub fn n_combinations(&self) -> usize {
        let n_scaler = self.scaler.as_ref().map(|s| s.len()).unwrap_or(1);
        let n_imputer = self.imputer.as_ref().map(|i| i.len()).unwrap_or(1);
        let n_poly = self.polynomial.as_ref().map(|p| p.len()).unwrap_or(1);

        n_scaler * n_imputer * n_poly
    }

    /// Iterate over all preprocessing combinations.
    pub fn iter(&self) -> PreprocessingGridIterator {
        PreprocessingGridIterator {
            grid: self.clone(),
            current_scaler: 0,
            current_imputer: 0,
            current_poly: 0,
        }
    }
}

/// Iterator over preprocessing parameter combinations.
pub struct PreprocessingGridIterator {
    grid: PreprocessingGrid,
    current_scaler: usize,
    current_imputer: usize,
    current_poly: usize,
}

impl Iterator for PreprocessingGridIterator {
    type Item = PreprocessingParams;

    fn next(&mut self) -> Option<Self::Item> {
        let scalers: Vec<ScalerType> = self
            .grid
            .scaler
            .as_ref()
            .map(|s| s.types.clone())
            .unwrap_or_else(|| vec![ScalerType::None]);

        let imputers: Vec<Option<ImputeStrategy>> = self
            .grid
            .imputer
            .as_ref()
            .map(|i| i.strategies.iter().copied().map(Some).collect())
            .unwrap_or_else(|| vec![None]);

        let poly_degrees: Vec<usize> = self
            .grid
            .polynomial
            .as_ref()
            .map(|p| p.degrees.clone())
            .unwrap_or_else(|| vec![1]);

        // Check if we've exhausted all combinations
        if self.current_scaler >= scalers.len() {
            return None;
        }

        let params = PreprocessingParams {
            scaler: scalers[self.current_scaler],
            imputer: imputers[self.current_imputer],
            poly_degree: poly_degrees[self.current_poly],
        };

        // Advance: poly -> imputer -> scaler
        self.current_poly += 1;
        if self.current_poly >= poly_degrees.len() {
            self.current_poly = 0;
            self.current_imputer += 1;
            if self.current_imputer >= imputers.len() {
                self.current_imputer = 0;
                self.current_scaler += 1;
            }
        }

        Some(params)
    }
}

// ============================================================================
// Model Types (existing)
// ============================================================================

/// A single parameter combination for linear regression training.
#[derive(Clone, Debug)]
pub struct ParamCombination {
    /// Learning rate for SGD optimizer.
    pub learning_rate: f64,
    /// L2 regularization lambda (0.0 for no regularization).
    pub lambda: f64,
    /// Batch size for training.
    pub batch_size: usize,
    /// Maximum training epochs.
    pub max_epochs: usize,
    /// Polynomial degree for feature expansion (1 = no expansion).
    pub poly_degree: usize,
}

/// Parameter grid for SGD optimizer.
#[derive(Clone, Debug)]
pub struct SGDGrid {
    /// Learning rates to search.
    pub learning_rates: Vec<f64>,
}

impl SGDGrid {
    /// Create a new SGD grid with given learning rates.
    pub fn new(learning_rates: Vec<f64>) -> Self {
        Self { learning_rates }
    }

    /// Returns the number of learning rate values.
    pub fn len(&self) -> usize {
        self.learning_rates.len()
    }

    /// Returns true if there are no learning rates.
    pub fn is_empty(&self) -> bool {
        self.learning_rates.is_empty()
    }
}

impl Default for SGDGrid {
    fn default() -> Self {
        Self {
            learning_rates: vec![0.01],
        }
    }
}

/// Parameter grid for regularizers.
#[derive(Clone, Debug, Default)]
pub enum RegularizerGrid {
    /// No regularization (single combination with lambda=0).
    #[default]
    None,
    /// L2 regularization with lambda values to search.
    L2 { lambdas: Vec<f64> },
}

impl RegularizerGrid {
    /// Returns the number of lambda values.
    pub fn len(&self) -> usize {
        match self {
            RegularizerGrid::None => 1, // Single combination
            RegularizerGrid::L2 { lambdas } => lambdas.len(),
        }
    }

    /// Returns true if there are no lambda values (only possible for L2 with empty vec).
    pub fn is_empty(&self) -> bool {
        matches!(self, RegularizerGrid::L2 { lambdas } if lambdas.is_empty())
    }

    /// Returns the lambda values for this grid.
    pub fn lambdas(&self) -> Vec<f64> {
        match self {
            RegularizerGrid::None => vec![0.0],
            RegularizerGrid::L2 { lambdas } => lambdas.clone(),
        }
    }
}

/// Parameter grid for trainer hyperparameters.
#[derive(Clone, Debug)]
pub struct TrainerGrid {
    /// Batch sizes to search.
    pub batch_sizes: Vec<usize>,
    /// Max epochs to search.
    pub max_epochs: Vec<usize>,
}

impl TrainerGrid {
    /// Create a new trainer grid.
    pub fn new(batch_sizes: Vec<usize>, max_epochs: Vec<usize>) -> Self {
        Self {
            batch_sizes,
            max_epochs,
        }
    }

    /// Returns the number of (batch_size, max_epochs) combinations.
    pub fn len(&self) -> usize {
        self.batch_sizes.len() * self.max_epochs.len()
    }

    /// Returns true if there are no combinations (empty batch_sizes or max_epochs).
    pub fn is_empty(&self) -> bool {
        self.batch_sizes.is_empty() || self.max_epochs.is_empty()
    }
}

impl Default for TrainerGrid {
    fn default() -> Self {
        Self {
            batch_sizes: vec![32],
            max_epochs: vec![1000],
        }
    }
}

/// Parameter grid for polynomial preprocessing.
#[derive(Clone, Debug)]
pub struct PolynomialGrid {
    /// Polynomial degrees to search.
    pub degrees: Vec<usize>,
}

impl PolynomialGrid {
    /// Create a new polynomial grid.
    pub fn new(degrees: Vec<usize>) -> Self {
        Self { degrees }
    }

    /// Returns the number of degree values.
    pub fn len(&self) -> usize {
        self.degrees.len()
    }

    /// Returns true if there are no degree values.
    pub fn is_empty(&self) -> bool {
        self.degrees.is_empty()
    }
}

/// Combined parameter grid for linear regression.
///
/// # Example
///
/// ```rust
/// use machinelearne_rs::model_selection::{
///     LinearRegressionGrid, SGDGrid, RegularizerGrid, TrainerGrid
/// };
///
/// let grid = LinearRegressionGrid::new()
///     .with_optimizer(SGDGrid::new(vec![0.001, 0.01, 0.1]))
///     .with_regularizer(RegularizerGrid::L2 { lambdas: vec![0.0, 0.1] })
///     .with_trainer(TrainerGrid::new(vec![16, 32], vec![100, 500]));
///
/// // Total: 3 * 2 * 2 * 2 = 24 combinations
/// assert_eq!(grid.n_combinations(), 24);
/// ```
#[derive(Clone, Debug, Default)]
pub struct LinearRegressionGrid {
    /// Optimizer parameters.
    pub optimizer: SGDGrid,
    /// Regularizer parameters.
    pub regularizer: RegularizerGrid,
    /// Trainer parameters.
    pub trainer: TrainerGrid,
    /// Optional polynomial preprocessing.
    pub polynomial: Option<PolynomialGrid>,
}

impl LinearRegressionGrid {
    /// Create a new empty grid with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set optimizer grid with learning rates.
    pub fn with_learning_rates(mut self, learning_rates: Vec<f64>) -> Self {
        self.optimizer = SGDGrid::new(learning_rates);
        self
    }

    /// Set optimizer grid.
    pub fn with_optimizer(mut self, optimizer: SGDGrid) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set L2 regularization with lambda values.
    pub fn with_lambdas(mut self, lambdas: Vec<f64>) -> Self {
        self.regularizer = RegularizerGrid::L2 { lambdas };
        self
    }

    /// Set regularizer grid.
    pub fn with_regularizer(mut self, regularizer: RegularizerGrid) -> Self {
        self.regularizer = regularizer;
        self
    }

    /// Set trainer grid.
    pub fn with_trainer(mut self, trainer: TrainerGrid) -> Self {
        self.trainer = trainer;
        self
    }

    /// Set polynomial grid.
    pub fn with_polynomial(mut self, polynomial: PolynomialGrid) -> Self {
        self.polynomial = Some(polynomial);
        self
    }

    /// Count total parameter combinations.
    pub fn n_combinations(&self) -> usize {
        let n_lr = self.optimizer.learning_rates.len();
        let n_lambda = self.regularizer.len();
        let n_trainer = self.trainer.len();
        let n_poly = self.polynomial.as_ref().map(|p| p.len()).unwrap_or(1);

        n_lr * n_lambda * n_trainer * n_poly
    }

    /// Iterate over all parameter combinations.
    ///
    /// Returns an iterator that yields each unique combination of parameters.
    pub fn iter(&self) -> ParamGridIterator {
        ParamGridIterator {
            grid: self.clone(),
            current_lr: 0,
            current_lambda: 0,
            current_batch: 0,
            current_epochs: 0,
            current_poly: 0,
        }
    }
}

/// Iterator over parameter combinations.
pub struct ParamGridIterator {
    grid: LinearRegressionGrid,
    current_lr: usize,
    current_lambda: usize,
    current_batch: usize,
    current_epochs: usize,
    current_poly: usize,
}

impl Iterator for ParamGridIterator {
    type Item = ParamCombination;

    fn next(&mut self) -> Option<Self::Item> {
        let lambdas = self.grid.regularizer.lambdas();
        let poly_degrees: Vec<usize> = self
            .grid
            .polynomial
            .as_ref()
            .map(|p| p.degrees.clone())
            .unwrap_or_else(|| vec![1]);

        // Check if we've exhausted all combinations
        if self.current_lr >= self.grid.optimizer.learning_rates.len() {
            return None;
        }

        let combination = ParamCombination {
            learning_rate: self.grid.optimizer.learning_rates[self.current_lr],
            lambda: lambdas[self.current_lambda],
            batch_size: self.grid.trainer.batch_sizes[self.current_batch],
            max_epochs: self.grid.trainer.max_epochs[self.current_epochs],
            poly_degree: poly_degrees[self.current_poly],
        };

        // Advance to next combination (poly -> epochs -> batch -> lambda -> lr)
        self.current_poly += 1;
        if self.current_poly >= poly_degrees.len() {
            self.current_poly = 0;
            self.current_epochs += 1;
            if self.current_epochs >= self.grid.trainer.max_epochs.len() {
                self.current_epochs = 0;
                self.current_batch += 1;
                if self.current_batch >= self.grid.trainer.batch_sizes.len() {
                    self.current_batch = 0;
                    self.current_lambda += 1;
                    if self.current_lambda >= lambdas.len() {
                        self.current_lambda = 0;
                        self.current_lr += 1;
                    }
                }
            }
        }

        Some(combination)
    }
}

// ============================================================================
// MLP Types
// ============================================================================

use crate::model::Activation;

/// Architecture configuration for MLP (hidden layer sizes).
///
/// The full layer sizes will be: [n_features, hidden_layers..., n_outputs]
#[derive(Clone, Debug, PartialEq)]
pub struct MLPArchitecture {
    /// Hidden layer sizes (e.g., vec![64, 32] for two hidden layers)
    pub hidden_layers: Vec<usize>,
}

impl MLPArchitecture {
    /// Create a new architecture with the given hidden layer sizes.
    pub fn new(hidden_layers: Vec<usize>) -> Self {
        Self { hidden_layers }
    }

    /// Create an architecture with a single hidden layer.
    pub fn single(size: usize) -> Self {
        Self {
            hidden_layers: vec![size],
        }
    }

    /// Create an architecture with two hidden layers.
    pub fn double(size1: usize, size2: usize) -> Self {
        Self {
            hidden_layers: vec![size1, size2],
        }
    }
}

/// Activation configuration for MLP layers.
#[derive(Clone, Debug, PartialEq)]
pub struct MLPActivations {
    /// Activation functions for each layer transition (hidden + output).
    pub activations: Vec<Activation>,
}

impl MLPActivations {
    /// Create activation configuration.
    pub fn new(activations: Vec<Activation>) -> Self {
        Self { activations }
    }

    /// Use the same activation for all hidden layers, with Identity for output.
    pub fn uniform(hidden_activation: Activation, num_hidden: usize) -> Self {
        let mut activations = vec![hidden_activation; num_hidden];
        activations.push(Activation::Identity); // Output layer
        Self { activations }
    }

    /// ReLU for all hidden layers, Identity for output.
    pub fn relu_all(num_hidden: usize) -> Self {
        Self::uniform(Activation::ReLU, num_hidden)
    }

    /// Tanh for all hidden layers, Identity for output.
    pub fn tanh_all(num_hidden: usize) -> Self {
        Self::uniform(Activation::Tanh, num_hidden)
    }

    /// Sigmoid for all hidden layers, Identity for output.
    pub fn sigmoid_all(num_hidden: usize) -> Self {
        Self::uniform(Activation::Sigmoid, num_hidden)
    }
}

/// A single parameter combination for MLP training.
#[derive(Clone, Debug)]
pub struct MLPParamCombination {
    /// Architecture (hidden layer sizes).
    pub architecture: MLPArchitecture,
    /// Activation functions per layer.
    pub activations: MLPActivations,
    /// Learning rate for SGD optimizer.
    pub learning_rate: f64,
    /// L2 regularization lambda (0.0 for no regularization).
    pub lambda: f64,
    /// Batch size for training.
    pub batch_size: usize,
    /// Maximum training epochs.
    pub max_epochs: usize,
}

/// Parameter grid for MLP architectures.
#[derive(Clone, Debug, Default)]
pub struct MLPArchitectureGrid {
    /// Hidden layer configurations to search.
    pub architectures: Vec<MLPArchitecture>,
}

impl MLPArchitectureGrid {
    /// Create a new architecture grid.
    pub fn new(architectures: Vec<MLPArchitecture>) -> Self {
        Self { architectures }
    }

    /// Returns the number of architectures.
    pub fn len(&self) -> usize {
        self.architectures.len()
    }

    /// Returns true if there are no architectures.
    pub fn is_empty(&self) -> bool {
        self.architectures.is_empty()
    }
}

/// Parameter grid for MLP activations.
#[derive(Clone, Debug, Default)]
pub struct MLPActivationsGrid {
    /// Activation configurations to search.
    pub activation_configs: Vec<MLPActivations>,
}

impl MLPActivationsGrid {
    /// Create a new activations grid.
    pub fn new(activation_configs: Vec<MLPActivations>) -> Self {
        Self { activation_configs }
    }

    /// Create grid from a list of activations for hidden layers (output is always Identity).
    /// Each activation will be used for all hidden layers.
    pub fn from_hidden_activations(hidden_activations: Vec<Activation>, num_hidden: usize) -> Self {
        let activation_configs = hidden_activations
            .into_iter()
            .map(|act| MLPActivations::uniform(act, num_hidden))
            .collect();
        Self { activation_configs }
    }

    /// Returns the number of activation configurations.
    pub fn len(&self) -> usize {
        self.activation_configs.len()
    }

    /// Returns true if there are no activation configurations.
    pub fn is_empty(&self) -> bool {
        self.activation_configs.is_empty()
    }
}

/// Combined parameter grid for MLP models.
///
/// # Example
///
/// ```rust
/// use machinelearne_rs::model_selection::{
///     MLPGrid, MLPArchitecture, MLPActivations, SGDGrid, RegularizerGrid, TrainerGrid
/// };
/// use machinelearne_rs::model::Activation;
///
/// let grid = MLPGrid::new()
///     .with_architectures(vec![
///         MLPArchitecture::single(8),
///         MLPArchitecture::double(16, 8),
///     ])
///     .with_hidden_activations(vec![Activation::ReLU, Activation::Tanh], 1)
///     .with_learning_rates(vec![0.001, 0.01])
///     .with_lambdas(vec![0.0, 0.01]);
///
/// // Total combinations depends on architecture/activation compatibility
/// println!("Total combinations: {}", grid.n_combinations());
/// ```
#[derive(Clone, Debug, Default)]
pub struct MLPGrid {
    /// Architecture options.
    pub architecture: MLPArchitectureGrid,
    /// Activation options.
    pub activations: MLPActivationsGrid,
    /// Optimizer parameters.
    pub optimizer: SGDGrid,
    /// Regularizer parameters.
    pub regularizer: RegularizerGrid,
    /// Trainer parameters.
    pub trainer: TrainerGrid,
}

impl MLPGrid {
    /// Create a new empty grid with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set architecture grid.
    pub fn with_architectures(mut self, architectures: Vec<MLPArchitecture>) -> Self {
        self.architecture = MLPArchitectureGrid::new(architectures);
        self
    }

    /// Set activations grid with full control over each config.
    pub fn with_activations(mut self, activation_configs: Vec<MLPActivations>) -> Self {
        self.activations = MLPActivationsGrid::new(activation_configs);
        self
    }

    /// Set hidden layer activations (applied uniformly, output is always Identity).
    /// Note: This sets up activations for the first architecture's hidden layer count.
    pub fn with_hidden_activations(
        mut self,
        hidden_activations: Vec<Activation>,
        num_hidden: usize,
    ) -> Self {
        self.activations =
            MLPActivationsGrid::from_hidden_activations(hidden_activations, num_hidden);
        self
    }

    /// Set optimizer grid with learning rates.
    pub fn with_learning_rates(mut self, learning_rates: Vec<f64>) -> Self {
        self.optimizer = SGDGrid::new(learning_rates);
        self
    }

    /// Set optimizer grid.
    pub fn with_optimizer(mut self, optimizer: SGDGrid) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set L2 regularization with lambda values.
    pub fn with_lambdas(mut self, lambdas: Vec<f64>) -> Self {
        self.regularizer = RegularizerGrid::L2 { lambdas };
        self
    }

    /// Set regularizer grid.
    pub fn with_regularizer(mut self, regularizer: RegularizerGrid) -> Self {
        self.regularizer = regularizer;
        self
    }

    /// Set trainer grid.
    pub fn with_trainer(mut self, trainer: TrainerGrid) -> Self {
        self.trainer = trainer;
        self
    }

    /// Count total parameter combinations.
    pub fn n_combinations(&self) -> usize {
        let n_arch = self.architecture.len().max(1);
        let n_act = self.activations.len().max(1);
        let n_lr = self.optimizer.learning_rates.len();
        let n_lambda = self.regularizer.len();
        let n_trainer = self.trainer.len();

        n_arch * n_act * n_lr * n_lambda * n_trainer
    }

    /// Iterate over all parameter combinations.
    ///
    /// Only yields combinations where activations match the architecture.
    pub fn iter(&self) -> MLPParamGridIterator {
        let architectures: Vec<_> = if self.architecture.is_empty() {
            vec![MLPArchitecture::new(vec![])] // Default: no hidden layers (linear)
        } else {
            self.architecture.architectures.clone()
        };

        let activation_configs: Vec<_> = if self.activations.is_empty() {
            vec![MLPActivations::new(vec![Activation::Identity])] // Default: single identity
        } else {
            self.activations.activation_configs.clone()
        };

        MLPParamGridIterator {
            grid: self.clone(),
            architectures,
            activation_configs,
            current_arch: 0,
            current_act: 0,
            current_lr: 0,
            current_lambda: 0,
            current_batch: 0,
            current_epochs: 0,
        }
    }
}

/// Iterator over MLP parameter combinations.
pub struct MLPParamGridIterator {
    grid: MLPGrid,
    architectures: Vec<MLPArchitecture>,
    activation_configs: Vec<MLPActivations>,
    current_arch: usize,
    current_act: usize,
    current_lr: usize,
    current_lambda: usize,
    current_batch: usize,
    current_epochs: usize,
}

impl Iterator for MLPParamGridIterator {
    type Item = MLPParamCombination;

    fn next(&mut self) -> Option<Self::Item> {
        let lambdas = self.grid.regularizer.lambdas();

        // Check if we've exhausted all combinations
        if self.current_arch >= self.architectures.len() {
            return None;
        }

        let arch = &self.architectures[self.current_arch];
        let num_layers = arch.hidden_layers.len() + 1; // +1 for output layer

        // Find a compatible activation config
        let mut found_compatible = false;

        while self.current_act < self.activation_configs.len() {
            let act_config = &self.activation_configs[self.current_act];
            if act_config.activations.len() == num_layers {
                found_compatible = true;
                break;
            }
            self.current_act += 1;
        }

        // If no compatible activation found, skip this architecture
        if !found_compatible {
            self.current_arch += 1;
            self.current_act = 0;
            return self.next();
        }

        let activations = self.activation_configs[self.current_act].clone();

        let combination = MLPParamCombination {
            architecture: arch.clone(),
            activations,
            learning_rate: self.grid.optimizer.learning_rates[self.current_lr],
            lambda: lambdas[self.current_lambda],
            batch_size: self.grid.trainer.batch_sizes[self.current_batch],
            max_epochs: self.grid.trainer.max_epochs[self.current_epochs],
        };

        // Advance to next combination
        self.current_epochs += 1;
        if self.current_epochs >= self.grid.trainer.max_epochs.len() {
            self.current_epochs = 0;
            self.current_batch += 1;
            if self.current_batch >= self.grid.trainer.batch_sizes.len() {
                self.current_batch = 0;
                self.current_lambda += 1;
                if self.current_lambda >= lambdas.len() {
                    self.current_lambda = 0;
                    self.current_lr += 1;
                    if self.current_lr >= self.grid.optimizer.learning_rates.len() {
                        self.current_lr = 0;
                        self.current_act += 1;
                        if self.current_act >= self.activation_configs.len() {
                            self.current_act = 0;
                            self.current_arch += 1;
                        }
                    }
                }
            }
        }

        Some(combination)
    }
}

// ============================================================================
// Combined Pipeline Types
// ============================================================================

/// Concrete model parameters for a single configuration.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelParams {
    /// Learning rate for SGD optimizer.
    pub learning_rate: f64,
    /// L2 regularization lambda (0.0 for no regularization).
    pub lambda: f64,
    /// Batch size for training.
    pub batch_size: usize,
    /// Maximum training epochs.
    pub max_epochs: usize,
}

impl From<&ParamCombination> for ModelParams {
    fn from(combo: &ParamCombination) -> Self {
        Self {
            learning_rate: combo.learning_rate,
            lambda: combo.lambda,
            batch_size: combo.batch_size,
            max_epochs: combo.max_epochs,
        }
    }
}

/// Concrete pipeline parameters combining preprocessing and model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PipelineParams {
    /// Preprocessing parameters.
    pub preprocessing: PreprocessingParams,
    /// Model parameters.
    pub model: ModelParams,
    /// Cross-validation mean score (populated after evaluation).
    pub mean_score: f64,
    /// Cross-validation std score (populated after evaluation).
    pub std_score: f64,
}

/// Combined parameter grid for full pipeline search.
///
/// # Example
///
/// ```rust
/// use machinelearne_rs::model_selection::{
///     PipelineGrid, PreprocessingGrid, ScalerGrid, ScalerType,
///     LinearRegressionGrid, TrainerGrid
/// };
///
/// let preprocessing = PreprocessingGrid::new()
///     .with_scaler(ScalerGrid::new(vec![ScalerType::Standard, ScalerType::MinMax]));
///
/// let model = LinearRegressionGrid::new()
///     .with_learning_rates(vec![0.01, 0.1]);
///
/// let pipeline_grid = PipelineGrid::new(preprocessing, model);
///
/// // Total: 2 * 2 = 4 combinations
/// assert_eq!(pipeline_grid.n_combinations(), 4);
/// ```
#[derive(Clone, Debug, Default)]
pub struct PipelineGrid {
    /// Preprocessing parameter grid.
    pub preprocessing: PreprocessingGrid,
    /// Model parameter grid.
    pub model: LinearRegressionGrid,
}

impl PipelineGrid {
    /// Create a new pipeline grid.
    pub fn new(preprocessing: PreprocessingGrid, model: LinearRegressionGrid) -> Self {
        Self {
            preprocessing,
            model,
        }
    }

    /// Create with just model grid (no preprocessing search).
    pub fn model_only(model: LinearRegressionGrid) -> Self {
        Self {
            preprocessing: PreprocessingGrid::default(),
            model,
        }
    }

    /// Count total pipeline combinations.
    pub fn n_combinations(&self) -> usize {
        self.preprocessing.n_combinations() * self.model.n_combinations()
    }

    /// Iterate over all pipeline parameter combinations.
    pub fn iter(&self) -> PipelineGridIterator {
        let mut model_iter = self.model.iter();
        let current_model_params = model_iter.next();
        let done = current_model_params.is_none();

        PipelineGridIterator {
            preprocessing_grid: self.preprocessing.clone(),
            preprocessing_iter: self.preprocessing.iter(),
            model_iter,
            current_model_params,
            done,
        }
    }
}

/// Iterator over pipeline parameter combinations.
pub struct PipelineGridIterator {
    preprocessing_grid: PreprocessingGrid,
    preprocessing_iter: PreprocessingGridIterator,
    model_iter: ParamGridIterator,
    current_model_params: Option<ParamCombination>,
    done: bool,
}

impl Iterator for PipelineGridIterator {
    type Item = PipelineParams;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Get next preprocessing params
        if let Some(preproc) = self.preprocessing_iter.next() {
            let model = self.current_model_params.as_ref().unwrap();
            return Some(PipelineParams {
                preprocessing: preproc,
                model: ModelParams::from(model),
                mean_score: 0.0,
                std_score: 0.0,
            });
        }

        // Preprocessing exhausted, advance to next model params
        self.current_model_params = self.model_iter.next();
        if self.current_model_params.is_none() {
            self.done = true;
            return None;
        }

        // Reset preprocessing iterator for new model params
        self.preprocessing_iter = self.preprocessing_grid.iter();

        // Get first preprocessing combination with new model params
        if let Some(preproc) = self.preprocessing_iter.next() {
            let model = self.current_model_params.as_ref().unwrap();
            return Some(PipelineParams {
                preprocessing: preproc,
                model: ModelParams::from(model),
                mean_score: 0.0,
                std_score: 0.0,
            });
        }

        self.done = true;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_grid() {
        let grid = SGDGrid::new(vec![0.001, 0.01, 0.1]);
        assert_eq!(grid.len(), 3);
        assert!(!grid.is_empty());
    }

    #[test]
    fn test_sgd_grid_empty() {
        let grid = SGDGrid::new(vec![]);
        assert_eq!(grid.len(), 0);
        assert!(grid.is_empty());
    }

    #[test]
    fn test_regularizer_grid_none() {
        let grid = RegularizerGrid::None;
        assert_eq!(grid.len(), 1);
        assert_eq!(grid.lambdas(), vec![0.0]);
    }

    #[test]
    fn test_regularizer_grid_l2() {
        let grid = RegularizerGrid::L2 {
            lambdas: vec![0.0, 0.01, 0.1],
        };
        assert_eq!(grid.len(), 3);
        assert_eq!(grid.lambdas(), vec![0.0, 0.01, 0.1]);
    }

    #[test]
    fn test_trainer_grid() {
        let grid = TrainerGrid::new(vec![16, 32], vec![100, 500]);
        assert_eq!(grid.len(), 4); // 2 * 2
    }

    #[test]
    fn test_linear_regression_grid_n_combinations() {
        let grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.001, 0.01, 0.1])
            .with_lambdas(vec![0.0, 0.1])
            .with_trainer(TrainerGrid::new(vec![16, 32], vec![100, 500]));

        // 3 * 2 * 2 * 2 = 24
        assert_eq!(grid.n_combinations(), 24);
    }

    #[test]
    fn test_linear_regression_grid_n_combinations_with_poly() {
        let grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.001, 0.01])
            .with_polynomial(PolynomialGrid::new(vec![1, 2, 3]));

        // 2 * 1 * 1 * 3 = 6
        assert_eq!(grid.n_combinations(), 6);
    }

    #[test]
    fn test_param_grid_iterator() {
        let grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1, 0.01])
            .with_lambdas(vec![0.0, 1.0]);

        let combinations: Vec<_> = grid.iter().collect();
        assert_eq!(combinations.len(), 4);

        // Verify all combinations are present (count unique values)
        let lr_count = combinations
            .iter()
            .map(|c| c.learning_rate.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(lr_count, 2);

        let lambda_count = combinations
            .iter()
            .map(|c| c.lambda.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(lambda_count, 2);
    }

    #[test]
    fn test_param_grid_iterator_count() {
        let grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1, 0.01, 0.001])
            .with_lambdas(vec![0.0, 0.1, 1.0])
            .with_trainer(TrainerGrid::new(vec![16, 32], vec![100]));

        // 3 * 3 * 2 * 1 = 18
        let count = grid.iter().count();
        assert_eq!(count, 18);
        assert_eq!(count, grid.n_combinations());
    }

    #[test]
    fn test_param_grid_iterator_order() {
        let grid = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.1, 0.01])
            .with_lambdas(vec![0.0, 1.0]);

        let combinations: Vec<_> = grid.iter().collect();

        // First all lambdas for first lr, then all lambdas for second lr
        assert_eq!(combinations[0].learning_rate, 0.1);
        assert_eq!(combinations[0].lambda, 0.0);

        assert_eq!(combinations[1].learning_rate, 0.1);
        assert_eq!(combinations[1].lambda, 1.0);

        assert_eq!(combinations[2].learning_rate, 0.01);
        assert_eq!(combinations[2].lambda, 0.0);

        assert_eq!(combinations[3].learning_rate, 0.01);
        assert_eq!(combinations[3].lambda, 1.0);
    }

    #[test]
    fn test_param_grid_default() {
        let grid = LinearRegressionGrid::new();
        assert_eq!(grid.n_combinations(), 1);

        let combination = grid.iter().next().unwrap();
        assert_eq!(combination.learning_rate, 0.01);
        assert_eq!(combination.lambda, 0.0);
        assert_eq!(combination.batch_size, 32);
        assert_eq!(combination.max_epochs, 1000);
        assert_eq!(combination.poly_degree, 1);
    }

    // ========================================================================
    // Preprocessing Grid Tests
    // ========================================================================

    #[test]
    fn test_scaler_grid() {
        let grid = ScalerGrid::new(vec![ScalerType::Standard, ScalerType::MinMax]);
        assert_eq!(grid.len(), 2);
        assert!(!grid.is_empty());
    }

    #[test]
    fn test_imputer_grid() {
        let grid = ImputerGrid::new(vec![ImputeStrategy::Mean, ImputeStrategy::Median]);
        assert_eq!(grid.len(), 2);
        assert!(!grid.is_empty());
    }

    #[test]
    fn test_preprocessing_grid_n_combinations() {
        let grid = PreprocessingGrid::new()
            .with_scaler(ScalerGrid::new(vec![
                ScalerType::Standard,
                ScalerType::MinMax,
                ScalerType::Robust,
            ]))
            .with_polynomial(PolynomialGrid::new(vec![1, 2]));

        // 3 * 1 * 2 = 6
        assert_eq!(grid.n_combinations(), 6);
    }

    #[test]
    fn test_preprocessing_grid_iterator() {
        let grid = PreprocessingGrid::new().with_scaler(ScalerGrid::new(vec![
            ScalerType::Standard,
            ScalerType::MinMax,
        ]));

        let combinations: Vec<_> = grid.iter().collect();
        assert_eq!(combinations.len(), 2);

        // Check all scalers are present
        let scalers: std::collections::HashSet<_> = combinations.iter().map(|c| c.scaler).collect();
        assert_eq!(scalers.len(), 2);
    }

    #[test]
    fn test_preprocessing_grid_with_imputer() {
        let grid = PreprocessingGrid::new()
            .with_scaler(ScalerGrid::new(vec![ScalerType::Standard]))
            .with_imputer(ImputerGrid::new(vec![
                ImputeStrategy::Mean,
                ImputeStrategy::Median,
            ]));

        // 1 * 2 * 1 = 2
        assert_eq!(grid.n_combinations(), 2);

        let combinations: Vec<_> = grid.iter().collect();
        assert_eq!(combinations.len(), 2);
    }

    #[test]
    fn test_preprocessing_grid_default() {
        let grid = PreprocessingGrid::new();
        assert_eq!(grid.n_combinations(), 1); // Just the defaults

        let params = grid.iter().next().unwrap();
        assert_eq!(params.scaler, ScalerType::None);
        assert_eq!(params.imputer, None);
        assert_eq!(params.poly_degree, 1);
    }

    // ========================================================================
    // Pipeline Grid Tests
    // ========================================================================

    #[test]
    fn test_pipeline_grid_n_combinations() {
        let preprocessing = PreprocessingGrid::new().with_scaler(ScalerGrid::new(vec![
            ScalerType::Standard,
            ScalerType::MinMax,
        ]));

        let model = LinearRegressionGrid::new().with_learning_rates(vec![0.01, 0.1]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        // 2 * 2 = 4
        assert_eq!(pipeline_grid.n_combinations(), 4);
    }

    #[test]
    fn test_pipeline_grid_iterator() {
        let preprocessing = PreprocessingGrid::new().with_scaler(ScalerGrid::new(vec![
            ScalerType::Standard,
            ScalerType::MinMax,
        ]));

        let model = LinearRegressionGrid::new().with_learning_rates(vec![0.01]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        let combinations: Vec<_> = pipeline_grid.iter().collect();
        assert_eq!(combinations.len(), 2);

        // Verify scalers are present
        let scalers: std::collections::HashSet<_> = combinations
            .iter()
            .map(|c| c.preprocessing.scaler)
            .collect();
        assert_eq!(scalers.len(), 2);
    }

    #[test]
    fn test_pipeline_grid_model_only() {
        let model = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.01, 0.1])
            .with_lambdas(vec![0.0, 0.1]);

        let pipeline_grid = PipelineGrid::model_only(model);

        // 2 * 2 = 4 (no preprocessing search, so 1 preprocessing combination)
        assert_eq!(pipeline_grid.n_combinations(), 4);
    }

    #[test]
    fn test_pipeline_grid_iterator_multiple_model_params() {
        let preprocessing =
            PreprocessingGrid::new().with_scaler(ScalerGrid::new(vec![ScalerType::Standard]));

        let model = LinearRegressionGrid::new()
            .with_learning_rates(vec![0.01, 0.1])
            .with_lambdas(vec![0.0, 1.0]);

        let pipeline_grid = PipelineGrid::new(preprocessing, model);

        // 1 * 4 = 4
        assert_eq!(pipeline_grid.n_combinations(), 4);

        let combinations: Vec<_> = pipeline_grid.iter().collect();
        assert_eq!(combinations.len(), 4);

        // Verify all learning rates are present
        let lrs: std::collections::HashSet<_> = combinations
            .iter()
            .map(|c| c.model.learning_rate.to_bits())
            .collect();
        assert_eq!(lrs.len(), 2);

        // Verify all lambdas are present
        let lambdas: std::collections::HashSet<_> = combinations
            .iter()
            .map(|c| c.model.lambda.to_bits())
            .collect();
        assert_eq!(lambdas.len(), 2);
    }

    #[test]
    fn test_model_params_from_param_combination() {
        let combo = ParamCombination {
            learning_rate: 0.01,
            lambda: 0.1,
            batch_size: 32,
            max_epochs: 100,
            poly_degree: 2,
        };

        let params = ModelParams::from(&combo);
        assert_eq!(params.learning_rate, 0.01);
        assert_eq!(params.lambda, 0.1);
        assert_eq!(params.batch_size, 32);
        assert_eq!(params.max_epochs, 100);
    }

    #[test]
    fn test_scaler_grid_empty() {
        let grid = ScalerGrid::default();
        assert!(grid.is_empty());
        assert_eq!(grid.len(), 0);
    }

    #[test]
    fn test_imputer_grid_empty() {
        let grid = ImputerGrid::default();
        assert!(grid.is_empty());
        assert_eq!(grid.len(), 0);
    }

    #[test]
    fn test_pipeline_params_default() {
        let params = PipelineParams::default();
        assert_eq!(params.preprocessing.scaler, ScalerType::None);
        assert_eq!(params.preprocessing.poly_degree, 0); // usize default is 0
        assert_eq!(params.model.learning_rate, 0.0);
        assert_eq!(params.mean_score, 0.0);
        assert_eq!(params.std_score, 0.0);
    }

    #[test]
    fn test_model_params_default() {
        let params = ModelParams::default();
        assert_eq!(params.learning_rate, 0.0);
        assert_eq!(params.lambda, 0.0);
        assert_eq!(params.batch_size, 0);
        assert_eq!(params.max_epochs, 0);
    }

    #[test]
    fn test_preprocessing_params_default() {
        let params = PreprocessingParams::default();
        assert_eq!(params.scaler, ScalerType::None);
        assert_eq!(params.imputer, None);
        assert_eq!(params.poly_degree, 0); // usize default is 0
    }

    #[test]
    fn test_preprocessing_grid_empty_iterator() {
        let grid = PreprocessingGrid::default();
        assert_eq!(grid.n_combinations(), 1); // Returns default params

        let params = grid.iter().next().unwrap();
        assert_eq!(params.scaler, ScalerType::None);
    }

    #[test]
    fn test_scaler_type_variants() {
        // Test all variants for coverage
        let scalers = vec![
            ScalerType::Standard,
            ScalerType::MinMax,
            ScalerType::Robust,
            ScalerType::MaxAbs,
            ScalerType::None,
        ];

        for scaler in scalers {
            let grid = ScalerGrid::new(vec![scaler]);
            assert_eq!(grid.len(), 1);
        }
    }

    #[test]
    fn test_impute_strategy_variants() {
        // Test all variants for coverage
        let strategies = vec![
            ImputeStrategy::Mean,
            ImputeStrategy::Median,
            ImputeStrategy::MostFrequent,
            ImputeStrategy::Constant(0.0),
        ];

        for strategy in strategies {
            let grid = ImputerGrid::new(vec![strategy]);
            assert_eq!(grid.len(), 1);
        }
    }

    #[test]
    fn test_sgd_grid_default() {
        let grid = SGDGrid::default();
        assert_eq!(grid.learning_rates, vec![0.01]);
        assert_eq!(grid.len(), 1);
        assert!(!grid.is_empty());
    }

    #[test]
    fn test_sgd_grid_new_empty() {
        let grid = SGDGrid::new(vec![]);
        assert!(grid.is_empty());
        assert_eq!(grid.len(), 0);
    }

    // ========================================================================
    // MLP Grid Tests
    // ========================================================================

    #[test]
    fn test_mlp_architecture() {
        let arch = MLPArchitecture::single(16);
        assert_eq!(arch.hidden_layers, vec![16]);

        let arch = MLPArchitecture::double(32, 16);
        assert_eq!(arch.hidden_layers, vec![32, 16]);

        let arch = MLPArchitecture::new(vec![64, 32, 16]);
        assert_eq!(arch.hidden_layers, vec![64, 32, 16]);
    }

    #[test]
    fn test_mlp_activations() {
        let act = MLPActivations::relu_all(2);
        assert_eq!(act.activations.len(), 3); // 2 hidden + 1 output
        assert_eq!(act.activations[0], Activation::ReLU);
        assert_eq!(act.activations[1], Activation::ReLU);
        assert_eq!(act.activations[2], Activation::Identity);

        let act = MLPActivations::tanh_all(1);
        assert_eq!(act.activations.len(), 2);
        assert_eq!(act.activations[0], Activation::Tanh);
        assert_eq!(act.activations[1], Activation::Identity);
    }

    #[test]
    fn test_mlp_grid_n_combinations() {
        let grid = MLPGrid::new()
            .with_architectures(vec![
                MLPArchitecture::single(8),
                MLPArchitecture::single(16),
            ])
            .with_hidden_activations(vec![Activation::ReLU], 1)
            .with_learning_rates(vec![0.001, 0.01])
            .with_lambdas(vec![0.0]);

        // 2 arch * 1 act * 2 lr * 1 lambda * 1 trainer = 4
        assert_eq!(grid.n_combinations(), 4);
    }

    #[test]
    fn test_mlp_grid_iterator() {
        let grid = MLPGrid::new()
            .with_architectures(vec![MLPArchitecture::single(8)])
            .with_hidden_activations(vec![Activation::ReLU, Activation::Tanh], 1)
            .with_learning_rates(vec![0.01]);

        let combinations: Vec<_> = grid.iter().collect();
        // 1 arch * 2 act * 1 lr * 1 lambda * 1 trainer = 2
        assert_eq!(combinations.len(), 2);

        // Check that both activation configs are present by counting unique strings
        let activation_names: std::collections::HashSet<_> = combinations
            .iter()
            .map(|c| format!("{:?}", c.activations.activations[0]))
            .collect();
        assert_eq!(activation_names.len(), 2);
    }

    #[test]
    fn test_mlp_grid_compatible_activations() {
        // Test that incompatible activations are skipped
        let grid = MLPGrid::new()
            .with_architectures(vec![
                MLPArchitecture::single(8),     // 2 layers (1 hidden + 1 output)
                MLPArchitecture::double(16, 8), // 3 layers (2 hidden + 1 output)
            ])
            .with_activations(vec![
                MLPActivations::relu_all(1), // For single hidden layer
                MLPActivations::relu_all(2), // For double hidden layer
            ])
            .with_learning_rates(vec![0.01]);

        let combinations: Vec<_> = grid.iter().collect();
        // Each architecture should match with exactly one activation config
        assert_eq!(combinations.len(), 2);

        // Verify first combination has correct layer count
        assert_eq!(combinations[0].architecture.hidden_layers.len(), 1);
        assert_eq!(combinations[0].activations.activations.len(), 2);

        // Verify second combination
        assert_eq!(combinations[1].architecture.hidden_layers.len(), 2);
        assert_eq!(combinations[1].activations.activations.len(), 3);
    }

    #[test]
    fn test_mlp_grid_default() {
        let grid = MLPGrid::new();
        // Default grid has 1 combination (default architecture + default activation)
        assert_eq!(grid.n_combinations(), 1);
    }

    #[test]
    fn test_mlp_param_combination() {
        let grid = MLPGrid::new()
            .with_architectures(vec![MLPArchitecture::single(8)])
            .with_hidden_activations(vec![Activation::ReLU], 1)
            .with_learning_rates(vec![0.01])
            .with_lambdas(vec![0.1])
            .with_trainer(TrainerGrid::new(vec![32], vec![100]));

        let combo = grid.iter().next().unwrap();
        assert_eq!(combo.architecture.hidden_layers, vec![8]);
        assert_eq!(combo.activations.activations.len(), 2);
        assert_eq!(combo.learning_rate, 0.01);
        assert_eq!(combo.lambda, 0.1);
        assert_eq!(combo.batch_size, 32);
        assert_eq!(combo.max_epochs, 100);
    }
}
