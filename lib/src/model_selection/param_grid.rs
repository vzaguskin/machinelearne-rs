//! Type-safe parameter grids for hyperparameter search.
//!
//! This module provides builder-pattern types for defining parameter grids
//! to search over during hyperparameter tuning.

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
}
