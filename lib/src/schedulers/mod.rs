//! Learning rate schedulers for adaptive learning rate adjustment.
//!
//! This module provides various learning rate scheduling strategies that
//! adjust the learning rate during training based on epoch number or metrics.
//!
//! # Available Schedulers
//!
//! - [`StepLR`]: Decay learning rate by a factor every N epochs
//! - [`ExponentialLR`]: Decay learning rate exponentially each epoch
//! - [`CosineAnnealingLR`]: Follow a cosine curve from initial to minimum LR
//! - [`ReduceLROnPlateau`]: Reduce LR when a metric has stopped improving
//!
//! # Example
//!
//! ```rust,ignore
//! use machinelearne_rs::schedulers::StepLR;
//!
//! let scheduler = StepLR::new(0.01, 30, 0.1); // lr=0.01, step_size=30, gamma=0.1
//!
//! for epoch in 0..100 {
//!     let lr = scheduler.step(epoch, &metrics);
//!     // Use lr for this epoch
//! }
//! ```

use std::collections::HashMap;

/// Trait for learning rate schedulers.
///
/// Learning rate schedulers adjust the learning rate during training
/// to improve convergence and final model performance.
pub trait LRScheduler {
    /// Returns the learning rate for the given epoch.
    ///
    /// # Arguments
    /// * `epoch` - Current epoch number (0-indexed)
    /// * `metrics` - Current metrics (used by some schedulers like ReduceLROnPlateau)
    ///
    /// # Returns
    /// The learning rate to use for this epoch.
    fn step(&mut self, epoch: usize, metrics: &HashMap<String, f64>) -> f64;

    /// Returns the current (most recently computed) learning rate.
    fn current_lr(&self) -> f64;
}

/// Step learning rate scheduler.
///
/// Decays the learning rate by `gamma` every `step_size` epochs.
///
/// # Formula
/// ```text
/// lr = initial_lr * gamma^(epoch / step_size)
/// ```
///
/// # Example
///
/// ```
/// use machinelearne_rs::schedulers::{LRScheduler, StepLR};
/// use std::collections::HashMap;
///
/// let mut scheduler = StepLR::new(0.1, 30, 0.1);
/// let metrics = HashMap::new();
///
/// // Epochs 0-29: lr = 0.1
/// // Epochs 30-59: lr = 0.01
/// // Epochs 60-89: lr = 0.001
/// assert!((scheduler.step(0, &metrics) - 0.1).abs() < 1e-10);
/// assert!((scheduler.step(30, &metrics) - 0.01).abs() < 1e-10);
/// assert!((scheduler.step(60, &metrics) - 0.001).abs() < 1e-10);
/// ```
#[derive(Clone, Debug)]
pub struct StepLR {
    initial_lr: f64,
    step_size: usize,
    gamma: f64,
    current_lr: f64,
}

impl StepLR {
    /// Creates a new StepLR scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Initial learning rate
    /// * `step_size` - Number of epochs between each decay
    /// * `gamma` - Multiplicative factor for learning rate decay
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_lr: initial_lr,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, epoch: usize, _metrics: &HashMap<String, f64>) -> f64 {
        let num_decays = epoch / self.step_size;
        self.current_lr = self.initial_lr * self.gamma.powi(num_decays as i32);
        self.current_lr
    }

    fn current_lr(&self) -> f64 {
        self.current_lr
    }
}

/// Exponential learning rate scheduler.
///
/// Decays the learning rate by `gamma` every epoch.
///
/// # Formula
/// ```text
/// lr = initial_lr * gamma^epoch
/// ```
///
/// # Example
///
/// ```
/// use machinelearne_rs::schedulers::{LRScheduler, ExponentialLR};
/// use std::collections::HashMap;
///
/// let mut scheduler = ExponentialLR::new(0.1, 0.9);
/// let metrics = HashMap::new();
///
/// assert!((scheduler.step(0, &metrics) - 0.1).abs() < 1e-10);
/// assert!((scheduler.step(1, &metrics) - 0.09).abs() < 1e-10);
/// assert!((scheduler.step(2, &metrics) - 0.081).abs() < 1e-10);
/// ```
#[derive(Clone, Debug)]
pub struct ExponentialLR {
    initial_lr: f64,
    gamma: f64,
    current_lr: f64,
}

impl ExponentialLR {
    /// Creates a new ExponentialLR scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Initial learning rate
    /// * `gamma` - Multiplicative factor for learning rate decay per epoch
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self {
            initial_lr,
            gamma,
            current_lr: initial_lr,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self, epoch: usize, _metrics: &HashMap<String, f64>) -> f64 {
        self.current_lr = self.initial_lr * self.gamma.powi(epoch as i32);
        self.current_lr
    }

    fn current_lr(&self) -> f64 {
        self.current_lr
    }
}

/// Cosine annealing learning rate scheduler.
///
/// Follows a cosine curve from `initial_lr` down to `eta_min` over `T_max` epochs,
/// then restarts.
///
/// # Formula
/// ```text
/// lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
/// ```
///
/// # Example
///
/// ```
/// use machinelearne_rs::schedulers::{LRScheduler, CosineAnnealingLR};
/// use std::collections::HashMap;
///
/// let mut scheduler = CosineAnnealingLR::new(0.1, 100, 0.0);
/// let metrics = HashMap::new();
///
/// // Start at initial_lr
/// assert!((scheduler.step(0, &metrics) - 0.1).abs() < 1e-10);
/// // At T_max/2, should be at eta_min
/// assert!((scheduler.step(50, &metrics) - 0.0).abs() < 1e-10);
/// // At T_max, should be back at initial_lr
/// assert!((scheduler.step(100, &metrics) - 0.1).abs() < 1e-10);
/// ```
#[derive(Clone, Debug)]
pub struct CosineAnnealingLR {
    initial_lr: f64,
    t_max: usize,
    eta_min: f64,
    current_lr: f64,
}

impl CosineAnnealingLR {
    /// Creates a new CosineAnnealingLR scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Initial (maximum) learning rate
    /// * `t_max` - Period of the cosine cycle (in epochs)
    /// * `eta_min` - Minimum learning rate
    pub fn new(initial_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self {
            initial_lr,
            t_max,
            eta_min,
            current_lr: initial_lr,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, epoch: usize, _metrics: &HashMap<String, f64>) -> f64 {
        use std::f64::consts::PI;
        let progress = (epoch % self.t_max) as f64 / self.t_max as f64;
        self.current_lr =
            self.eta_min + (self.initial_lr - self.eta_min) * (1.0 + (PI * progress).cos()) / 2.0;
        self.current_lr
    }

    fn current_lr(&self) -> f64 {
        self.current_lr
    }
}

/// Reduce learning rate on plateau scheduler.
///
/// Reduces the learning rate when a metric has stopped improving.
/// This is useful for fine-tuning when training plateaus.
///
/// # Example
///
/// ```rust,ignore
/// use machinelearne_rs::schedulers::{LRScheduler, ReduceLROnPlateau};
/// use std::collections::HashMap;
///
/// let mut scheduler = ReduceLROnPlateau::new(0.1, 0.5, 5, 0.001, "val_loss", true);
///
/// // If val_loss doesn't improve by 0.001 for 5 epochs, lr is multiplied by 0.5
/// ```
#[derive(Clone, Debug)]
pub struct ReduceLROnPlateau {
    #[allow(dead_code)]
    initial_lr: f64,
    factor: f64,
    patience: usize,
    min_delta: f64,
    metric_name: String,
    mode_max: bool, // true = maximize, false = minimize
    current_lr: f64,
    best_metric: f64,
    epochs_without_improvement: usize,
}

impl ReduceLROnPlateau {
    /// Creates a new ReduceLROnPlateau scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Initial learning rate
    /// * `factor` - Factor by which to reduce the learning rate
    /// * `patience` - Number of epochs with no improvement before reducing LR
    /// * `min_delta` - Minimum change to qualify as an improvement
    /// * `metric_name` - Name of the metric to monitor (e.g., "val_loss")
    /// * `mode_max` - If true, higher metric is better; if false, lower is better
    pub fn new(
        initial_lr: f64,
        factor: f64,
        patience: usize,
        min_delta: f64,
        metric_name: &str,
        mode_max: bool,
    ) -> Self {
        Self {
            initial_lr,
            factor,
            patience,
            min_delta,
            metric_name: metric_name.to_string(),
            mode_max,
            current_lr: initial_lr,
            best_metric: if mode_max {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            },
            epochs_without_improvement: 0,
        }
    }

    /// Creates a scheduler that monitors for decreasing metrics (like loss).
    pub fn for_minimize(
        initial_lr: f64,
        factor: f64,
        patience: usize,
        min_delta: f64,
        metric_name: &str,
    ) -> Self {
        Self::new(initial_lr, factor, patience, min_delta, metric_name, false)
    }

    /// Creates a scheduler that monitors for increasing metrics (like accuracy).
    pub fn for_maximize(
        initial_lr: f64,
        factor: f64,
        patience: usize,
        min_delta: f64,
        metric_name: &str,
    ) -> Self {
        Self::new(initial_lr, factor, patience, min_delta, metric_name, true)
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn step(&mut self, _epoch: usize, metrics: &HashMap<String, f64>) -> f64 {
        // Get the monitored metric
        if let Some(&metric_value) = metrics.get(&self.metric_name) {
            let improved = if self.mode_max {
                metric_value > self.best_metric + self.min_delta
            } else {
                metric_value < self.best_metric - self.min_delta
            };

            if improved {
                self.best_metric = metric_value;
                self.epochs_without_improvement = 0;
            } else {
                self.epochs_without_improvement += 1;
            }

            // Reduce LR if patience exceeded
            if self.epochs_without_improvement >= self.patience {
                self.current_lr *= self.factor;
                self.epochs_without_improvement = 0; // Reset counter after reduction
            }
        }

        self.current_lr
    }

    fn current_lr(&self) -> f64 {
        self.current_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_step_lr() {
        let mut scheduler = StepLR::new(0.1, 30, 0.1);
        let metrics = HashMap::new();

        // Check initial behavior
        assert!((scheduler.step(0, &metrics) - 0.1).abs() < 1e-10);
        assert!((scheduler.step(29, &metrics) - 0.1).abs() < 1e-10);

        // First step
        assert!((scheduler.step(30, &metrics) - 0.01).abs() < 1e-10);
        assert!((scheduler.step(59, &metrics) - 0.01).abs() < 1e-10);

        // Second step
        assert!((scheduler.step(60, &metrics) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_lr() {
        let mut scheduler = ExponentialLR::new(0.1, 0.9);
        let metrics = HashMap::new();

        assert!((scheduler.step(0, &metrics) - 0.1).abs() < 1e-10);
        assert!((scheduler.step(1, &metrics) - 0.09).abs() < 1e-10);
        assert!((scheduler.step(2, &metrics) - 0.081).abs() < 1e-10);
        assert!((scheduler.step(10, &metrics) - 0.1 * 0.9_f64.powi(10)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let mut scheduler = CosineAnnealingLR::new(0.1, 100, 0.0);
        let metrics = HashMap::new();

        // Start at initial_lr
        assert!((scheduler.step(0, &metrics) - 0.1).abs() < 1e-10);

        // At T_max/2, cos(pi/2)=0, so lr = initial_lr * (1+0)/2 = initial_lr/2
        let mid = scheduler.step(50, &metrics);
        assert!((mid - 0.05).abs() < 1e-10);

        // Near end of cycle (epoch 99), should be very close to 0
        let near_min = scheduler.step(99, &metrics);
        assert!(near_min < 0.001);

        // At T_max, cycles back to initial_lr
        let cycled = scheduler.step(100, &metrics);
        assert!((cycled - 0.1).abs() < 1e-10);

        // At T_max*1.5
        let rising = scheduler.step(150, &metrics);
        assert!((rising - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing_with_eta_min() {
        let mut scheduler = CosineAnnealingLR::new(0.1, 100, 0.001);
        let metrics = HashMap::new();

        assert!((scheduler.step(0, &metrics) - 0.1).abs() < 1e-10);

        // At T_max/2
        let mid = scheduler.step(50, &metrics);
        // lr = eta_min + (initial_lr - eta_min) * (1 + 0) / 2
        // lr = 0.001 + (0.1 - 0.001) * 0.5 = 0.001 + 0.0495 = 0.0505
        assert!((mid - 0.0505).abs() < 1e-10);

        // Near end of cycle, should be very close to eta_min
        let near_min = scheduler.step(99, &metrics);
        assert!(near_min < 0.002);
    }

    #[test]
    fn test_reduce_lr_on_plateau_minimize() {
        let mut scheduler = ReduceLROnPlateau::for_minimize(0.1, 0.5, 3, 0.01, "val_loss");
        let mut metrics = HashMap::new();

        // Initial LR with first metric
        metrics.insert("val_loss".to_string(), 1.0);
        let lr = scheduler.step(0, &metrics);
        assert!((lr - 0.1).abs() < 1e-10);

        // Improving - resets counter
        metrics.insert("val_loss".to_string(), 0.5);
        scheduler.step(1, &metrics);
        assert!((scheduler.current_lr() - 0.1).abs() < 1e-10);

        // No improvement for 3 epochs
        metrics.insert("val_loss".to_string(), 0.5); // Same as best, not improvement
        scheduler.step(2, &metrics);
        scheduler.step(3, &metrics);
        scheduler.step(4, &metrics);
        // After 3 epochs without improvement, next step should reduce

        metrics.insert("val_loss".to_string(), 0.5);
        let lr = scheduler.step(5, &metrics);
        assert!((lr - 0.05).abs() < 1e-10, "Expected lr=0.05, got {}", lr);
    }

    #[test]
    fn test_reduce_lr_on_plateau_maximize() {
        let mut scheduler = ReduceLROnPlateau::for_maximize(0.1, 0.5, 2, 0.01, "val_accuracy");
        let mut metrics = HashMap::new();

        // Initial
        metrics.insert("val_accuracy".to_string(), 0.8);
        scheduler.step(0, &metrics);
        assert!((scheduler.current_lr() - 0.1).abs() < 1e-10);

        // Improving
        metrics.insert("val_accuracy".to_string(), 0.85);
        scheduler.step(0, &metrics);
        assert!((scheduler.current_lr() - 0.1).abs() < 1e-10);

        // No improvement for 2 epochs
        metrics.insert("val_accuracy".to_string(), 0.84); // Not > 0.85 + 0.01
        scheduler.step(0, &metrics);
        scheduler.step(0, &metrics);

        // Next step should reduce
        metrics.insert("val_accuracy".to_string(), 0.84);
        let lr = scheduler.step(0, &metrics);
        assert!((lr - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_scheduler_current_lr() {
        let mut scheduler = StepLR::new(0.1, 10, 0.5);
        let metrics = HashMap::new();

        scheduler.step(5, &metrics);
        assert!((scheduler.current_lr() - 0.1).abs() < 1e-10);

        scheduler.step(15, &metrics);
        assert!((scheduler.current_lr() - 0.05).abs() < 1e-10);
    }
}
