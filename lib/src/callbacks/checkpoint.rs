//! Checkpoint callback for saving model state during training.
//!
//! This module provides checkpointing functionality to save model parameters
//! and training metadata at regular intervals or when metrics improve.

use super::{Callback, TrainingState};
use crate::backend::Backend;
use crate::model::TrainableModel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Trait for models that can be checkpointed.
///
/// Models implement this trait to provide a way to serialize their parameters
/// for checkpoint saving.
pub trait Checkpointable<B: Backend>: TrainableModel<B> {
    /// Serializes the model's current parameters to bytes.
    fn serialize_params(&self) -> Vec<u8>;
}

/// Metadata stored alongside checkpoint files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Epoch number when checkpoint was saved.
    pub epoch: usize,
    /// Training loss at checkpoint time.
    pub loss: f64,
    /// Learning rate at checkpoint time.
    pub learning_rate: f64,
    /// Custom metrics at checkpoint time.
    pub metrics: HashMap<String, f64>,
    /// Unix timestamp when checkpoint was saved.
    pub timestamp: u64,
    /// Checkpoint file name (relative to checkpoint directory).
    pub checkpoint_file: String,
}

/// Information about a saved checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub path: PathBuf,
    pub metadata: CheckpointMetadata,
}

/// Strategy for when to save checkpoints.
#[derive(Debug, Clone)]
pub enum SaveStrategy {
    /// Save every N epochs.
    EveryNEpochs(usize),
    /// Save when a metric improves (lower is better).
    BestMetric {
        metric_name: String,
        mode: MetricMode,
    },
    /// Save both periodically and on metric improvement.
    Combined {
        every_n: usize,
        metric_name: String,
        mode: MetricMode,
    },
}

/// Mode for metric comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricMode {
    /// Lower metric values are better (e.g., loss).
    Minimize,
    /// Higher metric values are better (e.g., accuracy).
    Maximize,
}

/// Callback for saving model checkpoints during training.
///
/// # Features
/// - Save checkpoints periodically (every N epochs)
/// - Save checkpoints when metrics improve
/// - Keep only the best N checkpoints (older ones are deleted)
/// - Stores metadata in JSON format alongside model files
///
/// # Example
///
/// ```rust,ignore
/// use machinelearne_rs::callbacks::{CheckpointCallback, SaveStrategy, MetricMode};
///
/// // Save best 3 checkpoints based on validation loss
/// let checkpoint = CheckpointCallback::new(
///     "checkpoints",
///     SaveStrategy::BestMetric {
///         metric_name: "val_loss".to_string(),
///         mode: MetricMode::Minimize,
///     },
///     3,  // keep best 3
/// );
/// ```
pub struct CheckpointCallback<B: Backend, M: Checkpointable<B>> {
    /// Directory to store checkpoints.
    checkpoint_dir: PathBuf,
    /// Strategy for when to save.
    strategy: SaveStrategy,
    /// Maximum number of checkpoints to keep.
    keep_best_n: usize,
    /// Prefix for checkpoint file names.
    prefix: String,
    /// List of saved checkpoints with their metrics.
    saved_checkpoints: Vec<CheckpointInfo>,
    /// Best metric value seen so far.
    best_metric: Option<f64>,
    /// Marker for backend type.
    _phantom: std::marker::PhantomData<(B, M)>,
}

impl<B: Backend, M: Checkpointable<B>> CheckpointCallback<B, M> {
    /// Creates a new checkpoint callback.
    ///
    /// # Arguments
    /// * `checkpoint_dir` - Directory to store checkpoints
    /// * `strategy` - When to save checkpoints
    /// * `keep_best_n` - Maximum number of checkpoints to keep (0 = keep all)
    pub fn new<P: AsRef<Path>>(
        checkpoint_dir: P,
        strategy: SaveStrategy,
        keep_best_n: usize,
    ) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.as_ref().to_path_buf(),
            strategy,
            keep_best_n,
            prefix: "checkpoint".to_string(),
            saved_checkpoints: Vec::new(),
            best_metric: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Sets a custom prefix for checkpoint file names.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Creates the checkpoint directory if it doesn't exist.
    fn ensure_dir(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.checkpoint_dir)
    }

    /// Generates a checkpoint file name for the given epoch.
    fn generate_filename(&self, epoch: usize) -> String {
        format!("{}_{:05}.bin", self.prefix, epoch)
    }

    /// Saves a checkpoint with the given state.
    fn save_checkpoint(
        &mut self,
        model: &M,
        epoch: usize,
        loss: f64,
        learning_rate: f64,
        metrics: &HashMap<String, f64>,
    ) -> std::io::Result<()> {
        self.ensure_dir()?;

        let filename = self.generate_filename(epoch);
        let checkpoint_path = self.checkpoint_dir.join(&filename);
        let metadata_path = self
            .checkpoint_dir
            .join(format!("{}_{:05}.json", self.prefix, epoch));

        // Save model parameters
        let params_bytes = model.serialize_params();
        fs::write(&checkpoint_path, params_bytes)?;

        // Create and save metadata
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let metadata = CheckpointMetadata {
            epoch,
            loss,
            learning_rate,
            metrics: metrics.clone(),
            timestamp,
            checkpoint_file: filename.clone(),
        };

        let metadata_file = File::create(&metadata_path)?;
        let writer = BufWriter::new(metadata_file);
        serde_json::to_writer_pretty(writer, &metadata).map_err(std::io::Error::other)?;

        // Track the checkpoint
        self.saved_checkpoints.push(CheckpointInfo {
            path: checkpoint_path,
            metadata,
        });

        // Cleanup old checkpoints if needed
        if self.keep_best_n > 0 {
            self.cleanup_old_checkpoints()?;
        }

        Ok(())
    }

    /// Removes old checkpoints, keeping only the best N.
    fn cleanup_old_checkpoints(&mut self) -> std::io::Result<()> {
        if self.saved_checkpoints.len() <= self.keep_best_n {
            return Ok(());
        }

        // Sort by the metric we're optimizing (or by epoch if no metric)
        let metric_name = match &self.strategy {
            SaveStrategy::BestMetric { metric_name, .. } => Some(metric_name.clone()),
            SaveStrategy::Combined { metric_name, .. } => Some(metric_name.clone()),
            SaveStrategy::EveryNEpochs(_) => None,
        };

        if let Some(name) = metric_name {
            let mode = match &self.strategy {
                SaveStrategy::BestMetric { mode, .. } => *mode,
                SaveStrategy::Combined { mode, .. } => *mode,
                _ => MetricMode::Minimize,
            };

            // Sort checkpoints by metric value
            self.saved_checkpoints.sort_by(|a, b| {
                let val_a = a.metadata.metrics.get(&name).unwrap_or(&f64::INFINITY);
                let val_b = b.metadata.metrics.get(&name).unwrap_or(&f64::INFINITY);
                match mode {
                    MetricMode::Minimize => val_a
                        .partial_cmp(val_b)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    MetricMode::Maximize => val_b
                        .partial_cmp(val_a)
                        .unwrap_or(std::cmp::Ordering::Equal),
                }
            });
        } else {
            // Sort by loss (lower is better)
            self.saved_checkpoints.sort_by(|a, b| {
                a.metadata
                    .loss
                    .partial_cmp(&b.metadata.loss)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Remove checkpoints beyond keep_best_n
        while self.saved_checkpoints.len() > self.keep_best_n {
            let removed = self.saved_checkpoints.pop(); // Remove worst (last after sorting)
            if let Some(info) = removed {
                // Remove the checkpoint file and metadata file
                let _ = fs::remove_file(&info.path);
                if let Some(metadata_path) = info.path.to_str() {
                    let json_path = metadata_path.replace(".bin", ".json");
                    let _ = fs::remove_file(&json_path);
                }
            }
        }

        Ok(())
    }

    /// Checks if a checkpoint should be saved based on the strategy.
    fn should_save(&self, epoch: usize, metrics: &HashMap<String, f64>) -> bool {
        match &self.strategy {
            SaveStrategy::EveryNEpochs(n) => (epoch + 1).is_multiple_of(*n),
            SaveStrategy::BestMetric { metric_name, mode } => {
                if let Some(&value) = metrics.get(metric_name) {
                    match (self.best_metric, mode) {
                        (None, _) => true,
                        (Some(best), MetricMode::Minimize) => value < best,
                        (Some(best), MetricMode::Maximize) => value > best,
                    }
                } else {
                    false
                }
            }
            SaveStrategy::Combined {
                every_n,
                metric_name,
                mode,
            } => {
                // Check periodic
                let periodic = (epoch + 1).is_multiple_of(*every_n);

                // Check metric improvement
                let metric_improved = if let Some(&value) = metrics.get(metric_name) {
                    match (self.best_metric, mode) {
                        (None, _) => true,
                        (Some(best), MetricMode::Minimize) => value < best,
                        (Some(best), MetricMode::Maximize) => value > best,
                    }
                } else {
                    false
                };

                periodic || metric_improved
            }
        }
    }

    /// Updates the best metric value.
    fn update_best_metric(&mut self, metrics: &HashMap<String, f64>) {
        let metric_name = match &self.strategy {
            SaveStrategy::BestMetric { metric_name, .. } => Some(metric_name),
            SaveStrategy::Combined { metric_name, .. } => Some(metric_name),
            SaveStrategy::EveryNEpochs(_) => None,
        };

        if let Some(name) = metric_name {
            if let Some(&value) = metrics.get(name) {
                let mode = match &self.strategy {
                    SaveStrategy::BestMetric { mode, .. } => *mode,
                    SaveStrategy::Combined { mode, .. } => *mode,
                    _ => MetricMode::Minimize,
                };

                let is_better = match (self.best_metric, mode) {
                    (None, _) => true,
                    (Some(best), MetricMode::Minimize) => value < best,
                    (Some(best), MetricMode::Maximize) => value > best,
                };

                if is_better {
                    self.best_metric = Some(value);
                }
            }
        }
    }
}

impl<B: Backend, M: Checkpointable<B>> Callback<B, M> for CheckpointCallback<B, M> {
    fn on_train_start(&mut self, _state: &TrainingState<B, M>) {
        // Ensure checkpoint directory exists at training start
        let _ = self.ensure_dir();
    }

    fn on_epoch_end(&mut self, state: &mut TrainingState<B, M>) {
        // Check if we should save
        if self.should_save(state.epoch, &state.metrics) {
            self.update_best_metric(&state.metrics);

            if let Err(e) = self.save_checkpoint(
                state.model,
                state.epoch,
                state.loss,
                state.learning_rate,
                &state.metrics,
            ) {
                eprintln!("Warning: Failed to save checkpoint: {}", e);
            }
        } else {
            self.update_best_metric(&state.metrics);
        }
    }

    fn on_train_end(&mut self, _state: &TrainingState<B, M>) {
        // Final cleanup
        if self.keep_best_n > 0 {
            let _ = self.cleanup_old_checkpoints();
        }
    }
}

/// Utility functions for checkpoint management.
pub mod utils {
    use super::*;
    use std::fs;

    /// Finds the latest checkpoint in a directory.
    pub fn find_latest_checkpoint<P: AsRef<Path>>(
        dir: P,
    ) -> std::io::Result<Option<CheckpointInfo>> {
        let dir = dir.as_ref();
        if !dir.exists() {
            return Ok(None);
        }

        let mut latest: Option<CheckpointInfo> = None;

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "json") {
                if let Ok(file) = File::open(&path) {
                    let reader = BufReader::new(file);
                    if let Ok(metadata) = serde_json::from_reader::<_, CheckpointMetadata>(reader) {
                        let bin_path = path.with_extension("bin");
                        if bin_path.exists() {
                            let info = CheckpointInfo {
                                path: bin_path,
                                metadata,
                            };

                            let is_later = match &latest {
                                None => true,
                                Some(current) => info.metadata.epoch > current.metadata.epoch,
                            };

                            if is_later {
                                latest = Some(info);
                            }
                        }
                    }
                }
            }
        }

        Ok(latest)
    }

    /// Finds the best checkpoint based on a metric.
    pub fn find_best_checkpoint<P: AsRef<Path>>(
        dir: P,
        metric_name: &str,
        mode: MetricMode,
    ) -> std::io::Result<Option<CheckpointInfo>> {
        let dir = dir.as_ref();
        if !dir.exists() {
            return Ok(None);
        }

        let mut best: Option<CheckpointInfo> = None;

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "json") {
                if let Ok(file) = File::open(&path) {
                    let reader = BufReader::new(file);
                    if let Ok(metadata) = serde_json::from_reader::<_, CheckpointMetadata>(reader) {
                        let bin_path = path.with_extension("bin");
                        if bin_path.exists() {
                            if let Some(&metric_value) = metadata.metrics.get(metric_name) {
                                let info = CheckpointInfo {
                                    path: bin_path,
                                    metadata,
                                };

                                let is_better = match &best {
                                    None => true,
                                    Some(current) => {
                                        let current_value = current
                                            .metadata
                                            .metrics
                                            .get(metric_name)
                                            .unwrap_or(&f64::INFINITY);
                                        match mode {
                                            MetricMode::Minimize => metric_value < *current_value,
                                            MetricMode::Maximize => metric_value > *current_value,
                                        }
                                    }
                                };

                                if is_better {
                                    best = Some(info);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(best)
    }

    /// Lists all checkpoints in a directory.
    pub fn list_checkpoints<P: AsRef<Path>>(dir: P) -> std::io::Result<Vec<CheckpointInfo>> {
        let dir = dir.as_ref();
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "json") {
                if let Ok(file) = File::open(&path) {
                    let reader = BufReader::new(file);
                    if let Ok(metadata) = serde_json::from_reader::<_, CheckpointMetadata>(reader) {
                        let bin_path = path.with_extension("bin");
                        if bin_path.exists() {
                            checkpoints.push(CheckpointInfo {
                                path: bin_path,
                                metadata,
                            });
                        }
                    }
                }
            }
        }

        // Sort by epoch
        checkpoints.sort_by_key(|c| c.metadata.epoch);

        Ok(checkpoints)
    }

    /// Loads checkpoint metadata from a JSON file.
    pub fn load_metadata<P: AsRef<Path>>(path: P) -> std::io::Result<CheckpointMetadata> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(std::io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::scalar::{Scalar, ScalarOps};
    use crate::backend::{CpuBackend, Tensor1D};
    use crate::model::linear::{LinearModel, Unfitted};
    use tempfile::tempdir;

    // Implement Checkpointable for LinearModel
    impl<B: Backend> Checkpointable<B> for LinearModel<B, Unfitted>
    where
        Tensor1D<B>: Clone,
        Scalar<B>: Clone,
    {
        fn serialize_params(&self) -> Vec<u8> {
            // Convert to serializable representation and serialize
            #[cfg(feature = "serde")]
            {
                use crate::model::linear::SerializableLinearParams;
                let repr: SerializableLinearParams = self.params().into();
                bincode::serialize(&repr).unwrap_or_default()
            }
            #[cfg(not(feature = "serde"))]
            {
                Vec::new()
            }
        }
    }

    #[test]
    fn test_checkpoint_metadata_serialization() {
        let mut metrics = HashMap::new();
        metrics.insert("val_loss".to_string(), 0.5);
        metrics.insert("val_accuracy".to_string(), 0.95);

        let metadata = CheckpointMetadata {
            epoch: 10,
            loss: 0.25,
            learning_rate: 0.001,
            metrics,
            timestamp: 1234567890,
            checkpoint_file: "checkpoint_00010.bin".to_string(),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: CheckpointMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.epoch, 10);
        assert!((deserialized.loss - 0.25).abs() < 1e-10);
        assert!((deserialized.learning_rate - 0.001).abs() < 1e-10);
        assert_eq!(deserialized.metrics.len(), 2);
    }

    #[test]
    fn test_save_strategy_every_n() {
        let strategy = SaveStrategy::EveryNEpochs(5);
        let callback: CheckpointCallback<CpuBackend, LinearModel<CpuBackend, Unfitted>> =
            CheckpointCallback::new("/tmp/test", strategy, 0);

        let metrics = HashMap::new();

        assert!(!callback.should_save(0, &metrics)); // epoch 1
        assert!(!callback.should_save(1, &metrics)); // epoch 2
        assert!(!callback.should_save(2, &metrics)); // epoch 3
        assert!(!callback.should_save(3, &metrics)); // epoch 4
        assert!(callback.should_save(4, &metrics)); // epoch 5
        assert!(!callback.should_save(5, &metrics)); // epoch 6
    }

    #[test]
    fn test_save_strategy_best_metric() {
        let strategy = SaveStrategy::BestMetric {
            metric_name: "val_loss".to_string(),
            mode: MetricMode::Minimize,
        };
        let mut callback: CheckpointCallback<CpuBackend, LinearModel<CpuBackend, Unfitted>> =
            CheckpointCallback::new("/tmp/test", strategy.clone(), 0);

        let mut metrics = HashMap::new();

        // First value - should save
        metrics.insert("val_loss".to_string(), 1.0);
        assert!(callback.should_save(0, &metrics));
        callback.update_best_metric(&metrics);

        // Worse value - should not save
        metrics.insert("val_loss".to_string(), 1.5);
        assert!(!callback.should_save(1, &metrics));

        // Better value - should save
        metrics.insert("val_loss".to_string(), 0.5);
        assert!(callback.should_save(2, &metrics));
    }

    #[test]
    fn test_find_latest_checkpoint() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path();

        // Create test checkpoints
        for epoch in [1, 5, 3] {
            let metadata = CheckpointMetadata {
                epoch,
                loss: 0.1 * epoch as f64,
                learning_rate: 0.01,
                metrics: HashMap::new(),
                timestamp: 1000 * epoch as u64,
                checkpoint_file: format!("checkpoint_{:05}.bin", epoch),
            };

            let json_path = dir_path.join(format!("checkpoint_{:05}.json", epoch));
            let bin_path = dir_path.join(format!("checkpoint_{:05}.bin", epoch));

            fs::write(&json_path, serde_json::to_string(&metadata).unwrap()).unwrap();
            fs::write(&bin_path, b"dummy").unwrap();
        }

        let latest = utils::find_latest_checkpoint(dir_path).unwrap();
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().metadata.epoch, 5);
    }

    #[test]
    fn test_find_best_checkpoint() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path();

        // Create test checkpoints with different val_loss values
        for (epoch, val_loss) in [(1, 0.5), (2, 0.3), (3, 0.4)] {
            let mut metrics = HashMap::new();
            metrics.insert("val_loss".to_string(), val_loss);

            let metadata = CheckpointMetadata {
                epoch,
                loss: val_loss,
                learning_rate: 0.01,
                metrics,
                timestamp: 1000 * epoch as u64,
                checkpoint_file: format!("checkpoint_{:05}.bin", epoch),
            };

            let json_path = dir_path.join(format!("checkpoint_{:05}.json", epoch));
            let bin_path = dir_path.join(format!("checkpoint_{:05}.bin", epoch));

            fs::write(&json_path, serde_json::to_string(&metadata).unwrap()).unwrap();
            fs::write(&bin_path, b"dummy").unwrap();
        }

        let best = utils::find_best_checkpoint(dir_path, "val_loss", MetricMode::Minimize).unwrap();
        assert!(best.is_some());
        assert_eq!(best.unwrap().metadata.epoch, 2); // epoch 2 has lowest val_loss (0.3)
    }

    #[test]
    fn test_find_best_checkpoint_maximize() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path();

        // Create test checkpoints with different val_accuracy values
        for (epoch, val_acc) in [(1, 0.8), (2, 0.9), (3, 0.85)] {
            let mut metrics = HashMap::new();
            metrics.insert("val_accuracy".to_string(), val_acc);

            let metadata = CheckpointMetadata {
                epoch,
                loss: 0.1,
                learning_rate: 0.01,
                metrics,
                timestamp: 1000 * epoch as u64,
                checkpoint_file: format!("checkpoint_{:05}.bin", epoch),
            };

            let json_path = dir_path.join(format!("checkpoint_{:05}.json", epoch));
            let bin_path = dir_path.join(format!("checkpoint_{:05}.bin", epoch));

            fs::write(&json_path, serde_json::to_string(&metadata).unwrap()).unwrap();
            fs::write(&bin_path, b"dummy").unwrap();
        }

        let best =
            utils::find_best_checkpoint(dir_path, "val_accuracy", MetricMode::Maximize).unwrap();
        assert!(best.is_some());
        assert_eq!(best.unwrap().metadata.epoch, 2); // epoch 2 has highest val_accuracy (0.9)
    }

    #[test]
    fn test_save_strategy_combined() {
        let strategy = SaveStrategy::Combined {
            every_n: 5,
            metric_name: "val_loss".to_string(),
            mode: MetricMode::Minimize,
        };
        let mut callback: CheckpointCallback<CpuBackend, LinearModel<CpuBackend, Unfitted>> =
            CheckpointCallback::new("/tmp/test", strategy, 0);

        let mut metrics = HashMap::new();

        // First value - should save (metric improvement)
        metrics.insert("val_loss".to_string(), 1.0);
        assert!(callback.should_save(0, &metrics));
        callback.update_best_metric(&metrics);

        // Epoch 4 (5th epoch) - should save (periodic)
        assert!(callback.should_save(4, &metrics));

        // Epoch 5 (6th epoch) with no improvement - should not save
        assert!(!callback.should_save(5, &metrics));

        // Better value at non-periodic epoch - should save (metric improvement)
        metrics.insert("val_loss".to_string(), 0.5);
        assert!(callback.should_save(6, &metrics));
    }

    #[test]
    fn test_save_strategy_maximize() {
        let strategy = SaveStrategy::BestMetric {
            metric_name: "val_accuracy".to_string(),
            mode: MetricMode::Maximize,
        };
        let mut callback: CheckpointCallback<CpuBackend, LinearModel<CpuBackend, Unfitted>> =
            CheckpointCallback::new("/tmp/test", strategy, 0);

        let mut metrics = HashMap::new();

        // First value - should save
        metrics.insert("val_accuracy".to_string(), 0.8);
        assert!(callback.should_save(0, &metrics));
        callback.update_best_metric(&metrics);

        // Lower value - should not save
        metrics.insert("val_accuracy".to_string(), 0.7);
        assert!(!callback.should_save(1, &metrics));

        // Higher value - should save
        metrics.insert("val_accuracy".to_string(), 0.9);
        assert!(callback.should_save(2, &metrics));
    }

    #[test]
    fn test_with_prefix() {
        let callback: CheckpointCallback<CpuBackend, LinearModel<CpuBackend, Unfitted>> =
            CheckpointCallback::new("/tmp/test", SaveStrategy::EveryNEpochs(1), 0)
                .with_prefix("model_v1");

        assert_eq!(callback.prefix, "model_v1");
    }

    #[test]
    fn test_list_checkpoints() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path();

        // Create test checkpoints
        for epoch in [3, 1, 2] {
            let metadata = CheckpointMetadata {
                epoch,
                loss: 0.1,
                learning_rate: 0.01,
                metrics: HashMap::new(),
                timestamp: 1000 * epoch as u64,
                checkpoint_file: format!("checkpoint_{:05}.bin", epoch),
            };

            let json_path = dir_path.join(format!("checkpoint_{:05}.json", epoch));
            let bin_path = dir_path.join(format!("checkpoint_{:05}.bin", epoch));

            fs::write(&json_path, serde_json::to_string(&metadata).unwrap()).unwrap();
            fs::write(&bin_path, b"dummy").unwrap();
        }

        let checkpoints = utils::list_checkpoints(dir_path).unwrap();
        assert_eq!(checkpoints.len(), 3);

        // Should be sorted by epoch
        assert_eq!(checkpoints[0].metadata.epoch, 1);
        assert_eq!(checkpoints[1].metadata.epoch, 2);
        assert_eq!(checkpoints[2].metadata.epoch, 3);
    }

    #[test]
    fn test_load_metadata() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path();

        let metadata = CheckpointMetadata {
            epoch: 42,
            loss: 0.123,
            learning_rate: 0.001,
            metrics: HashMap::new(),
            timestamp: 999999,
            checkpoint_file: "checkpoint_00042.bin".to_string(),
        };

        let json_path = dir_path.join("checkpoint_00042.json");
        fs::write(&json_path, serde_json::to_string(&metadata).unwrap()).unwrap();

        let loaded = utils::load_metadata(&json_path).unwrap();
        assert_eq!(loaded.epoch, 42);
        assert!((loaded.loss - 0.123).abs() < 1e-10);
        assert!((loaded.learning_rate - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_list_checkpoints_empty_dir() {
        let dir = tempdir().unwrap();
        let checkpoints = utils::list_checkpoints(dir.path()).unwrap();
        assert!(checkpoints.is_empty());
    }

    #[test]
    fn test_find_latest_checkpoint_empty_dir() {
        let dir = tempdir().unwrap();
        let latest = utils::find_latest_checkpoint(dir.path()).unwrap();
        assert!(latest.is_none());
    }

    #[test]
    fn test_find_best_checkpoint_missing_metric() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path();

        // Create checkpoint without the metric we're looking for
        let metadata = CheckpointMetadata {
            epoch: 1,
            loss: 0.1,
            learning_rate: 0.01,
            metrics: HashMap::new(), // no metrics
            timestamp: 1000,
            checkpoint_file: "checkpoint_00001.bin".to_string(),
        };

        let json_path = dir_path.join("checkpoint_00001.json");
        let bin_path = dir_path.join("checkpoint_00001.bin");

        fs::write(&json_path, serde_json::to_string(&metadata).unwrap()).unwrap();
        fs::write(&bin_path, b"dummy").unwrap();

        let best = utils::find_best_checkpoint(dir_path, "val_loss", MetricMode::Minimize).unwrap();
        assert!(best.is_none()); // no checkpoints have the metric
    }

    #[test]
    fn test_find_latest_checkpoint_nonexistent_dir() {
        let latest = utils::find_latest_checkpoint("/nonexistent/path").unwrap();
        assert!(latest.is_none());
    }

    #[test]
    fn test_find_best_checkpoint_nonexistent_dir() {
        let best =
            utils::find_best_checkpoint("/nonexistent/path", "val_loss", MetricMode::Minimize)
                .unwrap();
        assert!(best.is_none());
    }

    #[test]
    fn test_should_save_missing_metric() {
        let strategy = SaveStrategy::BestMetric {
            metric_name: "val_loss".to_string(),
            mode: MetricMode::Minimize,
        };
        let callback: CheckpointCallback<CpuBackend, LinearModel<CpuBackend, Unfitted>> =
            CheckpointCallback::new("/tmp/test", strategy, 0);

        // Empty metrics - should not save
        let metrics = HashMap::new();
        assert!(!callback.should_save(0, &metrics));
    }

    #[test]
    fn test_checkpoint_callback_train_lifecycle() {
        use crate::callbacks::Callback;
        use crate::model::linear::LinearRegression;

        let dir = tempdir().unwrap();
        let dir_path = dir.path();

        let strategy = SaveStrategy::EveryNEpochs(1);
        let mut callback: CheckpointCallback<CpuBackend, LinearModel<CpuBackend, Unfitted>> =
            CheckpointCallback::new(dir_path, strategy, 3);

        let model = LinearRegression::<CpuBackend>::new(2);

        // on_train_start
        let state = TrainingState::new(0, 0, 5, 1, 0.5, &model, 0.01);
        callback.on_train_start(&state);

        // Directory should be created
        assert!(dir_path.exists());

        // on_epoch_end for multiple epochs
        for epoch in 0..5 {
            let mut state =
                TrainingState::new(epoch, 0, 5, 1, 0.5 - epoch as f64 * 0.1, &model, 0.01);
            callback.on_epoch_end(&mut state);
        }

        // on_train_end - should cleanup old checkpoints
        let state = TrainingState::new(5, 0, 5, 1, 0.0, &model, 0.01);
        callback.on_train_end(&state);

        // Should have at most 3 checkpoints (keep_best_n=3)
        let checkpoints: Vec<_> = fs::read_dir(dir_path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
            .collect();
        assert!(checkpoints.len() <= 3);
    }

    #[test]
    fn test_checkpoint_cleanup_sorts_by_loss() {
        use crate::callbacks::Callback;
        use crate::model::linear::LinearRegression;

        let dir = tempdir().unwrap();
        let dir_path = dir.path();

        // Use EveryNEpochs strategy (sorts by loss for cleanup)
        let strategy = SaveStrategy::EveryNEpochs(1);
        let mut callback: CheckpointCallback<CpuBackend, LinearModel<CpuBackend, Unfitted>> =
            CheckpointCallback::new(dir_path, strategy, 2);

        let model = LinearRegression::<CpuBackend>::new(2);

        callback.on_train_start(&TrainingState::new(0, 0, 3, 1, 0.5, &model, 0.01));

        // Save 3 checkpoints with different losses
        for epoch in 0..3 {
            let loss = match epoch {
                0 => 0.5, // Highest - should be removed
                1 => 0.1, // Lowest
                2 => 0.3, // Middle
                _ => 0.0,
            };
            let mut state = TrainingState::new(epoch, 0, 3, 1, loss, &model, 0.01);
            callback.on_epoch_end(&mut state);
        }

        // Cleanup on train end should keep 2 best (by loss)
        callback.on_train_end(&TrainingState::new(3, 0, 3, 1, 0.0, &model, 0.01));

        let checkpoints = utils::list_checkpoints(dir_path).unwrap();
        assert_eq!(checkpoints.len(), 2);

        // Should have epochs 1 (loss=0.1) and 2 (loss=0.3)
        let epochs: Vec<_> = checkpoints.iter().map(|c| c.metadata.epoch).collect();
        assert!(epochs.contains(&1));
        assert!(epochs.contains(&2));
    }

    #[test]
    fn test_checkpoint_save_fails_gracefully() {
        // Test that error in save_checkpoint is handled
        use crate::model::linear::LinearRegression;

        let model = LinearRegression::<CpuBackend>::new(2);

        // Create callback pointing to an invalid path
        let strategy = SaveStrategy::EveryNEpochs(1);
        let mut callback: CheckpointCallback<CpuBackend, LinearModel<CpuBackend, Unfitted>> =
            CheckpointCallback::new("/nonexistent/path/that/cannot/be/created", strategy, 0);

        let mut state = TrainingState::new(0, 0, 1, 1, 0.5, &model, 0.01);

        // Should not panic even though save will fail
        callback.on_epoch_end(&mut state);
    }
}
