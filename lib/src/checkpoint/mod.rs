//! Checkpoint restoration utilities for resuming training.
//!
//! This module provides functions for loading model checkpoints and
//! resuming training from saved state.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::callbacks::checkpoint::utils;
pub use crate::callbacks::checkpoint::{CheckpointInfo, CheckpointMetadata, MetricMode};

/// Error type for checkpoint operations.
#[derive(Debug)]
pub enum CheckpointError {
    /// The checkpoint file was not found.
    NotFound(PathBuf),
    /// The checkpoint metadata could not be parsed.
    InvalidMetadata(String),
    /// The model parameters could not be loaded.
    InvalidParams(String),
    /// An I/O error occurred.
    Io(std::io::Error),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::NotFound(path) => {
                write!(f, "Checkpoint not found: {}", path.display())
            }
            CheckpointError::InvalidMetadata(msg) => {
                write!(f, "Invalid checkpoint metadata: {}", msg)
            }
            CheckpointError::InvalidParams(msg) => write!(f, "Invalid model parameters: {}", msg),
            CheckpointError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for CheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CheckpointError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self {
        CheckpointError::Io(e)
    }
}

/// Result type for checkpoint operations.
pub type CheckpointResult<T> = Result<T, CheckpointError>;

/// Represents a loaded checkpoint with its metadata and raw parameters.
#[derive(Debug)]
pub struct LoadedCheckpoint {
    /// Path to the checkpoint file.
    pub path: PathBuf,
    /// Checkpoint metadata.
    pub metadata: CheckpointMetadata,
    /// Raw model parameter bytes.
    pub params_bytes: Vec<u8>,
}

impl LoadedCheckpoint {
    /// Loads a checkpoint from a directory by finding the latest one.
    pub fn load_latest<P: AsRef<Path>>(dir: P) -> CheckpointResult<Self> {
        let info = utils::find_latest_checkpoint(&dir)?
            .ok_or_else(|| CheckpointError::NotFound(dir.as_ref().to_path_buf()))?;

        Self::from_info(&info)
    }

    /// Loads the best checkpoint based on a metric.
    pub fn load_best<P: AsRef<Path>>(
        dir: P,
        metric_name: &str,
        mode: MetricMode,
    ) -> CheckpointResult<Self> {
        let info = utils::find_best_checkpoint(&dir, metric_name, mode)?
            .ok_or_else(|| CheckpointError::NotFound(dir.as_ref().to_path_buf()))?;

        Self::from_info(&info)
    }

    /// Loads a checkpoint from a specific path (without .bin extension).
    pub fn load<P: AsRef<Path>>(path: P) -> CheckpointResult<Self> {
        let path = path.as_ref();
        let bin_path = path.with_extension("bin");
        let json_path = path.with_extension("json");

        if !bin_path.exists() {
            return Err(CheckpointError::NotFound(bin_path));
        }

        let metadata = utils::load_metadata(&json_path)
            .map_err(|e| CheckpointError::InvalidMetadata(e.to_string()))?;

        let params_bytes = std::fs::read(&bin_path)?;

        Ok(Self {
            path: bin_path,
            metadata,
            params_bytes,
        })
    }

    /// Loads a checkpoint from a CheckpointInfo.
    fn from_info(info: &CheckpointInfo) -> CheckpointResult<Self> {
        let params_bytes = std::fs::read(&info.path)?;

        Ok(Self {
            path: info.path.clone(),
            metadata: info.metadata.clone(),
            params_bytes,
        })
    }

    /// Returns the epoch number of this checkpoint.
    pub fn epoch(&self) -> usize {
        self.metadata.epoch
    }

    /// Returns the loss value at this checkpoint.
    pub fn loss(&self) -> f64 {
        self.metadata.loss
    }

    /// Returns the learning rate at this checkpoint.
    pub fn learning_rate(&self) -> f64 {
        self.metadata.learning_rate
    }

    /// Returns a metric value from this checkpoint, if present.
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metadata.metrics.get(name).copied()
    }

    /// Returns all metrics from this checkpoint.
    pub fn metrics(&self) -> &HashMap<String, f64> {
        &self.metadata.metrics
    }
}

/// Trait for models that can be restored from a checkpoint.
///
/// This trait allows models to be loaded from checkpoint parameter bytes.
pub trait RestorableFromCheckpoint: Sized {
    /// Restores the model from checkpoint parameter bytes.
    fn from_checkpoint_bytes(bytes: &[u8]) -> CheckpointResult<Self>;
}

/// Finds the latest checkpoint in a directory.
pub fn find_latest_checkpoint<P: AsRef<Path>>(dir: P) -> CheckpointResult<CheckpointInfo> {
    utils::find_latest_checkpoint(&dir)?
        .ok_or_else(|| CheckpointError::NotFound(dir.as_ref().to_path_buf()))
}

/// Finds the best checkpoint in a directory based on a metric.
pub fn find_best_checkpoint<P: AsRef<Path>>(
    dir: P,
    metric_name: &str,
    mode: MetricMode,
) -> CheckpointResult<CheckpointInfo> {
    utils::find_best_checkpoint(&dir, metric_name, mode)?
        .ok_or_else(|| CheckpointError::NotFound(dir.as_ref().to_path_buf()))
}

/// Lists all checkpoints in a directory.
pub fn list_checkpoints<P: AsRef<Path>>(dir: P) -> CheckpointResult<Vec<CheckpointInfo>> {
    utils::list_checkpoints(&dir).map_err(CheckpointError::Io)
}

/// Loads checkpoint metadata from a JSON file.
pub fn load_checkpoint_metadata<P: AsRef<Path>>(path: P) -> CheckpointResult<CheckpointMetadata> {
    utils::load_metadata(&path).map_err(|e| CheckpointError::InvalidMetadata(e.to_string()))
}

/// Loads a checkpoint and returns the raw parameter bytes.
///
/// Use this when you need access to the raw bytes to restore model parameters.
pub fn load_checkpoint_bytes<P: AsRef<Path>>(
    path: P,
) -> CheckpointResult<(Vec<u8>, CheckpointMetadata)> {
    let checkpoint = LoadedCheckpoint::load(path)?;
    Ok((checkpoint.params_bytes, checkpoint.metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::fs;
    use tempfile::tempdir;

    fn create_test_checkpoint(dir: &Path, epoch: usize, val_loss: f64) -> PathBuf {
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

        let json_path = dir.join(format!("checkpoint_{:05}.json", epoch));
        let bin_path = dir.join(format!("checkpoint_{:05}.bin", epoch));

        fs::write(&json_path, serde_json::to_string(&metadata).unwrap()).unwrap();
        fs::write(&bin_path, format!("params_{}", epoch).as_bytes()).unwrap();

        bin_path
    }

    #[test]
    fn test_find_latest_checkpoint_empty_dir() {
        let dir = tempdir().unwrap();
        let result = find_latest_checkpoint(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_find_best_checkpoint_empty_dir() {
        let dir = tempdir().unwrap();
        let result = find_best_checkpoint(dir.path(), "val_loss", MetricMode::Minimize);
        assert!(result.is_err());
    }

    #[test]
    fn test_list_checkpoints_empty_dir() {
        let dir = tempdir().unwrap();
        let checkpoints = list_checkpoints(dir.path()).unwrap();
        assert!(checkpoints.is_empty());
    }

    #[test]
    fn test_loaded_checkpoint_load() {
        let dir = tempdir().unwrap();
        create_test_checkpoint(dir.path(), 5, 0.25);

        let path = dir.path().join("checkpoint_00005");
        let checkpoint = LoadedCheckpoint::load(&path).unwrap();

        assert_eq!(checkpoint.epoch(), 5);
        assert!((checkpoint.loss() - 0.25).abs() < 1e-10);
        assert!((checkpoint.learning_rate() - 0.01).abs() < 1e-10);
        assert_eq!(checkpoint.params_bytes, format!("params_{}", 5).as_bytes());
    }

    #[test]
    fn test_loaded_checkpoint_load_not_found() {
        let result = LoadedCheckpoint::load("/nonexistent/path");
        assert!(result.is_err());
        match result.unwrap_err() {
            CheckpointError::NotFound(_) => {}
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn test_loaded_checkpoint_load_latest() {
        let dir = tempdir().unwrap();
        create_test_checkpoint(dir.path(), 1, 0.5);
        create_test_checkpoint(dir.path(), 5, 0.3);
        create_test_checkpoint(dir.path(), 3, 0.4);

        let checkpoint = LoadedCheckpoint::load_latest(dir.path()).unwrap();
        assert_eq!(checkpoint.epoch(), 5);
    }

    #[test]
    fn test_loaded_checkpoint_load_best() {
        let dir = tempdir().unwrap();
        create_test_checkpoint(dir.path(), 1, 0.5);
        create_test_checkpoint(dir.path(), 2, 0.2); // Best
        create_test_checkpoint(dir.path(), 3, 0.4);

        let checkpoint =
            LoadedCheckpoint::load_best(dir.path(), "val_loss", MetricMode::Minimize).unwrap();
        assert_eq!(checkpoint.epoch(), 2);
    }

    #[test]
    fn test_loaded_checkpoint_get_metric() {
        let dir = tempdir().unwrap();
        create_test_checkpoint(dir.path(), 1, 0.5);

        let path = dir.path().join("checkpoint_00001");
        let checkpoint = LoadedCheckpoint::load(&path).unwrap();

        assert_eq!(checkpoint.get_metric("val_loss"), Some(0.5));
        assert_eq!(checkpoint.get_metric("nonexistent"), None);
    }

    #[test]
    fn test_loaded_checkpoint_metrics() {
        let dir = tempdir().unwrap();
        create_test_checkpoint(dir.path(), 1, 0.5);

        let path = dir.path().join("checkpoint_00001");
        let checkpoint = LoadedCheckpoint::load(&path).unwrap();

        let metrics = checkpoint.metrics();
        assert_eq!(metrics.len(), 1);
        assert!(metrics.contains_key("val_loss"));
    }

    #[test]
    fn test_find_latest_checkpoint_with_data() {
        let dir = tempdir().unwrap();
        create_test_checkpoint(dir.path(), 1, 0.5);
        create_test_checkpoint(dir.path(), 5, 0.3);

        let info = find_latest_checkpoint(dir.path()).unwrap();
        assert_eq!(info.metadata.epoch, 5);
    }

    #[test]
    fn test_find_best_checkpoint_with_data() {
        let dir = tempdir().unwrap();
        create_test_checkpoint(dir.path(), 1, 0.5);
        create_test_checkpoint(dir.path(), 2, 0.1); // Best
        create_test_checkpoint(dir.path(), 3, 0.3);

        let info = find_best_checkpoint(dir.path(), "val_loss", MetricMode::Minimize).unwrap();
        assert_eq!(info.metadata.epoch, 2);
    }

    #[test]
    fn test_list_checkpoints_with_data() {
        let dir = tempdir().unwrap();
        create_test_checkpoint(dir.path(), 1, 0.5);
        create_test_checkpoint(dir.path(), 3, 0.3);

        let checkpoints = list_checkpoints(dir.path()).unwrap();
        assert_eq!(checkpoints.len(), 2);
    }

    #[test]
    fn test_load_checkpoint_metadata() {
        let dir = tempdir().unwrap();
        create_test_checkpoint(dir.path(), 5, 0.25);

        let json_path = dir.path().join("checkpoint_00005.json");
        let metadata = load_checkpoint_metadata(&json_path).unwrap();

        assert_eq!(metadata.epoch, 5);
        assert!((metadata.loss - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_load_checkpoint_bytes() {
        let dir = tempdir().unwrap();
        create_test_checkpoint(dir.path(), 3, 0.4);

        let path = dir.path().join("checkpoint_00003");
        let (bytes, metadata) = load_checkpoint_bytes(&path).unwrap();

        assert_eq!(bytes, format!("params_{}", 3).as_bytes());
        assert_eq!(metadata.epoch, 3);
    }

    #[test]
    fn test_checkpoint_error_display() {
        let err = CheckpointError::NotFound(PathBuf::from("/path/to/checkpoint"));
        assert!(err.to_string().contains("not found"));

        let err = CheckpointError::InvalidMetadata("bad json".to_string());
        assert!(err.to_string().contains("Invalid checkpoint metadata"));

        let err = CheckpointError::InvalidParams("bad params".to_string());
        assert!(err.to_string().contains("Invalid model parameters"));

        let err = CheckpointError::Io(std::io::Error::new(std::io::ErrorKind::Other, "io error"));
        assert!(err.to_string().contains("I/O error"));
    }

    #[test]
    fn test_checkpoint_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "test");
        let checkpoint_err: CheckpointError = io_err.into();
        match checkpoint_err {
            CheckpointError::Io(_) => {}
            _ => panic!("Expected Io error"),
        }
    }

    #[test]
    fn test_checkpoint_error_source() {
        let err = CheckpointError::Io(std::io::Error::new(std::io::ErrorKind::Other, "test"));
        assert!(err.source().is_some());

        let err = CheckpointError::NotFound(PathBuf::from("/path"));
        assert!(err.source().is_none());
    }

    #[test]
    fn test_loaded_checkpoint_load_best_maximize() {
        let dir = tempdir().unwrap();
        // Create checkpoints with accuracy metric
        for (epoch, val_acc) in [(1, 0.8), (2, 0.95), (3, 0.85)] {
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

            let json_path = dir.path().join(format!("checkpoint_{:05}.json", epoch));
            let bin_path = dir.path().join(format!("checkpoint_{:05}.bin", epoch));

            fs::write(&json_path, serde_json::to_string(&metadata).unwrap()).unwrap();
            fs::write(&bin_path, format!("params_{}", epoch).as_bytes()).unwrap();
        }

        let checkpoint =
            LoadedCheckpoint::load_best(dir.path(), "val_accuracy", MetricMode::Maximize).unwrap();
        assert_eq!(checkpoint.epoch(), 2); // highest accuracy
    }
}
