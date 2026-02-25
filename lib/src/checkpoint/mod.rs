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
    use tempfile::tempdir;

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
}
