//! Logging callback for training progress.
//!
//! Provides JSON Lines format logging compatible with visualization tools.

use crate::backend::Backend;
use crate::callbacks::{Callback, TrainingState};
use crate::model::TrainableModel;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// Logging callback that writes training progress to file and/or console.
///
/// Logs are written in JSON Lines format (one JSON object per line),
/// making them easy to parse and compatible with tools like TensorBoard
/// (after conversion).
///
/// # Example
///
/// ```rust,ignore
/// use machinelearne_rs::callbacks::LoggingCallback;
///
/// let logger = LoggingCallback::new("training_log.jsonl")
///     .with_console(true);
/// ```
pub struct LoggingCallback {
    /// Output file (optional).
    file: Option<BufWriter<File>>,
    /// File path for reopening on flush.
    #[allow(dead_code)]
    file_path: Option<PathBuf>,
    /// Whether to also print to console.
    console: bool,
    /// Log every N batches (0 = only at epoch end).
    log_frequency: usize,
    /// Buffer for batch-level logs.
    #[allow(dead_code)]
    buffer: Vec<String>,
}

impl LoggingCallback {
    /// Creates a new logging callback that writes to the specified file.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .ok()
            .map(BufWriter::new);

        Self {
            file,
            file_path: Some(path),
            console: false,
            log_frequency: 0,
            buffer: Vec::new(),
        }
    }

    /// Creates a console-only logger (no file output).
    pub fn console_only() -> Self {
        Self {
            file: None,
            file_path: None,
            console: true,
            log_frequency: 0,
            buffer: Vec::new(),
        }
    }

    /// Enables or disables console output.
    pub fn with_console(mut self, enabled: bool) -> Self {
        self.console = enabled;
        self
    }

    /// Sets how often to log batch-level information.
    ///
    /// - `0`: Only log at epoch end (default)
    /// - `n`: Log every n batches
    pub fn with_log_frequency(mut self, frequency: usize) -> Self {
        self.log_frequency = frequency;
        self
    }

    /// Formats a log entry as JSON.
    fn format_entry(
        epoch: usize,
        batch: Option<usize>,
        loss: f64,
        learning_rate: f64,
        metrics: &std::collections::HashMap<String, f64>,
    ) -> String {
        let timestamp = chrono_lite_timestamp();
        let metrics_json: Vec<String> = metrics
            .iter()
            .map(|(k, v)| format!("\"{}\": {:.6}", k, v))
            .collect();

        let batch_part = match batch {
            Some(b) => format!(", \"batch\": {}", b),
            None => String::new(),
        };

        let metrics_part = if metrics_json.is_empty() {
            String::new()
        } else {
            format!(", {}", metrics_json.join(", "))
        };

        format!(
            "{{\"epoch\": {}{}, \"loss\": {:.6}, \"learning_rate\": {:.6}, \"timestamp\": {}{}}}\n",
            epoch, batch_part, loss, learning_rate, timestamp, metrics_part
        )
    }

    /// Writes a log entry.
    fn log(&mut self, entry: &str) {
        if let Some(ref mut file) = self.file {
            let _ = file.write_all(entry.as_bytes());
        }
        if self.console {
            print!("{}", entry);
        }
    }

    /// Flushes buffered logs to disk.
    fn flush(&mut self) {
        if let Some(ref mut file) = self.file {
            let _ = file.flush();
        }
    }
}

/// Simple timestamp function without chrono dependency.
fn chrono_lite_timestamp() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

impl<B: Backend, M: TrainableModel<B>> Callback<B, M> for LoggingCallback {
    fn on_train_start(&mut self, _state: &TrainingState<B, M>) {
        let header = "{\"event\": \"train_start\"}\n";
        self.log(header);
    }

    fn on_train_end(&mut self, _state: &TrainingState<B, M>) {
        let footer = "{\"event\": \"train_end\"}\n";
        self.log(footer);
        self.flush();
    }

    fn on_epoch_end(&mut self, state: &mut TrainingState<B, M>) {
        let entry = Self::format_entry(
            state.epoch,
            None,
            state.loss,
            state.learning_rate,
            &state.metrics,
        );
        self.log(&entry);
        self.flush();
    }

    fn on_batch_end(&mut self, state: &mut TrainingState<B, M>) {
        if self.log_frequency > 0 && state.batch.is_multiple_of(self.log_frequency) {
            let entry = Self::format_entry(
                state.epoch,
                Some(state.batch),
                state.loss,
                state.learning_rate,
                &state.metrics,
            );
            self.log(&entry);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_format_entry_without_batch() {
        let mut metrics = HashMap::new();
        metrics.insert("val_loss".to_string(), 0.25);

        let entry = LoggingCallback::format_entry(5, None, 0.5, 0.01, &metrics);

        assert!(entry.contains("\"epoch\": 5"));
        assert!(entry.contains("\"loss\": 0.500000"));
        assert!(entry.contains("\"learning_rate\": 0.010000"));
        assert!(entry.contains("\"val_loss\": 0.250000"));
        assert!(!entry.contains("\"batch\""));
    }

    #[test]
    fn test_format_entry_with_batch() {
        let metrics = HashMap::new();
        let entry = LoggingCallback::format_entry(3, Some(10), 0.3, 0.001, &metrics);

        assert!(entry.contains("\"epoch\": 3"));
        assert!(entry.contains("\"batch\": 10"));
        assert!(entry.contains("\"loss\": 0.300000"));
    }

    #[test]
    fn test_console_only_logger() {
        let logger = LoggingCallback::console_only();
        assert!(logger.file.is_none());
        assert!(logger.console);
    }

    #[test]
    fn test_format_entry_is_valid_json() {
        let mut metrics = HashMap::new();
        metrics.insert("val_loss".to_string(), 0.123);
        metrics.insert("val_accuracy".to_string(), 0.95);

        let entry = LoggingCallback::format_entry(10, Some(5), 0.5, 0.01, &metrics);

        // Should be valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(entry.trim()).expect("Entry should be valid JSON");

        assert_eq!(parsed["epoch"], 10);
        assert_eq!(parsed["batch"], 5);
        assert!((parsed["loss"].as_f64().unwrap() - 0.5).abs() < 0.001);
        assert!((parsed["learning_rate"].as_f64().unwrap() - 0.01).abs() < 0.001);
        // Metrics are at top level
        assert!((parsed["val_loss"].as_f64().unwrap() - 0.123).abs() < 0.001);
        assert!((parsed["val_accuracy"].as_f64().unwrap() - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_format_entry_empty_metrics() {
        let metrics = HashMap::new();
        let entry = LoggingCallback::format_entry(0, None, 1.0, 0.1, &metrics);

        let parsed: serde_json::Value = serde_json::from_str(entry.trim()).unwrap();
        assert_eq!(parsed["epoch"], 0);
        assert!((parsed["loss"].as_f64().unwrap() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_log_frequency_filtering() {
        // Create a callback that logs every 5 batches
        let mut logger = LoggingCallback::console_only();
        logger.log_frequency = 5;

        // Batches 0, 5, 10 should be logged (is_multiple_of)
        assert!(5usize.is_multiple_of(logger.log_frequency));
        assert!(10usize.is_multiple_of(logger.log_frequency));
        assert!(!3usize.is_multiple_of(logger.log_frequency));
        assert!(!7usize.is_multiple_of(logger.log_frequency));
    }

    #[test]
    fn test_file_logger_creation() {
        use tempfile::NamedTempFile;

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        let logger = LoggingCallback::new(&path);
        assert!(logger.file.is_some());
        assert!(logger.file_path.is_some());
        assert!(!logger.console); // Default is no console output
    }
}
