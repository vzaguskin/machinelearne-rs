//! Command buffer accumulator for batching GPU operations.
//!
//! Instead of submitting each operation immediately, commands are
//! accumulated and submitted in batches, reducing GPU-CPU synchronization
//! overhead significantly.
//!
//! ## How it works
//!
//! 1. Each tensor operation creates its buffers and bind groups
//! 2. Instead of creating an encoder and submitting, it queues an `ExecutableCommand`
//! 3. On flush (triggered by `to_vec()`, `sum()`, etc.), ONE encoder is created
//! 4. All compute passes are added to this single encoder
//! 5. ONE submission to the GPU
//!
//! This reduces per-operation overhead dramatically for workloads with many operations.

use std::cell::RefCell;
use std::sync::Arc;
use wgpu::{BindGroup, CommandEncoder, Device, Queue};

use super::shaders::ComputePipeline;

/// Default number of operations before auto-flush.
/// Increased from 50 to 500 to reduce sync overhead during training.
const DEFAULT_FLUSH_THRESHOLD: usize = 500;

/// Default maximum memory for queued commands (in bytes).
/// Set to 256MB to prevent memory exhaustion while allowing good batching.
const DEFAULT_MEMORY_THRESHOLD: usize = 256 * 1024 * 1024;

/// Estimated memory per command (bind groups, buffers referenced).
/// This is a rough estimate; actual memory varies by operation.
const ESTIMATED_MEMORY_PER_COMMAND: usize = 1024; // 1KB per command estimate

/// An executable compute command ready to be added to a command encoder.
///
/// Contains everything needed to dispatch one compute pass.
pub struct ExecutableCommand {
    /// The compute pipeline to use.
    pub pipeline: Arc<ComputePipeline>,
    /// The bind group containing resources.
    pub bind_group: Arc<BindGroup>,
    /// Dynamic offsets for bind group (for uniform buffers with dynamic offsets).
    pub dynamic_offsets: Vec<u32>,
    /// Number of workgroups in X dimension.
    pub workgroups_x: u32,
    /// Number of workgroups in Y dimension.
    pub workgroups_y: u32,
    /// Number of workgroups in Z dimension.
    pub workgroups_z: u32,
    /// Optional label for debugging.
    pub label: Option<String>,
}

impl ExecutableCommand {
    /// Creates a new executable command.
    pub fn new(
        pipeline: Arc<ComputePipeline>,
        bind_group: Arc<BindGroup>,
        workgroups_x: u32,
        workgroups_y: u32,
        workgroups_z: u32,
        label: Option<&str>,
    ) -> Self {
        Self {
            pipeline,
            bind_group,
            dynamic_offsets: Vec::new(),
            workgroups_x,
            workgroups_y,
            workgroups_z,
            label: label.map(|s| s.to_string()),
        }
    }

    /// Creates a new executable command with dynamic offsets.
    pub fn with_dynamic_offsets(
        pipeline: Arc<ComputePipeline>,
        bind_group: Arc<BindGroup>,
        dynamic_offsets: Vec<u32>,
        workgroups_x: u32,
        workgroups_y: u32,
        workgroups_z: u32,
        label: Option<&str>,
    ) -> Self {
        Self {
            pipeline,
            bind_group,
            dynamic_offsets,
            workgroups_x,
            workgroups_y,
            workgroups_z,
            label: label.map(|s| s.to_string()),
        }
    }

    /// Creates a 1D dispatch (single workgroup dimension).
    pub fn dispatch_1d(
        pipeline: Arc<ComputePipeline>,
        bind_group: Arc<BindGroup>,
        workgroups: u32,
        label: Option<&str>,
    ) -> Self {
        Self::new(pipeline, bind_group, workgroups, 1, 1, label)
    }

    /// Creates a 1D dispatch with dynamic offsets.
    pub fn dispatch_1d_dynamic(
        pipeline: Arc<ComputePipeline>,
        bind_group: Arc<BindGroup>,
        dynamic_offset: u32,
        workgroups: u32,
        label: Option<&str>,
    ) -> Self {
        Self::with_dynamic_offsets(
            pipeline,
            bind_group,
            vec![dynamic_offset],
            workgroups,
            1,
            1,
            label,
        )
    }

    /// Creates a 2D dispatch (two workgroup dimensions).
    pub fn dispatch_2d(
        pipeline: Arc<ComputePipeline>,
        bind_group: Arc<BindGroup>,
        workgroups_x: u32,
        workgroups_y: u32,
        label: Option<&str>,
    ) -> Self {
        Self::new(pipeline, bind_group, workgroups_x, workgroups_y, 1, label)
    }

    /// Executes this command by adding a compute pass to the encoder.
    pub fn execute(&self, encoder: &mut CommandEncoder) {
        let label = self.label.as_deref();
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline.pipeline);
        compute_pass.set_bind_group(0, &*self.bind_group, &self.dynamic_offsets);
        compute_pass.dispatch_workgroups(self.workgroups_x, self.workgroups_y, self.workgroups_z);
    }
}

/// Command accumulator for batching GPU operations.
///
/// Thread-local accumulator that queues operations and submits them
/// in batches to reduce synchronization overhead.
pub struct CommandAccumulator {
    /// Pending commands waiting to be executed.
    pending_commands: Vec<ExecutableCommand>,
    /// Number of operations before auto-flush.
    flush_threshold: usize,
    /// Maximum estimated memory for queued commands (in bytes).
    memory_threshold: usize,
    /// Estimated memory usage of pending commands.
    estimated_memory: usize,
    /// Total operations queued (for statistics).
    total_ops: u64,
    /// Total flushes performed.
    total_flushes: u64,
    /// Debug mode: flush after every operation.
    debug_mode: bool,
}

impl CommandAccumulator {
    /// Creates a new command accumulator with default flush threshold.
    pub fn new() -> Self {
        Self::with_threshold(DEFAULT_FLUSH_THRESHOLD)
    }

    /// Creates a new command accumulator with specified flush threshold.
    pub fn with_threshold(threshold: usize) -> Self {
        Self::with_threshold_and_memory(threshold, DEFAULT_MEMORY_THRESHOLD)
    }

    /// Creates a new command accumulator with specified flush and memory thresholds.
    pub fn with_threshold_and_memory(flush_threshold: usize, memory_threshold: usize) -> Self {
        Self {
            pending_commands: Vec::with_capacity(flush_threshold.min(1000)),
            flush_threshold,
            memory_threshold,
            estimated_memory: 0,
            total_ops: 0,
            total_flushes: 0,
            debug_mode: false,
        }
    }

    /// Adds an executable command to the accumulator.
    pub fn add_command(&mut self, command: ExecutableCommand) {
        self.pending_commands.push(command);
        self.estimated_memory += ESTIMATED_MEMORY_PER_COMMAND;
        self.total_ops += 1;
    }

    /// Returns true if we should flush after adding a command.
    /// This is true in debug mode (eager flush) or when threshold is reached.
    pub fn should_flush_after_add(&self) -> bool {
        self.debug_mode
            || self.pending_commands.len() >= self.flush_threshold
            || self.estimated_memory >= self.memory_threshold
    }

    /// Returns the number of pending commands.
    pub fn pending_count(&self) -> usize {
        self.pending_commands.len()
    }

    /// Returns true if auto-flush threshold is reached (count or memory).
    pub fn should_flush(&self) -> bool {
        self.pending_commands.len() >= self.flush_threshold
            || self.estimated_memory >= self.memory_threshold
    }

    /// Returns the estimated memory usage of pending commands.
    pub fn estimated_memory(&self) -> usize {
        self.estimated_memory
    }

    /// Flushes all pending commands to the GPU.
    ///
    /// Creates a single command encoder, adds all compute passes,
    /// and submits once. This is much more efficient than per-operation
    /// submissions.
    pub fn flush(&mut self, device: &Device, queue: &Queue) {
        if self.pending_commands.is_empty() {
            return;
        }

        // Create a single command encoder for all operations
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CommandAccumulator::flush"),
        });

        // Add all compute passes to this single encoder
        for command in &self.pending_commands {
            command.execute(&mut encoder);
        }

        // Submit once for all operations
        queue.submit(std::iter::once(encoder.finish()));

        // Clear pending commands and reset memory estimate
        self.pending_commands.clear();
        self.estimated_memory = 0;
        self.total_flushes += 1;
    }

    /// Returns statistics about the accumulator.
    pub fn stats(&self) -> AccumulatorStats {
        AccumulatorStats {
            pending_ops: self.pending_commands.len(),
            estimated_memory: self.estimated_memory,
            total_ops: self.total_ops,
            total_flushes: self.total_flushes,
            flush_threshold: self.flush_threshold,
            memory_threshold: self.memory_threshold,
            debug_mode: self.debug_mode,
        }
    }

    /// Sets the flush threshold.
    pub fn set_flush_threshold(&mut self, threshold: usize) {
        self.flush_threshold = threshold;
    }

    /// Sets the memory threshold.
    pub fn set_memory_threshold(&mut self, threshold: usize) {
        self.memory_threshold = threshold;
    }

    /// Enables or disables debug mode for eager flushing.
    pub fn set_debug_mode(&mut self, enabled: bool) {
        self.debug_mode = enabled;
    }

    /// Returns whether debug mode is enabled.
    pub fn is_debug_mode(&self) -> bool {
        self.debug_mode
    }
}

impl Default for CommandAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the command accumulator.
#[derive(Debug, Clone, Copy)]
pub struct AccumulatorStats {
    /// Number of operations currently pending.
    pub pending_ops: usize,
    /// Estimated memory usage of pending commands (bytes).
    pub estimated_memory: usize,
    /// Total operations queued since creation.
    pub total_ops: u64,
    /// Total flushes performed.
    pub total_flushes: u64,
    /// Current flush threshold (operation count).
    pub flush_threshold: usize,
    /// Current memory threshold (bytes).
    pub memory_threshold: usize,
    /// Debug mode enabled (eager flushing).
    pub debug_mode: bool,
}

// Thread-local command accumulator.
// Each thread has its own accumulator to avoid synchronization overhead.
thread_local! {
    pub static COMMAND_ACCUMULATOR: RefCell<CommandAccumulator> =
        RefCell::new(CommandAccumulator::new());
}

/// Executes a function with access to the thread-local command accumulator.
pub fn with_accumulator<F, R>(f: F) -> R
where
    F: FnOnce(&mut CommandAccumulator) -> R,
{
    COMMAND_ACCUMULATOR.with(|acc| f(&mut acc.borrow_mut()))
}

/// Flushes the thread-local command accumulator.
pub fn flush_accumulator(device: &Device, queue: &Queue) {
    COMMAND_ACCUMULATOR.with(|acc| acc.borrow_mut().flush(device, queue));
}

/// Returns the pending operation count for the thread-local accumulator.
#[allow(dead_code)]
pub fn pending_ops_count() -> usize {
    COMMAND_ACCUMULATOR.with(|acc| acc.borrow().pending_count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_new() {
        let acc = CommandAccumulator::new();
        assert_eq!(acc.pending_count(), 0);
        assert_eq!(acc.flush_threshold, DEFAULT_FLUSH_THRESHOLD);
        assert!(!acc.should_flush());
    }

    #[test]
    fn test_accumulator_with_threshold() {
        let acc = CommandAccumulator::with_threshold(10);
        assert_eq!(acc.flush_threshold, 10);
    }

    #[test]
    fn test_stats() {
        let acc = CommandAccumulator::new();
        let stats = acc.stats();
        assert_eq!(stats.pending_ops, 0);
        assert_eq!(stats.estimated_memory, 0);
        assert_eq!(stats.total_ops, 0);
        assert_eq!(stats.total_flushes, 0);
        assert_eq!(stats.memory_threshold, DEFAULT_MEMORY_THRESHOLD);
    }

    #[test]
    fn test_memory_threshold() {
        let acc = CommandAccumulator::with_threshold_and_memory(100, 1024);
        assert_eq!(acc.flush_threshold, 100);
        assert_eq!(acc.memory_threshold, 1024);
    }

    #[test]
    fn test_memory_based_flush() {
        let mut acc = CommandAccumulator::with_threshold_and_memory(100, 100); // Very low memory threshold

        // Adding commands should eventually trigger memory-based flush
        // (though we can't test actual flushing without a device)
        assert!(!acc.should_flush()); // Empty

        // Simulate adding enough commands to exceed memory threshold
        // With ESTIMATED_MEMORY_PER_COMMAND = 1024, 1 command = 1024 bytes
        // So with memory_threshold = 100, first command should trigger
        acc.estimated_memory = 200; // Manually set for test
        assert!(acc.should_flush());
    }
}
