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
const DEFAULT_FLUSH_THRESHOLD: usize = 50;

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
    /// Total operations queued (for statistics).
    total_ops: u64,
    /// Total flushes performed.
    total_flushes: u64,
}

impl CommandAccumulator {
    /// Creates a new command accumulator with default flush threshold.
    pub fn new() -> Self {
        Self::with_threshold(DEFAULT_FLUSH_THRESHOLD)
    }

    /// Creates a new command accumulator with specified flush threshold.
    pub fn with_threshold(threshold: usize) -> Self {
        Self {
            pending_commands: Vec::with_capacity(threshold),
            flush_threshold: threshold,
            total_ops: 0,
            total_flushes: 0,
        }
    }

    /// Adds an executable command to the accumulator.
    pub fn add_command(&mut self, command: ExecutableCommand) {
        self.pending_commands.push(command);
        self.total_ops += 1;
    }

    /// Returns the number of pending commands.
    pub fn pending_count(&self) -> usize {
        self.pending_commands.len()
    }

    /// Returns true if auto-flush threshold is reached.
    pub fn should_flush(&self) -> bool {
        self.pending_commands.len() >= self.flush_threshold
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

        // Clear pending commands
        self.pending_commands.clear();
        self.total_flushes += 1;
    }

    /// Returns statistics about the accumulator.
    pub fn stats(&self) -> AccumulatorStats {
        AccumulatorStats {
            pending_ops: self.pending_commands.len(),
            total_ops: self.total_ops,
            total_flushes: self.total_flushes,
            flush_threshold: self.flush_threshold,
        }
    }

    /// Sets the flush threshold.
    pub fn set_flush_threshold(&mut self, threshold: usize) {
        self.flush_threshold = threshold;
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
    /// Total operations queued since creation.
    pub total_ops: u64,
    /// Total flushes performed.
    pub total_flushes: u64,
    /// Current flush threshold.
    pub flush_threshold: usize,
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
        assert_eq!(stats.total_ops, 0);
        assert_eq!(stats.total_flushes, 0);
    }
}
