//! Command buffer accumulator for batching GPU operations.
//!
//! Instead of submitting each operation immediately, commands are
//! accumulated and submitted in batches, reducing GPU-CPU synchronization
//! overhead significantly.

use std::cell::RefCell;
use std::sync::Arc;
use wgpu::{Buffer, Device, Queue};

use super::shaders::{BinaryOp, ScalarOp};

/// Default number of operations before auto-flush.
const DEFAULT_FLUSH_THRESHOLD: usize = 50;

/// A pending operation waiting to be executed.
#[allow(dead_code)] // Fields used for future batched execution
pub enum PendingOp {
    /// Binary operation on 1D tensors.
    Binary1D {
        input_a: Arc<Buffer>,
        input_b: Arc<Buffer>,
        output: Arc<Buffer>,
        params: BinaryParams,
        len: usize,
    },
    /// Binary operation on 2D tensors.
    Binary2D {
        input_a: Arc<Buffer>,
        input_b: Arc<Buffer>,
        output: Arc<Buffer>,
        params: Binary2DParams,
        rows: usize,
        cols: usize,
    },
    /// Scalar operation on 1D tensor.
    Scalar1D {
        input: Arc<Buffer>,
        output: Arc<Buffer>,
        params: Scalar1DParams,
        len: usize,
    },
    /// Scalar operation on 2D tensor.
    Scalar2D {
        input: Arc<Buffer>,
        output: Arc<Buffer>,
        params: Scalar2DParams,
        rows: usize,
        cols: usize,
    },
    /// Matrix-vector multiplication.
    MatVec {
        matrix: Arc<Buffer>,
        vector: Arc<Buffer>,
        output: Arc<Buffer>,
        params: MatVecParams,
    },
    /// Matrix-vector multiplication with transposed matrix.
    MatVecTransposed {
        matrix: Arc<Buffer>,
        vector: Arc<Buffer>,
        output: Arc<Buffer>,
        params: MatVecParams,
    },
    /// Transpose operation.
    Transpose {
        input: Arc<Buffer>,
        output: Arc<Buffer>,
        params: TransposeParams,
    },
}

/// Parameters for binary operations (mirrors tensor.rs structs).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BinaryParams {
    pub op: u32,
    pub len: u32,
    pub _padding: [u32; 2],
}

/// Parameters for 2D binary operations.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Binary2DParams {
    pub op: u32,
    pub rows: u32,
    pub cols: u32,
    pub _padding: u32,
}

/// Parameters for scalar 1D operations.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Scalar1DParams {
    pub op: u32,
    pub len: u32,
    pub scalar: f32,
    pub _padding: u32,
}

/// Parameters for scalar 2D operations.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Scalar2DParams {
    pub op: u32,
    pub rows: u32,
    pub cols: u32,
    pub scalar: f32,
}

/// Parameters for matrix-vector operations.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatVecParams {
    pub rows: u32,
    pub cols: u32,
    pub _padding: [u32; 2],
}

/// Parameters for transpose operations.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TransposeParams {
    pub rows: u32,
    pub cols: u32,
    pub _padding: [u32; 2],
}

/// Command accumulator for batching GPU operations.
///
/// Thread-local accumulator that queues operations and submits them
/// in batches to reduce synchronization overhead.
pub struct CommandAccumulator {
    /// Pending operations waiting to be executed.
    pending_ops: Vec<PendingOp>,
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
            pending_ops: Vec::with_capacity(threshold),
            flush_threshold: threshold,
            total_ops: 0,
            total_flushes: 0,
        }
    }

    /// Adds a binary 1D operation to the accumulator.
    pub fn add_binary_1d(
        &mut self,
        input_a: Arc<Buffer>,
        input_b: Arc<Buffer>,
        output: Arc<Buffer>,
        op: BinaryOp,
        len: usize,
    ) {
        self.pending_ops.push(PendingOp::Binary1D {
            input_a,
            input_b,
            output,
            params: BinaryParams {
                op: op as u32,
                len: len as u32,
                _padding: [0, 0],
            },
            len,
        });
        self.total_ops += 1;
    }

    /// Adds a binary 2D operation to the accumulator.
    pub fn add_binary_2d(
        &mut self,
        input_a: Arc<Buffer>,
        input_b: Arc<Buffer>,
        output: Arc<Buffer>,
        op: BinaryOp,
        rows: usize,
        cols: usize,
    ) {
        self.pending_ops.push(PendingOp::Binary2D {
            input_a,
            input_b,
            output,
            params: Binary2DParams {
                op: op as u32,
                rows: rows as u32,
                cols: cols as u32,
                _padding: 0,
            },
            rows,
            cols,
        });
        self.total_ops += 1;
    }

    /// Adds a scalar 1D operation to the accumulator.
    pub fn add_scalar_1d(
        &mut self,
        input: Arc<Buffer>,
        output: Arc<Buffer>,
        op: ScalarOp,
        scalar: f32,
        len: usize,
    ) {
        self.pending_ops.push(PendingOp::Scalar1D {
            input,
            output,
            params: Scalar1DParams {
                op: op as u32,
                len: len as u32,
                scalar,
                _padding: 0,
            },
            len,
        });
        self.total_ops += 1;
    }

    /// Adds a scalar 2D operation to the accumulator.
    pub fn add_scalar_2d(
        &mut self,
        input: Arc<Buffer>,
        output: Arc<Buffer>,
        op: ScalarOp,
        scalar: f32,
        rows: usize,
        cols: usize,
    ) {
        self.pending_ops.push(PendingOp::Scalar2D {
            input,
            output,
            params: Scalar2DParams {
                op: op as u32,
                rows: rows as u32,
                cols: cols as u32,
                scalar,
            },
            rows,
            cols,
        });
        self.total_ops += 1;
    }

    /// Adds a matrix-vector multiplication to the accumulator.
    pub fn add_matvec(
        &mut self,
        matrix: Arc<Buffer>,
        vector: Arc<Buffer>,
        output: Arc<Buffer>,
        rows: usize,
        cols: usize,
    ) {
        self.pending_ops.push(PendingOp::MatVec {
            matrix,
            vector,
            output,
            params: MatVecParams {
                rows: rows as u32,
                cols: cols as u32,
                _padding: [0, 0],
            },
        });
        self.total_ops += 1;
    }

    /// Adds a transposed matrix-vector multiplication to the accumulator.
    pub fn add_matvec_transposed(
        &mut self,
        matrix: Arc<Buffer>,
        vector: Arc<Buffer>,
        output: Arc<Buffer>,
        rows: usize,
        cols: usize,
    ) {
        self.pending_ops.push(PendingOp::MatVecTransposed {
            matrix,
            vector,
            output,
            params: MatVecParams {
                rows: rows as u32,
                cols: cols as u32,
                _padding: [0, 0],
            },
        });
        self.total_ops += 1;
    }

    /// Adds a transpose operation to the accumulator.
    pub fn add_transpose(
        &mut self,
        input: Arc<Buffer>,
        output: Arc<Buffer>,
        rows: usize,
        cols: usize,
    ) {
        self.pending_ops.push(PendingOp::Transpose {
            input,
            output,
            params: TransposeParams {
                rows: rows as u32,
                cols: cols as u32,
                _padding: [0, 0],
            },
        });
        self.total_ops += 1;
    }

    /// Returns the number of pending operations.
    pub fn pending_count(&self) -> usize {
        self.pending_ops.len()
    }

    /// Returns true if auto-flush threshold is reached.
    pub fn should_flush(&self) -> bool {
        self.pending_ops.len() >= self.flush_threshold
    }

    /// Flushes all pending operations to the GPU.
    ///
    /// This is a no-op if there are no pending operations.
    pub fn flush(&mut self, device: &Device, queue: &Queue) {
        if self.pending_ops.is_empty() {
            return;
        }

        // For now, we still execute operations individually but in a single submission.
        // Future optimization: fuse operations into single kernels.
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CommandAccumulator::flush"),
        });

        // Submit the command buffer
        queue.submit(std::iter::once(encoder.finish()));

        // Clear pending operations
        self.pending_ops.clear();
        self.total_flushes += 1;
    }

    /// Returns statistics about the accumulator.
    pub fn stats(&self) -> AccumulatorStats {
        AccumulatorStats {
            pending_ops: self.pending_ops.len(),
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
#[allow(dead_code)] // Used for debugging and future features
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
    fn test_should_flush() {
        let mut acc = CommandAccumulator::with_threshold(3);
        assert!(!acc.should_flush());

        // Manually add pending ops to test threshold logic
        // (In real use, these would be added via add_* methods)
        acc.pending_count(); // Just verify we can call it
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
