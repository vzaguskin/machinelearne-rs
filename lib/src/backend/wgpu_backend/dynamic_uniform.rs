//! Dynamic uniform buffer for efficient bind group reuse.
//!
//! Instead of creating a new uniform buffer for each operation's parameters,
//! we use a single large buffer and write params to different offsets.
//! This allows us to reuse bind groups by only changing the dynamic offset.
//!
//! ## How it works
//!
//! 1. Allocate a large uniform buffer (e.g., 64KB)
//! 2. Write params to the buffer at sequential offsets
//! 3. Use `set_bind_group` with dynamic offsets to reference the params
//! 4. Reset the buffer when it's full (after flush)
//!
//! ## Benefits
//!
//! - Reduces bind group creation (can reuse bind groups with different offsets)
//! - Reduces buffer allocation overhead
//! - Better cache locality for params

use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

/// Size of the dynamic uniform buffer (64KB - should handle ~250 operations with 256-byte alignment)
const DYNAMIC_BUFFER_SIZE: u64 = 64 * 1024;

/// Alignment for uniform buffer offsets (256 bytes is the WebGPU minimum)
pub const UNIFORM_ALIGNMENT: u64 = 256;

/// Dynamic uniform buffer manager.
///
/// Manages a single large GPU buffer for uniform parameters.
/// Parameters are written sequentially and referenced by offset.
pub struct DynamicUniformBuffer {
    /// The underlying GPU buffer.
    buffer: Arc<Buffer>,
    /// Current write offset.
    offset: u64,
    /// Number of operations since last reset.
    operation_count: u64,
    /// Pending writes (offset, data) - batched until flush.
    pending_writes: Vec<(u64, Vec<u8>)>,
}

impl DynamicUniformBuffer {
    /// Creates a new dynamic uniform buffer.
    pub fn new(device: &Device) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DynamicUniformBuffer"),
            size: DYNAMIC_BUFFER_SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        DynamicUniformBuffer {
            buffer: Arc::new(buffer),
            offset: 0,
            operation_count: 0,
            pending_writes: Vec::new(),
        }
    }

    /// Allocates space for params and returns the offset.
    ///
    /// The data is queued for writing and will be written on the next flush.
    /// Returns `None` if the buffer is full and needs to be reset.
    pub fn allocate(&mut self, data: &[u8]) -> Option<u32> {
        // Align the size to UNIFORM_ALIGNMENT
        let aligned_size = Self::align_size(data.len() as u64);

        if self.offset + aligned_size > DYNAMIC_BUFFER_SIZE {
            return None;
        }

        let current_offset = self.offset;

        // Queue the write
        self.pending_writes.push((current_offset, data.to_vec()));

        self.offset += aligned_size;
        self.operation_count += 1;

        Some(current_offset as u32)
    }

    /// Writes all pending data to the GPU buffer.
    pub fn flush_writes(&self, queue: &Queue) {
        for (offset, data) in &self.pending_writes {
            queue.write_buffer(&self.buffer, *offset, data);
        }
    }

    /// Aligns size to uniform buffer alignment requirement.
    fn align_size(size: u64) -> u64 {
        ((size + UNIFORM_ALIGNMENT - 1) / UNIFORM_ALIGNMENT) * UNIFORM_ALIGNMENT
    }

    /// Resets the buffer for reuse (call after GPU operations complete).
    pub fn reset(&mut self) {
        self.offset = 0;
        self.operation_count = 0;
        self.pending_writes.clear();
    }

    /// Returns the current offset (for debugging).
    pub fn current_offset(&self) -> u64 {
        self.offset
    }

    /// Returns the number of operations since last reset.
    pub fn operation_count(&self) -> u64 {
        self.operation_count
    }

    /// Returns the buffer utilization (0.0 to 1.0).
    pub fn utilization(&self) -> f32 {
        self.offset as f32 / DYNAMIC_BUFFER_SIZE as f32
    }

    /// Returns the underlying buffer.
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// Returns true if the buffer has pending writes.
    pub fn has_pending_writes(&self) -> bool {
        !self.pending_writes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_size() {
        assert_eq!(DynamicUniformBuffer::align_size(16), 256);
        assert_eq!(DynamicUniformBuffer::align_size(256), 256);
        assert_eq!(DynamicUniformBuffer::align_size(257), 512);
        assert_eq!(DynamicUniformBuffer::align_size(512), 512);
    }

    #[test]
    fn test_utilization() {
        // This test just checks the math works
        let util: f64 = 256.0 / (64.0 * 1024.0);
        assert!((util - 0.0039).abs() < 0.001);
    }
}
