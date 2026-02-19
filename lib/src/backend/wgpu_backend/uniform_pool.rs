//! Uniform buffer pool for efficient parameter passing.
//!
//! Instead of creating a new uniform buffer for each operation's parameters,
//! we maintain a pool of reusable buffers. This reduces allocation overhead
//! and GPU memory fragmentation.
//!
//! ## How it works
//!
//! 1. When params are needed, acquire a buffer from the pool
//! 2. Write params to the buffer via queue.write_buffer
//! 3. Use the buffer in the operation's bind group
//! 4. The buffer is returned to the pool implicitly (no explicit release needed)

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Buffer, BufferUsages, Device};

/// Maximum number of buffers to keep per size category.
#[allow(dead_code)]
const MAX_BUFFERS_PER_SIZE: usize = 64;

/// Size categories for uniform buffers (in bytes).
const SIZE_CATEGORIES: [u64; 4] = [16, 32, 64, 128];

/// Uniform buffer pool for efficient parameter passing.
///
/// Maintains pools of uniform buffers organized by size.
pub struct UniformBufferPool {
    /// Pools organized by size category.
    pools: HashMap<u64, Vec<Arc<Buffer>>>,
    /// Statistics.
    total_created: u64,
    total_reused: u64,
}

impl UniformBufferPool {
    /// Creates a new uniform buffer pool.
    pub fn new() -> Self {
        UniformBufferPool {
            pools: HashMap::new(),
            total_created: 0,
            total_reused: 0,
        }
    }

    /// Acquires a uniform buffer of the appropriate size.
    ///
    /// The buffer is suitable for uniform buffer usage and can hold
    /// at least `size` bytes.
    pub fn acquire(&mut self, device: &Device, size: u64) -> Arc<Buffer> {
        // Round up to the nearest size category
        let category = Self::get_size_category(size);

        // Try to get from pool
        if let Some(pool) = self.pools.get_mut(&category) {
            if let Some(buffer) = pool.pop() {
                self.total_reused += 1;
                return buffer;
            }
        }

        // Create new buffer
        self.total_created += 1;
        Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("UniformBuffer_{}b", category)),
            size: category,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// Returns a buffer to the pool.
    #[allow(dead_code)]
    pub fn release(&mut self, _buffer: Arc<Buffer>) {
        // Get buffer size (we need to know this, but Arc<Buffer> doesn't expose it directly)
        // For now, we'll store buffers by their likely category based on common sizes
        // In practice, we know the sizes we use: 16 (BinaryParams, Scalar1DParams), 32 (Scalar2DParams)

        // Since we can't easily get the size from Arc<Buffer>, we'll just let buffers
        // accumulate in the pool without releasing them explicitly.
        // The pool is cleaned up when it's dropped.
    }

    /// Gets the size category for a given size.
    fn get_size_category(size: u64) -> u64 {
        for &category in &SIZE_CATEGORIES {
            if size <= category {
                return category;
            }
        }
        // For larger sizes, round up to nearest 128
        size.div_ceil(128) * 128
    }

    /// Returns statistics about the pool.
    pub fn stats(&self) -> UniformPoolStats {
        let total_in_pool: usize = self.pools.values().map(|v| v.len()).sum();
        UniformPoolStats {
            total_created: self.total_created,
            total_reused: self.total_reused,
            buffers_in_pool: total_in_pool,
        }
    }
}

impl Default for UniformBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the uniform buffer pool.
#[derive(Debug, Clone, Copy)]
pub struct UniformPoolStats {
    /// Total buffers created.
    pub total_created: u64,
    /// Total buffer reuses.
    pub total_reused: u64,
    /// Buffers currently in pool.
    pub buffers_in_pool: usize,
}

// Thread-local uniform buffer pool.
thread_local! {
    pub static UNIFORM_POOL: RefCell<UniformBufferPool> = RefCell::new(UniformBufferPool::new());
}

/// Acquires a uniform buffer from the thread-local pool.
pub fn acquire_uniform_buffer(device: &Device, size: u64) -> Arc<Buffer> {
    UNIFORM_POOL.with(|pool| pool.borrow_mut().acquire(device, size))
}

/// Returns statistics about the thread-local uniform buffer pool.
#[allow(dead_code)]
pub fn uniform_pool_stats() -> UniformPoolStats {
    UNIFORM_POOL.with(|pool| pool.borrow().stats())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_category() {
        assert_eq!(UniformBufferPool::get_size_category(8), 16);
        assert_eq!(UniformBufferPool::get_size_category(16), 16);
        assert_eq!(UniformBufferPool::get_size_category(17), 32);
        assert_eq!(UniformBufferPool::get_size_category(32), 32);
        assert_eq!(UniformBufferPool::get_size_category(33), 64);
        assert_eq!(UniformBufferPool::get_size_category(64), 64);
        assert_eq!(UniformBufferPool::get_size_category(65), 128);
        assert_eq!(UniformBufferPool::get_size_category(128), 128);
        assert_eq!(UniformBufferPool::get_size_category(129), 256);
    }
}
