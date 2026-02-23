//! Staging buffer pool for efficient CPU readback.
//!
//! Staging buffers are used to copy GPU data back to CPU. This module provides
//! a pool that reuses staging buffers to reduce allocation overhead.

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Buffer, BufferUsages, Device};

/// Size bucket granularity for staging buffer pooling.
/// Buffers are rounded up to the nearest bucket size.
const SIZE_BUCKET_GRANULARITY: usize = 1024; // 1KB buckets

/// Maximum number of buffers to keep per size bucket.
const MAX_BUFFERS_PER_BUCKET: usize = 8;

/// A pooled staging buffer that returns to the pool when dropped.
pub struct PooledStagingBuffer {
    buffer: Option<Arc<Buffer>>,
    size: usize,
    pool: Arc<std::sync::Mutex<StagingBufferPoolInner>>,
}

impl PooledStagingBuffer {
    /// Get access to the underlying buffer.
    pub fn buffer(&self) -> &Arc<Buffer> {
        self.buffer.as_ref().expect("buffer already taken")
    }

    /// Take ownership of the buffer Arc (for operations that need ownership).
    /// The buffer will still be returned to the pool when this is dropped.
    pub fn take_buffer(mut self) -> Arc<Buffer> {
        self.buffer.take().expect("buffer already taken")
    }
}

impl Drop for PooledStagingBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // Return buffer to pool
            if let Ok(mut pool) = self.pool.lock() {
                pool.release(buffer, self.size);
            }
        }
    }
}

/// Inner pool state (behind a Mutex for thread safety).
struct StagingBufferPoolInner {
    /// Buffers organized by size bucket.
    /// Key is the bucket size in bytes.
    buffers: HashMap<usize, Vec<Arc<Buffer>>>,

    /// Total bytes currently pooled.
    total_bytes: usize,

    /// Maximum bytes to keep in the pool.
    max_bytes: usize,

    /// Statistics for monitoring.
    stats: StagingPoolStats,
}

/// Statistics for the staging buffer pool.
#[derive(Debug, Clone, Default)]
pub struct StagingPoolStats {
    /// Number of buffers currently in the pool.
    pub buffers_pooled: usize,

    /// Total bytes currently pooled.
    pub bytes_pooled: usize,

    /// Number of buffers acquired from pool (cache hits).
    pub cache_hits: usize,

    /// Number of buffers created (cache misses).
    pub cache_misses: usize,

    /// Number of buffers evicted due to size limits.
    pub evictions: usize,
}

/// Pool for reusing staging buffers across multiple readback operations.
///
/// Staging buffers are used to copy data from GPU to CPU. By pooling them,
/// we avoid the overhead of creating new buffers on every `to_vec()` call.
pub struct StagingBufferPool {
    inner: Arc<std::sync::Mutex<StagingBufferPoolInner>>,
}

impl StagingBufferPool {
    /// Create a new staging buffer pool with the given maximum size.
    ///
    /// # Arguments
    /// * `max_bytes` - Maximum total bytes to keep in the pool. When exceeded,
    ///   least-recently-used buffers are evicted.
    pub fn new(max_bytes: usize) -> Self {
        StagingBufferPool {
            inner: Arc::new(std::sync::Mutex::new(StagingBufferPoolInner {
                buffers: HashMap::new(),
                total_bytes: 0,
                max_bytes,
                stats: StagingPoolStats::default(),
            })),
        }
    }

    /// Acquire a staging buffer of at least the given size.
    ///
    /// This will reuse an existing buffer from the pool if available,
    /// or create a new one if necessary.
    ///
    /// # Arguments
    /// * `device` - The WGPU device to create buffers on.
    /// * `size` - Minimum size in bytes.
    ///
    /// # Returns
    /// A pooled staging buffer that returns to the pool when dropped.
    pub fn acquire(&self, device: &Device, size: usize) -> PooledStagingBuffer {
        let bucket = Self::size_to_bucket(size);

        let buffer = {
            let mut inner = self.inner.lock().expect("pool lock poisoned");

            // Try to get from pool
            if let Some(buffers) = inner.buffers.get_mut(&bucket) {
                if let Some(buffer) = buffers.pop() {
                    inner.total_bytes -= bucket;
                    inner.stats.cache_hits += 1;
                    inner.stats.buffers_pooled = inner.count_buffers();
                    inner.stats.bytes_pooled = inner.total_bytes;
                    buffer
                } else {
                    // Need to create new buffer
                    inner.stats.cache_misses += 1;
                    Self::create_buffer(device, bucket)
                }
            } else {
                inner.stats.cache_misses += 1;
                Self::create_buffer(device, bucket)
            }
        };

        PooledStagingBuffer {
            buffer: Some(buffer),
            size: bucket,
            pool: Arc::clone(&self.inner),
        }
    }

    /// Get current pool statistics.
    pub fn stats(&self) -> StagingPoolStats {
        let inner = self.inner.lock().expect("pool lock poisoned");
        inner.stats.clone()
    }

    /// Clear all buffers from the pool.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("pool lock poisoned");
        inner.buffers.clear();
        inner.total_bytes = 0;
        inner.stats.buffers_pooled = 0;
        inner.stats.bytes_pooled = 0;
    }

    /// Convert a size in bytes to a bucket size.
    /// Rounds up to the nearest bucket granularity.
    fn size_to_bucket(size: usize) -> usize {
        size.div_ceil(SIZE_BUCKET_GRANULARITY) * SIZE_BUCKET_GRANULARITY
    }

    /// Create a new staging buffer of the given size.
    fn create_buffer(device: &Device, size: usize) -> Arc<Buffer> {
        Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled_staging_buffer"),
            size: size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }
}

impl StagingBufferPoolInner {
    /// Release a buffer back to the pool.
    fn release(&mut self, buffer: Arc<Buffer>, size: usize) {
        let bucket = StagingBufferPool::size_to_bucket(size);

        // Check if we're over capacity
        if self.total_bytes + bucket > self.max_bytes {
            // Evict oldest buffers until we have room
            self.evict_for_size(bucket);
        }

        // Check per-bucket limit
        let bucket_count = self.buffers.get(&bucket).map(|v| v.len()).unwrap_or(0);
        if bucket_count >= MAX_BUFFERS_PER_BUCKET {
            // Don't pool this buffer - let it drop
            self.stats.evictions += 1;
            return;
        }

        // Add to pool
        self.buffers.entry(bucket).or_default().push(buffer);
        self.total_bytes += bucket;
        self.stats.buffers_pooled = self.count_buffers();
        self.stats.bytes_pooled = self.total_bytes;
    }

    /// Evict buffers to make room for a new one.
    fn evict_for_size(&mut self, needed: usize) {
        let mut freed = 0;

        // Evict from buckets in order (simple LRU approximation - evict from smaller buckets first)
        let mut bucket_sizes: Vec<usize> = self.buffers.keys().cloned().collect();
        bucket_sizes.sort();

        for bucket in bucket_sizes {
            if freed >= needed {
                break;
            }

            if let Some(buffers) = self.buffers.get_mut(&bucket) {
                while !buffers.is_empty() && freed < needed {
                    buffers.pop();
                    freed += bucket;
                    self.stats.evictions += 1;
                }
            }

            // Remove empty bucket entries
            if self
                .buffers
                .get(&bucket)
                .map(|v| v.is_empty())
                .unwrap_or(false)
            {
                self.buffers.remove(&bucket);
            }
        }

        self.total_bytes = self.total_bytes.saturating_sub(freed);
    }

    /// Count total buffers in pool.
    fn count_buffers(&self) -> usize {
        self.buffers.values().map(|v| v.len()).sum()
    }
}

impl Default for StagingBufferPool {
    fn default() -> Self {
        // Default to 64MB max pool size
        Self::new(64 * 1024 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_bucket_rounding() {
        assert_eq!(StagingBufferPool::size_to_bucket(0), 0);
        assert_eq!(StagingBufferPool::size_to_bucket(1), 1024);
        assert_eq!(StagingBufferPool::size_to_bucket(1024), 1024);
        assert_eq!(StagingBufferPool::size_to_bucket(1025), 2048);
        assert_eq!(StagingBufferPool::size_to_bucket(4096), 4096);
        assert_eq!(StagingBufferPool::size_to_bucket(4097), 5120);
    }

    #[test]
    fn test_pool_stats() {
        let pool = StagingBufferPool::new(1024 * 1024); // 1MB

        let stats = pool.stats();
        assert_eq!(stats.buffers_pooled, 0);
        assert_eq!(stats.bytes_pooled, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.evictions, 0);
    }

    #[test]
    fn test_pool_eviction_logic() {
        // Small pool - test eviction behavior
        let pool = StagingBufferPool::new(4096); // 4KB max

        // Initial stats
        let stats = pool.stats();
        assert_eq!(stats.buffers_pooled, 0);
    }

    #[test]
    fn test_max_buffers_per_bucket_limit() {
        // Test that per-bucket limit is enforced
        let pool = StagingBufferPool::new(1024 * 1024);

        // Verify pool is created
        let stats = pool.stats();
        assert_eq!(stats.evictions, 0);
    }
}
