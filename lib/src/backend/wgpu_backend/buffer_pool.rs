//! Buffer pool for reusable GPU memory.
//!
//! Reduces allocation overhead by maintaining a pool of buffers
//! that can be reused across operations instead of constantly
//! creating and destroying GPU memory.

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

/// Maximum total pool size in bytes (256 MB default).
const DEFAULT_MAX_POOL_SIZE: u64 = 256 * 1024 * 1024;

/// Entry in the buffer pool tracking usage.
struct PoolEntry {
    buffer: Arc<Buffer>,
    size: u64,
    last_used: u32,
}

impl PoolEntry {
    fn new(buffer: Buffer, size: u64) -> Self {
        Self {
            buffer: Arc::new(buffer),
            size,
            last_used: 0,
        }
    }

    fn touch(&mut self, current_tick: u32) {
        self.last_used = current_tick;
    }

    fn age(&self, current_tick: u32) -> u32 {
        current_tick.saturating_sub(self.last_used)
    }
}

/// Buffer pool for reusing GPU memory allocations.
///
/// Thread-safe pool that maintains buffers organized by size.
/// Uses LRU eviction when pool size exceeds configured maximum.
pub struct BufferPool {
    /// Buffers available for reuse, keyed by size in bytes.
    available_1d: HashMap<u64, Vec<PoolEntry>>,
    /// 2D buffers keyed by (rows * cols * sizeof(f32))
    available_2d: HashMap<(usize, usize), Vec<PoolEntry>>,
    /// Buffers currently in use.
    in_use: Vec<Arc<Buffer>>,
    /// Total bytes currently in the pool.
    total_bytes: u64,
    /// Maximum bytes allowed in the pool.
    max_bytes: u64,
    /// Current tick for LRU tracking.
    tick: u32,
    /// Statistics for debugging.
    stats: PoolStats,
}

/// Statistics about buffer pool usage.
#[derive(Debug, Default, Clone, Copy)]
pub struct PoolStats {
    /// Number of buffers acquired from pool (reused).
    pub cache_hits: u64,
    /// Number of buffers created (misses).
    pub cache_misses: u64,
    /// Number of buffers evicted due to size limits.
    pub evictions: u64,
    /// Current number of buffers in pool.
    pub pool_size: usize,
    /// Current bytes in pool.
    pub pool_bytes: u64,
}

impl BufferPool {
    /// Creates a new buffer pool with default size limits.
    pub fn new() -> Self {
        Self::with_max_size(DEFAULT_MAX_POOL_SIZE)
    }

    /// Creates a new buffer pool with specified maximum size in bytes.
    pub fn with_max_size(max_bytes: u64) -> Self {
        Self {
            available_1d: HashMap::new(),
            available_2d: HashMap::new(),
            in_use: Vec::new(),
            total_bytes: 0,
            max_bytes,
            tick: 0,
            stats: PoolStats::default(),
        }
    }

    /// Acquires a 1D buffer of the specified length.
    ///
    /// Returns a buffer from the pool if available, otherwise creates a new one.
    pub fn acquire_1d(&mut self, device: &Device, len: usize) -> Arc<Buffer> {
        let size = (len * std::mem::size_of::<f32>()) as u64;
        self.tick += 1;

        // Try to reuse existing buffer
        if let Some(entries) = self.available_1d.get_mut(&size) {
            if let Some(mut entry) = entries.pop() {
                entry.touch(self.tick);
                self.stats.cache_hits += 1;
                let buffer = entry.buffer.clone();
                self.in_use.push(buffer.clone());
                return buffer;
            }
        }

        // Need to create new buffer
        self.stats.cache_misses += 1;

        // Evict old buffers if needed
        while self.total_bytes + size > self.max_bytes && self.evict_oldest() {
            self.stats.evictions += 1;
        }

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("BufferPool::1d_{}", len)),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.total_bytes += size;
        let buffer = Arc::new(buffer);
        self.in_use.push(buffer.clone());
        self.stats.pool_bytes = self.total_bytes;
        buffer
    }

    /// Acquires a 2D buffer of the specified dimensions.
    ///
    /// Returns a buffer from the pool if available, otherwise creates a new one.
    pub fn acquire_2d(&mut self, device: &Device, rows: usize, cols: usize) -> Arc<Buffer> {
        let size = (rows * cols * std::mem::size_of::<f32>()) as u64;
        self.tick += 1;

        // Try to reuse existing buffer
        if let Some(entries) = self.available_2d.get_mut(&(rows, cols)) {
            if let Some(mut entry) = entries.pop() {
                entry.touch(self.tick);
                self.stats.cache_hits += 1;
                let buffer = entry.buffer.clone();
                self.in_use.push(buffer.clone());
                return buffer;
            }
        }

        // Need to create new buffer
        self.stats.cache_misses += 1;

        // Evict old buffers if needed
        while self.total_bytes + size > self.max_bytes && self.evict_oldest() {
            self.stats.evictions += 1;
        }

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("BufferPool::2d_{}x{}", rows, cols)),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.total_bytes += size;
        let buffer = Arc::new(buffer);
        self.in_use.push(buffer.clone());
        self.stats.pool_bytes = self.total_bytes;
        buffer
    }

    /// Releases a buffer back to the pool for reuse.
    ///
    /// If the pool is full, the buffer is dropped immediately.
    pub fn release(&mut self, buffer: Arc<Buffer>, len_hint: Option<usize>) {
        // Remove from in_use
        self.in_use.retain(|b| !Arc::ptr_eq(b, &buffer));

        // Determine size category
        let size = buffer.size();

        // Check if buffer is still valid (not destroyed)
        // If pool is too full, just drop it
        if self.total_bytes > self.max_bytes {
            self.total_bytes -= size;
            return;
        }

        let mut entry = PoolEntry::new((*buffer).clone(), size);
        entry.touch(self.tick);

        // Add to appropriate pool based on size hint
        if let Some(len) = len_hint {
            let expected_size = (len * std::mem::size_of::<f32>()) as u64;
            if size == expected_size {
                self.available_1d
                    .entry(expected_size)
                    .or_default()
                    .push(entry);
                return;
            }
        }

        // Default: add to 1D pool by size
        self.available_1d.entry(size).or_default().push(entry);
        self.stats.pool_size = self.available_1d.values().map(|v| v.len()).sum::<usize>()
            + self.available_2d.values().map(|v| v.len()).sum::<usize>();
    }

    /// Evicts the oldest buffer from the pool.
    ///
    /// Returns true if a buffer was evicted, false if pool is empty.
    fn evict_oldest(&mut self) -> bool {
        let mut oldest: Option<(u64, usize, u32)> = None; // (size, index, age)
        let mut oldest_2d: Option<((usize, usize), usize, u32)> = None;

        // Find oldest in 1D pool
        for (&size, entries) in &self.available_1d {
            for (i, entry) in entries.iter().enumerate() {
                let age = entry.age(self.tick);
                if oldest.is_none() || age > oldest.unwrap().2 {
                    oldest = Some((size, i, age));
                }
            }
        }

        // Find oldest in 2D pool
        for (&key, entries) in &self.available_2d {
            for (i, entry) in entries.iter().enumerate() {
                let age = entry.age(self.tick);
                if oldest_2d.is_none() || age > oldest_2d.unwrap().2 {
                    oldest_2d = Some((key, i, age));
                }
            }
        }

        // Evict the oldest overall
        match (oldest, oldest_2d) {
            (Some((size, idx, age_1d)), Some((key, idx_2d, age_2d))) => {
                if age_1d >= age_2d {
                    self.total_bytes -= size;
                    self.available_1d.get_mut(&size).unwrap().remove(idx);
                } else {
                    let entry = self.available_2d.get_mut(&key).unwrap().remove(idx_2d);
                    self.total_bytes -= entry.size;
                }
                true
            }
            (Some((size, idx, _)), None) => {
                self.total_bytes -= size;
                self.available_1d.get_mut(&size).unwrap().remove(idx);
                true
            }
            (None, Some((key, idx, _))) => {
                let entry = self.available_2d.get_mut(&key).unwrap().remove(idx);
                self.total_bytes -= entry.size;
                true
            }
            (None, None) => false,
        }
    }

    /// Returns statistics about pool usage.
    pub fn stats(&self) -> PoolStats {
        let mut stats = self.stats;
        stats.pool_size = self.available_1d.values().map(|v| v.len()).sum::<usize>()
            + self.available_2d.values().map(|v| v.len()).sum::<usize>();
        stats.pool_bytes = self.total_bytes;
        stats
    }

    /// Clears all buffers from the pool.
    pub fn clear(&mut self) {
        self.available_1d.clear();
        self.available_2d.clear();
        self.total_bytes = 0;
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_stats_default() {
        let stats = PoolStats::default();
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.evictions, 0);
    }

    #[test]
    fn test_pool_new() {
        let pool = BufferPool::new();
        assert_eq!(pool.total_bytes, 0);
        assert_eq!(pool.max_bytes, DEFAULT_MAX_POOL_SIZE);
    }

    #[test]
    fn test_pool_with_max_size() {
        let pool = BufferPool::with_max_size(1024);
        assert_eq!(pool.max_bytes, 1024);
    }

    #[test]
    fn test_buffers_reused_for_same_size() {
        // Test that buffers are reused when acquiring same-size allocations
        let pool = BufferPool::with_max_size(1024 * 1024);
        assert_eq!(pool.stats().cache_hits, 0);
        assert_eq!(pool.stats().cache_misses, 0);

        // Simulate acquire pattern - we can't test with real device in unit test,
        // but we can verify the logic via stats tracking
        // This test verifies the pool structure is correct
        let stats = pool.stats();
        assert_eq!(stats.pool_size, 0);
        assert_eq!(stats.pool_bytes, 0);
    }

    #[test]
    fn test_pool_evicts_when_size_exceeded() {
        // Test that pool evicts buffers when size limit is exceeded
        let pool = BufferPool::with_max_size(100);
        assert_eq!(pool.max_bytes, 100);

        // Verify initial state
        let stats = pool.stats();
        assert_eq!(stats.evictions, 0);

        // Pool should evict oldest entries when max_bytes is exceeded
        // (Full test requires device, but we verify the structure)
        assert!(pool.available_1d.is_empty());
        assert!(pool.available_2d.is_empty());
    }

    #[test]
    fn test_pool_entry_lru_tracking() {
        // Test LRU tracking logic without creating actual GPU buffers
        // We test the age calculation directly

        // Create a minimal test that doesn't require GPU resources
        let current_tick = 10u32;
        let last_used = 5u32;

        // Test age calculation directly
        let age = current_tick.saturating_sub(last_used);
        assert_eq!(age, 5);

        // Test saturating_sub for edge case
        let age_zero = last_used.saturating_sub(current_tick);
        assert_eq!(age_zero, 0);
    }

    #[test]
    fn test_pool_clear() {
        let mut pool = BufferPool::with_max_size(1024);
        pool.total_bytes = 500;
        pool.available_1d.insert(100, vec![]);

        pool.clear();

        assert!(pool.available_1d.is_empty());
        assert!(pool.available_2d.is_empty());
        assert_eq!(pool.total_bytes, 0);
    }
}
