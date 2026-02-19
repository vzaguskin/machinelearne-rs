//! Bind group cache for reducing GPU API overhead.
//!
//! Each GPU operation requires creating a bind group that binds buffers
//! to shader slots. Creating bind groups is expensive (GPU driver overhead).
//!
//! This cache stores bind groups keyed by their buffer pointers, allowing
//! reuse when the same tensor combination appears again.
//!
//! ## Cache Key Strategy
//!
//! Keys are based on:
//! - Pipeline layout pointer (identifies the shader/bind group layout)
//! - Buffer pointers (identifies the specific tensor data)
//!
//! ## Limitations
//!
//! Since output buffers are typically new for each operation, cache hit rate
//! depends on reusing the same input tensors. This happens in:
//! - Training loops (same weights across batches)
//! - Multiple operations on the same tensor

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, Buffer};

/// Maximum cached bind groups before eviction.
const MAX_CACHE_SIZE: usize = 256;

/// Key for bind group cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BindGroupKey {
    /// Pipeline layout pointer as usize.
    layout_ptr: usize,
    /// Hashed buffer pointers.
    buffer_hash: BufferPtrHash,
}

/// Hashed representation of buffer pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BufferPtrHash(u64);

impl BufferPtrHash {
    /// Creates a hash from buffer pointers.
    fn from_buffers(buffers: &[&Buffer]) -> Self {
        // Simple hash combining buffer pointers
        let mut hash: u64 = 0;
        for buffer in buffers {
            // Rotate and XOR to combine hashes
            hash = hash.rotate_left(7);
            hash ^= *buffer as *const Buffer as usize as u64;
        }
        BufferPtrHash(hash)
    }
}

/// LRU entry for bind group cache.
struct CacheEntry {
    /// The cached bind group.
    bind_group: Arc<BindGroup>,
    /// Access generation (for LRU eviction).
    generation: u64,
}

/// Bind group cache with LRU eviction.
pub struct BindGroupCache {
    /// Cache storage.
    cache: HashMap<BindGroupKey, CacheEntry>,
    /// Current generation (incremented on each access).
    generation: u64,
    /// Statistics.
    hits: u64,
    misses: u64,
}

impl BindGroupCache {
    /// Creates a new bind group cache.
    pub fn new() -> Self {
        BindGroupCache {
            cache: HashMap::new(),
            generation: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Gets or creates a bind group.
    ///
    /// If a matching bind group exists in the cache, returns it.
    /// Otherwise, calls the factory function to create a new one and caches it.
    pub fn get_or_create<F>(
        &mut self,
        layout: &BindGroupLayout,
        buffers: &[&Buffer],
        factory: F,
    ) -> Arc<BindGroup>
    where
        F: FnOnce() -> BindGroup,
    {
        let key = BindGroupKey {
            layout_ptr: layout as *const BindGroupLayout as usize,
            buffer_hash: BufferPtrHash::from_buffers(buffers),
        };

        self.generation += 1;

        if let Some(entry) = self.cache.get_mut(&key) {
            entry.generation = self.generation;
            self.hits += 1;
            return entry.bind_group.clone();
        }

        self.misses += 1;

        // Check if we need to evict
        if self.cache.len() >= MAX_CACHE_SIZE {
            self.evict_lru();
        }

        // Create new bind group
        let bind_group = Arc::new(factory());
        self.cache.insert(
            key,
            CacheEntry {
                bind_group: bind_group.clone(),
                generation: self.generation,
            },
        );

        bind_group
    }

    /// Evicts the least recently used entry.
    fn evict_lru(&mut self) {
        if self.cache.is_empty() {
            return;
        }

        // Find the entry with the oldest generation
        let oldest_key = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.generation)
            .map(|(key, _)| *key);

        if let Some(key) = oldest_key {
            self.cache.remove(&key);
        }
    }

    /// Returns cache statistics.
    pub fn stats(&self) -> BindGroupCacheStats {
        BindGroupCacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.cache.len(),
        }
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Default for BindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the bind group cache.
#[derive(Debug, Clone, Copy)]
pub struct BindGroupCacheStats {
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
    /// Current entries in cache.
    pub entries: usize,
}

impl BindGroupCacheStats {
    /// Returns the cache hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_ptr_hash() {
        // Two different hash combinations should produce different results
        let hash1 = BufferPtrHash(123);
        let hash2 = BufferPtrHash(456);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_cache_stats() {
        let stats = BindGroupCacheStats {
            hits: 75,
            misses: 25,
            entries: 10,
        };
        assert!((stats.hit_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_zero() {
        let stats = BindGroupCacheStats {
            hits: 0,
            misses: 0,
            entries: 0,
        };
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_new() {
        let cache = BindGroupCache::new();
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn test_cache_clear() {
        let cache = BindGroupCache::new();
        // Clear on empty cache should work
        let mut cache = cache;
        cache.clear();
        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn test_cache_default() {
        let cache = BindGroupCache::default();
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_bind_group_key_equality() {
        let key1 = BindGroupKey {
            layout_ptr: 0x1000,
            buffer_hash: BufferPtrHash(123),
        };
        let key2 = BindGroupKey {
            layout_ptr: 0x1000,
            buffer_hash: BufferPtrHash(123),
        };
        let key3 = BindGroupKey {
            layout_ptr: 0x2000,
            buffer_hash: BufferPtrHash(123),
        };
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_buffer_ptr_hash_equality() {
        let hash1 = BufferPtrHash(123);
        let hash2 = BufferPtrHash(123);
        let hash3 = BufferPtrHash(456);
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_cache_stats_all_hits() {
        let stats = BindGroupCacheStats {
            hits: 100,
            misses: 0,
            entries: 5,
        };
        assert!((stats.hit_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_all_misses() {
        let stats = BindGroupCacheStats {
            hits: 0,
            misses: 100,
            entries: 5,
        };
        assert!((stats.hit_rate() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_buffer_ptr_hash_from_empty() {
        // Empty buffer slice should produce consistent hash
        let hash1 = BufferPtrHash::from_buffers(&[]);
        let hash2 = BufferPtrHash::from_buffers(&[]);
        assert_eq!(hash1, hash2);
    }
}
