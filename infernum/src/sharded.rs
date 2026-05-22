//! Per-replica KV cache for tensor-parallel inference.
//!
//! [`ShardedKvCache`] is used by [`infernum_cuda::ShardedGraphEngine`] to
//! hold one inner KV cache per GPU replica.

/// Per-replica KV cache wrapper for sharded models.
///
/// Used by `ShardedGraphEngine` in `infernum_cuda` — each rank holds one
/// inner cache. The engine sees it as an opaque single cache.
pub struct ShardedKvCache<K> {
    pub(crate) inner: Vec<K>,
}

impl<K: Send> ShardedKvCache<K> {
    /// Iterate mutably over the per-rank caches.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut K> {
        self.inner.iter_mut()
    }
}

unsafe impl<K: Send> Send for ShardedKvCache<K> {}
