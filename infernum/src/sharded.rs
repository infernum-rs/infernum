//! Backend-agnostic sharded model for tensor-parallel inference.
//!
//! [`ShardedModel`] wraps N copies of any model (one per device) and
//! implements [`Model`](crate::Model) by running all replicas in lock-step
//! via scoped threads. All-reduce happens inside each replica's forward
//! call via the backend's [`Comm`](crate::Comm); `ShardedModel` itself is
//! only responsible for thread dispatch and returning rank 0's logits.

use std::path::Path;
use std::thread;

use crate::backend::MultiDeviceOps;
use crate::block_allocator::{BlockConfig, BlockTable};
use crate::model::{Model, ModelConfig};
use crate::runtime_state::RuntimeStateInit;
use crate::shard::ShardConfig;
use crate::Result;

/// A model sharded across multiple devices for tensor-parallel inference.
///
/// Wraps N replicas of `M`, each holding a shard of the model weights on a
/// different device. Implements [`Model`] so it can be used with a standard
/// [`Engine`](infernum_runtime::Engine) — the caller doesn't need to know
/// about sharding.
pub struct ShardedModel<B: MultiDeviceOps, M: Send> {
    replicas: Vec<(B::DeviceHandle, M)>,
}

impl<B: MultiDeviceOps, M: Send> ShardedModel<B, M> {
    /// Load a sharded model across `num_devices` devices.
    ///
    /// Creates one device per rank, establishes communicators, and loads
    /// one shard per device in parallel threads using the provided closure.
    ///
    /// The closure receives `(device, shard_config, communicator)` and
    /// returns a loaded model shard.
    ///
    /// # Errors
    /// Returns an error if device creation, communicator setup, or weight
    /// loading fails on any device.
    ///
    /// # Panics
    /// Panics if a device thread panics.
    pub fn new<F>(num_devices: usize, load_shard: F) -> Result<Self>
    where
        F: Fn(&B::DeviceHandle, ShardConfig, B::Comm) -> Result<M> + Send + Sync,
    {
        let comm_id = B::create_comm_id()?;

        let replicas = thread::scope(|s| {
            let handles: Vec<_> = (0..num_devices)
                .map(|rank| {
                    let load_shard = &load_shard;
                    s.spawn(move || {
                        let device = B::create_device(rank)?;
                        let comm = B::create_comm(&device, rank, num_devices, comm_id)?;
                        let shard = ShardConfig {
                            rank,
                            world_size: num_devices,
                        };
                        let model = load_shard(&device, shard, comm)?;
                        Ok::<_, crate::Error>((device, model))
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| h.join().expect("device thread panicked"))
                .collect::<Result<Vec<_>>>()
        })?;

        Ok(Self { replicas })
    }

    /// Load a sharded model from a pretrained directory.
    ///
    /// Convenience wrapper that takes a model path and a closure matching
    /// the common `(device, model_path, shard_config, comm) -> Model`
    /// pattern.
    ///
    /// # Errors
    /// Returns an error if any shard fails to load.
    ///
    /// # Panics
    /// Panics if a device thread panics.
    pub fn from_pretrained<F>(
        model_path: impl AsRef<Path>,
        num_devices: usize,
        load_shard: F,
    ) -> Result<Self>
    where
        F: Fn(&B::DeviceHandle, &Path, ShardConfig, B::Comm) -> Result<M> + Send + Sync,
    {
        let model_path = model_path.as_ref();
        Self::new(num_devices, |device, shard, comm| {
            load_shard(device, model_path, shard, comm)
        })
    }
}

// --- Per-replica KV cache wrapper ---

/// Per-replica KV cache wrapper for sharded models.
///
/// `ShardedModel`'s `Model::KvCache` wraps one inner cache per device
/// replica. The engine sees it as an opaque single cache.
pub struct ShardedKvCache<K> {
    inner: Vec<K>,
}

impl<K: Send> ShardedKvCache<K> {
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut K> {
        self.inner.iter_mut()
    }
}

unsafe impl<K: Send> Send for ShardedKvCache<K> {}

// --- Model implementation ---

impl<B, M> Model for ShardedModel<B, M>
where
    B: MultiDeviceOps,
    M: Model<B = B> + Send + Sync,
    M::KvCache: Send,
{
    type B = B;
    type KvCache = ShardedKvCache<M::KvCache>;

    fn device(&self) -> &B::DeviceHandle {
        &self.replicas[0].0
    }

    fn config(&self) -> ModelConfig {
        Model::config(&self.replicas[0].1)
    }

    fn allocate_kv_cache(&self, block_config: &BlockConfig) -> Result<Self::KvCache> {
        let mut caches = Vec::with_capacity(self.replicas.len());
        for (_, model) in &self.replicas {
            caches.push(Model::allocate_kv_cache(model, block_config)?);
        }
        Ok(ShardedKvCache { inner: caches })
    }

    fn forward(&self, input_ids: &[u32]) -> Result<B::Logits> {
        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .map(|(_, model)| s.spawn(move || Model::forward(model, input_ids)))
                .collect();

            collect_rank0(handles)
        })
    }

    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut B::RuntimeState,
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<B::Logits> {
        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .zip(kv_cache.iter_mut())
                .map(|((_, model), kv)| {
                    s.spawn(move || {
                        let mut state = B::RuntimeState::new_placeholder();
                        Model::forward_prefill(
                            model,
                            input_ids,
                            kv,
                            &mut state,
                            block_table,
                            start_pos,
                        )
                    })
                })
                .collect();

            collect_rank0(handles)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode(
        &self,
        token_ids: &B::Tensor,
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut B::RuntimeState,
        block_tables: &B::Tensor,
        seq_lens: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<B::Logits> {
        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .zip(kv_cache.iter_mut())
                .map(|((_, model), kv)| {
                    s.spawn(move || {
                        let mut state = B::RuntimeState::new_placeholder();
                        Model::forward_batch_decode(
                            model,
                            token_ids,
                            kv,
                            &mut state,
                            block_tables,
                            seq_lens,
                            positions,
                            batch_size,
                            max_blocks_per_seq,
                            max_seq_len,
                        )
                    })
                })
                .collect();

            collect_rank0(handles)
        })
    }
}

/// Collect results from parallel threads, returning rank 0's `Result`.
///
/// Other ranks' results are discarded — all-reduce has already made
/// the logits identical across all devices.
fn collect_rank0<T>(handles: Vec<thread::ScopedJoinHandle<'_, Result<T>>>) -> Result<T> {
    let results: Vec<Result<T>> = handles
        .into_iter()
        .map(|h| h.join().expect("device thread panicked"))
        .collect();

    results.into_iter().next().expect("no rank 0 in handles")
}
