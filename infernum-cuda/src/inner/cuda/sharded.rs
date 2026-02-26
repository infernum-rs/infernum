//! Generic sharded model wrapper for tensor-parallel inference.
//!
//! [`ShardedModel`] wraps N copies of any [`ShardedLoadable`] model (one per
//! GPU) and implements [`Model`] by running all replicas in lock-step via
//! scoped threads. NCCL all-reduce happens inside each replica's forward
//! call; `ShardedModel` itself is only responsible for thread dispatch and
//! returning rank 0's logits.

use std::path::Path;
use std::sync::Arc;
use std::thread;

use super::nccl::NcclCommunicator;
use super::BlockTable;
use super::{CudaContext, CudaTensor, PagedKvCache, ShardConfig};
use crate::inner::backend_impl::CudaBackend;
use crate::inner::cuda_logits::CudaLogits;
use crate::inner::cuda_runtime_state::CudaRuntimeState;
use crate::model::{Model, ShardedLoadable};
use infernum::Result;

/// A model sharded across multiple GPUs for tensor-parallel inference.
///
/// Wraps N replicas of `M`, each holding a shard of the model weights on a
/// different GPU. Implements [`Model`] so it can be used with a standard
/// [`Engine`](infernum_runtime::Engine) — the caller doesn't need to know
/// about sharding.
pub struct ShardedModel<M: Model> {
    replicas: Vec<(CudaContext, M)>,
}

impl<M: ShardedLoadable> ShardedModel<M> {
    /// Load a sharded model across `num_gpus` GPUs.
    ///
    /// Creates CUDA devices 0..`num_gpus`, establishes NCCL communicators,
    /// and loads one shard per GPU in parallel threads.
    ///
    /// # Errors
    /// Returns an error if device creation, NCCL initialisation, or weight
    /// loading fails on any GPU.
    #[allow(clippy::missing_panics_doc)]
    pub fn from_pretrained(model_path: impl AsRef<Path>, num_gpus: usize) -> Result<Self> {
        let model_path = model_path.as_ref();
        // Generate a shared NCCL ID on the main thread, then let each worker
        // thread create its own CudaContext + NcclCommunicator via from_rank.
        // This ensures each comm is initialised on the same thread (and CUDA
        // context) that will later call all-reduce, avoiding ncclInvalidUsage.
        let nccl_id = super::nccl::NcclId::new()?;

        let replicas: Vec<(CudaContext, M)> = thread::scope(|s| {
            let handles: Vec<_> = (0..num_gpus)
                .map(|rank| {
                    let world_size = num_gpus;
                    s.spawn(move || {
                        let ctx = CudaContext::new(rank)?;
                        let comm = NcclCommunicator::from_rank(
                            Arc::clone(ctx.device()),
                            rank,
                            world_size,
                            nccl_id,
                        )?;
                        let shard = ShardConfig { rank, world_size };
                        let model = M::load_shard(&ctx, model_path, shard, comm)?;
                        Ok::<_, crate::Error>((ctx, model))
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| h.join().expect("GPU thread panicked"))
                .collect::<Result<Vec<_>>>()
        })?;

        Ok(Self { replicas })
    }
}

impl<M: Model + Send + Sync> Model for ShardedModel<M> {
    fn config(&self) -> crate::model::ModelConfig {
        self.replicas[0].1.config()
    }

    fn devices(&self) -> Vec<&CudaContext> {
        self.replicas.iter().map(|(ctx, _)| ctx).collect()
    }

    fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor> {
        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .map(|(_, model)| s.spawn(move || model.forward(input_ids)))
                .collect();

            collect_rank0(handles)
        })
    }

    fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        paged_kvs: &mut [PagedKvCache],
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor> {
        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .zip(paged_kvs.iter_mut())
                .map(|((_, model), kv)| {
                    s.spawn(move || {
                        model.forward_batch_decode(
                            token_ids,
                            std::slice::from_mut(kv),
                            block_tables,
                            positions,
                        )
                    })
                })
                .collect();

            collect_rank0(handles)
        })
    }

    fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kvs: &mut [PagedKvCache],
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<CudaTensor> {
        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .zip(paged_kvs.iter_mut())
                .map(|((_, model), kv)| {
                    s.spawn(move || {
                        model.forward_prefill_paged(
                            input_ids,
                            std::slice::from_mut(kv),
                            block_table,
                            start_pos,
                        )
                    })
                })
                .collect();

            collect_rank0(handles)
        })
    }
}

// --- infernum::Model trait implementation (backend-generic) ---

/// Per-replica KV cache wrapper for sharded models.
///
/// `ShardedModel`'s `infernum::Model::KvCache` wraps one inner cache
/// per GPU replica. The engine sees it as an opaque single cache.
pub struct ShardedKvCache<K> {
    inner: Vec<K>,
}

impl<K: Send> ShardedKvCache<K> {
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut K> {
        self.inner.iter_mut()
    }
}

unsafe impl<K: Send> Send for ShardedKvCache<K> {}

impl<M> infernum::Model for ShardedModel<M>
where
    M: Model + infernum::Model<B = CudaBackend, KvCache = PagedKvCache> + Send + Sync,
{
    type B = CudaBackend;
    type KvCache = ShardedKvCache<PagedKvCache>;

    fn config(&self) -> infernum::ModelConfig {
        Model::config(&self.replicas[0].1)
    }

    fn allocate_kv_cache(&self, block_config: &infernum::BlockConfig) -> Result<Self::KvCache> {
        let mut caches = Vec::with_capacity(self.replicas.len());
        for (_, model) in &self.replicas {
            caches.push(infernum::Model::allocate_kv_cache(model, block_config)?);
        }
        Ok(ShardedKvCache { inner: caches })
    }

    fn forward(&self, input_ids: &[u32]) -> Result<CudaLogits> {
        let tensor: CudaTensor = thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .map(|(_, model)| s.spawn(move || Model::forward(model, input_ids)))
                .collect();

            collect_rank0(handles)
        })?;
        Ok(CudaLogits::new(tensor))
    }

    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut CudaRuntimeState,
        block_table: &infernum::BlockTable,
        start_pos: usize,
    ) -> Result<CudaLogits> {
        let tensor: CudaTensor = thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .zip(kv_cache.iter_mut())
                .map(|((_, model), kv)| {
                    s.spawn(move || {
                        Model::forward_prefill_paged(
                            model,
                            input_ids,
                            std::slice::from_mut(kv),
                            block_table,
                            start_pos,
                        )
                    })
                })
                .collect();

            collect_rank0(handles)
        })?;
        Ok(CudaLogits::new(tensor))
    }

    fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut CudaRuntimeState,
        block_tables: &[infernum::BlockTable],
        positions: &[usize],
    ) -> Result<CudaLogits> {
        let tensor: CudaTensor = thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .zip(kv_cache.iter_mut())
                .map(|((_, model), kv)| {
                    s.spawn(move || {
                        Model::forward_batch_decode(
                            model,
                            token_ids,
                            std::slice::from_mut(kv),
                            block_tables,
                            positions,
                        )
                    })
                })
                .collect();

            collect_rank0(handles)
        })?;
        Ok(CudaLogits::new(tensor))
    }
}

/// Collect results from parallel threads, returning rank 0's `Result`.
///
/// Other ranks' results are discarded — NCCL all-reduce has already made
/// the logits identical across all devices.
fn collect_rank0<T>(handles: Vec<thread::ScopedJoinHandle<'_, Result<T>>>) -> Result<T> {
    let results: Vec<Result<T>> = handles
        .into_iter()
        .map(|h| h.join().expect("GPU thread panicked"))
        .collect();

    results.into_iter().next().expect("no rank 0 in handles")
}
