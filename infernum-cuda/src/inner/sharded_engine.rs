//! Tensor-parallel inference engine for multi-GPU setups.
//!
//! Provides [`ShardedGraphEngine<C>`], which wraps N [`CudaGraphEngine<C>`]
//! instances (one per GPU), establishes NCCL communicators between them, and
//! fans out forward calls via scoped threads.
//!
//! Each replica holds a shard of the model weights (column-parallel and
//! row-parallel splits loaded by `CudaGraphEngine::from_config_comm_and_dir`).
//! All-reduce synchronisation happens inside each replica's graph execution
//! at the `AllReduceSumOp` nodes injected by the graph builder.

#[cfg(feature = "nccl")]
mod inner {
    use std::path::Path;
    use std::thread;

    use infernum::backend::MultiDeviceOps as _;
    use infernum::block_allocator::{BlockConfig, BlockTable};
    use infernum::model::{Model, ModelConfig};
    use infernum::runtime_state::RuntimeStateInit as _;
    use infernum::shard::ShardConfig;
    use infernum::{DType, Result};

    use crate::cuda::PagedKvCache;
    use crate::cuda_graph_engine::CudaGraphEngineConfig;
    use crate::cuda_logits::CudaLogits;
    use crate::inner::cuda_graph_engine::CudaGraphEngine;
    use crate::{CudaBackend, CudaRuntimeState};

    /// Tensor-parallel CUDA graph engine spanning multiple GPUs.
    ///
    /// Implements [`infernum::Model`] so it can be used with
    /// `infernum_runtime::Runtime` in place of a single-GPU `CudaGraphEngine`.
    ///
    /// ## Construction
    ///
    /// Use [`ShardedGraphEngine::from_pretrained`] — it creates one
    /// `CudaGraphEngine` per GPU, wires up NCCL communicators, and loads the
    /// sharded weights on each device in parallel.
    pub struct ShardedGraphEngine<C: CudaGraphEngineConfig + Clone> {
        replicas: Vec<CudaGraphEngine<C>>,
    }

    // SAFETY: CudaGraphEngine is Send (its CUDA handles are thread-safe with
    // proper synchronisation). Each replica is accessed only from its own
    // scoped thread inside the forward methods.
    unsafe impl<C: CudaGraphEngineConfig + Clone> Send for ShardedGraphEngine<C> {}
    unsafe impl<C: CudaGraphEngineConfig + Clone> Sync for ShardedGraphEngine<C> {}

    impl<C: CudaGraphEngineConfig + Clone> ShardedGraphEngine<C> {
        /// Load a tensor-parallel model from a `SafeTensors` directory.
        ///
        /// Creates one CUDA context per device, establishes an NCCL
        /// communicator group, and loads each shard in a dedicated thread.
        ///
        /// # Errors
        ///
        /// Returns an error if device creation, NCCL setup, or weight loading
        /// fails on any device.
        ///
        /// # Panics
        ///
        /// Panics if a device thread panics.
        pub fn from_pretrained(config: C, num_devices: usize, model_dir: &Path) -> Result<Self> {
            let comm_id = CudaBackend::create_comm_id()?;
            let comm_id_raw = *comm_id.to_raw();
            let model_dir = model_dir.to_owned();

            let replicas = thread::scope(|s| {
                let handles: Vec<_> = (0..num_devices)
                    .map(|rank| {
                        let config = config.clone();
                        let model_dir = model_dir.clone();
                        s.spawn(move || {
                            let ctx = CudaBackend::create_device(rank)?;
                            let comm_id = crate::cuda::NcclId::from_raw(comm_id_raw);
                            let comm = CudaBackend::create_comm(&ctx, rank, num_devices, comm_id)?;
                            let shard = ShardConfig {
                                rank,
                                world_size: num_devices,
                            };
                            CudaGraphEngine::from_config_comm_and_dir(
                                config, ctx, comm, shard, &model_dir,
                            )
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
    }

    /// Per-rank KV cache wrapper for [`ShardedGraphEngine`].
    pub struct ShardedPagedKvCache {
        pub(crate) inner: Vec<PagedKvCache>,
    }

    unsafe impl Send for ShardedPagedKvCache {}

    impl<C: CudaGraphEngineConfig + Clone + Sync> Model for ShardedGraphEngine<C> {
        type B = CudaBackend;
        type KvCache = ShardedPagedKvCache;

        fn config(&self) -> ModelConfig {
            let c = self.replicas[0].config();
            ModelConfig {
                num_layers: c.num_hidden_layers(),
                max_seq_len: c.max_position_embeddings(),
                num_kv_heads: c.num_kv_heads(),
                head_dim: c.head_dim(),
                eos_token_id: c.eos_token_id(),
                cache_dtype: DType::BF16,
            }
        }

        fn device(&self) -> &crate::cuda::CudaContext {
            self.replicas[0].cuda_context()
        }

        fn allocate_kv_cache(&self, block_config: &BlockConfig) -> Result<ShardedPagedKvCache> {
            let inner = self
                .replicas
                .iter()
                .map(|e| e.allocate_kv_cache(block_config))
                .collect::<Result<Vec<_>>>()?;
            Ok(ShardedPagedKvCache { inner })
        }

        fn forward(&self, input_ids: &[u32]) -> Result<CudaLogits> {
            thread::scope(|s| {
                let handles: Vec<_> = self
                    .replicas
                    .iter()
                    .map(|e| s.spawn(move || Model::forward(e, input_ids)))
                    .collect();
                collect_rank0(handles)
            })
        }

        fn forward_prefill(
            &self,
            input_ids: &[u32],
            kv_cache: &mut ShardedPagedKvCache,
            _runtime_state: &mut CudaRuntimeState,
            block_table: &BlockTable,
            start_pos: usize,
        ) -> Result<CudaLogits> {
            thread::scope(|s| {
                let handles: Vec<_> = self
                    .replicas
                    .iter()
                    .zip(kv_cache.inner.iter_mut())
                    .map(|(e, kv)| {
                        s.spawn(move || {
                            let mut state = CudaRuntimeState::new_placeholder();
                            e.forward_prefill(input_ids, kv, &mut state, block_table, start_pos)
                        })
                    })
                    .collect();
                collect_rank0(handles)
            })
        }

        #[allow(clippy::too_many_arguments)]
        fn forward_batch_decode(
            &self,
            token_ids: &crate::cuda::CudaTensor,
            kv_cache: &mut ShardedPagedKvCache,
            _runtime_state: &mut CudaRuntimeState,
            block_tables: &crate::cuda::CudaTensor,
            seq_lens: &crate::cuda::CudaTensor,
            positions: &crate::cuda::CudaTensor,
            batch_size: usize,
            max_blocks_per_seq: usize,
            max_seq_len: usize,
        ) -> Result<CudaLogits> {
            // Each replica's forward_batch_decode downloads token_ids,
            // block_tables, and positions to host and re-uploads to its own
            // device. Passing the rank-0 CUDA tensors is safe: cudarc uses the
            // tensor's own device context for dtoh copies.
            thread::scope(|s| {
                let handles: Vec<_> = self
                    .replicas
                    .iter()
                    .zip(kv_cache.inner.iter_mut())
                    .map(|(e, kv)| {
                        s.spawn(move || {
                            let mut state = CudaRuntimeState::new_placeholder();
                            e.forward_batch_decode(
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

    fn collect_rank0<T>(handles: Vec<thread::ScopedJoinHandle<'_, Result<T>>>) -> Result<T> {
        handles
            .into_iter()
            .map(|h| h.join().expect("device thread panicked"))
            .next()
            .expect("no replicas")
    }
}

#[cfg(feature = "nccl")]
pub use inner::{ShardedGraphEngine, ShardedPagedKvCache};
