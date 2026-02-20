//! Multi-GPU parallel inference engine
//!
//! [`ParallelEngine`] coordinates tensor-parallel inference across multiple
//! GPUs. Each GPU holds a shard of the model weights and runs the full
//! forward pass; NCCL all-reduce synchronises partial results within
//! each model's forward call.
//!
//! The API mirrors [`Engine`](crate::Engine) — callers don't need to
//! know about parallelism.

use std::thread;

use infernum::cuda::ops::{argmax_last_scalar, sample_top_p};
use infernum::cuda::{CudaContext, KvCache};
use infernum::{CudaTensor, GenerateOptions, Model, ModelConfig, Result, SamplingParams};

/// Per-GPU replica: model + KV cache + device context.
struct Replica<M: Model> {
    #[allow(dead_code)]
    ctx: CudaContext,
    model: M,
    kv_cache: KvCache<M::CacheDtype>,
}

/// Multi-GPU inference engine using tensor parallelism.
///
/// Owns one [`Replica`] per GPU. All replicas execute the same forward pass
/// in lock-step using scoped threads; NCCL all-reduce happens inside each
/// model's forward call. Only rank 0's logits are used for sampling.
pub struct ParallelEngine<M: Model> {
    replicas: Vec<Replica<M>>,
    model_config: ModelConfig,
}

impl<M: Model + Send + Sync> ParallelEngine<M>
where
    M::CacheDtype: Send,
{
    /// Create a parallel engine from pre-loaded sharded models.
    ///
    /// `models` must contain one `(CudaContext, Model)` per GPU, ordered by
    /// rank. Each model should have been loaded with the matching
    /// [`ShardConfig`](infernum::ShardConfig).
    ///
    /// # Panics
    /// Panics if `models` is empty.
    ///
    /// # Errors
    /// Returns an error if KV cache allocation fails on any GPU.
    pub fn new(models: Vec<(CudaContext, M)>) -> Result<Self> {
        assert!(!models.is_empty(), "need at least one model");

        infernum::fusion::init();
        let model_config = models[0].1.config();

        let replicas = models
            .into_iter()
            .map(|(ctx, model)| {
                let kv_cache = KvCache::new(
                    &ctx,
                    model_config.num_layers,
                    model_config.max_seq_len,
                    model_config.num_kv_heads,
                    model_config.head_dim,
                )?;
                Ok(Replica {
                    ctx,
                    model,
                    kv_cache,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            replicas,
            model_config,
        })
    }

    /// Get the model configuration.
    #[must_use]
    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    /// Get a reference to rank 0's model.
    #[must_use]
    pub fn model(&self) -> &M {
        &self.replicas[0].model
    }

    /// Generate tokens, blocking until complete.
    ///
    /// All GPUs run in lock-step; sampling uses rank 0's logits only.
    ///
    /// # Errors
    /// Returns an error if a forward pass fails on any GPU.
    pub fn generate(&mut self, input_ids: &[u32], options: &GenerateOptions) -> Result<Vec<u32>> {
        if options.use_kv_cache {
            self.generate_kv_cached(input_ids, options)
        } else {
            self.generate_naive(input_ids, options)
        }
    }

    /// Generate with KV cache (prefill + decode).
    fn generate_kv_cached(
        &mut self,
        input_ids: &[u32],
        options: &GenerateOptions,
    ) -> Result<Vec<u32>> {
        for replica in &mut self.replicas {
            replica.kv_cache.reset()?;
        }

        let mut tokens = input_ids.to_vec();
        let mut rng_state = options.sampling.as_ref().map(|s| s.seed);
        let sampling = options.sampling.as_ref();

        // Prefill: all GPUs process the full prompt in parallel
        let logits = self.parallel_forward_prefill(input_ids)?;
        let recent = recent_token_window(&tokens, sampling);
        let mut next_token = select_token(&logits, sampling, &mut rng_state, recent)?;

        if options.eos_token_id == Some(next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Decode loop
        for _ in 1..options.max_new_tokens {
            let logits = self.parallel_forward_decode(next_token)?;
            let recent = recent_token_window(&tokens, sampling);
            next_token = select_token(&logits, sampling, &mut rng_state, recent)?;

            if options.eos_token_id == Some(next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Generate without KV cache (full recompute each step).
    fn generate_naive(&self, input_ids: &[u32], options: &GenerateOptions) -> Result<Vec<u32>> {
        let mut tokens = input_ids.to_vec();
        let mut rng_state = options.sampling.as_ref().map(|s| s.seed);
        let sampling = options.sampling.as_ref();

        for _ in 0..options.max_new_tokens {
            let logits = self.parallel_forward(&tokens)?;
            let recent = recent_token_window(&tokens, sampling);
            let next_token = select_token(&logits, sampling, &mut rng_state, recent)?;

            if options.eos_token_id == Some(next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Run a full forward pass (no KV cache) on all GPUs in parallel.
    ///
    /// Returns rank 0's logits.
    fn parallel_forward(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .map(|replica| s.spawn(move || replica.model.forward(input_ids)))
                .collect();

            collect_rank0(handles)
        })
    }

    /// Run prefill (forward with KV cache) on all GPUs in parallel.
    ///
    /// Returns rank 0's logits.
    fn parallel_forward_prefill(&mut self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter_mut()
                .map(|replica| {
                    s.spawn(move || {
                        replica
                            .model
                            .forward_with_kv_cache(input_ids, &mut replica.kv_cache)
                    })
                })
                .collect();

            collect_rank0(handles)
        })
    }

    /// Run a single decode step on all GPUs in parallel.
    ///
    /// Returns rank 0's logits.
    fn parallel_forward_decode(&mut self, token_id: u32) -> Result<CudaTensor<f32>> {
        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter_mut()
                .map(|replica| {
                    s.spawn(move || {
                        replica
                            .model
                            .forward_next_token(token_id, &mut replica.kv_cache)
                    })
                })
                .collect();

            collect_rank0(handles)
        })
    }
}

/// Collect results from parallel threads, returning rank 0's value.
///
/// All threads are joined (panics propagate). Returns rank 0's `Result`.
fn collect_rank0<T>(handles: Vec<thread::ScopedJoinHandle<'_, Result<T>>>) -> Result<T> {
    let results: Vec<Result<T>> = handles
        .into_iter()
        .map(|h| h.join().expect("GPU thread panicked"))
        .collect();

    // We must consume the Vec to move rank 0's result out.
    results.into_iter().next().expect("no rank 0 in handles")
}

/// Select the next token from logits, either via greedy argmax or sampling.
fn select_token(
    logits: &CudaTensor<f32>,
    sampling: Option<&SamplingParams>,
    rng_state: &mut Option<u64>,
    recent_tokens: &[u32],
) -> Result<u32> {
    if let (Some(params), Some(state)) = (sampling, rng_state) {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        sample_top_p(
            logits,
            params.temperature,
            params.top_p,
            *state,
            params.repetition_penalty,
            recent_tokens,
        )
    } else {
        argmax_last_scalar(logits)
    }
}

/// Extract the recent token window for repetition penalty.
fn recent_token_window<'a>(tokens: &'a [u32], sampling: Option<&SamplingParams>) -> &'a [u32] {
    let window = sampling.map_or(0, |s| s.repetition_penalty_window);
    let start = tokens.len().saturating_sub(window);
    &tokens[start..]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- collect_rank0 ----

    #[test]
    fn test_collect_rank0_returns_first_result() {
        let result: Result<i32> = thread::scope(|s| {
            let handles = vec![s.spawn(|| Ok(10)), s.spawn(|| Ok(20)), s.spawn(|| Ok(30))];
            collect_rank0(handles)
        });
        assert_eq!(result.unwrap(), 10);
    }

    #[test]
    fn test_collect_rank0_single_handle() {
        let result: Result<&str> = thread::scope(|s| {
            let handles = vec![s.spawn(|| Ok("hello"))];
            collect_rank0(handles)
        });
        assert_eq!(result.unwrap(), "hello");
    }

    #[test]
    fn test_collect_rank0_propagates_rank0_error() {
        let result: Result<i32> = thread::scope(|s| {
            let handles = vec![
                s.spawn(|| Err(infernum::Error::InvalidShape("rank0 failed".into()))),
                s.spawn(|| Ok(20)),
            ];
            collect_rank0(handles)
        });
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "no rank 0")]
    fn test_collect_rank0_panics_on_empty() {
        let _: Result<i32> = thread::scope(|s| {
            let handles: Vec<thread::ScopedJoinHandle<'_, Result<i32>>> = vec![];
            collect_rank0(handles)
        });
    }

    // ---- recent_token_window ----

    #[test]
    fn test_recent_token_window_no_sampling() {
        let tokens = [1, 2, 3, 4, 5];
        let window = recent_token_window(&tokens, None);
        assert!(window.is_empty());
    }

    #[test]
    fn test_recent_token_window_zero_window() {
        let tokens = [1, 2, 3];
        let sampling = SamplingParams {
            repetition_penalty_window: 0,
            ..default_sampling()
        };
        let window = recent_token_window(&tokens, Some(&sampling));
        assert!(window.is_empty());
    }

    #[test]
    fn test_recent_token_window_partial() {
        let tokens = [10, 20, 30, 40, 50];
        let sampling = SamplingParams {
            repetition_penalty_window: 3,
            ..default_sampling()
        };
        let window = recent_token_window(&tokens, Some(&sampling));
        assert_eq!(window, &[30, 40, 50]);
    }

    #[test]
    fn test_recent_token_window_larger_than_tokens() {
        let tokens = [1, 2];
        let sampling = SamplingParams {
            repetition_penalty_window: 100,
            ..default_sampling()
        };
        let window = recent_token_window(&tokens, Some(&sampling));
        assert_eq!(window, &[1, 2]);
    }

    #[test]
    fn test_recent_token_window_exact_match() {
        let tokens = [1, 2, 3];
        let sampling = SamplingParams {
            repetition_penalty_window: 3,
            ..default_sampling()
        };
        let window = recent_token_window(&tokens, Some(&sampling));
        assert_eq!(window, &[1, 2, 3]);
    }

    fn default_sampling() -> SamplingParams {
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            seed: 42,
            repetition_penalty: 1.0,
            repetition_penalty_window: 0,
        }
    }
}
