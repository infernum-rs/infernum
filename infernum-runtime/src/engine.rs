//! Token-level inference engine
//!
//! The [`Engine`] manages a model and its KV cache, providing token-level
//! generation (tokens in, tokens out). It is generic over any [`Model`]
//! implementation.

use std::sync::mpsc;
use std::thread;

use infernum::cuda::ops::{argmax_last_scalar, sample_top_p};
use infernum::cuda::{CudaContext, KvCache};
use infernum::{CudaTensor, GenerateOptions, Model, ModelConfig, Result, SamplingParams};

/// Token-level inference engine.
///
/// Wraps a model and manages its KV cache. Handles the prefill/decode loop
/// and sampling logic. Does not know about text — that is the Runtime's job.
pub struct Engine<M: Model> {
    model: M,
    model_config: ModelConfig,
    kv_cache: KvCache,
}

impl<M: Model> Engine<M> {
    /// Create a new engine wrapping the given model.
    ///
    /// # Errors
    /// Returns an error if KV cache allocation fails.
    pub fn new(ctx: &CudaContext, model: M) -> Result<Self> {
        let model_config = model.config();
        let kv_cache = KvCache::new(
            ctx,
            model_config.num_layers,
            model_config.max_seq_len,
            model_config.num_kv_heads,
            model_config.head_dim,
        )?;
        Ok(Self {
            model,
            model_config,
            kv_cache,
        })
    }

    /// Get a reference to the underlying model.
    #[must_use]
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get the model configuration.
    #[must_use]
    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    /// Generate tokens, blocking until complete.
    ///
    /// # Arguments
    /// * `input_ids` - Prompt token IDs
    /// * `options` - Generation options (sampling, max tokens, KV cache, etc.)
    ///
    /// # Returns
    /// The full token sequence (prompt + generated tokens)
    ///
    /// # Errors
    /// Returns an error if a forward pass fails.
    pub fn generate(&mut self, input_ids: &[u32], options: &GenerateOptions) -> Result<Vec<u32>> {
        if options.use_kv_cache {
            self.generate_kv_cached(input_ids, options)
        } else {
            self.generate_naive(input_ids, options)
        }
    }

    /// Generate tokens with streaming via a channel.
    ///
    /// Runs the generation loop in a scoped thread, sending each new token
    /// through a channel. The provided `consumer` closure receives the
    /// [`mpsc::Receiver`] and is called on the current thread while tokens
    /// are being produced.
    ///
    /// # Arguments
    /// * `input_ids` - Prompt token IDs
    /// * `options` - Generation options (sampling, max tokens, KV cache, etc.)
    /// * `consumer` - Closure that processes tokens from the receiver
    ///
    /// # Returns
    /// The return value of the `consumer` closure.
    #[allow(clippy::missing_panics_doc)]
    pub fn generate_stream<F, R>(
        &mut self,
        input_ids: &[u32],
        options: &GenerateOptions,
        consumer: F,
    ) -> R
    where
        M: Send,
        F: FnOnce(mpsc::Receiver<Result<u32>>) -> R,
    {
        let (tx, rx) = mpsc::channel();
        let input_ids = input_ids.to_vec();
        let options = options.clone();

        let mut result = None;

        thread::scope(|s| {
            s.spawn(|| {
                let gen_result = if options.use_kv_cache {
                    self.stream_kv_cached(&input_ids, &options, &tx)
                } else {
                    self.stream_naive(&input_ids, &options, &tx)
                };

                if let Err(e) = gen_result {
                    let _ = tx.send(Err(e));
                }
                // Drop tx so the receiver sees the channel close
                drop(tx);
            });

            result = Some(consumer(rx));
        });

        result.expect("thread::scope completed without producing a result")
    }

    /// Generate with KV cache (prefill + single-token decode steps).
    fn generate_kv_cached(
        &mut self,
        input_ids: &[u32],
        options: &GenerateOptions,
    ) -> Result<Vec<u32>> {
        self.kv_cache.reset();
        let mut tokens = input_ids.to_vec();
        let mut rng_state = options.sampling.as_ref().map(|s| s.seed);

        // Prefill: process entire prompt
        let logits = self
            .model
            .forward_with_kv_cache(input_ids, &mut self.kv_cache)?;
        let mut next_token = select_token(&logits, options.sampling.as_ref(), &mut rng_state)?;

        if options.eos_token_id == Some(next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Decode: one token at a time
        for _ in 1..options.max_new_tokens {
            let logits = self
                .model
                .forward_next_token(next_token, &mut self.kv_cache)?;
            next_token = select_token(&logits, options.sampling.as_ref(), &mut rng_state)?;

            if options.eos_token_id == Some(next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Generate without KV cache (recomputes full sequence each step).
    fn generate_naive(&self, input_ids: &[u32], options: &GenerateOptions) -> Result<Vec<u32>> {
        let mut tokens = input_ids.to_vec();
        let mut rng_state = options.sampling.as_ref().map(|s| s.seed);

        for _ in 0..options.max_new_tokens {
            let logits = self.model.forward(&tokens)?;
            let next_token = select_token(&logits, options.sampling.as_ref(), &mut rng_state)?;

            if options.eos_token_id == Some(next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Stream with KV cache, sending each token through the channel.
    fn stream_kv_cached(
        &mut self,
        input_ids: &[u32],
        options: &GenerateOptions,
        tx: &mpsc::Sender<Result<u32>>,
    ) -> Result<()> {
        self.kv_cache.reset();
        let mut rng_state = options.sampling.as_ref().map(|s| s.seed);

        // Prefill
        let logits = self
            .model
            .forward_with_kv_cache(input_ids, &mut self.kv_cache)?;
        let mut next_token = select_token(&logits, options.sampling.as_ref(), &mut rng_state)?;

        if options.eos_token_id == Some(next_token) {
            return Ok(());
        }
        if tx.send(Ok(next_token)).is_err() {
            return Ok(());
        }

        // Decode
        for _ in 1..options.max_new_tokens {
            let logits = self
                .model
                .forward_next_token(next_token, &mut self.kv_cache)?;
            next_token = select_token(&logits, options.sampling.as_ref(), &mut rng_state)?;

            if options.eos_token_id == Some(next_token) {
                break;
            }
            if tx.send(Ok(next_token)).is_err() {
                break;
            }
        }

        Ok(())
    }

    /// Stream without KV cache, sending each token through the channel.
    fn stream_naive(
        &self,
        input_ids: &[u32],
        options: &GenerateOptions,
        tx: &mpsc::Sender<Result<u32>>,
    ) -> Result<()> {
        let mut tokens = input_ids.to_vec();
        let mut rng_state = options.sampling.as_ref().map(|s| s.seed);

        for _ in 0..options.max_new_tokens {
            let logits = self.model.forward(&tokens)?;
            let next_token = select_token(&logits, options.sampling.as_ref(), &mut rng_state)?;

            if options.eos_token_id == Some(next_token) {
                break;
            }
            tokens.push(next_token);
            if tx.send(Ok(next_token)).is_err() {
                break;
            }
        }

        Ok(())
    }
}

/// Select the next token from logits, either via greedy argmax or sampling.
fn select_token(
    logits: &CudaTensor<f32>,
    sampling: Option<&SamplingParams>,
    rng_state: &mut Option<u64>,
) -> Result<u32> {
    if let (Some(params), Some(state)) = (sampling, rng_state) {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        sample_top_p(logits, params.temperature, params.top_p, *state)
    } else {
        argmax_last_scalar(logits)
    }
}
