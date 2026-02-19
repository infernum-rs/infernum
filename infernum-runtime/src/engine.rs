//! Token-level inference engine
//!
//! The [`Engine`] manages a model and its KV cache, providing token-level
//! generation (tokens in, tokens out). It is generic over any [`Model`]
//! implementation.

use infernum::cuda::ops::{argmax_last, sample_top_p};
use infernum::cuda::{CudaContext, KvCache};
use infernum::{Model, ModelConfig, Result, SamplingParams};

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

    /// Greedy generation with KV cache.
    ///
    /// # Arguments
    /// * `input_ids` - Prompt token IDs
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `eos_token_id` - Optional EOS token ID to stop generation early
    ///
    /// # Returns
    /// The full token sequence (prompt + generated tokens)
    ///
    /// # Errors
    /// Returns an error if a forward pass fails.
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        self.kv_cache.reset();
        let mut tokens = input_ids.to_vec();

        // Prefill: process entire prompt
        let logits = self
            .model
            .forward_with_kv_cache(input_ids, &mut self.kv_cache)?;
        let all_argmax = argmax_last(&logits)?;
        let mut next_token = all_argmax[0];

        if eos_token_id == Some(next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Decode: one token at a time
        for _ in 1..max_new_tokens {
            let logits = self
                .model
                .forward_next_token(next_token, &mut self.kv_cache)?;
            let all_argmax = argmax_last(&logits)?;
            next_token = all_argmax[0];

            if eos_token_id == Some(next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Generation with KV cache and nucleus (top-p) sampling.
    ///
    /// # Arguments
    /// * `input_ids` - Prompt token IDs
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `eos_token_id` - Optional EOS token ID to stop generation early
    /// * `params` - Sampling parameters (temperature, top-p, seed)
    ///
    /// # Returns
    /// The full token sequence (prompt + generated tokens)
    ///
    /// # Errors
    /// Returns an error if a forward pass fails.
    pub fn generate_sampled(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
        params: &SamplingParams,
    ) -> Result<Vec<u32>> {
        self.kv_cache.reset();
        let mut tokens = input_ids.to_vec();
        let mut rng_state = params.seed;

        // Prefill
        let logits = self
            .model
            .forward_with_kv_cache(input_ids, &mut self.kv_cache)?;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;

        let next_token = sample_top_p(&logits, params.temperature, params.top_p, rng_state)?;

        if eos_token_id == Some(next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);
        let mut last_token = next_token;

        // Decode
        for _ in 1..max_new_tokens {
            let logits = self
                .model
                .forward_next_token(last_token, &mut self.kv_cache)?;

            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let next_token = sample_top_p(&logits, params.temperature, params.top_p, rng_state)?;

            if eos_token_id == Some(next_token) {
                break;
            }
            tokens.push(next_token);
            last_token = next_token;
        }

        Ok(tokens)
    }
}
