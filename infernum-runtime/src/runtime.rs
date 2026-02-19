//! Text-level inference runtime
//!
//! The [`Runtime`] wraps an [`Engine`] and a [`Tokenizer`], providing
//! a text-in, text-out interface for generation.

use std::io::{self, Write};

use infernum::cuda::CudaContext;
use infernum::{Model, Result, SamplingParams, Tokenizer};

use crate::Engine;

/// Text-level inference runtime.
///
/// Combines a model (via Engine) with a tokenizer to provide text-in, text-out
/// generation. Each Runtime instance serves one model.
pub struct Runtime<M: Model, T: Tokenizer> {
    engine: Engine<M>,
    tokenizer: T,
}

impl<M: Model, T: Tokenizer> Runtime<M, T> {
    /// Create a new runtime from a CUDA context, model, and tokenizer.
    ///
    /// # Errors
    /// Returns an error if KV cache allocation fails.
    pub fn new(ctx: &CudaContext, model: M, tokenizer: T) -> Result<Self> {
        let engine = Engine::new(ctx, model)?;
        Ok(Self { engine, tokenizer })
    }

    /// Get a reference to the underlying engine.
    #[must_use]
    pub fn engine(&self) -> &Engine<M> {
        &self.engine
    }

    /// Get a reference to the tokenizer.
    #[must_use]
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Generate text using greedy decoding.
    ///
    /// # Arguments
    /// * `prompt` - Input text
    /// * `max_new_tokens` - Maximum number of tokens to generate
    ///
    /// # Returns
    /// The generated text (prompt + completion)
    ///
    /// # Errors
    /// Returns an error if tokenization or generation fails.
    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt, true)?;
        let eos = Some(self.engine.model_config().eos_token_id);
        let output_ids = self.engine.generate(&input_ids, max_new_tokens, eos)?;
        self.tokenizer.decode(&output_ids)
    }

    /// Generate text using nucleus (top-p) sampling.
    ///
    /// # Arguments
    /// * `prompt` - Input text
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `params` - Sampling parameters
    ///
    /// # Returns
    /// The generated text (prompt + completion)
    ///
    /// # Errors
    /// Returns an error if tokenization or generation fails.
    pub fn generate_sampled(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        params: &SamplingParams,
    ) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt, true)?;
        let eos = Some(self.engine.model_config().eos_token_id);
        let output_ids = self
            .engine
            .generate_sampled(&input_ids, max_new_tokens, eos, params)?;
        self.tokenizer.decode(&output_ids)
    }

    /// Generate text with streaming output (prints tokens as they are generated).
    ///
    /// # Arguments
    /// * `prompt` - Input text
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `params` - Optional sampling parameters (greedy if `None`)
    ///
    /// # Returns
    /// The full generated token sequence (prompt + completion)
    ///
    /// # Errors
    /// Returns an error if tokenization or generation fails.
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        params: Option<&SamplingParams>,
    ) -> Result<Vec<u32>> {
        let input_ids = self.tokenizer.encode(prompt, true)?;
        let eos = Some(self.engine.model_config().eos_token_id);

        let output_ids = if let Some(params) = params {
            self.engine
                .generate_sampled(&input_ids, max_new_tokens, eos, params)?
        } else {
            self.engine.generate(&input_ids, max_new_tokens, eos)?
        };

        // Stream the newly generated tokens
        let prompt_len = input_ids.len();
        for &tok in &output_ids[prompt_len..] {
            let text = self.tokenizer.decode_token(tok)?;
            print!("{text}");
            io::stdout().flush()?;
        }

        Ok(output_ids)
    }
}
