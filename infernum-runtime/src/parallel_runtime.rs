//! Text-level multi-GPU inference runtime
//!
//! [`ParallelRuntime`] wraps a [`ParallelEngine`] and a [`Tokenizer`],
//! providing a text-in, text-out interface identical to [`Runtime`](crate::Runtime)
//! but using tensor parallelism across multiple GPUs.

use std::io::{self, Write};

use infernum::{GenerateOptions, Model, Result, Tokenizer};

use crate::ParallelEngine;

/// Text-level multi-GPU inference runtime.
///
/// Combines a parallel engine (multi-GPU) with a tokenizer to provide
/// text-in, text-out generation. The API mirrors [`Runtime`](crate::Runtime).
pub struct ParallelRuntime<M: Model, T: Tokenizer> {
    engine: ParallelEngine<M>,
    tokenizer: T,
}

impl<M, T> ParallelRuntime<M, T>
where
    M: Model + Send + Sync,
    M::CacheDtype: Send,
    T: Tokenizer,
{
    /// Create a new parallel runtime from pre-loaded sharded models and a tokenizer.
    ///
    /// # Panics
    /// Panics if `models` is empty.
    ///
    /// # Errors
    /// Returns an error if KV cache allocation fails.
    pub fn new(models: Vec<(infernum::cuda::CudaContext, M)>, tokenizer: T) -> Result<Self> {
        let engine = ParallelEngine::new(models)?;
        Ok(Self { engine, tokenizer })
    }

    /// Get a reference to the underlying parallel engine.
    #[must_use]
    pub fn engine(&self) -> &ParallelEngine<M> {
        &self.engine
    }

    /// Get a reference to the tokenizer.
    #[must_use]
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Generate text, blocking until complete.
    ///
    /// # Errors
    /// Returns an error if tokenization or generation fails.
    pub fn generate(&mut self, prompt: &str, options: &GenerateOptions) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt, true)?;
        let options = self.fill_eos(options);
        let output_ids = self.engine.generate(&input_ids, &options)?;
        self.tokenizer.decode(&output_ids)
    }

    /// Generate text, printing tokens as they complete.
    ///
    /// Unlike the single-GPU [`Runtime::generate_stream`](crate::Runtime::generate_stream),
    /// this does not stream token-by-token — it generates all tokens first,
    /// then prints them. True streaming will be added in a future version.
    ///
    /// # Errors
    /// Returns an error if tokenization or generation fails.
    pub fn generate_stream(&mut self, prompt: &str, options: &GenerateOptions) -> Result<Vec<u32>> {
        let input_ids = self.tokenizer.encode(prompt, true)?;
        let options = self.fill_eos(options);
        let all_ids = self.engine.generate(&input_ids, &options)?;

        for &tok in all_ids.iter().skip(input_ids.len()) {
            let text = self.tokenizer.decode_token(tok)?;
            print!("{text}");
            io::stdout().flush()?;
        }

        Ok(all_ids)
    }

    /// Clone options with EOS token filled from model config if not set.
    fn fill_eos(&self, options: &GenerateOptions) -> GenerateOptions {
        if options.eos_token_id.is_some() {
            return options.clone();
        }
        let mut options = options.clone();
        options.eos_token_id = Some(self.engine.model_config().eos_token_id);
        options
    }
}
