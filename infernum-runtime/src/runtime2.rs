//! Backend-generic text-level inference runtime.
//!
//! [`Runtime2`] wraps an [`Engine2`] and a [`Tokenizer`], providing
//! a text-in, text-out interface for generation. No CUDA-specific imports.

use std::io::{self, Write};

use infernum::{GenerateOptions, Model, ModelConfig, Result, Tokenizer};

use crate::engine2::Engine2;
use crate::scheduler2::{BatchConfig, GenerationEvent};

/// Backend-generic text-level inference runtime.
///
/// Combines a model (via `Engine2`) with a tokenizer to provide text-in,
/// text-out generation. Each `Runtime2` instance serves one model.
pub struct Runtime2<M: Model, T: Tokenizer> {
    engine: Engine2<M>,
    tokenizer: T,
}

impl<M: Model, T: Tokenizer> Runtime2<M, T> {
    /// Create a new runtime from a model and tokenizer.
    ///
    /// # Errors
    /// Returns an error if paged KV cache allocation fails.
    pub fn new(model: M, tokenizer: T) -> Result<Self> {
        let engine = Engine2::new(model)?;
        Ok(Self { engine, tokenizer })
    }

    /// Create a new runtime with a custom batch configuration.
    ///
    /// # Errors
    /// Returns an error if paged KV cache allocation fails.
    pub fn with_config(model: M, tokenizer: T, batch_config: BatchConfig) -> Result<Self> {
        let engine = Engine2::with_config(model, batch_config)?;
        Ok(Self { engine, tokenizer })
    }

    /// Create a new runtime, ignoring `max_seq_len` (kept for API compat).
    ///
    /// # Errors
    /// Returns an error if paged KV cache allocation fails.
    pub fn with_max_seq_len(model: M, tokenizer: T, _max_seq_len: Option<usize>) -> Result<Self> {
        Self::new(model, tokenizer)
    }

    /// Get a reference to the underlying engine.
    #[must_use]
    pub fn engine(&self) -> &Engine2<M> {
        &self.engine
    }

    /// Get a reference to the tokenizer.
    #[must_use]
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Get the model configuration.
    #[must_use]
    pub fn model_config(&self) -> &ModelConfig {
        self.engine.model_config()
    }

    /// Generate text, blocking until complete.
    ///
    /// # Arguments
    /// * `prompt` - Input text
    /// * `options` - Generation options (sampling, max tokens, etc.)
    ///
    /// # Returns
    /// The generated text (prompt + completion)
    ///
    /// # Errors
    /// Returns an error if tokenization or generation fails.
    pub fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt, true)?;
        let options = self.fill_eos(options);
        let output_ids = self.engine.generate(&input_ids, &options)?;
        self.tokenizer.decode(&output_ids)
    }

    /// Generate text with streaming output (prints tokens as they are generated).
    ///
    /// # Arguments
    /// * `prompt` - Input text
    /// * `options` - Generation options (sampling, max tokens, etc.)
    ///
    /// # Returns
    /// The full generated token sequence (prompt + completion)
    ///
    /// # Errors
    /// Returns an error if tokenization or generation fails.
    pub fn generate_stream(&self, prompt: &str, options: &GenerateOptions) -> Result<Vec<u32>> {
        let input_ids = self.tokenizer.encode(prompt, true)?;
        let options = self.fill_eos(options);
        let tokenizer = &self.tokenizer;

        self.engine.generate_stream(&input_ids, &options, |rx| {
            let mut tokens = input_ids.clone();
            let mut prev_len = tokenizer.decode(&tokens)?.len();

            for event in rx {
                match event {
                    GenerationEvent::Token(id) => {
                        tokens.push(id);
                        let full_text = tokenizer.decode(&tokens)?;
                        print!("{}", &full_text[prev_len..]);
                        io::stdout().flush()?;
                        prev_len = full_text.len();
                    }
                    GenerationEvent::Error(e) => return Err(e),
                    GenerationEvent::Finished(_) => break,
                }
            }

            Ok::<Vec<u32>, infernum::Error>(tokens)
        })
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
