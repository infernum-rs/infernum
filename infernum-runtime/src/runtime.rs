//! Text-level inference runtime
//!
//! The [`Runtime`] wraps an [`Engine`] and a [`Tokenizer`], providing
//! a text-in, text-out interface for generation.

use std::io::{self, Write};

use infernum::{GenerateOptions, Model, Result, Tokenizer};

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
    /// Create a new runtime from a model and tokenizer.
    ///
    /// # Errors
    /// Returns an error if KV cache allocation fails.
    pub fn new(model: M, tokenizer: T) -> Result<Self> {
        let engine = Engine::new(model)?;
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

    /// Generate text, blocking until complete.
    ///
    /// # Arguments
    /// * `prompt` - Input text
    /// * `options` - Generation options (sampling, max tokens, KV cache, etc.)
    ///
    /// # Returns
    /// The generated text (prompt + completion)
    ///
    /// # Errors
    /// Returns an error if tokenization or generation fails.
    pub fn generate(&mut self, prompt: &str, options: &GenerateOptions) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt, true)?;
        let options = self.fill_eos(options);
        let output_ids = self.engine.generate(&input_ids, &options)?;
        self.tokenizer.decode(&output_ids)
    }

    /// Generate text with streaming output (prints tokens as they are generated).
    ///
    /// # Arguments
    /// * `prompt` - Input text
    /// * `options` - Generation options (sampling, max tokens, KV cache, etc.)
    ///
    /// # Returns
    /// The full generated token sequence (prompt + completion)
    ///
    /// # Errors
    /// Returns an error if tokenization or generation fails.
    pub fn generate_stream(&mut self, prompt: &str, options: &GenerateOptions) -> Result<Vec<u32>>
    where
        M: Send,
    {
        let input_ids = self.tokenizer.encode(prompt, true)?;
        let options = self.fill_eos(options);
        let tokenizer = &self.tokenizer;

        self.engine.generate_stream(&input_ids, &options, |rx| {
            let mut tokens = input_ids.clone();
            let mut prev_len = tokenizer.decode(&tokens)?.len();

            for token_result in rx {
                let token = token_result?;
                tokens.push(token);
                let full_text = tokenizer.decode(&tokens)?;
                print!("{}", &full_text[prev_len..]);
                io::stdout().flush()?;
                prev_len = full_text.len();
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
