//! Tokenizer integration
//!
//! Defines the [`Tokenizer`] trait and concrete implementations.

mod llama_tokenizer;

use crate::Result;

pub use llama_tokenizer::LlamaTokenizer;

/// Trait for tokenizers that convert between text and token IDs.
///
/// Model crates provide concrete implementations (e.g., `LlamaTokenizer`).
/// The Runtime uses this trait to handle text in/out.
pub trait Tokenizer {
    /// Encode text to token IDs.
    ///
    /// # Arguments
    /// * `text` - The text to encode
    /// * `add_bos` - Whether to prepend the beginning-of-sequence token
    ///
    /// # Errors
    /// Returns an error if encoding fails.
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>>;

    /// Decode token IDs to text.
    ///
    /// # Errors
    /// Returns an error if decoding fails.
    fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Decode a single token ID to text.
    ///
    /// # Errors
    /// Returns an error if decoding fails.
    fn decode_token(&self, id: u32) -> Result<String>;

    /// Get the end-of-sequence token ID.
    fn eos_token_id(&self) -> u32;
}
