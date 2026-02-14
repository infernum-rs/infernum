//! Llama tokenizer using the `tokenizers` crate

use std::path::Path;
use tokenizers::Tokenizer;

use crate::{Error, Result};

/// Wrapper around the `tokenizers` crate for Llama models
pub struct LlamaTokenizer {
    tokenizer: Tokenizer,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl LlamaTokenizer {
    /// Load a tokenizer from a directory containing tokenizer.json
    ///
    /// # Errors
    /// Returns an error if the tokenizer cannot be loaded
    pub fn from_pretrained(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();

        // Try tokenizer.json first (fast tokenizer format)
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Tokenizer::from_file(&tokenizer_path).map_err(|e| Error::Tokenizer(e.to_string()))?
        } else {
            return Err(Error::Tokenizer(format!(
                "No tokenizer.json found in {}",
                model_path.display()
            )));
        };

        // Get special token IDs (defaults for Llama 3)
        let bos_token_id = tokenizer
            .token_to_id("<|begin_of_text|>")
            .or_else(|| tokenizer.token_to_id("<s>"))
            .unwrap_or(1);

        let eos_token_id = tokenizer
            .token_to_id("<|end_of_text|>")
            .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(2);

        Ok(Self {
            tokenizer,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Encode text to token IDs
    ///
    /// # Arguments
    /// * `text` - The text to encode
    /// * `add_bos` - Whether to prepend the BOS token
    ///
    /// # Errors
    /// Returns an error if encoding fails
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();

        if add_bos {
            ids.insert(0, self.bos_token_id);
        }

        Ok(ids)
    }

    /// Decode token IDs to text
    ///
    /// # Errors
    /// Returns an error if decoding fails
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Decode a single token ID to text
    ///
    /// # Errors
    /// Returns an error if decoding fails
    pub fn decode_token(&self, id: u32) -> Result<String> {
        self.decode(&[id])
    }

    /// Get the BOS token ID
    #[must_use]
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// Get the EOS token ID
    #[must_use]
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Check if a token ID is an EOS token
    #[must_use]
    pub fn is_eos(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id
    }

    /// Get the vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    // Tests require a tokenizer file, so we skip them in CI
}
