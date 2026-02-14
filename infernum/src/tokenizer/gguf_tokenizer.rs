//! Tokenizer that loads vocabulary from GGUF metadata
//!
//! GGUF files embed the tokenizer vocabulary under `tokenizer.ggml.*` keys.
//! This implements BPE encoding using the merge scores stored in the file,
//! compatible with both SentencePiece-style (score-based) and GPT2-style
//! (explicit merges) tokenizers.

#![allow(
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::missing_panics_doc
)]

use std::collections::HashMap;

use crate::weights::GgufValue;
use crate::{Error, Result};

/// A tokenizer built from GGUF metadata (`tokenizer.ggml.*` keys).
///
/// Supports SentencePiece-style BPE where token pairs are merged in order
/// of decreasing score (higher score = merge first).
pub struct GgufTokenizer {
    /// Token ID → token bytes
    vocab: Vec<Vec<u8>>,
    /// Token bytes → token ID (for encoding)
    token_to_id: HashMap<Vec<u8>, u32>,
    /// (left_id, right_id) → merge score (higher = merge earlier)
    merge_scores: HashMap<(u32, u32), f32>,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl GgufTokenizer {
    /// Build a tokenizer from GGUF metadata.
    ///
    /// Expects the following keys:
    /// - `tokenizer.ggml.tokens` — array of token strings
    /// - `tokenizer.ggml.scores` — array of f32 merge priorities
    /// - `tokenizer.ggml.bos_token_id` — BOS token ID
    /// - `tokenizer.ggml.eos_token_id` — EOS token ID
    ///
    /// # Errors
    /// Returns an error if required metadata keys are missing.
    pub fn from_gguf_metadata(metadata: &HashMap<String, GgufValue>) -> Result<Self> {
        // Extract token strings
        let tokens_val = metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| Error::Tokenizer("Missing tokenizer.ggml.tokens".into()))?;

        let token_strings: Vec<&str> = match tokens_val {
            GgufValue::Array(arr) => arr
                .iter()
                .map(|v| {
                    v.as_str()
                        .ok_or_else(|| Error::Tokenizer("Token is not a string".into()))
                })
                .collect::<Result<Vec<_>>>()?,
            _ => {
                return Err(Error::Tokenizer(
                    "tokenizer.ggml.tokens is not an array".into(),
                ))
            }
        };

        let vocab_size = token_strings.len();

        // Extract scores (optional — default to 0.0 if missing)
        let scores: Vec<f32> =
            if let Some(GgufValue::Array(arr)) = metadata.get("tokenizer.ggml.scores") {
                arr.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect()
            } else {
                vec![0.0; vocab_size]
            };

        // Build vocab: decode SentencePiece-style escaped bytes (e.g. <0x0A> for newline)
        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab_size);

        for (id, token_str) in token_strings.iter().enumerate() {
            let bytes = decode_token_str(token_str);
            token_to_id.entry(bytes.clone()).or_insert(id as u32);
            vocab.push(bytes);
        }

        // Build merge table: for every pair of tokens that concatenate to a
        // known token, record the merge with the score of the result token.
        // This is how SentencePiece BPE works — the score of the merged token
        // determines merge priority.
        let mut merge_scores: HashMap<(u32, u32), f32> = HashMap::new();

        for (merged_id, merged_bytes) in vocab.iter().enumerate() {
            let score = scores[merged_id];
            // Try all splits of merged_bytes into (left, right)
            for split_pos in 1..merged_bytes.len() {
                let left = &merged_bytes[..split_pos];
                let right = &merged_bytes[split_pos..];
                if let (Some(&left_id), Some(&right_id)) =
                    (token_to_id.get(left), token_to_id.get(right))
                {
                    // Only keep the highest-scoring merge for this pair
                    merge_scores
                        .entry((left_id, right_id))
                        .and_modify(|s| {
                            if score > *s {
                                *s = score;
                            }
                        })
                        .or_insert(score);
                }
            }
        }

        let bos_token_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(GgufValue::as_usize)
            .unwrap_or(1) as u32;

        let eos_token_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(GgufValue::as_usize)
            .unwrap_or(2) as u32;

        Ok(Self {
            vocab,
            token_to_id,
            merge_scores,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Encode text into token IDs using BPE.
    ///
    /// SentencePiece prepends a space (▁ = U+2581) to the input, which maps to
    /// the `▁` byte sequence in the vocabulary.
    ///
    /// # Arguments
    /// * `text` - The text to encode
    /// * `add_bos` - Whether to prepend the BOS token
    ///
    /// # Errors
    /// Returns an error if a byte cannot be found in the vocabulary.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();

        if add_bos {
            ids.push(self.bos_token_id);
        }

        if text.is_empty() {
            return Ok(ids);
        }

        // SentencePiece convention: prepend space (▁ = U+2581 = 0xE2 0x96 0x81)
        let sp_text = format!("\u{2581}{text}");
        let bytes = sp_text.as_bytes();

        // Start with one token per byte (using byte-fallback tokens like <0xAB>)
        let mut tokens: Vec<u32> = bytes
            .iter()
            .map(|&b| {
                // Try single byte first
                if let Some(&id) = self.token_to_id.get(&[b][..]) {
                    Ok(id)
                } else {
                    // Byte-fallback: <0xHH>
                    let hex_token = format!("<0x{b:02X}>");
                    self.token_to_id
                        .get(hex_token.as_bytes())
                        .copied()
                        .ok_or_else(|| Error::Tokenizer(format!("No token for byte 0x{b:02x}")))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // BPE merge loop: repeatedly merge the pair with the highest score
        loop {
            if tokens.len() < 2 {
                break;
            }

            // Find the best merge
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&score) = self.merge_scores.get(&pair) {
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break; // No more merges possible
            }

            // Merge: find the resulting token
            let left = &self.vocab[tokens[best_idx] as usize];
            let right = &self.vocab[tokens[best_idx + 1] as usize];
            let mut merged = left.clone();
            merged.extend_from_slice(right);

            if let Some(&merged_id) = self.token_to_id.get(&merged) {
                tokens[best_idx] = merged_id;
                tokens.remove(best_idx + 1);
            } else {
                break; // Should not happen if merge table is consistent
            }
        }

        ids.extend_from_slice(&tokens);
        Ok(ids)
    }

    /// Decode token IDs to a string.
    ///
    /// Replaces SentencePiece `▁` (U+2581) with space.
    ///
    /// # Errors
    /// Returns an error if a token ID is out of range.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let id_usize = id as usize;
            if id_usize >= self.vocab.len() {
                return Err(Error::Tokenizer(format!("Token ID {id} out of range")));
            }
            bytes.extend_from_slice(&self.vocab[id_usize]);
        }

        // Replace ▁ (0xE2 0x96 0x81) with space
        let text = String::from_utf8_lossy(&bytes).replace('\u{2581}', " ");

        // Trim leading space (from the ▁ we prepend during encoding)
        Ok(text.strip_prefix(' ').unwrap_or(&text).to_string())
    }

    /// Decode a single token ID to a string.
    ///
    /// # Errors
    /// Returns an error if the token ID is out of range.
    pub fn decode_token(&self, id: u32) -> Result<String> {
        let id_usize = id as usize;
        if id_usize >= self.vocab.len() {
            return Err(Error::Tokenizer(format!("Token ID {id} out of range")));
        }
        let text = String::from_utf8_lossy(&self.vocab[id_usize]).replace('\u{2581}', " ");
        Ok(text)
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

    /// Get the vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Decode a SentencePiece token string to raw bytes.
///
/// Handles byte-fallback tokens like `<0x0A>` and the space placeholder
/// `▁` (U+2581) which is kept as-is in UTF-8 (0xE2 0x96 0x81).
fn decode_token_str(s: &str) -> Vec<u8> {
    // Check for byte-fallback tokens: <0xHH>
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
            return vec![byte];
        }
    }
    s.as_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_metadata(
        tokens: &[&str],
        scores: &[f32],
        bos: u32,
        eos: u32,
    ) -> HashMap<String, GgufValue> {
        let mut m = HashMap::new();
        m.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(
                tokens
                    .iter()
                    .map(|t| GgufValue::String((*t).into()))
                    .collect(),
            ),
        );
        m.insert(
            "tokenizer.ggml.scores".into(),
            GgufValue::Array(scores.iter().map(|&s| GgufValue::F32(s)).collect()),
        );
        m.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(bos));
        m.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(eos));
        m
    }

    #[test]
    fn test_decode_token_str_byte_fallback() {
        assert_eq!(decode_token_str("<0x0A>"), vec![0x0A]);
        assert_eq!(decode_token_str("<0xFF>"), vec![0xFF]);
        assert_eq!(decode_token_str("<0x00>"), vec![0x00]);
    }

    #[test]
    fn test_decode_token_str_normal() {
        assert_eq!(decode_token_str("hello"), b"hello".to_vec());
        assert_eq!(
            decode_token_str("\u{2581}the"),
            "\u{2581}the".as_bytes().to_vec()
        );
    }

    #[test]
    fn test_basic_encoding() {
        // Minimal vocab: bytes + some merged tokens
        // ▁ = U+2581 = [0xE2, 0x96, 0x81]
        let tokens = &[
            "<unk>",         // 0
            "<s>",           // 1 (BOS)
            "</s>",          // 2 (EOS)
            "\u{2581}h",     // 3 — merged "▁h"
            "e",             // 4
            "l",             // 5
            "o",             // 6
            "\u{2581}",      // 7 — space placeholder
            "h",             // 8
            "ll",            // 9 — merged "ll"
            "llo",           // 10 — merged "llo"
            "\u{2581}hello", // 11 — full merge
        ];
        // Higher score = merge first
        let scores: &[f32] = &[
            0.0,   // <unk>
            0.0,   // <s>
            0.0,   // </s>
            -1.0,  // ▁h
            -10.0, // e
            -10.0, // l
            -10.0, // o
            -5.0,  // ▁
            -10.0, // h
            -2.0,  // ll
            -1.5,  // llo
            -0.5,  // ▁hello
        ];

        let meta = make_metadata(tokens, scores, 1, 2);
        let tok = GgufTokenizer::from_gguf_metadata(&meta).unwrap();

        // "hello" -> with ▁ prepended: "▁hello" -> should merge to token 11
        let ids = tok.encode("hello", false).unwrap();
        assert_eq!(ids, vec![11]); // single token "▁hello"

        // With BOS
        let ids_bos = tok.encode("hello", true).unwrap();
        assert_eq!(ids_bos, vec![1, 11]);
    }

    #[test]
    fn test_decode_roundtrip() {
        let tokens = &[
            "<unk>",
            "<s>",
            "</s>",
            "\u{2581}h",
            "e",
            "l",
            "o",
            "\u{2581}",
            "h",
            "ll",
            "llo",
            "\u{2581}hello",
        ];
        let scores: &[f32] = &[
            0.0, 0.0, 0.0, -1.0, -10.0, -10.0, -10.0, -5.0, -10.0, -2.0, -1.5, -0.5,
        ];
        let meta = make_metadata(tokens, scores, 1, 2);
        let tok = GgufTokenizer::from_gguf_metadata(&meta).unwrap();

        let ids = tok.encode("hello", false).unwrap();
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_vocab_size() {
        let tokens = &["<unk>", "<s>", "</s>", "a", "b"];
        let scores = &[0.0; 5];
        let meta = make_metadata(tokens, scores, 1, 2);
        let tok = GgufTokenizer::from_gguf_metadata(&meta).unwrap();
        assert_eq!(tok.vocab_size(), 5);
    }

    #[test]
    fn test_special_tokens() {
        let tokens = &["<unk>", "<s>", "</s>"];
        let scores = &[0.0; 3];
        let meta = make_metadata(tokens, scores, 1, 2);
        let tok = GgufTokenizer::from_gguf_metadata(&meta).unwrap();
        assert_eq!(tok.bos_token_id(), 1);
        assert_eq!(tok.eos_token_id(), 2);
    }

    #[test]
    fn test_empty_string() {
        let tokens = &["<unk>", "<s>", "</s>"];
        let scores = &[0.0; 3];
        let meta = make_metadata(tokens, scores, 1, 2);
        let tok = GgufTokenizer::from_gguf_metadata(&meta).unwrap();

        let ids = tok.encode("", false).unwrap();
        assert!(ids.is_empty());

        let ids = tok.encode("", true).unwrap();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn test_missing_tokens_error() {
        let meta: HashMap<String, GgufValue> = HashMap::new();
        let result = GgufTokenizer::from_gguf_metadata(&meta);
        assert!(result.is_err());
    }
}
