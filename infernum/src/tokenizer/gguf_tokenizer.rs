//! Tokenizer that loads vocabulary from GGUF metadata
//!
//! GGUF files embed the tokenizer vocabulary under `tokenizer.ggml.*` keys.
//! This implements BPE encoding using the merge scores stored in the file,
//! compatible with both SentencePiece-style (Llama) and GPT2-style (SmolLM,
//! StarCoder, etc.) tokenizers. The tokenizer model type is read from
//! `tokenizer.ggml.model` and the encode/decode paths branch accordingly.

#![allow(
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::missing_panics_doc
)]

use std::collections::HashMap;

use crate::gguf_meta::GgufValue;
use crate::{Error, Result};

/// Which tokenizer convention is used in the GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenizerModel {
    /// SentencePiece BPE (Llama, Mistral, etc.)
    /// Spaces are represented as `▁` (U+2581). Byte fallbacks use `<0xHH>`.
    SentencePiece,
    /// GPT2-style byte-level BPE (SmolLM, StarCoder, GPT-2, etc.)
    /// Each byte 0x00–0xFF is mapped to a printable Unicode character.
    /// Spaces become `Ġ` (U+0120).
    Gpt2,
}

/// Build the GPT2 byte→unicode mapping table.
///
/// Printable ASCII (0x21–0x7E) and Latin-1 (0xA1–0xAC, 0xAE–0xFF) map to
/// themselves. The remaining 68 bytes (control chars, space, DEL, 0x80–0xA0,
/// 0xAD) are mapped to U+0100–U+0143 in order.
fn gpt2_byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n: u32 = 0;
    for b in 0u16..=255 {
        let is_printable =
            (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        if is_printable {
            table[b as usize] = char::from(b as u8);
        } else {
            table[b as usize] = char::from_u32(256 + n).expect("GPT2 byte table Unicode codepoint");
            n += 1;
        }
    }
    table
}

/// Build the inverse GPT2 unicode→byte mapping table.
fn gpt2_unicode_to_byte() -> HashMap<char, u8> {
    let fwd = gpt2_byte_to_unicode();
    fwd.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect()
}

/// A tokenizer built from GGUF metadata (`tokenizer.ggml.*` keys).
///
/// Supports both SentencePiece-style (Llama) and GPT2-style byte-level BPE.
/// The model type is auto-detected from `tokenizer.ggml.model`.
pub struct GgufTokenizer {
    /// Token ID → token bytes (raw bytes after model-specific decoding)
    vocab: Vec<Vec<u8>>,
    /// Token bytes → token ID (for encoding)
    token_to_id: HashMap<Vec<u8>, u32>,
    /// (left_id, right_id) → merge score (higher = merge earlier)
    merge_scores: HashMap<(u32, u32), f32>,
    bos_token_id: u32,
    eos_token_id: u32,
    /// Which tokenizer convention is used
    model: TokenizerModel,
}

impl GgufTokenizer {
    /// Build a tokenizer from GGUF metadata.
    ///
    /// Expects the following keys:
    /// - `tokenizer.ggml.tokens` — array of token strings
    /// - `tokenizer.ggml.scores` — array of f32 merge priorities (for Unigram)
    /// - `tokenizer.ggml.merges` — array of "left right" merge strings (for BPE)
    /// - `tokenizer.ggml.bos_token_id` — BOS token ID
    /// - `tokenizer.ggml.eos_token_id` — EOS token ID
    ///
    /// # Errors
    /// Returns an error if required metadata keys are missing.
    pub fn from_gguf_metadata(metadata: &HashMap<String, GgufValue>) -> Result<Self> {
        // Detect tokenizer model type
        let model = match metadata
            .get("tokenizer.ggml.model")
            .and_then(GgufValue::as_str)
        {
            Some("gpt2") => TokenizerModel::Gpt2,
            _ => TokenizerModel::SentencePiece,
        };

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

        // Build vocab: decode token strings to raw bytes.
        // SentencePiece uses `<0xHH>` byte-fallback tokens.
        // GPT2 uses a Unicode remapping table (each byte maps to a printable char).
        let unicode_to_byte = if model == TokenizerModel::Gpt2 {
            Some(gpt2_unicode_to_byte())
        } else {
            None
        };

        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab_size);

        for (id, token_str) in token_strings.iter().enumerate() {
            let bytes = if let Some(ref u2b) = unicode_to_byte {
                decode_gpt2_token(token_str, u2b)
            } else {
                decode_token_str(token_str)
            };
            // Use insert() to prefer later tokens over earlier ones.
            // This ensures we use the actual character token (e.g., ',')
            // instead of byte-fallback tokens (e.g., '<0x2C>') which appear earlier.
            token_to_id.insert(bytes.clone(), id as u32);
            vocab.push(bytes);
        }

        // Build merge table from explicit merges if available (BPE tokenizers)
        // Otherwise fall back to score-based inference (Unigram tokenizers)
        let merge_scores: HashMap<(u32, u32), f32> =
            if let Some(GgufValue::Array(merges_arr)) = metadata.get("tokenizer.ggml.merges") {
                Self::build_merge_table_from_explicit_merges(
                    merges_arr,
                    &token_to_id,
                    unicode_to_byte.as_ref(),
                )?
            } else {
                Self::build_merge_table_from_scores(metadata, &vocab, &token_to_id, vocab_size)
            };

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
            model,
        })
    }

    /// Build merge table from explicit BPE merges.
    /// Each merge is "left right" format. Earlier merges have higher priority.
    fn build_merge_table_from_explicit_merges(
        merges_arr: &[GgufValue],
        token_to_id: &HashMap<Vec<u8>, u32>,
        unicode_to_byte: Option<&HashMap<char, u8>>,
    ) -> Result<HashMap<(u32, u32), f32>> {
        let mut merge_scores: HashMap<(u32, u32), f32> = HashMap::new();
        let total_merges = merges_arr.len();

        for (rank, merge_val) in merges_arr.iter().enumerate() {
            let merge_str = merge_val
                .as_str()
                .ok_or_else(|| Error::Tokenizer("Merge is not a string".into()))?;

            // Split by single space to get left and right tokens
            if let Some(space_idx) = merge_str.find(' ') {
                let left_str = &merge_str[..space_idx];
                let right_str = &merge_str[space_idx + 1..];

                let left_bytes = if let Some(u2b) = unicode_to_byte {
                    decode_gpt2_token(left_str, u2b)
                } else {
                    decode_token_str(left_str)
                };
                let right_bytes = if let Some(u2b) = unicode_to_byte {
                    decode_gpt2_token(right_str, u2b)
                } else {
                    decode_token_str(right_str)
                };

                if let (Some(&left_id), Some(&right_id)) =
                    (token_to_id.get(&left_bytes), token_to_id.get(&right_bytes))
                {
                    // Higher priority (lower rank) gets higher score.
                    // BPE merge tables can have 60k+ entries, but f32 has enough
                    // precision for ranking purposes.
                    #[allow(clippy::cast_precision_loss)]
                    let score = (total_merges - rank) as f32;
                    merge_scores.insert((left_id, right_id), score);
                }
            }
        }

        Ok(merge_scores)
    }

    /// Build merge table from token scores (for Unigram tokenizers).
    /// For every pair of tokens that concatenate to a known token,
    /// record the merge with the score of the result token.
    fn build_merge_table_from_scores(
        metadata: &HashMap<String, GgufValue>,
        vocab: &[Vec<u8>],
        token_to_id: &HashMap<Vec<u8>, u32>,
        vocab_size: usize,
    ) -> HashMap<(u32, u32), f32> {
        let scores: Vec<f32> =
            if let Some(GgufValue::Array(arr)) = metadata.get("tokenizer.ggml.scores") {
                arr.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect()
            } else {
                vec![0.0; vocab_size]
            };

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

        merge_scores
    }

    /// Encode text into token IDs using BPE.
    ///
    /// For SentencePiece models, prepends `▁` and replaces spaces with `▁`.
    /// For GPT2 models, each input byte is mapped to its single-byte token.
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

        let mut tokens: Vec<u32> = match self.model {
            TokenizerModel::SentencePiece => self.encode_sentencepiece(text)?,
            TokenizerModel::Gpt2 => self.encode_gpt2(text)?,
        };

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

    /// Initialize tokens for SentencePiece BPE.
    /// Prepends `▁`, replaces spaces with `▁`, then maps characters to token
    /// IDs with `<0xHH>` byte-fallback.
    fn encode_sentencepiece(&self, text: &str) -> Result<Vec<u32>> {
        let sp_text = format!("\u{2581}{}", text.replace(' ', "\u{2581}"));

        let mut tokens = Vec::new();
        for ch in sp_text.chars() {
            let char_str = ch.to_string();
            let char_bytes = char_str.as_bytes();

            if let Some(&id) = self.token_to_id.get(char_bytes) {
                tokens.push(id);
            } else {
                for &b in char_bytes {
                    if let Some(&id) = self.token_to_id.get(&[b][..]) {
                        tokens.push(id);
                    } else {
                        let hex_token = format!("<0x{b:02X}>");
                        let id = self
                            .token_to_id
                            .get(hex_token.as_bytes())
                            .copied()
                            .ok_or_else(|| {
                                Error::Tokenizer(format!("No token for byte 0x{b:02x}"))
                            })?;
                        tokens.push(id);
                    }
                }
            }
        }
        Ok(tokens)
    }

    /// Initialize tokens for GPT2 byte-level BPE.
    /// Each input byte maps to a single-byte token (the vocab was decoded
    /// through the GPT2 unicode→byte table at load time).
    fn encode_gpt2(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        for &b in text.as_bytes() {
            let id = self
                .token_to_id
                .get(&[b][..])
                .copied()
                .ok_or_else(|| Error::Tokenizer(format!("No token for byte 0x{b:02x}")))?;
            tokens.push(id);
        }
        Ok(tokens)
    }

    /// Decode token IDs to a string.
    ///
    /// For SentencePiece, replaces `▁` (U+2581) with space and trims the
    /// leading space. For GPT2, the vocab already contains raw bytes.
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

        match self.model {
            TokenizerModel::SentencePiece => {
                let text = String::from_utf8_lossy(&bytes).replace('\u{2581}', " ");
                // Trim leading space (from the ▁ we prepend during encoding)
                Ok(text.strip_prefix(' ').unwrap_or(&text).to_string())
            }
            TokenizerModel::Gpt2 => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        }
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

        let token_bytes = &self.vocab[id_usize];
        match self.model {
            TokenizerModel::SentencePiece => {
                Ok(String::from_utf8_lossy(token_bytes).replace('\u{2581}', " "))
            }
            TokenizerModel::Gpt2 => Ok(String::from_utf8_lossy(token_bytes).into_owned()),
        }
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

impl super::Tokenizer for GgufTokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        self.encode(text, add_bos)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.decode(ids)
    }

    fn decode_token(&self, id: u32) -> Result<String> {
        self.decode_token(id)
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id()
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

/// Decode a GPT2 token string to raw bytes.
///
/// GPT2 token strings use a Unicode remapping where each byte 0x00–0xFF is
/// represented as a printable Unicode character (e.g., space 0x20 → `Ġ`
/// U+0120). This reverses that mapping to get raw bytes.
fn decode_gpt2_token(s: &str, unicode_to_byte: &HashMap<char, u8>) -> Vec<u8> {
    s.chars()
        .map(|c| unicode_to_byte.get(&c).copied().unwrap_or(c as u8))
        .collect()
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
        make_metadata_with_merges(tokens, scores, bos, eos, &[])
    }

    fn make_metadata_with_merges(
        tokens: &[&str],
        scores: &[f32],
        bos: u32,
        eos: u32,
        merges: &[&str],
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
        if !merges.is_empty() {
            m.insert(
                "tokenizer.ggml.merges".into(),
                GgufValue::Array(
                    merges
                        .iter()
                        .map(|s| GgufValue::String((*s).into()))
                        .collect(),
                ),
            );
        }
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
        // Minimal vocab with explicit BPE merges
        // ▁ = U+2581 = [0xE2, 0x96, 0x81]
        let tokens = &[
            "<unk>",         // 0
            "<s>",           // 1 (BOS)
            "</s>",          // 2 (EOS)
            "\u{2581}",      // 3 — space placeholder
            "h",             // 4
            "e",             // 5
            "l",             // 6
            "o",             // 7
            "\u{2581}h",     // 8 — merged "▁h"
            "ll",            // 9 — merged "ll"
            "llo",           // 10 — merged "llo"
            "ello",          // 11 — merged "ello"
            "\u{2581}hello", // 12 — full merge
        ];
        let scores: &[f32] = &[0.0; 13]; // Scores not used when merges are explicit

        // BPE merge order (earlier = higher priority)
        let merges = &[
            "\u{2581}h ello", // merge to ▁hello (token 12)
            "l l",            // merge to ll (token 9)
            "ll o",           // merge to llo (token 10)
            "e llo",          // merge to ello (token 11)
            "\u{2581} h",     // merge to ▁h (token 8)
        ];

        let meta = make_metadata_with_merges(tokens, scores, 1, 2, merges);
        let tok = GgufTokenizer::from_gguf_metadata(&meta).unwrap();

        // "hello" -> prepend ▁ -> "▁hello"
        // Initial tokens: ▁(3), h(4), e(5), l(6), l(6), o(7)
        // Merges: l+l→ll, ll+o→llo, e+llo→ello, ▁h+ello→▁hello
        let ids = tok.encode("hello", false).unwrap();
        assert_eq!(ids, vec![12]); // single token "▁hello"

        // With BOS
        let ids_bos = tok.encode("hello", true).unwrap();
        assert_eq!(ids_bos, vec![1, 12]);
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

    // ─── GPT2 BPE tests ─────────────────────────────────────────────────

    /// Build the GPT2 byte→unicode table and create a helper to convert
    /// raw bytes to a GPT2 token string.
    fn bytes_to_gpt2_str(raw: &[u8]) -> String {
        let table = gpt2_byte_to_unicode();
        raw.iter().map(|&b| table[b as usize]).collect()
    }

    fn make_gpt2_metadata(
        tokens: &[&str],
        merges: &[&str],
        bos: u32,
        eos: u32,
    ) -> HashMap<String, GgufValue> {
        let mut m = HashMap::new();
        m.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("gpt2".into()),
        );
        m.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(
                tokens
                    .iter()
                    .map(|t| GgufValue::String((*t).into()))
                    .collect(),
            ),
        );
        m.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(bos));
        m.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(eos));
        if !merges.is_empty() {
            m.insert(
                "tokenizer.ggml.merges".into(),
                GgufValue::Array(
                    merges
                        .iter()
                        .map(|s| GgufValue::String((*s).into()))
                        .collect(),
                ),
            );
        }
        m
    }

    #[test]
    fn test_gpt2_byte_to_unicode_roundtrip() {
        let fwd = gpt2_byte_to_unicode();
        let inv = gpt2_unicode_to_byte();

        // Every byte should roundtrip
        for b in 0u8..=255 {
            let c = fwd[b as usize];
            assert_eq!(inv[&c], b, "Roundtrip failed for byte 0x{b:02X}");
        }

        // All 256 entries should map to unique chars
        let mut chars: Vec<char> = fwd.to_vec();
        chars.sort_unstable();
        chars.dedup();
        assert_eq!(chars.len(), 256);
    }

    #[test]
    fn test_gpt2_space_is_remapped() {
        let table = gpt2_byte_to_unicode();
        // Space (0x20) should map to Ġ (U+0120)
        assert_eq!(table[0x20], '\u{0120}');
    }

    #[test]
    fn test_gpt2_basic_encoding() {
        // Build a minimal GPT2 vocab:
        // Single-byte tokens for all 256 bytes + a few merges
        let table = gpt2_byte_to_unicode();
        let mut tokens: Vec<String> = (0u8..=255).map(|b| table[b as usize].to_string()).collect();

        // Token 256: "he" (merged h + e)
        tokens.push(bytes_to_gpt2_str(b"he"));
        // Token 257: "Ġt" (merged space + t) — "Ġ" is GPT2's space
        tokens.push(bytes_to_gpt2_str(b" t"));
        // Token 258: "Ġthe" — not directly built from merges in this test

        let token_strs: Vec<&str> = tokens.iter().map(String::as_str).collect();

        // Merges: "h e" → "he" (token 256), "Ġ t" → "Ġt" (token 257)
        let space_char = table[0x20];
        let merge1 = format!("h e");
        let merge2 = format!("{space_char} t");
        let merges = vec![merge1.as_str(), merge2.as_str()];

        let meta = make_gpt2_metadata(&token_strs, &merges, 0, 0);
        let tok = GgufTokenizer::from_gguf_metadata(&meta).unwrap();

        // "the" → bytes [0x74, 0x68, 0x65] → initial tokens [t, h, e]
        // Merge h+e → he(256), then no more merges → [t, he]
        let ids = tok.encode("the", false).unwrap();
        assert_eq!(ids, vec![b't' as u32, 256]);

        // " t" → bytes [0x20, 0x74] → merge → token 257
        let ids2 = tok.encode(" t", false).unwrap();
        assert_eq!(ids2, vec![257]);
    }

    #[test]
    fn test_gpt2_decode_roundtrip() {
        let table = gpt2_byte_to_unicode();
        let tokens: Vec<String> = (0u8..=255).map(|b| table[b as usize].to_string()).collect();
        let token_strs: Vec<&str> = tokens.iter().map(String::as_str).collect();

        let meta = make_gpt2_metadata(&token_strs, &[], 0, 0);
        let tok = GgufTokenizer::from_gguf_metadata(&meta).unwrap();

        let text = "Hello, world!";
        let ids = tok.encode(text, false).unwrap();
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_gpt2_no_space_prepend() {
        // GPT2 should NOT prepend a space like SentencePiece does
        let table = gpt2_byte_to_unicode();
        let tokens: Vec<String> = (0u8..=255).map(|b| table[b as usize].to_string()).collect();
        let token_strs: Vec<&str> = tokens.iter().map(String::as_str).collect();

        let meta = make_gpt2_metadata(&token_strs, &[], 0, 0);
        let tok = GgufTokenizer::from_gguf_metadata(&meta).unwrap();

        // "Hi" should encode to exactly [H, i] with no leading space token
        let ids = tok.encode("Hi", false).unwrap();
        assert_eq!(ids, vec![b'H' as u32, b'i' as u32]);
    }
}
