//! Integration tests for the Gemma model family.
//!
//! Gated behind `--features integration`. Tests download real models from
//! HuggingFace and verify end-to-end generation output.

#![cfg(feature = "integration")]
