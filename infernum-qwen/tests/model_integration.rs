//! Integration tests for Qwen model family
//!
//! These tests download real models from HuggingFace and verify end-to-end generation.
//! They are gated behind the `integration` feature.
#![cfg(feature = "integration")]
