//! Tokenizer integration

#[cfg(feature = "cuda")]
mod gguf_tokenizer;
mod llama_tokenizer;

#[cfg(feature = "cuda")]
pub use gguf_tokenizer::GgufTokenizer;
pub use llama_tokenizer::LlamaTokenizer;
