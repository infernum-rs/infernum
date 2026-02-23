//! Infernum Serve: `OpenAI`-compatible HTTP server for LLM inference
//!
//! This crate provides an Axum-based HTTP server that exposes models via
//! the `OpenAI` Chat Completions API. It is a **library** — users construct
//! their own `main.rs`, load models however they want, and hand them to
//! the server.
//!
//! # Example
//!
//! ```ignore
//! use infernum_serve::{Server, ModelEntry};
//!
//! #[tokio::main]
//! async fn main() {
//!     let model = /* load model */;
//!     let tokenizer = /* load tokenizer */;
//!     let template = /* chat template */;
//!
//!     let server = Server::builder()
//!         .add_model(ModelEntry::new("my-model", model, tokenizer, template, None))
//!         .bind(([0, 0, 0, 0], 8080))
//!         .build();
//!
//!     server.run().await.unwrap();
//! }
//! ```

#[cfg(feature = "cuda")]
mod server;
pub mod types;

#[cfg(feature = "cuda")]
pub use server::{ModelEntry, Server, ServerBuilder};
