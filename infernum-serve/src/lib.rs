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
mod server2;
pub mod types;

#[cfg(feature = "cuda")]
pub use infernum_runtime::BatchConfig;
#[cfg(feature = "cuda")]
pub use server::{ModelEntry, Server, ServerBuilder};

pub use infernum_runtime::BatchConfig2;
pub use server2::{ModelEntry2, Server2, ServerBuilder2};
