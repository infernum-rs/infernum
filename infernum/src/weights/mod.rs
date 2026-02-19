//! Weight loading utilities

#[cfg(feature = "cuda")]
mod gguf;
mod loader;
#[cfg(feature = "cuda")]
mod safetensors;

#[cfg(feature = "cuda")]
pub use gguf::{GgufLoader, GgufValue};
#[cfg(feature = "cuda")]
pub use loader::WeightLoader;
#[cfg(feature = "cuda")]
pub use safetensors::SafeTensorsLoader;
