//! Weight loading utilities

mod loader;
#[cfg(feature = "cuda")]
mod safetensors;

#[cfg(feature = "cuda")]
pub use loader::WeightLoader;
pub use loader::WeightNameMapper;
#[cfg(feature = "cuda")]
pub use safetensors::SafeTensorsLoader;
