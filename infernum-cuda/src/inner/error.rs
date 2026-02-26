//! Error handling for the CUDA backend
//!
//! All `From` impls for cudarc error types → `infernum::Error` are provided
//! by `infernum`'s `cuda-errors` feature (enabled automatically by this crate).
//! This module is reserved for any future CUDA-backend-specific error helpers.
