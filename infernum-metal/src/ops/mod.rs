//! Metal implementations of all backend op traits.
//!
//! Op modules are added incrementally as kernels are implemented.

pub mod activation;
pub mod arith;
pub mod bias;
pub mod cast;
pub mod data;
pub mod factory;
pub mod norm;
