//! Metal implementations of all backend op traits.
//!
//! Op modules are added incrementally as kernels are implemented.

pub mod activation;
pub mod arith;
pub mod attention;
pub mod bias;
pub mod cast;
pub mod data;
pub mod embed;
pub mod factory;
pub mod matmul;
pub mod matmul_ext;
pub mod moe;
pub mod norm;
pub mod rope;
pub mod tensor_ops;
