//! Operations (kernels) for tensor computations

mod add;
mod add_rmsnorm;
mod argmax;
mod attention;
mod bias_add;
mod cast;
mod embed;
mod fused_attention;
mod geglu;
mod gelu;
mod linear;
mod matmul;
mod mla_tensor_ops;
mod mul;
mod paged_attention;
mod quantized_matmul;
mod repeat_kv;
mod rmsnorm;
mod rope;
mod sample;
mod scale;
mod silu;
mod softmax;
mod swiglu;
mod transpose;

pub use add::{add, add_inplace};
pub use add_rmsnorm::add_rmsnorm;
pub use argmax::{argmax_last, argmax_last_scalar};
pub use attention::{attention, attention_kv};
pub use bias_add::{bias_add, bias_add_inplace};
pub use cast::{cast_bf16_to_f32, cast_f32_to_bf16, cast_to_f32};
pub use embed::{embedding_gather, embedding_gather_from_device};
pub use fused_attention::{
    fused_attention_decode, fused_attention_decode_indirect, fused_attention_prefill,
};
pub use geglu::geglu;
pub use gelu::{gelu, gelu_inplace, gelu_mul};
pub use linear::{linear, reinterpret_tensor, LinearWeight};
pub use matmul::{matmul, matmul_bf16_f32, GemmScalar};
pub use mla_tensor_ops::{broadcast_to_heads, concat_inner_dim, pad_inner_dim, split_inner_dim};
pub use mul::{mul, scale_f32_inplace};
pub use paged_attention::{
    gather_paged_kv, paged_attention_decode, paged_attention_decode_indirect,
};
pub use quantized_matmul::quantized_matmul;
pub use repeat_kv::repeat_kv;
pub use rmsnorm::{rms_norm, rms_norm_inplace};
pub use rope::{
    apply_rope, apply_rope_batched, apply_rope_batched_indirect, apply_rope_indirect,
    apply_rope_interleaved, apply_rope_interleaved_indirect, precompute_rope_cache,
    precompute_rope_cache_scaled, RopeScaling,
};
pub use sample::sample_top_p;
pub use scale::scale_inplace;
pub use silu::{silu, silu_inplace, silu_mul};
pub use softmax::{softmax, softmax_causal};
pub use swiglu::swiglu;
pub use transpose::{transpose_012_to_102, transpose_2d, transpose_2d_bf16, transpose_last_two};
