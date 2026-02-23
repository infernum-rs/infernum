//! Operations (kernels) for tensor computations

mod add;
mod add_rmsnorm;
mod argmax;
mod attention;
mod bias_add;
mod cast;
mod embed;
mod fused_attention;
mod matmul;
mod mul;
mod paged_attention;
mod quantized_matmul;
mod repeat_kv;
mod rmsnorm;
mod rope;
mod sample;
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
pub use matmul::{matmul, matmul_bf16_f32, GemmScalar};
pub use mul::mul;
pub use paged_attention::{gather_paged_kv, paged_attention_decode};
pub use quantized_matmul::quantized_matmul;
pub use repeat_kv::repeat_kv;
pub use rmsnorm::{rms_norm, rms_norm_inplace};
pub use rope::{
    apply_rope, apply_rope_batched, apply_rope_indirect, precompute_rope_cache,
    precompute_rope_cache_scaled, RopeScaling,
};
pub use sample::sample_top_p;
pub use silu::{silu, silu_inplace, silu_mul};
pub use softmax::{softmax, softmax_causal};
pub use swiglu::swiglu;
pub use transpose::{transpose_012_to_102, transpose_2d, transpose_2d_bf16, transpose_last_two};
