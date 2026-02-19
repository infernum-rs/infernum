//! Operations (kernels) for tensor computations

mod add;
mod add_rmsnorm;
mod argmax;
mod attention;
mod embed;
mod fused_attention;
mod matmul;
mod quantized_matmul;
mod repeat_kv;
mod rmsnorm;
mod rope;
mod sample;
mod silu;
mod softmax;
mod transpose;

pub use add::{add, add_inplace};
pub use add_rmsnorm::{add_rmsnorm, ADD_RMSNORM_FUSED};
pub use argmax::{argmax_last, argmax_last_scalar};
pub use attention::{attention, attention_kv, ATTENTION_KV_FUSED};
pub use embed::embedding_gather;
pub use fused_attention::{fused_attention_decode, fused_attention_prefill};
pub use matmul::matmul;
pub use quantized_matmul::quantized_matmul;
pub use repeat_kv::repeat_kv;
pub use rmsnorm::{rms_norm, rms_norm_inplace};
pub use rope::{apply_rope, precompute_rope_cache};
pub use sample::sample_top_p;
pub use silu::{silu, silu_inplace, silu_mul};
pub use softmax::{softmax, softmax_causal};
pub use transpose::{transpose_012_to_102, transpose_2d, transpose_last_two};
