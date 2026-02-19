//! Operations (kernels) for tensor computations

mod add;
mod argmax;
mod attention;
mod embed;
mod matmul;
mod quantized_matmul;
mod repeat_kv;
mod rmsnorm;
mod rope;
mod sample;
mod silu;
mod softmax;
mod transpose;

pub use add::add;
pub use argmax::argmax_last;
pub use attention::attention;
pub use embed::embedding_gather;
pub use matmul::matmul;
pub use quantized_matmul::quantized_matmul;
pub use repeat_kv::repeat_kv;
pub use rmsnorm::rms_norm;
pub use rope::{apply_rope, precompute_rope_cache};
pub use sample::sample_top_p;
pub use silu::{silu, silu_mul};
pub use softmax::{softmax, softmax_causal};
pub use transpose::{transpose_012_to_102, transpose_2d, transpose_last_two};
