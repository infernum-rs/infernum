//! Operations (kernels) for tensor computations

mod add;
mod attention;
mod embed;
mod matmul;
mod repeat_kv;
mod rmsnorm;
mod rope;
mod silu;
mod softmax;
mod transpose;

pub use add::add;
pub use attention::attention;
pub use embed::embedding_gather;
pub use matmul::matmul;
pub use repeat_kv::repeat_kv;
pub use rmsnorm::rms_norm;
pub use rope::{apply_rope, precompute_rope_cache};
pub use silu::{silu, silu_mul};
pub use softmax::{softmax, softmax_causal};
pub use transpose::{transpose_012_to_102, transpose_2d, transpose_last_two};
