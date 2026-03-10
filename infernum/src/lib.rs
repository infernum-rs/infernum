//! Infernum: A Rust-based LLM inference server
//!
//! This crate provides the core types and traits for LLM inference.
//! Backend-specific implementations (CUDA, CPU, etc.) are in separate crates.

// Allow proc macros to use `::infernum::` paths from within this crate.
extern crate self as infernum;

#[allow(clippy::missing_errors_doc, clippy::doc_markdown)]
pub mod backend;
pub mod block_allocator;
pub mod chat_template;
pub mod dtype;
pub mod error;
pub mod fusion;
pub mod gguf_meta;
pub mod graph;
pub mod logits;
pub mod model;
pub mod rope;
pub mod runtime_state;
pub mod sampling;
pub mod serde_helpers;
pub mod shard;
pub mod sharded;
pub mod tensor;
pub mod tokenizer;
pub mod transformer;
pub mod weights;

pub use backend::{
<<<<<<< HEAD
    ArithOps, AttentionOps, Backend, BiasOps, CastOps, Comm, ContextBackend, DecodeBufferOps,
    DecodeTensors, EmbedOps, GegluOps, KvCacheOps, MatmulExtOps, MatmulOps, MlaAttentionOps,
    MoeOps, MoeSigmoidOps, MultiDeviceOps, NormOps, PagedAttentionOps, PagedKvCacheOps,
    RopeInterleavedOps, RopeOps, SafeTensorsLoaderOps, SwigluOps, TensorDataOps, TensorFactory,
    TensorOps,
=======
    ArithOps, AttentionOps, Backend, BiasOps, CastOps, Comm, DecodeBufferOps, DecodeTensors,
    EmbedOps, FusedDecodeOps, GegluOps, KvCacheOps, MatmulExtOps, MatmulOps, MoeOps, MoeSigmoidOps,
    MultiDeviceOps, NormOps, PagedAttentionOps, PagedKvCacheOps, RopeInterleavedOps, RopeOps,
    SafeTensorsLoaderOps, SwigluOps, TensorDataOps, TensorFactory, TensorOps,
>>>>>>> e5dc174 (perf: fuse RoPE+KV-append and SwiGLU+down_proj dispatches)
};
pub use block_allocator::{BlockAllocator, BlockConfig, BlockTable};
pub use dtype::{DType, GPTQ_GROUP_SIZE, QUANTIZATION_BLOCK_SIZE};
pub use error::{path_to_utf8, Error, Result};
pub use gguf_meta::GgufValue;
pub use logits::Logits;
pub use model::{Model, ModelConfig};
pub use rope::{precompute_rope_data, precompute_rope_row, RopeScaling};
pub use runtime_state::{BatchConfig, RuntimeStateInit};
pub use sampling::{GenerateOptions, SamplingParams};
pub use shard::{shard_strategy_for_weight, GpuConfig, ShardConfig, ShardStrategy};
pub use sharded::{ShardedKvCache, ShardedModel};
pub use tensor::Tensor;
pub use tokenizer::GgufTokenizer;
pub use tokenizer::Tokenizer;
pub use weights::{QuantizationConfig, WeightLoader};

pub use chat_template::{ChatMessage, ChatTemplate, RawTemplate};
pub use graph::{
    plan, BufferSlot, ExecutionPlan, Graph, GraphArithOps, GraphAttentionOps, GraphBiasOps,
    GraphCastOps, GraphEmbedOps, GraphGegluOps, GraphMatmulExtOps, GraphMatmulOps, GraphNode,
    GraphNormOps, GraphPagedAttentionOps, GraphPagedKvCacheOps, GraphRopeInterleavedOps,
    GraphRopeOps, GraphSiluOps, GraphSoftcapOps, GraphSwigluOps, GraphTensorOps, NodeId, OpNode,
    OutputRef, WeightId, WeightMeta, WeightRef,
};
pub use serde_helpers::deserialize_u32_or_first;
