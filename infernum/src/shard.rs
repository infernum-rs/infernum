//! Tensor parallelism configuration types
//!
//! Describes how a model is sharded across multiple GPUs for tensor-parallel
//! inference. These types are used by model code and weight loaders regardless
//! of whether NCCL is available (the actual communication requires the `nccl`
//! feature).

/// Describes this GPU's position in a tensor-parallel group.
#[derive(Debug, Clone, Copy)]
pub struct ShardConfig {
    /// This GPU's rank (`0..world_size`)
    pub rank: usize,
    /// Total number of GPUs in the tensor-parallel group
    pub world_size: usize,
}

impl ShardConfig {
    /// Compute the shard range for a dimension of size `dim`.
    ///
    /// Returns `(start, shard_size)` where the shard covers
    /// `[start .. start + shard_size)`.
    ///
    /// # Panics
    /// Panics if `dim` is not evenly divisible by `world_size`.
    #[must_use]
    pub fn shard_range(&self, dim: usize) -> (usize, usize) {
        assert_eq!(
            dim % self.world_size,
            0,
            "Dimension {dim} is not evenly divisible by world_size {}",
            self.world_size
        );
        let shard_size = dim / self.world_size;
        let start = self.rank * shard_size;
        (start, shard_size)
    }
}

/// Single-GPU or tensor-parallel configuration.
///
/// Passed to model constructors to determine whether weights should be
/// sharded and whether all-reduce is needed in the forward pass.
#[derive(Debug, Clone, Copy)]
pub enum GpuConfig {
    /// Single GPU, no communication needed.
    Single,
    /// Part of a tensor-parallel group.
    Sharded(ShardConfig),
}

impl GpuConfig {
    /// Get the shard config, if sharded.
    #[must_use]
    pub fn shard(&self) -> Option<&ShardConfig> {
        match self {
            Self::Single => None,
            Self::Sharded(s) => Some(s),
        }
    }

    /// World size: 1 for single-GPU, N for sharded.
    #[must_use]
    pub fn world_size(&self) -> usize {
        match self {
            Self::Single => 1,
            Self::Sharded(s) => s.world_size,
        }
    }
}

/// How a specific weight tensor should be sliced during loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Replicate the full tensor on every rank (norms, embeddings).
    Replicate,
    /// Column-parallel: split along the output dimension (rows of the weight
    /// matrix). Each rank gets rows `[rank*N/tp .. (rank+1)*N/tp]`.
    /// Used for: `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`.
    Column,
    /// Row-parallel: split along the input dimension (columns of the weight
    /// matrix). Each rank gets columns `[rank*K/tp .. (rank+1)*K/tp]`.
    /// Used for: `o_proj`, `down_proj`. Requires all-reduce after matmul.
    Row,
}

/// Determine the shard strategy for a weight tensor by its name.
///
/// Standard transformer projection naming convention:
/// - Column-parallel (split output dim): `q_proj`, `k_proj`, `v_proj`,
///   `gate_proj`, `up_proj`
/// - Row-parallel (split input dim): `o_proj`, `down_proj`
/// - `MoE` experts: `w1`/`w3` Column, `w2` Row (Mixtral naming)
/// - Replicate: norms, embeddings, `RoPE` caches, `lm_head`, router gate,
///   scales, everything else
#[must_use]
pub fn shard_strategy_for_weight(name: &str) -> ShardStrategy {
    // Scale tensors are always replicated (per-tensor scalars)
    if name.ends_with("_scale") {
        return ShardStrategy::Replicate;
    }

    // Column-parallel projections (split along output dimension)
    if name.ends_with("q_proj.weight")
        || name.ends_with("k_proj.weight")
        || name.ends_with("v_proj.weight")
        || name.ends_with("gate_proj.weight")
        || name.ends_with("up_proj.weight")
    {
        return ShardStrategy::Column;
    }

    // Row-parallel projections (split along input dimension)
    if name.ends_with("o_proj.weight") || name.ends_with("down_proj.weight") {
        return ShardStrategy::Row;
    }

    // MoE expert projections (Mixtral SafeTensors naming)
    // w1 = gate_proj (column-parallel), w3 = up_proj (column-parallel)
    // w2 = down_proj (row-parallel)
    // Router gate falls through to Replicate below.
    if name.contains(".block_sparse_moe.experts.") {
        if name.ends_with(".w1.weight") || name.ends_with(".w3.weight") {
            return ShardStrategy::Column;
        }
        if name.ends_with(".w2.weight") {
            return ShardStrategy::Row;
        }
    }

    // Everything else: norms, embeddings, lm_head, router gate
    ShardStrategy::Replicate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_range() {
        let shard = ShardConfig {
            rank: 0,
            world_size: 4,
        };
        assert_eq!(shard.shard_range(128), (0, 32));

        let shard = ShardConfig {
            rank: 3,
            world_size: 4,
        };
        assert_eq!(shard.shard_range(128), (96, 32));
    }

    #[test]
    #[should_panic(expected = "not evenly divisible")]
    fn test_shard_range_indivisible() {
        let shard = ShardConfig {
            rank: 0,
            world_size: 3,
        };
        let _ = shard.shard_range(128);
    }

    #[test]
    fn test_gpu_config_single() {
        let config = GpuConfig::Single;
        assert!(config.shard().is_none());
        assert_eq!(config.world_size(), 1);
    }

    #[test]
    fn test_gpu_config_sharded() {
        let config = GpuConfig::Sharded(ShardConfig {
            rank: 1,
            world_size: 4,
        });
        assert!(config.shard().is_some());
        assert_eq!(config.world_size(), 4);
        assert_eq!(config.shard().unwrap().rank, 1);
    }

    #[test]
    fn test_shard_strategy_column_parallel() {
        let column_names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.5.self_attn.k_proj.weight",
            "model.layers.31.self_attn.v_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            // MoE expert projections: w1 = gate_proj, w3 = up_proj
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.7.w3.weight",
            "model.layers.31.block_sparse_moe.experts.3.w1.weight",
        ];
        for name in &column_names {
            assert!(
                matches!(shard_strategy_for_weight(name), ShardStrategy::Column),
                "{name} should be Column"
            );
        }
    }

    #[test]
    fn test_shard_strategy_row_parallel() {
        let row_names = [
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.31.mlp.down_proj.weight",
            // MoE expert projections: w2 = down_proj
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            "model.layers.31.block_sparse_moe.experts.5.w2.weight",
        ];
        for name in &row_names {
            assert!(
                matches!(shard_strategy_for_weight(name), ShardStrategy::Row),
                "{name} should be Row"
            );
        }
    }

    #[test]
    fn test_shard_strategy_replicate() {
        let replicate_names = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight_scale",
            // MoE router gate: replicated on all ranks
            "model.layers.0.block_sparse_moe.gate.weight",
            // Gemma-specific norms
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        ];
        for name in &replicate_names {
            assert!(
                matches!(shard_strategy_for_weight(name), ShardStrategy::Replicate),
                "{name} should be Replicate"
            );
        }
    }
}
