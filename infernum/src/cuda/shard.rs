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
        shard.shard_range(128);
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
}
