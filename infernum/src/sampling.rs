//! Sampling parameters for text generation

/// Parameters for nucleus (top-p) sampling
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for logit scaling (higher = more random). Must be > 0.
    pub temperature: f32,
    /// Nucleus probability threshold in (0, 1]. Only tokens within the top-p
    /// cumulative probability mass are considered.
    pub top_p: f32,
    /// Seed for the PRNG. Same seed + same input → same output.
    pub seed: u64,
    /// Repetition penalty factor (CTRL paper). Values > 1.0 penalise tokens
    /// that have already appeared in the recent context window. A value of
    /// 1.0 disables the penalty entirely.
    pub repetition_penalty: f32,
    /// Number of recent tokens to consider for repetition penalty. Only the
    /// last `repetition_penalty_window` tokens are checked.
    pub repetition_penalty_window: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
            repetition_penalty: 1.0,
            repetition_penalty_window: 64,
        }
    }
}

/// Options for text generation.
///
/// Controls the generation loop: how many tokens to produce, when to stop,
/// whether to use greedy or sampled decoding, and whether to use KV cache.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Optional EOS token ID to stop generation early.
    pub eos_token_id: Option<u32>,
    /// Sampling parameters. `None` means greedy (argmax) decoding.
    pub sampling: Option<SamplingParams>,
    /// Whether to use the KV cache for efficient generation.
    /// When `false`, recomputes the full sequence at each step.
    pub use_kv_cache: bool,
    /// Whether to use CUDA graph capture/replay for the decode loop.
    /// Reduces per-token kernel launch overhead. Requires `use_kv_cache`.
    pub use_cuda_graphs: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            eos_token_id: None,
            sampling: None,
            use_kv_cache: true,
            use_cuda_graphs: false,
        }
    }
}
