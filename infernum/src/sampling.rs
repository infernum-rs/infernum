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
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
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
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            eos_token_id: None,
            sampling: None,
            use_kv_cache: true,
        }
    }
}
