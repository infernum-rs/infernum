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
