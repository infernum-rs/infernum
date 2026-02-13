//! Sampling operations for token generation
//!
//! Provides nucleus (top-p) sampling with temperature scaling.
//! Logits are pulled to CPU for the sort + cumulative-softmax step,
//! which is fast enough for single-row vocab-sized vectors (~50K).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc
)]

use crate::cuda::CudaTensor;
use crate::tensor::Tensor;
use crate::Result;

/// Sample a token from logits using nucleus (top-p) sampling with temperature.
///
/// 1. Extract the last row of the `(seq_len, vocab_size)` logits matrix.
/// 2. Divide by `temperature` (higher → more random).
/// 3. Sort descending, compute cumulative softmax.
/// 4. Mask out tokens whose cumulative probability exceeds `top_p`.
/// 5. Re-normalise and sample from the remaining distribution.
///
/// # Arguments
/// * `logits` - 2D tensor of shape `(seq_len, vocab_size)`
/// * `temperature` - Scaling factor (must be > 0)
/// * `top_p` - Nucleus probability threshold in `(0, 1]`
/// * `rng_seed` - Seed for the lightweight xorshift PRNG
///
/// # Errors
/// Returns an error if the GPU → CPU transfer fails.
pub fn sample_top_p(
    logits: &CudaTensor<f32>,
    temperature: f32,
    top_p: f32,
    rng_seed: u64,
) -> Result<u32> {
    assert!(
        logits.shape().len() == 2,
        "sample_top_p expects 2D logits, got shape {:?}",
        logits.shape()
    );
    assert!(temperature > 0.0, "temperature must be > 0");
    assert!(top_p > 0.0 && top_p <= 1.0, "top_p must be in (0, 1]");

    let seq_len = logits.shape()[0];
    let vocab_size = logits.shape()[1];

    // Pull entire logits to CPU and take the last row
    let all_logits = logits.to_vec()?;
    let last_row = &all_logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];

    Ok(sample_from_logits(last_row, temperature, top_p, rng_seed))
}

/// Pure-CPU sampling from a single row of logits (no GPU dependency).
/// Useful for testing and as the inner implementation.
fn sample_from_logits(logits: &[f32], temperature: f32, top_p: f32, rng_seed: u64) -> u32 {
    let n = logits.len();

    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();

    // Stable softmax: subtract max, then exp
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Build (index, probability) pairs and sort descending by probability
    let mut indexed: Vec<(u32, f32)> = (0..n as u32).zip(probs).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Compute cumulative probability and find the nucleus
    let mut cumulative = 0.0_f32;
    let mut nucleus_end = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumulative += p;
        if cumulative >= top_p {
            nucleus_end = i + 1;
            break;
        }
    }

    let nucleus = &indexed[..nucleus_end];

    // Re-normalise within the nucleus
    let nucleus_sum: f32 = nucleus.iter().map(|&(_, p)| p).sum();

    // Sample using a lightweight xorshift PRNG
    let mut state = rng_seed.wrapping_add(1); // avoid zero state
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    let u = (state as f32) / (u64::MAX as f32); // uniform in [0, 1)

    let mut acc = 0.0_f32;
    for &(idx, p) in nucleus {
        acc += p / nucleus_sum;
        if u < acc {
            return idx;
        }
    }

    // Fallback to the last token in the nucleus (rounding)
    nucleus.last().unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_deterministic_with_seed() {
        // With a fixed seed, sampling should be deterministic
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let t1 = sample_from_logits(&logits, 1.0, 1.0, 42);
        let t2 = sample_from_logits(&logits, 1.0, 1.0, 42);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_sample_temperature_zero_point_one_picks_argmax() {
        // Very low temperature should almost always pick the highest-logit token
        let logits = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        for seed in 0..100 {
            let token = sample_from_logits(&logits, 0.1, 1.0, seed);
            assert_eq!(token, 2, "Low temperature should pick argmax (seed={seed})");
        }
    }

    #[test]
    fn test_sample_top_p_restricts_nucleus() {
        // Logits strongly favouring index 0 (prob ~0.99 after softmax at temp=1)
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        // With top_p = 0.5, the nucleus should contain only index 0
        for seed in 0..100 {
            let token = sample_from_logits(&logits, 1.0, 0.5, seed);
            assert_eq!(
                token, 0,
                "top_p=0.5 should restrict to dominant token (seed={seed})"
            );
        }
    }

    #[test]
    fn test_sample_top_p_one_allows_all() {
        // top_p = 1.0 means full distribution; different seeds can produce different tokens
        let logits = vec![1.0, 1.0, 1.0, 1.0]; // uniform
        let mut seen = std::collections::HashSet::new();
        for seed in 0..1000 {
            seen.insert(sample_from_logits(&logits, 1.0, 1.0, seed));
        }
        assert!(
            seen.len() > 1,
            "Uniform logits with top_p=1.0 should produce multiple distinct tokens"
        );
    }

    #[test]
    fn test_sample_returns_valid_index() {
        let logits = vec![1.0; 49152]; // typical vocab size
        let token = sample_from_logits(&logits, 1.0, 0.9, 12345);
        assert!((token as usize) < logits.len());
    }

    #[cfg(feature = "cuda")]
    mod gpu {
        use super::super::*;
        use crate::cuda::CudaContext;

        #[test]
        fn test_sample_top_p_gpu() {
            let ctx = CudaContext::new(0).expect("CUDA context");

            let data: Vec<f32> = vec![
                0.0, 0.0, 10.0, 0.0, // row 0
                0.0, 0.0, 10.0, 0.0, // row 1 (last row picked)
            ];
            let logits = CudaTensor::from_slice(&ctx, &[2, 4], &data).unwrap();

            let token = sample_top_p(&logits, 0.1, 1.0, 42).unwrap();
            assert_eq!(token, 2);
        }
    }
}
