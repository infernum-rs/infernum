//! Metal logits implementation with host-side sampling.
//!
//! Since Metal uses unified memory (`StorageModeShared`), logits data
//! is directly accessible from the CPU without an explicit download.

use infernum::logits::Logits;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::MetalTensor;

/// Metal logits: data lives in a shared Metal buffer, sampling is CPU-side.
pub struct MetalLogits {
    data: Vec<f32>,
    vocab_size: usize,
    batch_size: usize,
}

// SAFETY: MetalLogits owns a Vec<f32> (pure host data).
unsafe impl Send for MetalLogits {}

impl MetalLogits {
    /// Create from a `MetalTensor` containing logits.
    ///
    /// Reads the tensor data via unified memory and copies to a `Vec<f32>`.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_tensor(tensor: MetalTensor) -> Self {
        let shape = tensor.shape().to_vec();
        let batch_size = if shape.len() >= 2 { shape[0] } else { 1 };
        let vocab_size = *shape.last().unwrap();

        // Read logits from shared memory
        let data = if tensor.dtype() == infernum::DType::F32 {
            tensor.as_f32_slice().to_vec()
        } else {
            // Cast non-f32 logits to f32
            let bytes = tensor.as_bytes();
            match tensor.dtype() {
                infernum::DType::BF16 => {
                    let bf16s: &[half::bf16] = bytemuck::cast_slice(bytes);
                    bf16s.iter().map(|v| v.to_f32()).collect()
                }
                infernum::DType::F16 => {
                    let f16s: &[half::f16] = bytemuck::cast_slice(bytes);
                    f16s.iter().map(|v| v.to_f32()).collect()
                }
                other => panic!("Unsupported logits dtype: {other}"),
            }
        };

        Self {
            data,
            vocab_size,
            batch_size,
        }
    }
}

impl Logits for MetalLogits {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn argmax(&self, batch_index: usize) -> Result<u32> {
        let start = batch_index * self.vocab_size;
        let row = &self.data[start..start + self.vocab_size];
        let mut max_idx = 0u32;
        let mut max_val = f32::NEG_INFINITY;
        #[allow(clippy::cast_possible_truncation)]
        for (i, &v) in row.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i as u32;
            }
        }
        Ok(max_idx)
    }

    fn sample_top_p(
        &self,
        batch_index: usize,
        temperature: f32,
        top_p: f32,
        rng_seed: u64,
        repetition_penalty: f32,
        recent_tokens: &[u32],
    ) -> Result<u32> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let start = batch_index * self.vocab_size;
        let mut logits = self.data[start..start + self.vocab_size].to_vec();

        // Apply repetition penalty
        #[allow(clippy::float_cmp)]
        if repetition_penalty != 1.0 {
            for &tok in recent_tokens {
                let idx = tok as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= repetition_penalty;
                    } else {
                        logits[idx] *= repetition_penalty;
                    }
                }
            }
        }

        // Temperature
        #[allow(clippy::float_cmp)]
        if temperature != 1.0 {
            for l in &mut logits {
                *l /= temperature;
            }
        }

        // Softmax
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Sort by probability descending
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Top-p nucleus
        let mut cumulative = 0.0f32;
        let mut cutoff = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumulative += p;
            if cumulative >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        let nucleus = &indexed[..cutoff];

        // Renormalize
        let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
        let mut rng = StdRng::seed_from_u64(rng_seed);
        let r: f32 = rng.gen();
        let mut cumul = 0.0;
        for &(idx, p) in nucleus {
            cumul += p / nucleus_sum;
            if cumul >= r {
                #[allow(clippy::cast_possible_truncation)]
                return Ok(idx as u32);
            }
        }

        // Fallback: most probable
        #[allow(clippy::cast_possible_truncation)]
        Ok(nucleus[0].0 as u32)
    }
}
