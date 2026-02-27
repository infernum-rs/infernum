//! RoPE (Rotary Positional Embeddings) precomputation.
//!
//! Pure host-side computation of cos/sin caches. The results are
//! `Vec<f32>` arrays that can be uploaded to any backend via
//! `TensorFactory::from_f32_slice`.

#![allow(clippy::cast_precision_loss, clippy::doc_markdown)]

/// Precompute RoPE cos/sin cache for standard (unscaled) RoPE.
///
/// Returns `(cos_data, sin_data)` each of length `max_seq_len * (head_dim / 2)`,
/// stored in row-major order `[max_seq_len, head_dim / 2]`.
#[must_use]
pub fn precompute_rope_data(
    max_seq_len: usize,
    head_dim: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos_data = vec![0.0_f32; max_seq_len * half_dim];
    let mut sin_data = vec![0.0_f32; max_seq_len * half_dim];

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            cos_data[pos * half_dim + i] = angle.cos();
            sin_data[pos * half_dim + i] = angle.sin();
        }
    }

    (cos_data, sin_data)
}

/// RoPE scaling configuration (from `config.json` `rope_scaling` field).
#[derive(Debug, Clone)]
pub struct RopeScaling {
    /// Scaling type: `"yarn"`, `"linear"`, etc.
    pub rope_type: String,
    /// Extension factor (e.g. 4.0 means 4× the original context)
    pub factor: f32,
    /// Original context length before scaling
    pub original_max_position_embeddings: usize,
}

/// Precompute RoPE cos/sin cache with YaRN or linear scaling.
///
/// YaRN splits dimensions into three bands (high-frequency, low-frequency,
/// middle) and applies differential scaling so that short-wavelength
/// dimensions keep their resolution while long-wavelength dimensions
/// are interpolated. A magnitude correction `sqrt(1 + 0.1 * ln(factor))`
/// is baked into the cos/sin values.
///
/// For `rope_type == "linear"`, all frequencies are uniformly divided by
/// `factor`.
///
/// Returns `(cos_data, sin_data)` each of length `max_seq_len * (head_dim / 2)`.
#[must_use]
pub fn precompute_rope_data_scaled(
    max_seq_len: usize,
    head_dim: usize,
    base: f32,
    scaling: &RopeScaling,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let factor = scaling.factor;
    let orig_max_pos = scaling.original_max_position_embeddings as f32;

    let mut cos_data = vec![0.0_f32; max_seq_len * half_dim];
    let mut sin_data = vec![0.0_f32; max_seq_len * half_dim];

    if scaling.rope_type == "yarn" {
        // YaRN parameters (matches HF transformers defaults)
        let beta_low = 1.0_f32;
        let beta_high = 32.0_f32;
        let low_freq_wavelen = orig_max_pos / beta_low;
        let high_freq_wavelen = orig_max_pos / beta_high;

        // Magnitude correction
        let attn_scale = (1.0 + 0.1 * factor.ln()).sqrt();

        for i in 0..half_dim {
            let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let wavelen = 2.0 * std::f32::consts::PI / freq;

            let scaled_freq = if wavelen < high_freq_wavelen {
                freq
            } else if wavelen > low_freq_wavelen {
                freq / factor
            } else {
                let ramp = (low_freq_wavelen / wavelen - 1.0)
                    / (low_freq_wavelen / high_freq_wavelen - 1.0);
                freq * (1.0 - ramp) / factor + freq * ramp
            };

            for pos in 0..max_seq_len {
                let angle = pos as f32 * scaled_freq;
                cos_data[pos * half_dim + i] = angle.cos() * attn_scale;
                sin_data[pos * half_dim + i] = angle.sin() * attn_scale;
            }
        }
    } else {
        // "linear" or unknown: uniform interpolation (freq / factor)
        for i in 0..half_dim {
            let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let scaled_freq = freq / factor;
            for pos in 0..max_seq_len {
                let angle = pos as f32 * scaled_freq;
                cos_data[pos * half_dim + i] = angle.cos();
                sin_data[pos * half_dim + i] = angle.sin();
            }
        }
    }

    (cos_data, sin_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_data_shape() {
        let (cos, sin) = precompute_rope_data(128, 64, 10000.0);
        assert_eq!(cos.len(), 128 * 32);
        assert_eq!(sin.len(), 128 * 32);
    }

    #[test]
    fn test_rope_data_values() {
        let (cos, sin) = precompute_rope_data(2, 4, 10000.0);
        // pos=0: all angles are 0
        assert!((cos[0] - 1.0).abs() < 1e-6);
        assert!((cos[1] - 1.0).abs() < 1e-6);
        assert!(sin[0].abs() < 1e-6);
        assert!(sin[1].abs() < 1e-6);
    }

    #[test]
    fn test_rope_scaled_linear() {
        let scaling = RopeScaling {
            rope_type: "linear".to_string(),
            factor: 2.0,
            original_max_position_embeddings: 128,
        };
        let (cos, _sin) = precompute_rope_data_scaled(4, 4, 10000.0, &scaling);
        // With factor=2, frequencies are halved, so wavelengths double
        assert_eq!(cos.len(), 4 * 2);
    }
}
