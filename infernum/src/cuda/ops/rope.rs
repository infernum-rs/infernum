//! Rotary Positional Embeddings (RoPE)

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::missing_panics_doc
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::{CudaContext, CudaTensor};
use crate::dtype::TensorDType;
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/rope.ptx"));
const KERNEL_NAMES: &[&str] = &["rope_f32", "rope_f16", "rope_bf16"];

/// Precompute cosine and sine caches for RoPE
///
/// # Arguments
/// * `ctx` - CUDA context
/// * `max_seq_len` - Maximum sequence length
/// * `head_dim` - Dimension of each attention head
/// * `base` - Base frequency (default 10000.0)
///
/// # Returns
/// (cos_cache, sin_cache) tensors of shape (max_seq_len, head_dim/2)
///
/// # Errors
/// Returns an error if allocation fails
pub fn precompute_rope_cache(
    ctx: &CudaContext,
    max_seq_len: usize,
    head_dim: usize,
    base: f32,
) -> Result<(CudaTensor<f32>, CudaTensor<f32>)> {
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

    let cos_cache = CudaTensor::from_slice(ctx, &[max_seq_len, half_dim], &cos_data)?;
    let sin_cache = CudaTensor::from_slice(ctx, &[max_seq_len, half_dim], &sin_data)?;

    Ok((cos_cache, sin_cache))
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

/// Precompute RoPE cache with YaRN or linear scaling.
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
/// # Arguments
/// * `ctx` - CUDA context
/// * `max_seq_len` - Maximum sequence length
/// * `head_dim` - Dimension of each attention head
/// * `base` - Base frequency (typically 10000.0 or 1000000.0)
/// * `scaling` - Scaling configuration
///
/// # Returns
/// (cos_cache, sin_cache) tensors of shape `(max_seq_len, head_dim/2)`
///
/// # Errors
/// Returns an error if allocation fails
pub fn precompute_rope_cache_scaled(
    ctx: &CudaContext,
    max_seq_len: usize,
    head_dim: usize,
    base: f32,
    scaling: &RopeScaling,
) -> Result<(CudaTensor<f32>, CudaTensor<f32>)> {
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

    let cos_cache = CudaTensor::from_slice(ctx, &[max_seq_len, half_dim], &cos_data)?;
    let sin_cache = CudaTensor::from_slice(ctx, &[max_seq_len, half_dim], &sin_data)?;

    Ok((cos_cache, sin_cache))
}

/// Kernel name suffix for dtype
fn kernel_suffix<T: cudarc::driver::DeviceRepr>() -> &'static str {
    let type_name = std::any::type_name::<T>();
    if type_name.contains("f32") {
        "f32"
    } else if type_name.contains("f16") && !type_name.contains("bf16") {
        "f16"
    } else if type_name.contains("bf16") {
        "bf16"
    } else {
        panic!("Unsupported dtype for rope: {type_name}")
    }
}

/// Apply rotary positional embeddings (generic version)
fn apply_rope_generic<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &CudaTensor<T>,
    cos_cache: &CudaTensor<T>,
    sin_cache: &CudaTensor<T>,
    position_offset: usize,
) -> Result<CudaTensor<T>> {
    let shape = input.shape();
    assert_eq!(
        shape.len(),
        3,
        "Input must be 3D: (seq_len, num_heads, head_dim)"
    );

    let seq_len = shape[0];
    let num_heads = shape[1];
    let head_dim = shape[2];

    assert_eq!(head_dim % 2, 0, "head_dim must be even");

    let mut output = unsafe { CudaTensor::<T>::uninit(input.context(), shape)? };

    let device = input.context().device();
    let kernel_name = format!("rope_{}", kernel_suffix::<T>());

    let module_name = "rope";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            module_name,
            &all_kernel_names(),
        )?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (seq_len as u32, num_heads as u32, 1),
        block_dim: ((head_dim / 2) as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &input.cuda_slice(),
                &cos_cache.cuda_slice(),
                &sin_cache.cuda_slice(),
                seq_len as i32,
                num_heads as i32,
                head_dim as i32,
                position_offset as i32,
            ),
        )?;
    }

    Ok(output)
}

/// Apply rotary positional embeddings to Q and K tensors
///
/// Supports F32, F16, and BF16 tensor types.
///
/// # Arguments
/// * `input` - Input tensor of shape (seq_len, num_heads, head_dim)
/// * `cos_cache` - Precomputed cos cache of shape (max_seq_len, head_dim/2)
/// * `sin_cache` - Precomputed sin cache of shape (max_seq_len, head_dim/2)
/// * `position_offset` - Starting position (for incremental decoding)
///
/// # Errors
/// Returns an error if the operation fails
pub fn apply_rope<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &CudaTensor<T>,
    cos_cache: &CudaTensor<T>,
    sin_cache: &CudaTensor<T>,
    position_offset: usize,
) -> Result<CudaTensor<T>> {
    apply_rope_generic(input, cos_cache, sin_cache, position_offset)
}

const INDIRECT_KERNEL_NAMES: &[&str] = &[
    "rope_indirect_f32",
    "rope_indirect_f16",
    "rope_indirect_bf16",
];

/// Apply rotary positional embeddings using a GPU-resident position offset.
///
/// Identical to [`apply_rope`] but reads `position_offset` from the
/// [`SeqPosition`]'s device pointer instead of a host scalar.  This makes
/// the kernel capturable by a CUDA graph — the graph references the fixed
/// device address, and only the value at that address changes between
/// replays.
///
/// # Errors
/// Returns an error if the kernel launch fails.
pub fn apply_rope_indirect<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &CudaTensor<T>,
    cos_cache: &CudaTensor<T>,
    sin_cache: &CudaTensor<T>,
    position: &crate::cuda::SeqPosition,
) -> Result<CudaTensor<T>> {
    let shape = input.shape();
    assert_eq!(
        shape.len(),
        3,
        "Input must be 3D: (seq_len, num_heads, head_dim)"
    );

    let seq_len = shape[0];
    let num_heads = shape[1];
    let head_dim = shape[2];

    assert_eq!(head_dim % 2, 0, "head_dim must be even");

    let mut output = unsafe { CudaTensor::<T>::uninit(input.context(), shape)? };

    let device = input.context().device();
    let kernel_name = format!("rope_indirect_{}", kernel_suffix::<T>());

    let module_name = "rope";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            module_name,
            &all_kernel_names(),
        )?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (seq_len as u32, num_heads as u32, 1),
        block_dim: ((head_dim / 2) as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &input.cuda_slice(),
                &cos_cache.cuda_slice(),
                &sin_cache.cuda_slice(),
                seq_len as i32,
                num_heads as i32,
                head_dim as i32,
                position.device(),
            ),
        )?;
    }

    Ok(output)
}

const BATCHED_KERNEL_NAMES: &[&str] =
    &["rope_batched_f32", "rope_batched_f16", "rope_batched_bf16"];

/// All kernel names in the RoPE PTX module.
fn all_kernel_names() -> Vec<&'static str> {
    KERNEL_NAMES
        .iter()
        .chain(INDIRECT_KERNEL_NAMES.iter())
        .chain(BATCHED_KERNEL_NAMES.iter())
        .copied()
        .collect()
}

/// Apply rotary positional embeddings to a batch of single-token sequences,
/// each at a different position.
///
/// Input shape: `(batch_size, num_heads, head_dim)` — one token per sequence.
/// `positions` must have length `batch_size`.
///
/// # Errors
/// Returns an error if the kernel launch or GPU allocation fails.
pub fn apply_rope_batched<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &CudaTensor<T>,
    cos_cache: &CudaTensor<T>,
    sin_cache: &CudaTensor<T>,
    positions: &[usize],
) -> Result<CudaTensor<T>> {
    let shape = input.shape();
    assert_eq!(
        shape.len(),
        3,
        "Input must be 3D: (batch_size, num_heads, head_dim)"
    );

    let batch_size = shape[0];
    let num_heads = shape[1];
    let head_dim = shape[2];

    assert_eq!(head_dim % 2, 0, "head_dim must be even");
    assert_eq!(
        positions.len(),
        batch_size,
        "positions length ({}) must match batch_size ({batch_size})",
        positions.len()
    );

    let mut output = unsafe { CudaTensor::<T>::uninit(input.context(), shape)? };

    let device = input.context().device();
    let kernel_name = format!("rope_batched_{}", kernel_suffix::<T>());

    let module_name = "rope";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            module_name,
            &all_kernel_names(),
        )?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

    // Upload positions as i32 array
    let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let positions_gpu = device.htod_sync_copy(&positions_i32)?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, num_heads as u32, 1),
        block_dim: ((head_dim / 2) as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &input.cuda_slice(),
                &cos_cache.cuda_slice(),
                &sin_cache.cuda_slice(),
                batch_size as i32,
                num_heads as i32,
                head_dim as i32,
                &positions_gpu,
            ),
        )?;
    }

    Ok(output)
}

/// Apply rotary positional embeddings to a batch of single-token sequences,
/// reading positions from a pre-allocated GPU buffer.
///
/// Identical to [`apply_rope_batched`] but takes a `CudaSlice<i32>` of positions
/// already on the GPU (e.g., from [`BatchedGraphInputs`](crate::cuda::BatchedGraphInputs)),
/// avoiding the `htod_sync_copy` that would break CUDA graph capture.
///
/// `batch_size` is the number of sequences to process (may be larger than the
/// logical batch if padding is used for graph capture).
///
/// # Errors
/// Returns an error if the kernel launch or GPU allocation fails.
pub fn apply_rope_batched_indirect<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &CudaTensor<T>,
    cos_cache: &CudaTensor<T>,
    sin_cache: &CudaTensor<T>,
    positions_gpu: &cudarc::driver::CudaSlice<i32>,
    batch_size: usize,
) -> Result<CudaTensor<T>> {
    let shape = input.shape();
    assert_eq!(
        shape.len(),
        3,
        "Input must be 3D: (batch_size, num_heads, head_dim)"
    );
    assert_eq!(shape[0], batch_size);

    let num_heads = shape[1];
    let head_dim = shape[2];

    assert_eq!(head_dim % 2, 0, "head_dim must be even");

    let mut output = unsafe { CudaTensor::<T>::uninit(input.context(), shape)? };

    let device = input.context().device();
    let kernel_name = format!("rope_batched_{}", kernel_suffix::<T>());

    let module_name = "rope";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            module_name,
            &all_kernel_names(),
        )?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, num_heads as u32, 1),
        block_dim: ((head_dim / 2) as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &input.cuda_slice(),
                &cos_cache.cuda_slice(),
                &sin_cache.cuda_slice(),
                batch_size as i32,
                num_heads as i32,
                head_dim as i32,
                positions_gpu,
            ),
        )?;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_rope_identity_at_zero() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let seq_len = 2;
        let num_heads = 2;
        let head_dim = 4;

        let (cos_cache, sin_cache) = precompute_rope_cache(&ctx, 128, head_dim, 10000.0).unwrap();

        // At position 0, angle = 0, cos = 1, sin = 0
        // So output should equal input
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // seq=0, head=0
            5.0, 6.0, 7.0, 8.0, // seq=0, head=1
            9.0, 10.0, 11.0, 12.0, // seq=1, head=0
            13.0, 14.0, 15.0, 16.0, // seq=1, head=1
        ];

        let input =
            CudaTensor::from_slice(&ctx, &[seq_len, num_heads, head_dim], &input_data).unwrap();

        let output = apply_rope(&input, &cos_cache, &sin_cache, 0).unwrap();
        let result = output.to_vec().unwrap();

        // Position 0: cos(0) = 1, sin(0) = 0, so x' = x
        // Position 1 will have rotation applied
        assert!((result[0] - 1.0).abs() < 1e-5); // seq=0, no rotation
        assert!((result[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_sequential_halves_convention() {
        // Verify that RoPE uses the Llama sequential (half-half) convention:
        //   pairs are (x[i], x[i + head_dim/2])  for i in 0..head_dim/2
        // NOT the interleaved convention:
        //   pairs are (x[2i], x[2i+1])
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let head_dim = 4;
        let base = 10000.0_f32;

        let (cos_cache, sin_cache) = precompute_rope_cache(&ctx, 128, head_dim, base).unwrap();

        // Use position 1 to get non-trivial rotation angles
        // freq_0 = 1 / 10000^(0/4) = 1.0,      angle_0 = 1.0
        // freq_1 = 1 / 10000^(2/4) = 0.01,      angle_1 = 0.01
        let cos0 = 1.0_f32.cos();
        let sin0 = 1.0_f32.sin();
        let cos1 = 0.01_f32.cos();
        let sin1 = 0.01_f32.sin();

        // input: [a, b, c, d] with head_dim=4
        // Sequential convention pairs: (a, c) rotated by freq_0, (b, d) rotated by freq_1
        //   out[0] = a*cos0 - c*sin0
        //   out[1] = b*cos1 - d*sin1
        //   out[2] = c*cos0 + a*sin0
        //   out[3] = d*cos1 + b*sin1
        let a = 1.0_f32;
        let b = 2.0;
        let c = 3.0;
        let d = 4.0;
        let input_data = vec![a, b, c, d];

        let input = CudaTensor::from_slice(&ctx, &[1, 1, head_dim], &input_data).unwrap();
        let output = apply_rope(&input, &cos_cache, &sin_cache, 1).unwrap();
        let result = output.to_vec().unwrap();

        let expected = [
            a * cos0 - c * sin0,
            b * cos1 - d * sin1,
            c * cos0 + a * sin0,
            d * cos1 + b * sin1,
        ];

        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "Mismatch at index {i}: got {got}, expected {want} — \
                 wrong rotation convention?"
            );
        }
    }

    #[test]
    fn test_rope_indirect_matches_direct_at_zero() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let head_dim = 4;
        let (cos_cache, sin_cache) = precompute_rope_cache(&ctx, 128, head_dim, 10000.0).unwrap();

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = CudaTensor::from_slice(&ctx, &[1, 2, head_dim], &input_data).unwrap();

        let direct = apply_rope(&input, &cos_cache, &sin_cache, 0).unwrap();

        let mut pos = crate::cuda::SeqPosition::new(ctx.device()).unwrap();
        pos.set(0, ctx.device()).unwrap();
        let indirect = apply_rope_indirect(&input, &cos_cache, &sin_cache, &pos).unwrap();

        let direct_data = direct.to_vec().unwrap();
        let indirect_data = indirect.to_vec().unwrap();

        for (i, (&d, &ind)) in direct_data.iter().zip(indirect_data.iter()).enumerate() {
            assert!(
                (d - ind).abs() < 1e-6,
                "Mismatch at {i}: direct={d}, indirect={ind}"
            );
        }
    }

    #[test]
    fn test_rope_indirect_matches_direct_nonzero_position() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let head_dim = 4;
        let (cos_cache, sin_cache) = precompute_rope_cache(&ctx, 128, head_dim, 10000.0).unwrap();

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = CudaTensor::from_slice(&ctx, &[1, 1, head_dim], &input_data).unwrap();

        let position = 42;
        let direct = apply_rope(&input, &cos_cache, &sin_cache, position).unwrap();

        let mut pos = crate::cuda::SeqPosition::new(ctx.device()).unwrap();
        pos.set(position, ctx.device()).unwrap();
        let indirect = apply_rope_indirect(&input, &cos_cache, &sin_cache, &pos).unwrap();

        let direct_data = direct.to_vec().unwrap();
        let indirect_data = indirect.to_vec().unwrap();

        for (i, (&d, &ind)) in direct_data.iter().zip(indirect_data.iter()).enumerate() {
            assert!(
                (d - ind).abs() < 1e-6,
                "Mismatch at position {position}, index {i}: direct={d}, indirect={ind}"
            );
        }
    }

    #[test]
    fn test_rope_indirect_updates_with_position() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let head_dim = 4;
        let (cos_cache, sin_cache) = precompute_rope_cache(&ctx, 128, head_dim, 10000.0).unwrap();

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = CudaTensor::from_slice(&ctx, &[1, 1, head_dim], &input_data).unwrap();

        let mut pos = crate::cuda::SeqPosition::new(ctx.device()).unwrap();

        // Position 5
        pos.set(5, ctx.device()).unwrap();
        let out5 = apply_rope_indirect(&input, &cos_cache, &sin_cache, &pos).unwrap();
        let ref5 = apply_rope(&input, &cos_cache, &sin_cache, 5).unwrap();

        // Position 10
        pos.set(10, ctx.device()).unwrap();
        let out10 = apply_rope_indirect(&input, &cos_cache, &sin_cache, &pos).unwrap();
        let ref10 = apply_rope(&input, &cos_cache, &sin_cache, 10).unwrap();

        let out5_data = out5.to_vec().unwrap();
        let ref5_data = ref5.to_vec().unwrap();
        let out10_data = out10.to_vec().unwrap();
        let ref10_data = ref10.to_vec().unwrap();

        // Results at different positions should differ
        assert_ne!(
            out5_data, out10_data,
            "Different positions should produce different results"
        );

        // Each should match its direct counterpart
        for (i, (&a, &b)) in out5_data.iter().zip(ref5_data.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "pos=5 mismatch at {i}: {a} vs {b}");
        }
        for (i, (&a, &b)) in out10_data.iter().zip(ref10_data.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "pos=10 mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_rope_yarn_factor_1_matches_standard() {
        // YaRN with factor=1.0 should produce identical results to standard
        // RoPE (all dimensions fall in the high-frequency band, scale ≈ 1.0).
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let head_dim = 64;
        let max_seq = 128;
        let base = 10000.0;

        let (cos_std, sin_std) = precompute_rope_cache(&ctx, max_seq, head_dim, base).unwrap();
        let scaling = RopeScaling {
            rope_type: "yarn".to_string(),
            factor: 1.0,
            original_max_position_embeddings: 4096,
        };
        let (cos_yarn, sin_yarn) =
            precompute_rope_cache_scaled(&ctx, max_seq, head_dim, base, &scaling).unwrap();

        let cos_std_v = cos_std.to_vec().unwrap();
        let sin_std_v = sin_std.to_vec().unwrap();
        let cos_yarn_v = cos_yarn.to_vec().unwrap();
        let sin_yarn_v = sin_yarn.to_vec().unwrap();

        // scale = sqrt(1 + 0.1 * ln(1.0)) = sqrt(1.0) = 1.0
        for (i, (&a, &b)) in cos_std_v.iter().zip(cos_yarn_v.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "cos mismatch at {i}: standard={a}, yarn_factor1={b}"
            );
        }
        for (i, (&a, &b)) in sin_std_v.iter().zip(sin_yarn_v.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "sin mismatch at {i}: standard={a}, yarn_factor1={b}"
            );
        }
    }

    #[test]
    fn test_rope_yarn_magnitude_correction() {
        // Verify that YaRN applies the magnitude correction sqrt(1 + 0.1*ln(factor))
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let head_dim = 4;
        let max_seq = 4;
        let base = 10000.0;
        let factor = 4.0_f32;

        let scaling = RopeScaling {
            rope_type: "yarn".to_string(),
            factor,
            original_max_position_embeddings: 32768,
        };
        let (cos_yarn, sin_yarn) =
            precompute_rope_cache_scaled(&ctx, max_seq, head_dim, base, &scaling).unwrap();

        let cos_v = cos_yarn.to_vec().unwrap();
        let sin_v = sin_yarn.to_vec().unwrap();

        let expected_scale = (1.0 + 0.1 * factor.ln()).sqrt();

        // At position 0, cos should be scale * 1.0 and sin should be scale * 0.0
        let half_dim = head_dim / 2;
        for i in 0..half_dim {
            assert!(
                (cos_v[i] - expected_scale).abs() < 1e-5,
                "pos=0, dim={i}: cos={}, expected scale={}",
                cos_v[i],
                expected_scale
            );
            assert!(
                sin_v[i].abs() < 1e-5,
                "pos=0, dim={i}: sin={}, expected 0.0",
                sin_v[i]
            );
        }
    }

    #[test]
    fn test_rope_linear_scaling() {
        // Linear scaling: all frequencies divided by factor
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let head_dim = 4;
        let max_seq = 8;
        let base = 10000.0;
        let factor = 2.0_f32;

        let scaling = RopeScaling {
            rope_type: "linear".to_string(),
            factor,
            original_max_position_embeddings: 4096,
        };
        let (cos_lin, _sin_lin) =
            precompute_rope_cache_scaled(&ctx, max_seq, head_dim, base, &scaling).unwrap();

        let cos_v = cos_lin.to_vec().unwrap();
        let half_dim = head_dim / 2;

        // Position p with linear scaling should match position p/factor in standard
        // cos_lin[pos=4, dim=0] should equal cos_std[pos=2, dim=0]
        let (cos_std, _) = precompute_rope_cache(&ctx, max_seq, head_dim, base).unwrap();
        let cos_std_v = cos_std.to_vec().unwrap();

        let pos_lin = 4;
        let pos_std = 2; // 4 / factor
        for i in 0..half_dim {
            assert!(
                (cos_v[pos_lin * half_dim + i] - cos_std_v[pos_std * half_dim + i]).abs() < 1e-5,
                "linear scaling mismatch at dim {i}"
            );
        }
    }

    #[test]
    fn test_batched_rope_matches_scalar() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let num_heads = 2;
        let head_dim = 8;
        let positions = [5_usize, 12, 0];
        let batch_size = positions.len();

        let (cos_cache, sin_cache) = precompute_rope_cache(&ctx, 128, head_dim, 10000.0).unwrap();

        // Build batched input: (3, 2, 8)
        let row_size = num_heads * head_dim;
        let mut input_data = Vec::with_capacity(batch_size * row_size);
        for i in 0..batch_size * row_size {
            input_data.push((i as f32 + 1.0) * 0.1);
        }
        let input =
            CudaTensor::from_slice(&ctx, &[batch_size, num_heads, head_dim], &input_data).unwrap();

        let batched = apply_rope_batched(&input, &cos_cache, &sin_cache, &positions).unwrap();
        let batched_data = batched.to_vec().unwrap();

        // Compare each row against scalar apply_rope
        for (i, &pos) in positions.iter().enumerate() {
            let row_start = i * row_size;
            let row_input = CudaTensor::from_slice(
                &ctx,
                &[1, num_heads, head_dim],
                &input_data[row_start..row_start + row_size],
            )
            .unwrap();
            let scalar = apply_rope(&row_input, &cos_cache, &sin_cache, pos).unwrap();
            let scalar_data = scalar.to_vec().unwrap();

            for (j, (&got, &want)) in batched_data[row_start..row_start + row_size]
                .iter()
                .zip(scalar_data.iter())
                .enumerate()
            {
                assert!(
                    (got - want).abs() < 1e-6,
                    "Mismatch at batch={i}, pos={pos}, elem={j}: batched={got}, scalar={want}"
                );
            }
        }
    }

    #[test]
    fn test_batched_rope_single_sequence() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let num_heads = 2;
        let head_dim = 4;

        let (cos_cache, sin_cache) = precompute_rope_cache(&ctx, 128, head_dim, 10000.0).unwrap();

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &input_data).unwrap();

        let pos = 7;
        let batched = apply_rope_batched(&input, &cos_cache, &sin_cache, &[pos]).unwrap();
        let scalar = apply_rope(&input, &cos_cache, &sin_cache, pos).unwrap();

        let batched_data = batched.to_vec().unwrap();
        let scalar_data = scalar.to_vec().unwrap();

        for (i, (&got, &want)) in batched_data.iter().zip(scalar_data.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "Mismatch at {i}: batched={got}, scalar={want}"
            );
        }
    }

    #[test]
    fn test_batched_rope_indirect_matches_eager() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let num_heads = 2;
        let head_dim = 8;
        let positions = [5_usize, 12, 0];
        let batch_size = positions.len();

        let (cos_cache, sin_cache) = precompute_rope_cache(&ctx, 128, head_dim, 10000.0).unwrap();

        let row_size = num_heads * head_dim;
        let mut input_data = Vec::with_capacity(batch_size * row_size);
        for i in 0..batch_size * row_size {
            input_data.push((i as f32 + 1.0) * 0.1);
        }
        let input =
            CudaTensor::from_slice(&ctx, &[batch_size, num_heads, head_dim], &input_data).unwrap();

        // Eager path
        let eager = apply_rope_batched(&input, &cos_cache, &sin_cache, &positions).unwrap();
        let eager_data = eager.to_vec().unwrap();

        // Indirect path: positions pre-uploaded as i32
        let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let positions_gpu = ctx.device().htod_sync_copy(&positions_i32).unwrap();
        let indirect =
            apply_rope_batched_indirect(&input, &cos_cache, &sin_cache, &positions_gpu, batch_size)
                .unwrap();
        let indirect_data = indirect.to_vec().unwrap();

        for (i, (&e, &ind)) in eager_data.iter().zip(indirect_data.iter()).enumerate() {
            assert!(
                (e - ind).abs() < 1e-6,
                "Mismatch at {i}: eager={e}, indirect={ind}"
            );
        }
    }
}
