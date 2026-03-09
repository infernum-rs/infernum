//! Softmax ops for Metal — row-wise softmax via GPU kernels.
//!
//! Standalone softmax kernels for use by attention and other ops.
//! The kernels use threadgroup shared memory for max/sum reductions.

use infernum::tensor::Tensor;
use infernum::DType;

use crate::context::reduction_threadgroup_size;
use crate::tensor::MetalTensor;
use crate::MetalContext;

use metal::MTLSize;

// ---------------------------------------------------------------------------
// Packed param structs — must match MSL struct layout
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxParams {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CausalSoftmaxParams {
    rows: u32,
    cols: u32,
    offset: u32,
}

/// Row-wise softmax on GPU.
///
/// Input shape: `(rows, cols)`. Output shape: same.
/// Each row is independently normalized.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn softmax(ctx: &MetalContext, input: &MetalTensor) -> MetalTensor {
    let shape = input.shape();
    let rows = shape[0];
    let cols = *shape.last().unwrap();
    let out = MetalTensor::zeros(ctx, shape, DType::F32);

    let tg_size = reduction_threadgroup_size(cols);

    let params = SoftmaxParams {
        rows: rows as u32,
        cols: cols as u32,
    };

    let threadgroups = MTLSize::new(rows as u64, 1, 1);
    let threads_per_group = MTLSize::new(tg_size as u64, 1, 1);
    let shared_mem = tg_size * std::mem::size_of::<f32>();

    ctx.dispatch_threadgroups(
        "softmax_f32",
        &[
            (input.metal_buffer(), input.buffer_offset()),
            (out.metal_buffer(), out.buffer_offset()),
        ],
        bytemuck::bytes_of(&params),
        threadgroups,
        threads_per_group,
        shared_mem,
    );

    out
}

/// Row-wise causal softmax on GPU.
///
/// Input shape: `(rows, cols)`. For query at position `offset + row`,
/// only keys `[0..offset+row]` are attended to; future positions are masked.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn causal_softmax(ctx: &MetalContext, input: &MetalTensor, offset: usize) -> MetalTensor {
    let shape = input.shape();
    let rows = shape[0];
    let cols = *shape.last().unwrap();
    let out = MetalTensor::zeros(ctx, shape, DType::F32);

    let tg_size = reduction_threadgroup_size(cols);

    let params = CausalSoftmaxParams {
        rows: rows as u32,
        cols: cols as u32,
        offset: offset as u32,
    };

    let threadgroups = MTLSize::new(rows as u64, 1, 1);
    let threads_per_group = MTLSize::new(tg_size as u64, 1, 1);
    let shared_mem = tg_size * std::mem::size_of::<f32>();

    ctx.dispatch_threadgroups(
        "causal_softmax_f32",
        &[
            (input.metal_buffer(), input.buffer_offset()),
            (out.metal_buffer(), out.buffer_offset()),
        ],
        bytemuck::bytes_of(&params),
        threadgroups,
        threads_per_group,
        shared_mem,
    );

    out
}

/// GPU argmax: returns the index of the maximum value in a 1D f32 tensor.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn argmax(ctx: &MetalContext, input: &MetalTensor) -> u32 {
    let n = input.numel();
    let tg_size = reduction_threadgroup_size(n);

    // Output: single u32
    let out = MetalTensor::zeros(ctx, &[1], DType::U32);
    let n_u32 = n as u32;

    // Shared memory: tg_size floats (values) + tg_size uints (indices)
    let shared_mem = tg_size * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());

    ctx.dispatch_threadgroups(
        "argmax_f32",
        &[
            (input.metal_buffer(), input.buffer_offset()),
            (out.metal_buffer(), out.buffer_offset()),
        ],
        bytemuck::bytes_of(&n_u32),
        MTLSize::new(1, 1, 1),
        MTLSize::new(tg_size as u64, 1, 1),
        shared_mem,
    );

    // Read result from output buffer
    let bytes = out.as_bytes();
    let result: &[u32] = bytemuck::cast_slice(bytes);
    result[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalBackend;
    use crate::MetalContext;
    use infernum::backend::TensorFactory;

    fn ctx() -> MetalContext {
        MetalContext::new()
    }

    #[test]
    fn test_softmax_single_row() {
        let c = ctx();
        let input = MetalBackend::from_f32_slice(&c, &[1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = softmax(&c, &input);

        let result = out.as_f32_slice();
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax sum = {sum}, expected 1.0"
        );

        // Verify ordering: result[3] > result[2] > result[1] > result[0]
        assert!(result[3] > result[2]);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_softmax_multi_row() {
        let c = ctx();
        let input =
            MetalBackend::from_f32_slice(&c, &[2, 3], &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
        let out = softmax(&c, &input);
        let result = out.as_f32_slice();

        // Each row should sum to 1.0
        let sum0: f32 = result[..3].iter().sum();
        let sum1: f32 = result[3..].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5, "row 0 sum = {sum0}");
        assert!((sum1 - 1.0).abs() < 1e-5, "row 1 sum = {sum1}");
    }

    #[test]
    fn test_causal_softmax() {
        let c = ctx();
        // 3 queries, 3 keys, offset = 0
        // Row 0: attend to key[0] only → result = [1.0, 0.0, 0.0]
        // Row 1: attend to key[0,1]   → result = [p0, p1, 0.0]
        // Row 2: attend to all        → result = [p0, p1, p2]
        let input = MetalBackend::from_f32_slice(&c, &[3, 3], &[1.0; 9]).unwrap();
        let out = causal_softmax(&c, &input, 0);
        let result = out.as_f32_slice();

        // Row 0: only key 0 is visible → [1.0, 0.0, 0.0]
        assert!((result[0] - 1.0).abs() < 1e-5, "row0[0] = {}", result[0]);
        assert!(result[1].abs() < 1e-5, "row0[1] = {}", result[1]);
        assert!(result[2].abs() < 1e-5, "row0[2] = {}", result[2]);

        // Row 1: keys 0,1 visible, equal input → [0.5, 0.5, 0.0]
        assert!((result[3] - 0.5).abs() < 1e-4, "row1[0] = {}", result[3]);
        assert!((result[4] - 0.5).abs() < 1e-4, "row1[1] = {}", result[4]);
        assert!(result[5].abs() < 1e-5, "row1[2] = {}", result[5]);

        // Row 2: all keys visible, equal input → [1/3, 1/3, 1/3]
        let third = 1.0 / 3.0;
        assert!((result[6] - third).abs() < 1e-4, "row2[0] = {}", result[6]);
        assert!((result[7] - third).abs() < 1e-4, "row2[1] = {}", result[7]);
        assert!((result[8] - third).abs() < 1e-4, "row2[2] = {}", result[8]);
    }

    #[test]
    fn test_argmax_gpu() {
        let c = ctx();
        let input = MetalBackend::from_f32_slice(&c, &[5], &[1.0, 3.0, 0.5, 7.0, 2.0]).unwrap();
        let idx = argmax(&c, &input);
        assert_eq!(idx, 3, "argmax should be 3, got {idx}");
    }

    #[test]
    fn test_argmax_large_vocab() {
        let c = ctx();
        let n = 32000;
        let mut data = vec![0.0f32; n];
        data[12345] = 99.0;
        let input = MetalBackend::from_f32_slice(&c, &[n], &data).unwrap();
        let idx = argmax(&c, &input);
        assert_eq!(idx, 12345, "argmax should be 12345, got {idx}");
    }

    #[test]
    fn test_argmax_ties() {
        let c = ctx();
        // When ties exist, we expect the first (lowest index) maximum
        let input = MetalBackend::from_f32_slice(&c, &[4], &[5.0, 5.0, 5.0, 5.0]).unwrap();
        let idx = argmax(&c, &input);
        // First maximum should win due to strict > comparison
        assert_eq!(
            idx, 0,
            "argmax with ties should return first max, got {idx}"
        );
    }
}
