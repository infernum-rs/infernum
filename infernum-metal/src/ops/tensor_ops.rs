//! TensorOps implementation for Metal — tensor manipulation via GPU kernels.

use infernum::backend::TensorOps;
use infernum::tensor::Tensor;
use infernum::DType;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

// ---------------------------------------------------------------------------
// Packed param structs — must match MSL struct layout
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposeParams {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RepeatKvParams {
    seq: u32,
    kv_heads: u32,
    head_dim: u32,
    num_repeats: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CopyStridedParams {
    in_cols: u32,
    out_cols: u32,
    col_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PadInnerParams {
    width: u32,
    new_width: u32,
}

impl TensorOps for MetalBackend {
    #[allow(clippy::cast_possible_truncation)]
    fn transpose_2d(input: &MetalTensor) -> Result<MetalTensor> {
        let shape = input.shape();
        assert_eq!(shape.len(), 2, "transpose_2d: expected 2D");
        let (rows, cols) = (shape[0], shape[1]);
        let ctx = input.context();
        let out = MetalTensor::zeros(ctx, &[cols, rows], DType::F32);

        let params = TransposeParams {
            rows: rows as u32,
            cols: cols as u32,
        };

        ctx.dispatch_1d(
            "transpose_2d_f32",
            &[
                (input.metal_buffer(), input.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            rows * cols,
        );

        Ok(out)
    }

    fn split_inner_dim(
        tensor: &MetalTensor,
        dim1: usize,
        dim2: usize,
    ) -> Result<(MetalTensor, MetalTensor)> {
        let shape = tensor.shape();
        let outer = shape[0];
        let inner = shape[1];
        assert_eq!(
            dim1 + dim2,
            inner,
            "split_inner_dim: {dim1} + {dim2} != {inner}"
        );

        // CPU-side: split is a stride-skipping read pattern that's awkward for
        // copy_strided (which is designed for writes). Simple and correct.
        let ctx = tensor.context();
        let data = tensor.as_f32_slice();
        let mut a_data = vec![0.0f32; outer * dim1];
        let mut b_data = vec![0.0f32; outer * dim2];
        for r in 0..outer {
            a_data[r * dim1..(r + 1) * dim1].copy_from_slice(&data[r * inner..r * inner + dim1]);
            b_data[r * dim2..(r + 1) * dim2]
                .copy_from_slice(&data[r * inner + dim1..r * inner + dim1 + dim2]);
        }

        Ok((
            MetalTensor::from_f32(ctx, &[outer, dim1], &a_data),
            MetalTensor::from_f32(ctx, &[outer, dim2], &b_data),
        ))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn concat_inner_dim(a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(a_shape[0], b_shape[0], "concat_inner_dim: row mismatch");
        let outer = a_shape[0];
        let d1 = a_shape[1];
        let d2 = b_shape[1];
        let new_inner = d1 + d2;

        let ctx = a.context();
        let out = MetalTensor::zeros(ctx, &[outer, new_inner], a.dtype());

        let kernel = if a.dtype() == DType::F16 {
            "copy_strided_f16"
        } else {
            "copy_strided_f32"
        };

        // Copy part a at col_offset=0
        let params_a = CopyStridedParams {
            in_cols: d1 as u32,
            out_cols: new_inner as u32,
            col_offset: 0,
        };
        ctx.dispatch_1d(
            kernel,
            &[
                (a.metal_buffer(), a.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params_a),
            outer * d1,
        );

        // Copy part b at col_offset=d1
        let params_b = CopyStridedParams {
            in_cols: d2 as u32,
            out_cols: new_inner as u32,
            col_offset: d1 as u32,
        };
        ctx.dispatch_1d(
            kernel,
            &[
                (b.metal_buffer(), b.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params_b),
            outer * d2,
        );

        Ok(out)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn pad_inner_dim(tensor: &MetalTensor, new_width: usize) -> Result<MetalTensor> {
        let shape = tensor.shape();
        let outer = shape[0];
        let width = shape[1];
        assert!(new_width >= width, "pad_inner_dim: new_width < width");

        let ctx = tensor.context();
        let out = MetalTensor::zeros(ctx, &[outer, new_width], DType::F32);

        let params = PadInnerParams {
            width: width as u32,
            new_width: new_width as u32,
        };

        ctx.dispatch_1d(
            "pad_inner_f32",
            &[
                (tensor.metal_buffer(), tensor.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            outer * width,
        );

        Ok(out)
    }

    fn broadcast_to_heads(tensor: &MetalTensor, num_heads: usize) -> Result<MetalTensor> {
        // (batch, 1, head_dim) → (batch, num_heads, head_dim)
        // This is equivalent to repeat_kv with kv_heads=1
        let shape = tensor.shape();
        let batch = shape[0];
        let head_dim = shape[2];
        let data = tensor.as_f32_slice();

        let mut out = vec![0.0f32; batch * num_heads * head_dim];
        for b in 0..batch {
            let src = &data[b * head_dim..(b + 1) * head_dim];
            for h in 0..num_heads {
                out[(b * num_heads + h) * head_dim..(b * num_heads + h + 1) * head_dim]
                    .copy_from_slice(src);
            }
        }

        Ok(MetalTensor::from_f32(
            tensor.context(),
            &[batch, num_heads, head_dim],
            &out,
        ))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn repeat_kv(tensor: &MetalTensor, num_repeats: usize) -> Result<MetalTensor> {
        if num_repeats == 1 {
            return Ok(tensor.clone());
        }
        let shape = tensor.shape();
        let seq = shape[0];
        let kv_heads = shape[1];
        let head_dim = shape[2];
        let new_heads = kv_heads * num_repeats;

        let ctx = tensor.context();
        let out = MetalTensor::zeros(ctx, &[seq, new_heads, head_dim], tensor.dtype());

        let params = RepeatKvParams {
            seq: seq as u32,
            kv_heads: kv_heads as u32,
            head_dim: head_dim as u32,
            num_repeats: num_repeats as u32,
        };

        let kernel = if tensor.dtype() == DType::F16 {
            "repeat_kv_f16"
        } else {
            "repeat_kv_f32"
        };
        ctx.dispatch_1d(
            kernel,
            &[
                (tensor.metal_buffer(), tensor.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            seq * new_heads * head_dim,
        );

        Ok(out)
    }

    fn concat_rows(parts: &[MetalTensor]) -> Result<MetalTensor> {
        assert!(!parts.is_empty(), "concat_rows: empty slice");
        let cols = parts[0].shape()[parts[0].shape().len() - 1];
        let total_rows: usize = parts.iter().map(|p| p.numel() / cols).sum();

        // Simple memcpy per part — contiguous f32 tensors
        let mut out = Vec::with_capacity(total_rows * cols);
        for p in parts {
            out.extend_from_slice(p.as_f32_slice());
        }

        Ok(MetalTensor::from_f32(
            parts[0].context(),
            &[total_rows, cols],
            &out,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalContext;
    use infernum::backend::{TensorDataOps, TensorFactory};

    fn ctx() -> MetalContext {
        MetalContext::new()
    }

    #[test]
    fn test_transpose_2d() {
        let c = ctx();
        let t = MetalBackend::from_f32_slice(&c, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let out = MetalBackend::transpose_2d(&t).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_split_inner_dim() {
        let c = ctx();
        let t =
            MetalBackend::from_f32_slice(&c, &[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap();
        let (a, b) = MetalBackend::split_inner_dim(&t, 1, 3).unwrap();
        let a_data = MetalBackend::to_f32_vec(&a).unwrap();
        let b_data = MetalBackend::to_f32_vec(&b).unwrap();
        assert_eq!(a_data, [1.0, 5.0]);
        assert_eq!(b_data, [2.0, 3.0, 4.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_concat_inner_dim() {
        let c = ctx();
        let a = MetalBackend::from_f32_slice(&c, &[2, 1], &[1.0, 2.0]).unwrap();
        let b = MetalBackend::from_f32_slice(&c, &[2, 2], &[3.0, 4.0, 5.0, 6.0]).unwrap();
        let out = MetalBackend::concat_inner_dim(&a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_pad_inner_dim() {
        let c = ctx();
        let t = MetalBackend::from_f32_slice(&c, &[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = MetalBackend::pad_inner_dim(&t, 4).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn test_repeat_kv() {
        let c = ctx();
        // (1 seq, 2 heads, 2 dim)
        let t = MetalBackend::from_f32_slice(&c, &[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = MetalBackend::repeat_kv(&t, 2).unwrap();
        assert_eq!(out.shape(), &[1, 4, 2]);
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_concat_rows() {
        let c = ctx();
        let a = MetalBackend::from_f32_slice(&c, &[1, 3], &[1.0, 2.0, 3.0]).unwrap();
        let b = MetalBackend::from_f32_slice(&c, &[1, 3], &[4.0, 5.0, 6.0]).unwrap();
        let out = MetalBackend::concat_rows(&[a, b]).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
