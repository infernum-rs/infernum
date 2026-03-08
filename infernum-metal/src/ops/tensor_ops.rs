//! TensorOps implementation for Metal — tensor manipulation.

use infernum::backend::TensorOps;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl TensorOps for MetalBackend {
    fn transpose_2d(input: &MetalTensor) -> Result<MetalTensor> {
        let shape = input.shape();
        assert_eq!(shape.len(), 2, "transpose_2d: expected 2D");
        let (rows, cols) = (shape[0], shape[1]);
        let data = input.as_f32_slice();

        let mut out = vec![0.0f32; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(&device, &[cols, rows], &out))
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

        let data = tensor.as_f32_slice();
        let mut a = vec![0.0f32; outer * dim1];
        let mut b = vec![0.0f32; outer * dim2];

        for r in 0..outer {
            a[r * dim1..(r + 1) * dim1].copy_from_slice(&data[r * inner..r * inner + dim1]);
            b[r * dim2..(r + 1) * dim2]
                .copy_from_slice(&data[r * inner + dim1..r * inner + dim1 + dim2]);
        }

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok((
            MetalTensor::from_f32(&device, &[outer, dim1], &a),
            MetalTensor::from_f32(&device, &[outer, dim2], &b),
        ))
    }

    fn concat_inner_dim(a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(a_shape[0], b_shape[0], "concat_inner_dim: row mismatch");
        let outer = a_shape[0];
        let d1 = a_shape[1];
        let d2 = b_shape[1];
        let new_inner = d1 + d2;

        let a_data = a.as_f32_slice();
        let b_data = b.as_f32_slice();
        let mut out = vec![0.0f32; outer * new_inner];

        for r in 0..outer {
            out[r * new_inner..r * new_inner + d1].copy_from_slice(&a_data[r * d1..(r + 1) * d1]);
            out[r * new_inner + d1..r * new_inner + d1 + d2]
                .copy_from_slice(&b_data[r * d2..(r + 1) * d2]);
        }

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(&device, &[outer, new_inner], &out))
    }

    fn pad_inner_dim(tensor: &MetalTensor, new_width: usize) -> Result<MetalTensor> {
        let shape = tensor.shape();
        let outer = shape[0];
        let width = shape[1];
        assert!(new_width >= width, "pad_inner_dim: new_width < width");

        let data = tensor.as_f32_slice();
        let mut out = vec![0.0f32; outer * new_width];

        for r in 0..outer {
            out[r * new_width..r * new_width + width]
                .copy_from_slice(&data[r * width..(r + 1) * width]);
        }

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(&device, &[outer, new_width], &out))
    }

    fn broadcast_to_heads(tensor: &MetalTensor, num_heads: usize) -> Result<MetalTensor> {
        let shape = tensor.shape();
        // (batch, 1, head_dim) → (batch, num_heads, head_dim)
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

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(
            &device,
            &[batch, num_heads, head_dim],
            &out,
        ))
    }

    fn repeat_kv(tensor: &MetalTensor, num_repeats: usize) -> Result<MetalTensor> {
        if num_repeats == 1 {
            return Ok(tensor.clone());
        }
        let shape = tensor.shape();
        // (seq, kv_heads, head_dim)
        let seq = shape[0];
        let kv_heads = shape[1];
        let head_dim = shape[2];
        let data = tensor.as_f32_slice();

        let new_heads = kv_heads * num_repeats;
        let mut out = vec![0.0f32; seq * new_heads * head_dim];

        for s in 0..seq {
            for kv in 0..kv_heads {
                let src = &data[(s * kv_heads + kv) * head_dim..(s * kv_heads + kv + 1) * head_dim];
                for rep in 0..num_repeats {
                    let dst_head = kv * num_repeats + rep;
                    out[(s * new_heads + dst_head) * head_dim
                        ..(s * new_heads + dst_head + 1) * head_dim]
                        .copy_from_slice(src);
                }
            }
        }

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(
            &device,
            &[seq, new_heads, head_dim],
            &out,
        ))
    }

    fn concat_rows(parts: &[MetalTensor]) -> Result<MetalTensor> {
        assert!(!parts.is_empty(), "concat_rows: empty slice");
        let cols = parts[0].shape()[parts[0].shape().len() - 1];
        let total_rows: usize = parts.iter().map(|p| p.numel() / cols).sum();

        let mut out = Vec::with_capacity(total_rows * cols);
        for p in parts {
            out.extend_from_slice(p.as_f32_slice());
        }

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(&device, &[total_rows, cols], &out))
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
