//! TensorOps, BiasOps, TensorDataOps, TensorFactory implementations.

use infernum::backend::{BiasOps, DecodeBufferOps, TensorDataOps, TensorFactory, TensorOps};
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::CpuTensor;
use crate::CpuBackend;

impl TensorFactory for CpuBackend {
    fn from_f32_slice(_device: &(), shape: &[usize], data: &[f32]) -> Result<CpuTensor> {
        Ok(CpuTensor::from_f32(shape, data))
    }

    fn from_raw_bytes(
        _device: &(),
        shape: &[usize],
        dtype: DType,
        data: &[u8],
    ) -> Result<CpuTensor> {
        Ok(CpuTensor::from_raw(shape, dtype, data.to_vec()))
    }

    fn from_u32_slice(_device: &(), shape: &[usize], data: &[u32]) -> Result<CpuTensor> {
        Ok(CpuTensor::from_u32(shape, data))
    }

    fn from_i32_slice(_device: &(), shape: &[usize], data: &[i32]) -> Result<CpuTensor> {
        Ok(CpuTensor::from_i32(shape, data))
    }
}

impl DecodeBufferOps for CpuBackend {}

impl TensorOps for CpuBackend {
    fn transpose_2d(input: &CpuTensor) -> Result<CpuTensor> {
        let shape = input.shape();
        assert_eq!(shape.len(), 2, "transpose_2d: expected 2D tensor");
        let rows = shape[0];
        let cols = shape[1];
        let data = input.as_f32_slice();

        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = data[i * cols + j];
            }
        }
        Ok(CpuTensor::from_f32(&[cols, rows], &out))
    }

    fn split_inner_dim(
        tensor: &CpuTensor,
        dim1: usize,
        dim2: usize,
    ) -> Result<(CpuTensor, CpuTensor)> {
        let shape = tensor.shape();
        let outer: usize = shape[..shape.len() - 1].iter().product();
        let inner = *shape.last().unwrap();
        assert_eq!(
            inner,
            dim1 + dim2,
            "split_inner_dim: {inner} != {dim1} + {dim2}"
        );

        let data = tensor.as_f32_slice();
        let mut a = Vec::with_capacity(outer * dim1);
        let mut b = Vec::with_capacity(outer * dim2);

        for row in 0..outer {
            let start = row * inner;
            a.extend_from_slice(&data[start..start + dim1]);
            b.extend_from_slice(&data[start + dim1..start + inner]);
        }

        let mut shape_a = shape[..shape.len() - 1].to_vec();
        shape_a.push(dim1);
        let mut shape_b = shape[..shape.len() - 1].to_vec();
        shape_b.push(dim2);

        Ok((
            CpuTensor::from_f32(&shape_a, &a),
            CpuTensor::from_f32(&shape_b, &b),
        ))
    }

    fn concat_inner_dim(a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let outer: usize = a_shape[..a_shape.len() - 1].iter().product();
        let a_inner = *a_shape.last().unwrap();
        let b_inner = *b_shape.last().unwrap();

        let a_data = a.as_f32_slice();
        let b_data = b.as_f32_slice();
        let new_inner = a_inner + b_inner;
        let mut out = Vec::with_capacity(outer * new_inner);

        for row in 0..outer {
            out.extend_from_slice(&a_data[row * a_inner..(row + 1) * a_inner]);
            out.extend_from_slice(&b_data[row * b_inner..(row + 1) * b_inner]);
        }

        let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
        out_shape.push(new_inner);
        Ok(CpuTensor::from_f32(&out_shape, &out))
    }

    fn pad_inner_dim(tensor: &CpuTensor, new_width: usize) -> Result<CpuTensor> {
        let shape = tensor.shape();
        let outer: usize = shape[..shape.len() - 1].iter().product();
        let old_width = *shape.last().unwrap();
        assert!(
            new_width >= old_width,
            "pad_inner_dim: new_width < old_width"
        );

        let data = tensor.as_f32_slice();
        let mut out = vec![0.0f32; outer * new_width];
        for row in 0..outer {
            out[row * new_width..row * new_width + old_width]
                .copy_from_slice(&data[row * old_width..(row + 1) * old_width]);
        }

        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.push(new_width);
        Ok(CpuTensor::from_f32(&out_shape, &out))
    }

    fn broadcast_to_heads(tensor: &CpuTensor, num_heads: usize) -> Result<CpuTensor> {
        // (batch, 1, head_dim) → (batch, num_heads, head_dim)
        let shape = tensor.shape();
        assert_eq!(shape[1], 1, "broadcast_to_heads: expected dim 1 == 1");
        let batch = shape[0];
        let head_dim = shape[2];
        let data = tensor.as_f32_slice();

        let mut out = Vec::with_capacity(batch * num_heads * head_dim);
        for b in 0..batch {
            let row = &data[b * head_dim..(b + 1) * head_dim];
            for _ in 0..num_heads {
                out.extend_from_slice(row);
            }
        }
        Ok(CpuTensor::from_f32(&[batch, num_heads, head_dim], &out))
    }

    fn repeat_kv(tensor: &CpuTensor, num_repeats: usize) -> Result<CpuTensor> {
        if num_repeats == 1 {
            return Ok(tensor.clone());
        }
        let shape = tensor.shape();
        let seq_len = shape[0];
        let num_kv_heads = shape[1];
        let head_dim = shape[2];
        let data = tensor.as_f32_slice();

        let new_heads = num_kv_heads * num_repeats;
        let mut out = Vec::with_capacity(seq_len * new_heads * head_dim);
        for s in 0..seq_len {
            for h in 0..num_kv_heads {
                let head_data =
                    &data[(s * num_kv_heads + h) * head_dim..(s * num_kv_heads + h + 1) * head_dim];
                for _ in 0..num_repeats {
                    out.extend_from_slice(head_data);
                }
            }
        }
        Ok(CpuTensor::from_f32(&[seq_len, new_heads, head_dim], &out))
    }

    fn concat_rows(parts: &[CpuTensor]) -> Result<CpuTensor> {
        assert!(!parts.is_empty(), "concat_rows: empty parts");
        let cols = parts[0].shape().last().copied().unwrap_or(0);
        let mut data = Vec::new();
        for part in parts {
            data.extend_from_slice(part.as_f32_slice());
        }
        let rows = parts.len();
        Ok(CpuTensor::from_f32(&[rows, cols], &data))
    }
}

impl BiasOps for CpuBackend {
    fn bias_add_inplace(input: &mut CpuTensor, bias: &CpuTensor) -> Result<()> {
        let bias_data = bias.as_f32_slice().to_vec();
        let cols = bias_data.len();
        let data = input.as_f32_slice_mut();
        let rows = data.len() / cols;
        for row in 0..rows {
            for col in 0..cols {
                data[row * cols + col] += bias_data[col];
            }
        }
        Ok(())
    }
}

impl TensorDataOps for CpuBackend {
    fn to_f32_vec(tensor: &CpuTensor) -> Result<Vec<f32>> {
        Ok(tensor.to_f32_vec())
    }

    fn to_raw_bytes(tensor: &CpuTensor) -> Result<Vec<u8>> {
        Ok(tensor.as_bytes().to_vec())
    }
}
