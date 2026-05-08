//! `ArgmaxLastOps` implementation for the Metal backend.
//!
//! Computes the argmax over the last dimension of a 2D `[rows, cols]` tensor,
//! returning a `[rows]` U32 tensor. For single-row inputs, the existing GPU
//! argmax kernel is used directly. Multi-row inputs fall back to CPU via
//! unified memory reads.

use infernum::backend::ArgmaxLastOps;
use infernum::tensor::Tensor;
use infernum::DType;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl ArgmaxLastOps for MetalBackend {
    #[allow(clippy::cast_possible_truncation)]
    fn argmax_last_tensor(input: &MetalTensor) -> infernum::Result<MetalTensor> {
        let shape = input.shape();
        let (rows, cols) = match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => panic!("argmax_last_tensor expects 1D or 2D input, got shape {shape:?}"),
        };

        if rows == 1 {
            // Single row: use the existing GPU argmax kernel.
            let idx = super::softmax::argmax(input.context(), input);
            let ctx = input.context();
            let out = MetalTensor::from_raw_bytes(ctx, &[1], DType::U32, bytemuck::bytes_of(&idx));
            return Ok(out);
        }

        // Multi-row: compute argmax per row on CPU via shared memory.
        let data = input.as_f32_slice();
        let mut indices = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * cols;
            let row_data = &data[start..start + cols];
            let mut best_idx: u32 = 0;
            let mut best_val = row_data[0];
            for (j, &val) in row_data.iter().enumerate().skip(1) {
                if val > best_val {
                    best_val = val;
                    best_idx = j as u32;
                }
            }
            indices.push(best_idx);
        }

        let ctx = input.context();
        Ok(MetalTensor::from_raw_bytes(
            ctx,
            &[rows],
            DType::U32,
            bytemuck::cast_slice(&indices),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::MetalContext;
    use infernum::backend::TensorFactory;

    fn ctx() -> MetalContext {
        MetalContext::new()
    }

    #[test]
    fn test_argmax_last_single_row() {
        let c = ctx();
        let input = MetalBackend::from_f32_slice(&c, &[1, 5], &[1.0, 3.0, 0.5, 7.0, 2.0]).unwrap();
        let result = MetalBackend::argmax_last_tensor(&input).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.dtype(), DType::U32);
        let bytes = result.as_bytes();
        let indices: &[u32] = bytemuck::cast_slice(bytes);
        assert_eq!(indices[0], 3, "expected argmax 3, got {}", indices[0]);
    }

    #[test]
    fn test_argmax_last_multi_row() {
        let c = ctx();
        #[rustfmt::skip]
        let data = [
            1.0, 5.0, 3.0, 2.0,   // row 0: max at col 1
            0.0, 0.0, 0.0, 9.0,   // row 1: max at col 3
            7.0, 1.0, 4.0, 0.0,   // row 2: max at col 0
        ];
        let input = MetalBackend::from_f32_slice(&c, &[3, 4], &data).unwrap();
        let result = MetalBackend::argmax_last_tensor(&input).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.dtype(), DType::U32);
        let bytes = result.as_bytes();
        let indices: &[u32] = bytemuck::cast_slice(bytes);
        assert_eq!(indices[0], 1, "row 0: expected 1, got {}", indices[0]);
        assert_eq!(indices[1], 3, "row 1: expected 3, got {}", indices[1]);
        assert_eq!(indices[2], 0, "row 2: expected 0, got {}", indices[2]);
    }
}
