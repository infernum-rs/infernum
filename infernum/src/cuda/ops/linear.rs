use crate::cuda::ops::{cast_f32_to_bf16, cast_to_f32, matmul, quantized_matmul, GemmScalar};
use crate::cuda::{CudaTensor, QuantizedTensor};
use crate::dtype::TensorDType;
use crate::tensor::Tensor;
use crate::Result;
use cudarc::cublas::{CudaBlas, Gemm};
use cudarc::driver::DeviceRepr;

/// A linear layer weight that is either a dense matrix (pre-transposed for
/// standard matmul) or a quantized tensor (dequantized on-the-fly in the kernel).
pub enum LinearWeight<T: TensorDType> {
    /// Pre-transposed dense weight: shape `(in_features, out_features)`.
    Dense(CudaTensor<T>),
    /// Quantized weight: shape `(out_features, in_features)` — transposed inside kernel.
    Quantized(QuantizedTensor),
}

/// Reinterpret a `CudaTensor<A>` as `CudaTensor<B>` when both have the same
/// in-memory layout (same byte size per element). Zero-cost: no copy, no kernel.
///
/// # Panics
/// Panics if `size_of::<A>() != size_of::<B>()`.
#[must_use]
pub fn reinterpret_tensor<A: TensorDType + DeviceRepr, B: TensorDType + DeviceRepr>(
    tensor: CudaTensor<A>,
) -> CudaTensor<B> {
    assert_eq!(
        std::mem::size_of::<A>(),
        std::mem::size_of::<B>(),
        "reinterpret_tensor: size mismatch between {} and {}",
        A::DTYPE,
        B::DTYPE,
    );
    unsafe { tensor.reinterpret() }
}

/// Applies a linear layer: `output = input × weight`.
///
/// For `Dense` weights: pre-transposed as `(in_features, out_features)`, uses standard matmul.
/// For `Quantized` weights: stored as `(out_features, in_features)`, dequantized on-the-fly.
///
/// # Errors
/// Returns an error if the underlying matmul or cast operation fails.
///
/// # Panics
/// Panics if `Quantized` variant is used with a dtype other than f32 or bf16.
pub fn linear<T>(input: &CudaTensor<T>, weight: &LinearWeight<T>) -> Result<CudaTensor<T>>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default,
    CudaBlas: Gemm<T>,
{
    match weight {
        LinearWeight::Dense(w) => matmul(input, w),
        LinearWeight::Quantized(w) => {
            let input_f32 = if T::DTYPE == crate::dtype::DType::F32 {
                reinterpret_tensor(input.slice_view(0, input.shape()))
            } else {
                cast_to_f32(input)?
            };
            let output_f32 = quantized_matmul(&input_f32, w)?;
            match T::DTYPE {
                crate::dtype::DType::F32 => Ok(reinterpret_tensor(output_f32)),
                crate::dtype::DType::BF16 => Ok(reinterpret_tensor(cast_f32_to_bf16(&output_f32)?)),
                other => panic!("Quantized matmul not supported for dtype {other}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;
    use crate::dtype::DType;

    /// Pack f32 weights into GPTQ INT4 format on the host.
    fn pack_gptq_test(
        weights: &[f32],
        out_features: usize,
        in_features: usize,
        group_size: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        assert_eq!(weights.len(), out_features * in_features);
        assert_eq!(in_features % 8, 0);
        assert_eq!(in_features % group_size, 0);
        assert_eq!(out_features % 8, 0);

        let num_groups = in_features / group_size;
        let packed_rows = in_features / 8;
        let zero_point = 8_i32;

        let mut scales_f16 = vec![half::f16::from_f32(0.0); num_groups * out_features];
        let mut quantized = vec![0_i32; out_features * in_features];

        for n in 0..out_features {
            for g in 0..num_groups {
                let k_start = g * group_size;
                let k_end = k_start + group_size;
                let group_vals: Vec<f32> = (k_start..k_end)
                    .map(|k| weights[n * in_features + k])
                    .collect();
                let max_abs = group_vals.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
                scales_f16[g * out_features + n] = half::f16::from_f32(scale);

                for (j, &v) in group_vals.iter().enumerate() {
                    let q = ((v / scale).round() as i32 + zero_point).clamp(0, 15);
                    quantized[n * in_features + k_start + j] = q;
                }
            }
        }

        let mut qweight = vec![0_u8; packed_rows * out_features * 4];
        for pr in 0..packed_rows {
            for n in 0..out_features {
                let mut packed: u32 = 0;
                for j in 0..8 {
                    let k = pr * 8 + j;
                    let q = quantized[n * in_features + k] as u32;
                    packed |= (q & 0xF) << (j * 4);
                }
                let idx = (pr * out_features + n) * 4;
                qweight[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
            }
        }

        let mut scales_bytes = vec![0_u8; num_groups * out_features * 2];
        for (i, &s) in scales_f16.iter().enumerate() {
            let bytes = s.to_le_bytes();
            scales_bytes[i * 2] = bytes[0];
            scales_bytes[i * 2 + 1] = bytes[1];
        }

        let qzeros_cols = out_features / 8;
        let mut qzeros = vec![0_u8; num_groups * qzeros_cols * 4];
        for g in 0..num_groups {
            for col in 0..qzeros_cols {
                let mut packed: u32 = 0;
                for j in 0..8 {
                    packed |= (zero_point as u32 & 0xF) << (j * 4);
                }
                let idx = (g * qzeros_cols + col) * 4;
                qzeros[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
            }
        }

        (qweight, scales_bytes, qzeros)
    }

    #[test]
    fn test_linear_dense_f32() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let input = CudaTensor::from_slice(&ctx, &[1, 4], &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let w =
            CudaTensor::from_slice(&ctx, &[4, 2], &[1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
                .unwrap();
        let weight = LinearWeight::Dense(w);

        let output = linear(&input, &weight).unwrap();
        assert_eq!(output.shape(), &[1, 2]);
        let result = output.to_vec().unwrap();
        assert!((result[0] - 4.0).abs() < 1e-4); // 1+3
        assert!((result[1] - 6.0).abs() < 1e-4); // 2+4
    }

    #[test]
    fn test_linear_dense_bf16() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let input_data: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let w_data: Vec<half::bf16> = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let input = CudaTensor::from_slice(&ctx, &[1, 4], &input_data).unwrap();
        let weight = LinearWeight::Dense(CudaTensor::from_slice(&ctx, &[4, 2], &w_data).unwrap());

        let output = linear(&input, &weight).unwrap();
        assert_eq!(output.shape(), &[1, 2]);
        let result: Vec<f32> = output
            .to_vec()
            .unwrap()
            .into_iter()
            .map(half::bf16::to_f32)
            .collect();
        assert!((result[0] - 4.0).abs() < 0.1);
        assert!((result[1] - 6.0).abs() < 0.1);
    }

    #[test]
    fn test_linear_quantized_f32() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let k = 32;
        let n = 8;
        let group_size = 32;

        let w_data = vec![1.0_f32; n * k];
        let (qw, sc, qz) = pack_gptq_test(&w_data, n, k, group_size);

        let weight = LinearWeight::Quantized(
            QuantizedTensor::from_gptq_raw(
                &ctx,
                &[n, k],
                DType::GPTQ_INT4,
                &qw,
                &sc,
                &qz,
                group_size,
            )
            .unwrap(),
        );

        let input = CudaTensor::from_slice(&ctx, &[1, k], &vec![1.0_f32; k]).unwrap();
        let output = linear(&input, &weight).unwrap();
        assert_eq!(output.shape(), &[1, n]);

        let result = output.to_vec().unwrap();
        for &v in &result {
            assert!(
                v.is_finite(),
                "non-finite value in quantized f32 output: {v}"
            );
            assert!((v - 32.0).abs() < 8.0, "expected ~32.0, got {v}");
        }
    }

    #[test]
    fn test_linear_quantized_bf16() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let k = 32;
        let n = 8;
        let group_size = 32;

        let w_data = vec![1.0_f32; n * k];
        let (qw, sc, qz) = pack_gptq_test(&w_data, n, k, group_size);

        let weight: LinearWeight<half::bf16> = LinearWeight::Quantized(
            QuantizedTensor::from_gptq_raw(
                &ctx,
                &[n, k],
                DType::GPTQ_INT4,
                &qw,
                &sc,
                &qz,
                group_size,
            )
            .unwrap(),
        );

        let input_data: Vec<half::bf16> = vec![1.0_f32; k]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let input = CudaTensor::from_slice(&ctx, &[1, k], &input_data).unwrap();
        let output = linear(&input, &weight).unwrap();
        assert_eq!(output.shape(), &[1, n]);

        let result: Vec<f32> = output
            .to_vec()
            .unwrap()
            .into_iter()
            .map(half::bf16::to_f32)
            .collect();
        for &v in &result {
            assert!(
                v.is_finite(),
                "non-finite value in quantized bf16 output: {v}"
            );
            assert!((v - 32.0).abs() < 8.0, "expected ~32.0, got {v}");
        }
    }
}
