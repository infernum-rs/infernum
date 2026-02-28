//! CastOps implementation for CpuBackend.

use infernum::backend::CastOps;
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::CpuTensor;
use crate::CpuBackend;

impl CastOps for CpuBackend {
    fn cast_to_f32(input: &CpuTensor) -> Result<CpuTensor> {
        let f32_data = input.to_f32_vec();
        Ok(CpuTensor::from_f32(input.shape(), &f32_data))
    }

    fn cast_from_f32(input: &CpuTensor, target: DType) -> Result<CpuTensor> {
        let f32_data = input.as_f32_slice();
        match target {
            DType::F32 => Ok(input.clone()),
            DType::BF16 => {
                let bf16_data: Vec<half::bf16> =
                    f32_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let bytes: Vec<u8> = bytemuck::cast_slice(&bf16_data).to_vec();
                Ok(CpuTensor::from_raw(input.shape(), DType::BF16, bytes))
            }
            DType::F16 => {
                let f16_data: Vec<half::f16> =
                    f32_data.iter().map(|&v| half::f16::from_f32(v)).collect();
                let bytes: Vec<u8> = bytemuck::cast_slice(&f16_data).to_vec();
                Ok(CpuTensor::from_raw(input.shape(), DType::F16, bytes))
            }
            other => Err(infernum::Error::UnsupportedDtype(format!(
                "cast_from_f32: unsupported target dtype {other}"
            ))),
        }
    }
}
