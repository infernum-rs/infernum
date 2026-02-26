//! `CudaBackend` — implements the infernum `Backend` + op traits for CUDA.

use infernum::backend::{
    ArithOps, Backend, CastOps, GegluOps, MatmulOps, NormOps, SwigluOps, TensorOps,
};
use infernum::{DType, Result};

use crate::cuda::ops;
use crate::cuda::ops::LinearWeight;
use crate::cuda::CudaTensor;

/// Marker type for the CUDA backend.
pub struct CudaBackend;

impl Backend for CudaBackend {
    type Tensor = CudaTensor;
}

impl ArithOps for CudaBackend {
    fn add(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        ops::add(a, b)
    }

    fn add_inplace(a: &mut CudaTensor, b: &CudaTensor) -> Result<()> {
        ops::add_inplace(a, b)
    }

    fn mul(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        ops::mul(a, b)
    }

    fn scale_inplace(a: &mut CudaTensor, scale: f32) -> Result<()> {
        ops::scale_inplace(a, scale)
    }
}

impl MatmulOps for CudaBackend {
    type LinearWeight = LinearWeight;

    fn matmul(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        ops::matmul(a, b)
    }

    fn linear(input: &CudaTensor, weight: &LinearWeight) -> Result<CudaTensor> {
        ops::linear(input, weight)
    }
}

impl NormOps for CudaBackend {
    fn rms_norm(input: &CudaTensor, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
        ops::rms_norm(input, weight, eps)
    }

    fn rms_norm_inplace(input: &mut CudaTensor, weight: &CudaTensor, eps: f32) -> Result<()> {
        ops::rms_norm_inplace(input, weight, eps)
    }

    fn add_rmsnorm(
        residual: &CudaTensor,
        input: &CudaTensor,
        weight: &CudaTensor,
        eps: f32,
    ) -> Result<(CudaTensor, CudaTensor)> {
        ops::add_rmsnorm(residual, input, weight, eps)
    }
}

impl SwigluOps for CudaBackend {
    fn swiglu(gate: &CudaTensor, up: &CudaTensor) -> Result<CudaTensor> {
        ops::swiglu(gate, up)
    }
}

impl GegluOps for CudaBackend {
    fn geglu(gate: &CudaTensor, up: &CudaTensor) -> Result<CudaTensor> {
        ops::geglu(gate, up)
    }
}

impl CastOps for CudaBackend {
    fn cast_to_f32(input: &CudaTensor) -> Result<CudaTensor> {
        ops::cast_to_f32(input)
    }

    fn cast_from_f32(input: &CudaTensor, target: DType) -> Result<CudaTensor> {
        ops::cast_from_f32(input, target)
    }
}

impl TensorOps for CudaBackend {
    fn transpose_2d(input: &CudaTensor) -> Result<CudaTensor> {
        ops::transpose_2d(input)
    }
}
