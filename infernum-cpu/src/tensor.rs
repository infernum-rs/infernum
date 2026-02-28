//! CPU tensor implementation.

use std::sync::Arc;

use infernum::dtype::DType;
use infernum::tensor::Tensor;

/// A CPU-resident tensor backed by shared byte storage.
///
/// Uses `Arc<Vec<u8>>` so clones and `slice_view` are cheap (shared backing).
/// All compute is done in f32; bf16/f16 data is cast on load.
#[derive(Clone)]
pub struct CpuTensor {
    data: Arc<Vec<u8>>,
    offset: usize,
    shape: Vec<usize>,
    dtype: DType,
}

impl CpuTensor {
    /// Create a tensor from an f32 slice.
    #[must_use]
    pub fn from_f32(shape: &[usize], data: &[f32]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "data len {} != shape product {numel}",
            data.len()
        );
        Self {
            data: Arc::new(bytemuck::cast_slice(data).to_vec()),
            offset: 0,
            shape: shape.to_vec(),
            dtype: DType::F32,
        }
    }

    /// Create a tensor from a u32 slice.
    #[must_use]
    pub fn from_u32(shape: &[usize], data: &[u32]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel);
        Self {
            data: Arc::new(bytemuck::cast_slice(data).to_vec()),
            offset: 0,
            shape: shape.to_vec(),
            dtype: DType::U32,
        }
    }

    /// Create a tensor from an i32 slice (stored as U32).
    #[must_use]
    pub fn from_i32(shape: &[usize], data: &[i32]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel);
        Self {
            data: Arc::new(bytemuck::cast_slice(data).to_vec()),
            offset: 0,
            shape: shape.to_vec(),
            dtype: DType::U32,
        }
    }

    /// Create a tensor from raw bytes with a given dtype.
    #[must_use]
    pub fn from_raw(shape: &[usize], dtype: DType, data: Vec<u8>) -> Self {
        Self {
            data: Arc::new(data),
            offset: 0,
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// Create a zero-filled f32 tensor.
    #[must_use]
    pub fn zeros_f32(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            data: Arc::new(vec![0u8; numel * 4]),
            offset: 0,
            shape: shape.to_vec(),
            dtype: DType::F32,
        }
    }

    /// Get the data as an f32 slice.
    ///
    /// # Panics
    /// Panics if dtype is not F32.
    #[must_use]
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32, "expected F32 tensor");
        let start = self.offset;
        let end = start + self.numel() * 4;
        bytemuck::cast_slice(&self.data[start..end])
    }

    /// Get the data as a mutable f32 slice.
    ///
    /// # Panics
    /// Panics if dtype is not F32 or if the Arc has other references.
    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        assert_eq!(self.dtype, DType::F32, "expected F32 tensor");
        let start = self.offset;
        let numel = self.numel();
        let end = start + numel * 4;
        let data = Arc::make_mut(&mut self.data);
        bytemuck::cast_slice_mut(&mut data[start..end])
    }

    /// Get the data as a u32 slice.
    ///
    /// # Panics
    /// Panics if dtype is not U32.
    #[must_use]
    pub fn as_u32_slice(&self) -> &[u32] {
        assert_eq!(self.dtype, DType::U32, "expected U32 tensor");
        let start = self.offset;
        let end = start + self.numel() * 4;
        bytemuck::cast_slice(&self.data[start..end])
    }

    /// Get the data as an i32 slice.
    ///
    /// # Panics
    /// Panics if dtype is not U32.
    #[must_use]
    pub fn as_i32_slice(&self) -> &[i32] {
        assert_eq!(self.dtype, DType::U32, "expected U32 tensor (i32 view)");
        let start = self.offset;
        let end = start + self.numel() * 4;
        bytemuck::cast_slice(&self.data[start..end])
    }

    /// Get the raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        let start = self.offset;
        let end = start + self.size_in_bytes();
        &self.data[start..end]
    }

    /// Convert to an f32 Vec, casting from bf16/f16 if necessary.
    #[must_use]
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self.dtype {
            DType::F32 => self.as_f32_slice().to_vec(),
            DType::BF16 => {
                let bytes = self.as_bytes();
                let bf16s: &[half::bf16] = bytemuck::cast_slice(bytes);
                bf16s.iter().map(|v| v.to_f32()).collect()
            }
            DType::F16 => {
                let bytes = self.as_bytes();
                let f16s: &[half::f16] = bytemuck::cast_slice(bytes);
                f16s.iter().map(|v| v.to_f32()).collect()
            }
            other => panic!("to_f32_vec: unsupported dtype {other}"),
        }
    }
}

impl Tensor for CpuTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        let new_numel: usize = shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "reshape: {} elements != {new_numel} elements",
            self.numel()
        );
        Self {
            data: Arc::clone(&self.data),
            offset: self.offset,
            shape: shape.to_vec(),
            dtype: self.dtype,
        }
    }

    fn slice_view(&self, offset: usize, shape: &[usize]) -> Self {
        let elem_size = self.dtype.size_in_bytes();
        let byte_offset = self.offset + offset * elem_size;
        let new_numel: usize = shape.iter().product();
        assert!(
            byte_offset + new_numel * elem_size <= self.data.len(),
            "slice_view out of bounds"
        );
        Self {
            data: Arc::clone(&self.data),
            offset: byte_offset,
            shape: shape.to_vec(),
            dtype: self.dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_f32() {
        let t = CpuTensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape() {
        let t = CpuTensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = t.reshape(&[3, 2]);
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_slice_view() {
        let t = CpuTensor::from_f32(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = t.slice_view(2, &[3]);
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.as_f32_slice(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_zeros() {
        let t = CpuTensor::zeros_f32(&[2, 2]);
        assert_eq!(t.as_f32_slice(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clone_shares_data() {
        let t = CpuTensor::from_f32(&[3], &[1.0, 2.0, 3.0]);
        let c = t.clone();
        assert!(std::ptr::eq(t.data.as_ref(), c.data.as_ref()));
    }
}
