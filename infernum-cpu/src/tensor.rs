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

/// Thread-local scratch buffer reused across forward-pass ops.
///
/// Avoids per-token heap allocations in matmul and other compute-heavy ops.
/// The buffer grows to the high-water mark and is never shrunk, so after a
/// few warm-up tokens the allocator is never called again.
thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Borrow the thread-local scratch buffer, resized to at least `len` elements.
///
/// The closure receives a `&mut [f32]` of exactly `len` elements. It must not
/// call any other function that also borrows `SCRATCH` (no re-entrancy).
pub fn with_scratch<F, R>(len: usize, f: F) -> R
where
    F: FnOnce(&mut [f32]) -> R,
{
    SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < len {
            buf.resize(len, 0.0f32);
        }
        f(&mut buf[..len])
    })
}

impl CpuTensor {
    /// Create a tensor from an f32 slice (copies data).
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

    /// Create a tensor by taking ownership of a `Vec<f32>`, zero-copy.
    ///
    /// The vector is reinterpreted as raw bytes via `bytemuck`. This avoids
    /// the extra copy that `from_f32` performs and should be preferred in
    /// hot paths where the caller already owns the output buffer.
    #[must_use]
    pub fn from_f32_vec(shape: &[usize], mut data: Vec<f32>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "data len {} != shape product {numel}",
            data.len()
        );
        // Reinterpret Vec<f32> as Vec<u8> without any copy.
        // SAFETY:
        //  - f32 is always 4 bytes, so len*4 and cap*4 fit in usize.
        //  - The pointer, len, and capacity are all from a valid Vec<f32>,
        //    so the resulting Vec<u8> is valid and the memory is properly owned.
        //  - We `forget` the original Vec so it is not double-freed.
        let len = data.len() * 4;
        let cap = data.capacity() * 4;
        let ptr = data.as_mut_ptr().cast::<u8>();
        let bytes: Vec<u8> = unsafe {
            std::mem::forget(data);
            Vec::from_raw_parts(ptr, len, cap)
        };
        Self {
            data: Arc::new(bytes),
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

    /// Create a tensor that shares an existing `Arc<Vec<u8>>` (zero-copy).
    ///
    /// Used by the paged KV cache to avoid re-copying pool data on every
    /// token append.
    #[must_use]
    pub fn from_arc(shape: &[usize], dtype: DType, data: Arc<Vec<u8>>) -> Self {
        Self {
            data,
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

/// Block-quantized weight for CPU inference.
///
/// Stores quantized data and per-block scales separately, matching the
/// GGUF loader output. Layout is row-major: `out_features` rows of
/// `in_features` elements, with `in_features / 32` blocks per row.
///
/// Scales and mins are pre-decoded to f32 at load time so the forward
/// pass pays no f16→f32 conversion cost (and avoids repeated CPU-feature
/// detection inside the half crate).
#[derive(Clone)]
pub struct CpuQuantizedWeight {
    /// Logical shape: `[out_features, in_features]`
    pub shape: Vec<usize>,
    /// Quantization format (`Q8_0`, `Q4_0`, or `Q4_1`)
    pub dtype: DType,
    /// Raw quantized data — int8 bytes (Q8_0) or packed nibbles (Q4_0/Q4_1)
    pub data: Vec<u8>,
    /// Per-block scales decoded to f32 (one per block)
    pub scales: Vec<f32>,
    /// Per-block minimums decoded to f32 (one per block, Q4_1 only)
    pub mins: Option<Vec<f32>>,
}

/// Decode a buffer of f16 values stored as raw little-endian bytes into f32.
pub fn decode_f16_scales(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

/// A linear weight — dense f32 or block-quantized.
#[derive(Clone)]
pub enum CpuLinearWeight {
    /// Dense f32 weight.
    ///
    /// - `weight`: original layout `(in_features, out_features)` = `(K, N)`,
    ///   used by `as_dense_weight()` for fusion ops like `concat_inner_dim`.
    /// - `weight_nt`: pre-transposed to `(out_features, in_features)` = `(N, K)`,
    ///   used by matmul/GEMV for contiguous dot products (avoids per-call transpose).
    Dense {
        weight: CpuTensor,
        weight_nt: CpuTensor,
    },
    /// Block-quantized weight in `(out_features, in_features)` layout.
    Quantized(CpuQuantizedWeight),
}

impl CpuLinearWeight {
    /// Create a dense weight from a `(K, N)` tensor, pre-computing the transposed layout.
    ///
    /// The transposed copy is always stored as f32 (used by matmul kernels).
    /// The original tensor is kept as-is for fusion ops like `concat_inner_dim`.
    #[must_use]
    pub fn new_dense(weight: CpuTensor) -> Self {
        let shape = weight.shape();
        assert!(shape.len() == 2, "Dense weight must be 2D, got {shape:?}");
        let k = shape[0];
        let n = shape[1];
        let data = weight.to_f32_vec();

        // Transpose (K, N) → (N, K)
        let mut nt = vec![0.0f32; n * k];
        for row in 0..k {
            for col in 0..n {
                nt[col * k + row] = data[row * n + col];
            }
        }
        let weight_nt = CpuTensor::from_f32(&[n, k], &nt);

        Self::Dense { weight, weight_nt }
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
