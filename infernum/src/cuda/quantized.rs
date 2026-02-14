//! Quantized tensor storage for CUDA
//!
//! Stores weights in compressed quantized formats on the GPU.
//! Dequantization happens on-the-fly during matmul — the full f32
//! expansion never exists in GPU memory.

#![allow(
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions
)]

use cudarc::driver::{CudaSlice, DevicePtr};

use crate::cuda::CudaContext;
use crate::dtype::{DType, QUANTIZATION_BLOCK_SIZE};
use crate::Result;

/// A tensor stored in a quantized format on the GPU.
///
/// Unlike [`CudaTensor<T>`](super::CudaTensor), this is not parameterized by
/// element type — the `dtype` field determines how the raw bytes are interpreted.
///
/// For block-quantized formats (`Q8_0`, `Q4_0`):
/// - `data` holds the quantized values (int8 or packed int4)
/// - `scales` holds one f16 scale factor per block of [`QUANTIZATION_BLOCK_SIZE`] elements
///
/// For FP8 (F8E4M3):
/// - `data` holds the raw fp8 bytes (one per element)
/// - `scales` is empty (no block structure)
pub struct QuantizedTensor {
    /// Raw quantized data on the GPU
    data: CudaSlice<u8>,
    /// Per-block scale factors (f16, stored as raw bytes; empty for FP8)
    scales: CudaSlice<u8>,
    /// Logical shape of the tensor (number of elements per dimension)
    shape: Vec<usize>,
    /// Quantization format
    dtype: DType,
}

impl QuantizedTensor {
    /// Create a quantized tensor from raw host data.
    ///
    /// # Arguments
    /// * `ctx` — CUDA context
    /// * `shape` — logical element shape (e.g. `[out_features, in_features]`)
    /// * `dtype` — one of `Q8_0`, `Q4_0`, or `F8E4M3`
    /// * `data` — raw quantized bytes (int8, packed int4, or fp8)
    /// * `scales` — per-block f16 scales as raw bytes (empty slice for FP8)
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation or copy fails.
    ///
    /// # Panics
    /// Panics if `dtype` is not a quantized format or if data sizes are inconsistent.
    pub fn from_raw(
        ctx: &CudaContext,
        shape: &[usize],
        dtype: DType,
        data: &[u8],
        scales: &[u8],
    ) -> Result<Self> {
        assert!(
            dtype.is_quantized(),
            "QuantizedTensor requires a quantized dtype, got {dtype}"
        );

        let numel: usize = shape.iter().product();

        if dtype.is_block_quantized() {
            assert_eq!(
                numel % QUANTIZATION_BLOCK_SIZE,
                0,
                "Number of elements ({numel}) must be divisible by block size ({QUANTIZATION_BLOCK_SIZE})"
            );
            let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
            let expected_scale_bytes = num_blocks * 2; // f16 = 2 bytes
            assert_eq!(
                scales.len(),
                expected_scale_bytes,
                "Expected {expected_scale_bytes} scale bytes for {num_blocks} blocks, got {}",
                scales.len()
            );
        }

        Self::validate_data_size(numel, dtype, data);

        let gpu_data = ctx.device().htod_sync_copy(data)?;
        let gpu_scales = ctx.device().htod_sync_copy(scales)?;

        Ok(Self {
            data: gpu_data,
            scales: gpu_scales,
            shape: shape.to_vec(),
            dtype,
        })
    }

    /// Create a quantized tensor from pre-allocated GPU slices.
    ///
    /// # Panics
    /// Panics if `dtype` is not a quantized format.
    #[must_use]
    pub fn from_gpu(
        shape: &[usize],
        dtype: DType,
        data: CudaSlice<u8>,
        scales: CudaSlice<u8>,
    ) -> Self {
        assert!(
            dtype.is_quantized(),
            "QuantizedTensor requires a quantized dtype, got {dtype}"
        );
        Self {
            data,
            scales,
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// Logical shape of the tensor
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Quantization format
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Total number of logical elements
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of quantization blocks (for block-quantized formats)
    ///
    /// # Panics
    /// Panics if the dtype is not block-quantized.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        assert!(
            self.dtype.is_block_quantized(),
            "num_blocks() only valid for block-quantized types"
        );
        self.numel() / QUANTIZATION_BLOCK_SIZE
    }

    /// Raw device pointer to quantized data
    #[must_use]
    pub fn data_ptr(&self) -> *const u8 {
        *self.data.device_ptr() as *const u8
    }

    /// Raw device pointer to scale factors
    #[must_use]
    pub fn scales_ptr(&self) -> *const u8 {
        *self.scales.device_ptr() as *const u8
    }

    /// Reference to the underlying data slice
    #[must_use]
    pub fn data_slice(&self) -> &CudaSlice<u8> {
        &self.data
    }

    /// Reference to the underlying scales slice
    #[must_use]
    pub fn scales_slice(&self) -> &CudaSlice<u8> {
        &self.scales
    }

    /// Validate that the data byte count matches what we expect for the dtype
    fn validate_data_size(numel: usize, dtype: DType, data: &[u8]) {
        let expected = match dtype {
            DType::Q8_0 | DType::F8E4M3 => numel, // 1 byte per element
            DType::Q4_0 => numel / 2,             // 0.5 bytes per element (packed)
            _ => unreachable!(),
        };
        assert_eq!(
            data.len(),
            expected,
            "Expected {expected} data bytes for {numel} elements of {dtype}, got {}",
            data.len()
        );
    }

    /// Total GPU memory used by this tensor in bytes
    #[must_use]
    pub fn size_in_bytes(&self) -> usize {
        let numel = self.numel();
        match self.dtype {
            DType::Q8_0 => {
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                numel + num_blocks * 2 // data + scales
            }
            DType::Q4_0 => {
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                numel / 2 + num_blocks * 2 // packed data + scales
            }
            DType::F8E4M3 => numel, // 1 byte per element, no scales
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_quantized_tensor_q8_creation() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let numel = 64;
        let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
        let data = vec![42u8; numel];
        let scales = vec![0u8; num_blocks * 2]; // f16 = 2 bytes each

        let qt = QuantizedTensor::from_raw(&ctx, &[2, 32], DType::Q8_0, &data, &scales)
            .expect("Failed to create Q8_0 tensor");

        assert_eq!(qt.shape(), &[2, 32]);
        assert_eq!(qt.dtype(), DType::Q8_0);
        assert_eq!(qt.numel(), 64);
        assert_eq!(qt.num_blocks(), 2);
    }

    #[test]
    fn test_quantized_tensor_q4_creation() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let numel = 64;
        let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
        let data = vec![0u8; numel / 2]; // packed: 2 elements per byte
        let scales = vec![0u8; num_blocks * 2];

        let qt = QuantizedTensor::from_raw(&ctx, &[2, 32], DType::Q4_0, &data, &scales)
            .expect("Failed to create Q4_0 tensor");

        assert_eq!(qt.shape(), &[2, 32]);
        assert_eq!(qt.dtype(), DType::Q4_0);
        assert_eq!(qt.numel(), 64);
        assert_eq!(qt.num_blocks(), 2);
    }

    #[test]
    fn test_quantized_tensor_fp8_creation() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let numel = 64;
        let data = vec![0u8; numel]; // 1 byte per element
        let scales: Vec<u8> = vec![];

        let qt = QuantizedTensor::from_raw(&ctx, &[8, 8], DType::F8E4M3, &data, &scales)
            .expect("Failed to create F8E4M3 tensor");

        assert_eq!(qt.shape(), &[8, 8]);
        assert_eq!(qt.dtype(), DType::F8E4M3);
        assert_eq!(qt.numel(), 64);
    }

    #[test]
    fn test_quantized_tensor_size_in_bytes() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // Q8_0: 64 data bytes + 2 blocks × 2 scale bytes = 68
        let qt =
            QuantizedTensor::from_raw(&ctx, &[2, 32], DType::Q8_0, &vec![0u8; 64], &vec![0u8; 4])
                .unwrap();
        assert_eq!(qt.size_in_bytes(), 68);

        // Q4_0: 32 data bytes + 2 blocks × 2 scale bytes = 36
        let qt =
            QuantizedTensor::from_raw(&ctx, &[2, 32], DType::Q4_0, &vec![0u8; 32], &vec![0u8; 4])
                .unwrap();
        assert_eq!(qt.size_in_bytes(), 36);

        // F8E4M3: 64 bytes
        let qt =
            QuantizedTensor::from_raw(&ctx, &[8, 8], DType::F8E4M3, &vec![0u8; 64], &[]).unwrap();
        assert_eq!(qt.size_in_bytes(), 64);
    }

    #[test]
    #[should_panic(expected = "quantized dtype")]
    fn test_quantized_tensor_rejects_f32() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let _ = QuantizedTensor::from_raw(&ctx, &[2, 2], DType::F32, &[0u8; 4], &[]);
    }

    #[test]
    #[should_panic(expected = "divisible by block size")]
    fn test_quantized_tensor_q8_bad_numel() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        // 33 elements is not divisible by 32
        let _ = QuantizedTensor::from_raw(&ctx, &[33], DType::Q8_0, &[0u8; 33], &[0u8; 2]);
    }
}
