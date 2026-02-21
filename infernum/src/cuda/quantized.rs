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
use crate::dtype::{
    DType, GPTQ_GROUP_SIZE, Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_SIZE_BYTES, QUANTIZATION_BLOCK_SIZE,
};
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
///
/// For group-quantized formats (`GPTQ_INT4`, `AWQ_INT4`):
/// - `data` holds packed int4 weights (8 values per int32)
/// - `scales` holds f16 per-group scale factors
/// - `qzeros` holds packed int4 zero-points (8 values per int32)
/// - `group_size` is the number of elements per quantization group (typically 128)
pub struct QuantizedTensor {
    /// Raw quantized data on the GPU
    data: CudaSlice<u8>,
    /// Per-block/per-group scale factors (f16, stored as raw bytes; empty for FP8)
    scales: CudaSlice<u8>,
    /// Per-group zero-points (packed int32, stored as raw bytes; only for GPTQ/AWQ)
    qzeros: Option<CudaSlice<u8>>,
    /// Logical shape of the tensor (number of elements per dimension)
    shape: Vec<usize>,
    /// Quantization format
    dtype: DType,
    /// Number of elements per quantization group (only for GPTQ/AWQ)
    group_size: Option<usize>,
    /// Per-tensor scale factor (used by FP8 dynamic quantization; 1.0 for block-quantized)
    weight_scale: f32,
    /// Cached device-side weight scale (lazily allocated, avoids per-matmul host→device copies)
    d_weight_scale: Option<CudaSlice<f32>>,
    /// Per-channel (per-row) scale factors on GPU, shape [N] as f32.
    /// Used by compressed-tensors FP8 models. When set, `weight_scale` is ignored.
    d_channel_scales: Option<CudaSlice<f32>>,
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

        if dtype == DType::Q6_K {
            assert_eq!(
                numel % Q6_K_BLOCK_ELEMENTS,
                0,
                "Number of elements ({numel}) must be divisible by Q6_K block size ({Q6_K_BLOCK_ELEMENTS})"
            );
            // Q6_K stores packed super-blocks in data, no separate scales
        } else if dtype.is_block_quantized() {
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
            qzeros: None,
            shape: shape.to_vec(),
            dtype,
            group_size: None,
            weight_scale: 1.0,
            d_weight_scale: None,
            d_channel_scales: None,
        })
    }

    /// Create a quantized tensor from pre-allocated GPU slices.
    ///
    /// For GPTQ/AWQ formats, use [`from_gptq_raw`](Self::from_gptq_raw) instead.
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
            qzeros: None,
            shape: shape.to_vec(),
            dtype,
            group_size: None,
            weight_scale: 1.0,
            d_weight_scale: None,
            d_channel_scales: None,
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

    /// Per-tensor scale factor (FP8 dynamic quantization)
    #[must_use]
    pub fn weight_scale(&self) -> f32 {
        self.weight_scale
    }

    /// Set the per-tensor scale factor and upload it to the GPU.
    ///
    /// # Errors
    /// Returns an error if GPU allocation fails.
    pub fn set_weight_scale(&mut self, ctx: &CudaContext, scale: f32) -> Result<()> {
        self.weight_scale = scale;
        self.d_weight_scale = Some(ctx.device().htod_sync_copy(&[scale])?);
        Ok(())
    }

    /// Get the cached device-side weight scale buffer, if set.
    #[must_use]
    pub fn d_weight_scale(&self) -> Option<&CudaSlice<f32>> {
        self.d_weight_scale.as_ref()
    }

    /// Set per-channel (per-row) scale factors and upload them to the GPU.
    ///
    /// Used by compressed-tensors FP8 models where each output channel has
    /// its own scale factor. Shape must be `[N]` where N is `shape[0]`.
    ///
    /// # Errors
    /// Returns an error if GPU allocation fails.
    ///
    /// # Panics
    /// Panics if `scales.len()` does not match `shape[0]`.
    pub fn set_channel_scales(&mut self, ctx: &CudaContext, scales: &[f32]) -> Result<()> {
        assert_eq!(
            scales.len(),
            self.shape[0],
            "channel scales length ({}) must match shape[0] ({})",
            scales.len(),
            self.shape[0]
        );
        self.d_channel_scales = Some(ctx.device().htod_sync_copy(scales)?);
        Ok(())
    }

    /// Get the per-channel scale factors on GPU, if set.
    #[must_use]
    pub fn d_channel_scales(&self) -> Option<&CudaSlice<f32>> {
        self.d_channel_scales.as_ref()
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
        match self.dtype {
            DType::Q6_K => self.numel() / Q6_K_BLOCK_ELEMENTS,
            _ => self.numel() / QUANTIZATION_BLOCK_SIZE,
        }
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

    /// Raw device pointer to zero-points (GPTQ/AWQ only)
    ///
    /// # Panics
    /// Panics if this tensor does not have zero-points (i.e., is not GPTQ/AWQ).
    #[must_use]
    pub fn qzeros_ptr(&self) -> *const u8 {
        *self
            .qzeros
            .as_ref()
            .expect("qzeros_ptr() only valid for GPTQ/AWQ tensors")
            .device_ptr() as *const u8
    }

    /// Reference to the underlying zero-points slice (GPTQ/AWQ only)
    #[must_use]
    pub fn qzeros_slice(&self) -> Option<&CudaSlice<u8>> {
        self.qzeros.as_ref()
    }

    /// Quantization group size (GPTQ/AWQ only)
    #[must_use]
    pub fn group_size(&self) -> Option<usize> {
        self.group_size
    }

    /// Create a GPTQ/AWQ quantized tensor from raw host data.
    ///
    /// # Arguments
    /// * `ctx` — CUDA context
    /// * `shape` — logical weight shape `[in_features, out_features]` (the unpacked dimensions)
    /// * `dtype` — `GPTQ_INT4` or `AWQ_INT4`
    /// * `qweight` — packed int4 weights as raw bytes (int32 layout)
    /// * `scales` — per-group f16 scale factors as raw bytes
    /// * `qzeros` — packed int4 zero-points as raw bytes (int32 layout)
    /// * `group_size` — number of elements per quantization group
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation or copy fails.
    ///
    /// # Panics
    /// Panics if `dtype` is not a group-quantized format.
    pub fn from_gptq_raw(
        ctx: &CudaContext,
        shape: &[usize],
        dtype: DType,
        qweight: &[u8],
        scales: &[u8],
        qzeros: &[u8],
        group_size: usize,
    ) -> Result<Self> {
        assert!(
            dtype.is_group_quantized(),
            "from_gptq_raw requires GPTQ_INT4 or AWQ_INT4, got {dtype}"
        );
        assert!(
            group_size > 0,
            "group_size must be positive, got {group_size}"
        );

        let gpu_data = ctx.device().htod_sync_copy(qweight)?;
        let gpu_scales = ctx.device().htod_sync_copy(scales)?;
        let gpu_qzeros = ctx.device().htod_sync_copy(qzeros)?;

        Ok(Self {
            data: gpu_data,
            scales: gpu_scales,
            qzeros: Some(gpu_qzeros),
            shape: shape.to_vec(),
            dtype,
            group_size: Some(group_size),
            weight_scale: 1.0,
            d_weight_scale: None,
            d_channel_scales: None,
        })
    }

    /// Validate that the data byte count matches what we expect for the dtype
    fn validate_data_size(numel: usize, dtype: DType, data: &[u8]) {
        let expected = match dtype {
            DType::Q8_0 | DType::F8E4M3 => numel, // 1 byte per element
            DType::Q4_0 => numel / 2,             // 0.5 bytes per element (packed)
            DType::Q6_K => (numel / Q6_K_BLOCK_ELEMENTS) * Q6_K_BLOCK_SIZE_BYTES,
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
            DType::Q6_K => {
                let num_blocks = numel / Q6_K_BLOCK_ELEMENTS;
                num_blocks * Q6_K_BLOCK_SIZE_BYTES // packed super-blocks
            }
            DType::F8E4M3 => numel, // 1 byte per element, no scales
            DType::GPTQ_INT4 | DType::AWQ_INT4 => {
                let gs = self.group_size.unwrap_or(GPTQ_GROUP_SIZE);
                let in_features = self.shape[0];
                let out_features = self.shape[1];
                let num_groups = in_features / gs;
                // qweight: packed int4 → numel / 2 bytes (stored as int32)
                let qweight_bytes = numel / 2;
                // scales: f16 per group per out_feature
                let scales_bytes = num_groups * out_features * 2;
                // qzeros: packed int4 zero-points per group
                let qzeros_bytes = num_groups * out_features / 2;
                qweight_bytes + scales_bytes + qzeros_bytes
            }
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

    #[test]
    fn test_quantized_tensor_gptq_creation() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let in_features = 256;
        let out_features = 128;
        let group_size = 128;
        let num_groups = in_features / group_size;

        // qweight: [in_features/8, out_features] as int32 → (256/8)*128*4 bytes
        let qweight = vec![0u8; (in_features / 8) * out_features * 4];
        // scales: [num_groups, out_features] as f16 → 2*128*2 bytes
        let scales = vec![0u8; num_groups * out_features * 2];
        // qzeros: [num_groups, out_features/8] as int32 → 2*(128/8)*4 bytes
        let qzeros = vec![0u8; num_groups * (out_features / 8) * 4];

        let qt = QuantizedTensor::from_gptq_raw(
            &ctx,
            &[in_features, out_features],
            DType::GPTQ_INT4,
            &qweight,
            &scales,
            &qzeros,
            group_size,
        )
        .expect("Failed to create GPTQ tensor");

        assert_eq!(qt.shape(), &[in_features, out_features]);
        assert_eq!(qt.dtype(), DType::GPTQ_INT4);
        assert_eq!(qt.numel(), in_features * out_features);
        assert_eq!(qt.group_size(), Some(group_size));
        assert!(qt.qzeros_slice().is_some());
    }

    #[test]
    fn test_quantized_tensor_awq_creation() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let in_features = 256;
        let out_features = 128;
        let group_size = 128;
        let num_groups = in_features / group_size;

        let qweight = vec![0u8; in_features * (out_features / 8) * 4];
        let scales = vec![0u8; num_groups * out_features * 2];
        let qzeros = vec![0u8; num_groups * (out_features / 8) * 4];

        let qt = QuantizedTensor::from_gptq_raw(
            &ctx,
            &[in_features, out_features],
            DType::AWQ_INT4,
            &qweight,
            &scales,
            &qzeros,
            group_size,
        )
        .expect("Failed to create AWQ tensor");

        assert_eq!(qt.dtype(), DType::AWQ_INT4);
        assert_eq!(qt.group_size(), Some(group_size));
    }

    #[test]
    #[should_panic(expected = "GPTQ_INT4 or AWQ_INT4")]
    fn test_gptq_raw_rejects_non_group_quantized() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let _ = QuantizedTensor::from_gptq_raw(&ctx, &[32, 32], DType::Q8_0, &[], &[], &[], 128);
    }

    #[test]
    fn test_quantized_tensor_non_gptq_has_no_qzeros() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let qt =
            QuantizedTensor::from_raw(&ctx, &[8, 8], DType::F8E4M3, &vec![0u8; 64], &[]).unwrap();

        assert!(qt.qzeros_slice().is_none());
        assert_eq!(qt.group_size(), None);
    }
}
