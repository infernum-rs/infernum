//! Metal tensor implementation.
//!
//! Uses `MTLBuffer` with `StorageModeShared` (unified memory on Apple Silicon)
//! so the CPU can directly read/write buffer contents without explicit copies.

use std::sync::Arc;

use metal::Buffer;

use infernum::dtype::DType;
use infernum::tensor::Tensor;

use crate::context::MetalContext;

/// A Metal-resident tensor backed by a shared `MTLBuffer`.
///
/// Uses `Arc<Buffer>` so clones and `slice_view` are cheap (shared backing).
/// `StorageModeShared` gives both CPU and GPU access to the same memory.
///
/// Each tensor carries a [`MetalContext`] so that op trait methods (which
/// are static and receive no device handle) can access the command queue
/// and compute pipelines from any input tensor.
#[derive(Clone)]
pub struct MetalTensor {
    ctx: MetalContext,
    buffer: Arc<Buffer>,
    /// Byte offset into the buffer (for slice_view).
    offset: usize,
    shape: Vec<usize>,
    dtype: DType,
}

// SAFETY: Metal buffers with StorageModeShared are accessible from any thread.
// We synchronise via command buffer completion before CPU reads.
unsafe impl Send for MetalTensor {}
unsafe impl Sync for MetalTensor {}

impl MetalTensor {
    /// Create a tensor from raw bytes, copying into a new Metal buffer.
    ///
    /// # Panics
    /// Panics if `data.len()` doesn't match the expected size for the shape and dtype.
    #[must_use]
    pub fn from_raw_bytes(ctx: &MetalContext, shape: &[usize], dtype: DType, data: &[u8]) -> Self {
        let numel: usize = shape.iter().product();
        let expected_bytes = if dtype.is_quantized() {
            // For quantized types, trust the caller's data length
            data.len()
        } else {
            numel * dtype.size_in_bytes()
        };
        assert_eq!(
            data.len(),
            expected_bytes,
            "data len {} != expected {expected_bytes} for shape {shape:?} dtype {dtype}",
            data.len()
        );

        let buffer = ctx.device().new_buffer_with_data(
            data.as_ptr().cast(),
            data.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Self {
            ctx: ctx.clone(),
            buffer: Arc::new(buffer),
            offset: 0,
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// Create a tensor from an f32 slice.
    #[must_use]
    pub fn from_f32(ctx: &MetalContext, shape: &[usize], data: &[f32]) -> Self {
        Self::from_raw_bytes(ctx, shape, DType::F32, bytemuck::cast_slice(data))
    }

    /// Create a zero-initialized tensor.
    #[must_use]
    pub fn zeros(ctx: &MetalContext, shape: &[usize], dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * dtype.size_in_bytes();
        let buffer = ctx.device().new_buffer(
            size_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Zero-initialize via memset on the shared buffer
        unsafe {
            std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, size_bytes);
        }

        Self {
            ctx: ctx.clone(),
            buffer: Arc::new(buffer),
            offset: 0,
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// Reference to the Metal context embedded in this tensor.
    ///
    /// Op implementations use this to access the device, command queue,
    /// and compute pipeline states without needing an external device handle.
    #[must_use]
    pub fn context(&self) -> &MetalContext {
        &self.ctx
    }

    /// Direct pointer to the tensor's data in shared memory.
    ///
    /// # Safety
    /// The caller must ensure no GPU command buffer is concurrently writing
    /// to this buffer region.
    #[must_use]
    pub fn contents_ptr(&self) -> *mut u8 {
        unsafe { self.buffer.contents().cast::<u8>().add(self.offset) }
    }

    /// Read the tensor data as a byte slice (CPU-side, unified memory).
    ///
    /// # Safety
    /// The caller must ensure no GPU command buffer is concurrently writing
    /// to this buffer.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        let len = self.size_in_bytes();
        unsafe { std::slice::from_raw_parts(self.contents_ptr(), len) }
    }

    /// Read the tensor data as an f32 slice.
    ///
    /// # Panics
    /// Panics if the dtype is not F32.
    #[must_use]
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32, "as_f32_slice: expected F32 tensor");
        bytemuck::cast_slice(self.as_bytes())
    }

    /// Read the tensor data as an i32 slice (U32 dtype reinterpreted as i32).
    ///
    /// Block tables and positions are stored as U32 but contain signed values.
    ///
    /// # Panics
    /// Panics if the dtype is not U32.
    #[must_use]
    pub fn as_i32_slice(&self) -> &[i32] {
        assert_eq!(
            self.dtype,
            DType::U32,
            "as_i32_slice: expected U32 tensor (i32 view)"
        );
        bytemuck::cast_slice(self.as_bytes())
    }

    /// Mutable f32 slice into the Metal buffer (unified memory).
    ///
    /// # Safety
    /// The caller must ensure no GPU command buffer is concurrently
    /// reading or writing to this buffer region.
    ///
    /// # Panics
    /// Panics if the dtype is not F32.
    #[must_use]
    #[allow(clippy::cast_ptr_alignment)]
    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        assert_eq!(
            self.dtype,
            DType::F32,
            "as_f32_slice_mut: expected F32 tensor"
        );
        let len = self.numel();
        // Metal buffers with StorageModeShared are always page-aligned (4096 bytes),
        // so casting to f32 (4-byte alignment) is safe.
        unsafe { std::slice::from_raw_parts_mut(self.contents_ptr().cast::<f32>(), len) }
    }

    /// Reference to the underlying Metal buffer.
    #[must_use]
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Byte offset into the Metal buffer.
    #[must_use]
    pub fn buffer_offset(&self) -> usize {
        self.offset
    }

    /// Total size of this tensor's data in bytes.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn data_size_bytes(&self) -> usize {
        if self.dtype.is_quantized() {
            // For quantized, compute from buffer length minus offset
            self.buffer.length() as usize - self.offset
        } else {
            self.numel() * self.dtype.size_in_bytes()
        }
    }
}

impl Tensor for MetalTensor {
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
            "reshape: numel mismatch {} vs {new_numel}",
            self.numel()
        );
        Self {
            ctx: self.ctx.clone(),
            buffer: Arc::clone(&self.buffer),
            offset: self.offset,
            shape: shape.to_vec(),
            dtype: self.dtype,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn slice_view(&self, offset: usize, shape: &[usize]) -> Self {
        let byte_offset = offset * self.dtype.size_in_bytes();
        let new_offset = self.offset + byte_offset;
        let new_numel: usize = shape.iter().product();
        let new_end = new_offset + new_numel * self.dtype.size_in_bytes();
        assert!(
            new_end <= self.buffer.length() as usize,
            "slice_view: end {new_end} exceeds buffer length {}",
            self.buffer.length()
        );
        Self {
            ctx: self.ctx.clone(),
            buffer: Arc::clone(&self.buffer),
            offset: new_offset,
            shape: shape.to_vec(),
            dtype: self.dtype,
        }
    }
}
