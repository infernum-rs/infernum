//! KV cache for incremental decoding
//!
//! Pre-allocates GPU memory for key and value tensors across all layers,
//! allowing O(1) append and O(n) attention against the full history.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig, ValidAsZeroBits};

use super::CudaContext;
use super::CudaTensor;
use crate::dtype::TensorDType;
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/append_kv.ptx"));
const KERNEL_NAMES: &[&str] = &["append_kv_f32", "append_kv_f16", "append_kv_bf16"];

/// Kernel name suffix for dtype
fn kernel_suffix<T: cudarc::driver::DeviceRepr>() -> &'static str {
    let type_name = std::any::type_name::<T>();
    if type_name.contains("f32") {
        "f32"
    } else if type_name.contains("f16") && !type_name.contains("bf16") {
        "f16"
    } else if type_name.contains("bf16") {
        "bf16"
    } else {
        panic!("Unsupported dtype for append_kv: {type_name}")
    }
}

/// KV cache for one layer's key or value buffer.
///
/// Layout: `(max_seq_len, num_kv_heads, head_dim)`, row-major.
/// Only the first `current_len` positions contain valid data.
struct LayerBuffer<T: TensorDType> {
    k: CudaTensor<T>,
    v: CudaTensor<T>,
}

/// Pre-allocated KV cache for all transformer layers.
///
/// Holds key and value buffers on GPU, tracks current sequence length,
/// and provides `append` / `get` / `reset` operations.
///
/// Generic over `T` (f32, f16, bf16) to support different compute precisions.
pub struct KvCache<T: TensorDType = f32> {
    layers: Vec<LayerBuffer<T>>,
    ctx: CudaContext,
    current_len: usize,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

/// Launch the GPU kernel that copies `new_data` into `cache` at `current_len`.
#[allow(clippy::too_many_arguments)]
fn launch_append<T: TensorDType + cudarc::driver::DeviceRepr>(
    ctx: &CudaContext,
    cache: &mut CudaTensor<T>,
    new_data: &CudaTensor<T>,
    current_len: usize,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    new_seq_len: usize,
) -> Result<()> {
    let total = new_seq_len * num_kv_heads * head_dim;
    let device = ctx.device();

    let kernel_name = format!("append_kv_{}", kernel_suffix::<T>());
    let module_name = "append_kv";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

    let block_size = 256;
    let grid_size = (total + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                cache.cuda_slice_mut(),
                &new_data.cuda_slice(),
                current_len as i32,
                max_seq_len as i32,
                num_kv_heads as i32,
                head_dim as i32,
                new_seq_len as i32,
            ),
        )?;
    }

    Ok(())
}

impl<T: TensorDType + cudarc::driver::DeviceRepr + ValidAsZeroBits> KvCache<T> {
    /// Allocate a new KV cache.
    ///
    /// # Arguments
    /// * `ctx` — CUDA context
    /// * `num_layers` — number of transformer layers
    /// * `max_seq_len` — maximum sequence length the cache can hold
    /// * `num_kv_heads` — number of key-value heads (GQA-aware)
    /// * `head_dim` — dimension of each attention head
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation fails.
    pub fn new(
        ctx: &CudaContext,
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let shape = [max_seq_len, num_kv_heads, head_dim];
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(LayerBuffer {
                k: CudaTensor::zeros(ctx, &shape)?,
                v: CudaTensor::zeros(ctx, &shape)?,
            });
        }

        Ok(Self {
            layers,
            ctx: ctx.clone(),
            current_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
        })
    }

    /// Append new key/value tensors for a given layer.
    ///
    /// `k_new` and `v_new` must have shape `(new_seq_len, num_kv_heads, head_dim)`.
    /// `new_seq_len` can be > 1 (prefill) or 1 (decode step).
    ///
    /// # Panics
    /// Panics if the cache would overflow or shapes are wrong.
    ///
    /// # Errors
    /// Returns an error if the GPU kernel launch fails.
    pub fn append(
        &mut self,
        layer_idx: usize,
        k_new: &CudaTensor<T>,
        v_new: &CudaTensor<T>,
    ) -> Result<()> {
        let k_shape = k_new.shape();
        let v_shape = v_new.shape();
        assert_eq!(k_shape.len(), 3, "k_new must be 3D");
        assert_eq!(v_shape.len(), 3, "v_new must be 3D");

        let new_seq_len = k_shape[0];
        assert_eq!(k_shape[1], self.num_kv_heads, "k_new num_kv_heads mismatch");
        assert_eq!(k_shape[2], self.head_dim, "k_new head_dim mismatch");
        assert_eq!(v_shape, k_shape, "v_new shape must match k_new");

        assert!(
            self.current_len + new_seq_len <= self.max_seq_len,
            "KV cache overflow: current_len {} + new_seq_len {} > max_seq_len {}",
            self.current_len,
            new_seq_len,
            self.max_seq_len,
        );

        let buf = &mut self.layers[layer_idx];
        launch_append(
            &self.ctx,
            &mut buf.k,
            k_new,
            self.current_len,
            self.max_seq_len,
            self.num_kv_heads,
            self.head_dim,
            new_seq_len,
        )?;
        launch_append(
            &self.ctx,
            &mut buf.v,
            v_new,
            self.current_len,
            self.max_seq_len,
            self.num_kv_heads,
            self.head_dim,
            new_seq_len,
        )?;

        // Only advance current_len after the last layer appends
        // — the caller is responsible for calling `advance` after all layers.
        Ok(())
    }

    /// Advance the sequence position by `n` tokens.
    ///
    /// Must be called once per generation step, after all layers have appended.
    pub fn advance(&mut self, n: usize) {
        self.current_len += n;
    }

    /// Get the cached K and V slices for a given layer, up to the current length.
    ///
    /// Returns zero-copy views of shape `(current_len, num_kv_heads, head_dim)`
    /// that share GPU memory with the cache buffers.
    ///
    /// # Panics
    /// Panics if `current_len` is 0 (no data appended yet).
    #[must_use]
    pub fn get(&self, layer_idx: usize) -> (CudaTensor<T>, CudaTensor<T>) {
        assert!(self.current_len > 0, "KV cache is empty");
        self.get_up_to(layer_idx, self.current_len)
    }

    /// Get cached K and V slices for a given layer, up to `len` rows.
    ///
    /// Returns zero-copy views that share GPU memory with the cache buffers.
    /// This is useful when tokens have been appended but `advance()` has not
    /// yet been called (e.g., inside `attention_kv`).
    ///
    /// # Panics
    /// Panics if `len` is 0 or exceeds `max_seq_len`.
    #[must_use]
    pub fn get_up_to(&self, layer_idx: usize, len: usize) -> (CudaTensor<T>, CudaTensor<T>) {
        assert!(len > 0, "requested length must be > 0");
        assert!(
            len <= self.max_seq_len,
            "requested length exceeds max_seq_len"
        );

        let buf = &self.layers[layer_idx];
        let k_slice = self.slice_to_len(&buf.k, len);
        let v_slice = self.slice_to_len(&buf.v, len);
        (k_slice, v_slice)
    }

    /// Zero-copy view of the first `len` rows from a `(max_seq_len, heads, dim)` buffer.
    ///
    /// Returns a `CudaTensor` that shares the same GPU allocation as the cache
    /// buffer — no allocation or copy occurs.
    fn slice_to_len(&self, tensor: &CudaTensor<T>, len: usize) -> CudaTensor<T> {
        let shape = [len, self.num_kv_heads, self.head_dim];
        tensor.slice_view(0, &shape)
    }

    /// Reset the cache for reuse with a new sequence (no reallocation).
    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    /// Current sequence length stored in the cache.
    #[must_use]
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Maximum sequence length this cache can hold.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of layers in the cache.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_new() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let cache: KvCache<f32> =
            KvCache::new(&ctx, 2, 128, 4, 16).expect("Failed to create KV cache");

        assert_eq!(cache.current_len(), 0);
        assert_eq!(cache.max_seq_len(), 128);
        assert_eq!(cache.num_layers(), 2);
    }

    #[test]
    fn test_kv_cache_append_and_get_single() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let num_kv_heads = 2;
        let head_dim = 4;
        let mut cache = KvCache::new(&ctx, 1, 16, num_kv_heads, head_dim).unwrap();

        // Append a single token's K and V
        let k_data: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|i| (i as f32) + 100.0)
            .collect();

        let k = CudaTensor::from_slice(&ctx, &[1, num_kv_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[1, num_kv_heads, head_dim], &v_data).unwrap();

        cache.append(0, &k, &v).unwrap();
        cache.advance(1);

        assert_eq!(cache.current_len(), 1);

        let (k_out, v_out) = cache.get(0);
        assert_eq!(k_out.shape(), &[1, num_kv_heads, head_dim]);
        assert_eq!(v_out.shape(), &[1, num_kv_heads, head_dim]);

        let k_result = k_out.to_vec().unwrap();
        let v_result = v_out.to_vec().unwrap();
        assert_eq!(k_result, k_data);
        assert_eq!(v_result, v_data);
    }

    #[test]
    fn test_kv_cache_append_multiple_tokens() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let num_kv_heads = 2;
        let head_dim = 4;
        let mut cache = KvCache::new(&ctx, 1, 16, num_kv_heads, head_dim).unwrap();

        // Prefill: append 3 tokens at once
        let seq_len = 3;
        let k_data: Vec<f32> = (0..(seq_len * num_kv_heads * head_dim) as u32)
            .map(|i| i as f32)
            .collect();
        let v_data: Vec<f32> = (0..(seq_len * num_kv_heads * head_dim) as u32)
            .map(|i| (i as f32) + 100.0)
            .collect();

        let k = CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &v_data).unwrap();

        cache.append(0, &k, &v).unwrap();
        cache.advance(seq_len);

        assert_eq!(cache.current_len(), 3);

        let (k_out, v_out) = cache.get(0);
        assert_eq!(k_out.shape(), &[3, num_kv_heads, head_dim]);

        let k_result = k_out.to_vec().unwrap();
        let v_result = v_out.to_vec().unwrap();
        assert_eq!(k_result, k_data);
        assert_eq!(v_result, v_data);
    }

    #[test]
    fn test_kv_cache_incremental_append() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let num_kv_heads = 1;
        let head_dim = 2;
        let mut cache = KvCache::new(&ctx, 1, 16, num_kv_heads, head_dim).unwrap();

        // Token 0
        let k0 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[1.0, 2.0]).unwrap();
        let v0 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[10.0, 20.0]).unwrap();
        cache.append(0, &k0, &v0).unwrap();
        cache.advance(1);

        // Token 1
        let k1 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[3.0, 4.0]).unwrap();
        let v1 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[30.0, 40.0]).unwrap();
        cache.append(0, &k1, &v1).unwrap();
        cache.advance(1);

        assert_eq!(cache.current_len(), 2);

        let (k_out, v_out) = cache.get(0);
        assert_eq!(k_out.shape(), &[2, 1, 2]);

        let k_result = k_out.to_vec().unwrap();
        assert_eq!(k_result, vec![1.0, 2.0, 3.0, 4.0]);

        let v_result = v_out.to_vec().unwrap();
        assert_eq!(v_result, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let mut cache = KvCache::new(&ctx, 1, 16, 2, 4).unwrap();

        let k = CudaTensor::from_slice(&ctx, &[1, 2, 4], &[0.0; 8]).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[1, 2, 4], &[0.0; 8]).unwrap();
        cache.append(0, &k, &v).unwrap();
        cache.advance(1);
        assert_eq!(cache.current_len(), 1);

        cache.reset();
        assert_eq!(cache.current_len(), 0);
    }

    #[test]
    fn test_kv_cache_multi_layer() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let num_kv_heads = 1;
        let head_dim = 2;
        let mut cache = KvCache::new(&ctx, 2, 16, num_kv_heads, head_dim).unwrap();

        let k0 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[1.0, 2.0]).unwrap();
        let v0 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[10.0, 20.0]).unwrap();
        let k1 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[5.0, 6.0]).unwrap();
        let v1 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[50.0, 60.0]).unwrap();

        cache.append(0, &k0, &v0).unwrap();
        cache.append(1, &k1, &v1).unwrap();
        cache.advance(1);

        let (k_out_0, _) = cache.get(0);
        let (k_out_1, _) = cache.get(1);

        assert_eq!(k_out_0.to_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(k_out_1.to_vec().unwrap(), vec![5.0, 6.0]);
    }
}
