//! Metal device context.
//!
//! Holds the Metal device, command queue, and preloaded compute pipeline
//! states. Constructed once at startup and shared (via `Clone`) across
//! model layers and ops.
//!
//! Also provides [`dispatch_1d`] and [`dispatch_2d`] helpers that encode
//! a compute command, set buffers/parameters, dispatch with appropriate
//! threadgroup sizes, commit, and wait — reducing boilerplate across all
//! kernel implementations.

use std::collections::HashMap;
use std::sync::Arc;

use metal::{Device, MTLSize};

/// Compiled Metal shader library, embedded at build time.
///
/// `build.rs` compiles all `.metal` files in `kernels/` into a single
/// `infernum.metallib`. If the `kernels/` directory is empty the build
/// script does *not* produce this file, so we gate inclusion behind a
/// `cfg` check: when the file is missing we fall back to an empty slice.
///
/// The `option_env!` + `include_bytes!` trick lets us compile even when
/// no `.metal` files exist yet (early Phase 1).
const METALLIB_DATA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels/infernum.metallib"));

/// Metal device context shared across all ops.
///
/// Wraps an `Arc` so cloning is cheap. Models store one `MetalContext`
/// and pass `&MetalContext` to op trait methods as `DeviceHandle`.
#[derive(Clone)]
pub struct MetalContext {
    inner: Arc<MetalContextInner>,
}

struct MetalContextInner {
    device: Device,
    queue: metal::CommandQueue,
    /// Preloaded compute pipeline states, keyed by kernel function name.
    pipelines: HashMap<String, metal::ComputePipelineState>,
}

// SAFETY: Metal objects are thread-safe when accessed through command buffers.
// The `MetalContext` is read-only after construction (pipelines are immutable).
unsafe impl Send for MetalContextInner {}
unsafe impl Sync for MetalContextInner {}

/// Default number of threads per threadgroup for 1-D dispatches.
const DEFAULT_THREADGROUP_SIZE: u64 = 256;

impl MetalContext {
    /// Create a new Metal context using the system default GPU.
    ///
    /// Loads the embedded metallib (compiled from `kernels/*.metal` at
    /// build time) and pre-creates compute pipeline states for every
    /// kernel function found within it.
    ///
    /// # Panics
    /// Panics if no Metal device is available or if the metallib
    /// cannot be loaded.
    #[must_use]
    pub fn new() -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let queue = device.new_command_queue();

        let pipelines = if METALLIB_DATA.is_empty() {
            HashMap::new()
        } else {
            load_pipelines(&device, METALLIB_DATA)
        };

        Self {
            inner: Arc::new(MetalContextInner {
                device,
                queue,
                pipelines,
            }),
        }
    }

    /// Create a context with preloaded pipelines from a metallib.
    ///
    /// # Errors
    /// Returns an error if the metallib cannot be loaded or a pipeline
    /// cannot be created.
    pub fn with_metallib(metallib_data: &[u8]) -> infernum::Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device found".into()))?;
        let queue = device.new_command_queue();

        let pipelines = load_pipelines_checked(&device, metallib_data)?;

        Ok(Self {
            inner: Arc::new(MetalContextInner {
                device,
                queue,
                pipelines,
            }),
        })
    }

    /// Reference to the Metal device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// Reference to the command queue.
    #[must_use]
    pub fn queue(&self) -> &metal::CommandQueue {
        &self.inner.queue
    }

    /// Get a preloaded compute pipeline by kernel function name.
    ///
    /// # Panics
    /// Panics if the pipeline was not loaded at construction time.
    #[must_use]
    pub fn pipeline(&self, name: &str) -> &metal::ComputePipelineState {
        self.inner
            .pipelines
            .get(name)
            .unwrap_or_else(|| panic!("Metal pipeline '{name}' not found"))
    }

    /// Check if a pipeline exists (for optional kernels).
    #[must_use]
    pub fn has_pipeline(&self, name: &str) -> bool {
        self.inner.pipelines.contains_key(name)
    }

    /// Create a new command buffer from the queue.
    #[must_use]
    pub fn command_buffer(&self) -> &metal::CommandBufferRef {
        self.inner.queue.new_command_buffer()
    }

    // ------------------------------------------------------------------
    // Dispatch helpers
    // ------------------------------------------------------------------

    /// Dispatch a 1-D compute kernel.
    ///
    /// * `pipeline` — kernel function name (must be in the loaded metallib).
    /// * `buffers`  — `(buffer, offset)` pairs bound in order to buffer
    ///   indices 0, 1, 2, …
    /// * `params`   — raw bytes bound to the *next* buffer index after
    ///   `buffers`.  Pass `&[]` when the kernel has no extra parameters.
    /// * `n`        — total number of threads to launch.
    ///
    /// Creates a command buffer, encodes the compute command, commits,
    /// and **blocks** until completion.
    ///
    /// # Panics
    /// Panics if the pipeline is not found or if the command buffer
    /// reports an error.
    #[allow(clippy::cast_possible_truncation)]
    pub fn dispatch_1d(
        &self,
        pipeline: &str,
        buffers: &[(&metal::BufferRef, usize)],
        params: &[u8],
        n: usize,
    ) {
        let pso = self.pipeline(pipeline);
        let cmd = self.command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(pso);

        for (idx, &(buf, offset)) in buffers.iter().enumerate() {
            enc.set_buffer(idx as u64, Some(buf), offset as u64);
        }

        if !params.is_empty() {
            let param_idx = buffers.len() as u64;
            enc.set_bytes(param_idx, params.len() as u64, params.as_ptr().cast());
        }

        let tg = pso.thread_execution_width().min(DEFAULT_THREADGROUP_SIZE);
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(tg, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch a 2-D compute kernel.
    ///
    /// Same as [`dispatch_1d`](Self::dispatch_1d) but launches a
    /// `width × height` grid.
    #[allow(clippy::cast_possible_truncation)]
    pub fn dispatch_2d(
        &self,
        pipeline: &str,
        buffers: &[(&metal::BufferRef, usize)],
        params: &[u8],
        width: usize,
        height: usize,
    ) {
        let pso = self.pipeline(pipeline);
        let cmd = self.command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(pso);

        for (idx, &(buf, offset)) in buffers.iter().enumerate() {
            enc.set_buffer(idx as u64, Some(buf), offset as u64);
        }

        if !params.is_empty() {
            let param_idx = buffers.len() as u64;
            enc.set_bytes(param_idx, params.len() as u64, params.as_ptr().cast());
        }

        let max_w = pso.thread_execution_width();
        let max_h = pso.max_total_threads_per_threadgroup() / max_w;
        let grid = MTLSize::new(width as u64, height as u64, 1);
        let group = MTLSize::new(max_w.min(width as u64), max_h.min(height as u64), 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch a compute kernel with explicit threadgroup counts.
    ///
    /// Unlike [`dispatch_1d`](Self::dispatch_1d), this uses
    /// `dispatch_thread_groups` (not `dispatch_threads`), giving the
    /// caller full control over the grid layout. Supports threadgroup
    /// shared memory via `threadgroup_mem_len`.
    ///
    /// * `pipeline`            — kernel function name.
    /// * `buffers`             — `(buffer, offset)` pairs.
    /// * `params`              — raw bytes for extra parameters.
    /// * `threadgroups`        — number of threadgroups `(x, y, z)`.
    /// * `threads_per_group`   — threads per threadgroup `(x, y, z)`.
    /// * `threadgroup_mem_len` — bytes of threadgroup memory (0 = none).
    #[allow(clippy::cast_possible_truncation)]
    pub fn dispatch_threadgroups(
        &self,
        pipeline: &str,
        buffers: &[(&metal::BufferRef, usize)],
        params: &[u8],
        threadgroups: MTLSize,
        threads_per_group: MTLSize,
        threadgroup_mem_len: usize,
    ) {
        let pso = self.pipeline(pipeline);
        let cmd = self.command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(pso);

        for (idx, &(buf, offset)) in buffers.iter().enumerate() {
            enc.set_buffer(idx as u64, Some(buf), offset as u64);
        }

        if !params.is_empty() {
            let param_idx = buffers.len() as u64;
            enc.set_bytes(param_idx, params.len() as u64, params.as_ptr().cast());
        }

        if threadgroup_mem_len > 0 {
            // Threadgroup memory index matches MSL [[threadgroup(N)]] attribute.
            // All our kernels use [[threadgroup(0)]].
            enc.set_threadgroup_memory_length(0, threadgroup_mem_len as u64);
        }

        enc.dispatch_thread_groups(threadgroups, threads_per_group);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();
    }
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------------------------------------------------
// Internal helpers
// ------------------------------------------------------------------

/// Load all kernel functions from a metallib and create pipelines.
///
/// # Panics
/// Panics on any failure — used in the infallible `new()` path.
fn load_pipelines(device: &Device, data: &[u8]) -> HashMap<String, metal::ComputePipelineState> {
    let library = device
        .new_library_with_data(data)
        .expect("Failed to load embedded metallib");

    let mut pipelines = HashMap::new();
    for name in &library.function_names() {
        let name_str = name.clone();
        let function = library
            .get_function(&name_str, None)
            .unwrap_or_else(|e| panic!("Failed to get function '{name_str}': {e}"));
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap_or_else(|e| panic!("Failed to create pipeline for '{name_str}': {e}"));
        pipelines.insert(name_str, pipeline);
    }
    pipelines
}

/// Fallible version of `load_pipelines` for `with_metallib`.
fn load_pipelines_checked(
    device: &Device,
    data: &[u8],
) -> infernum::Result<HashMap<String, metal::ComputePipelineState>> {
    let library = device
        .new_library_with_data(data)
        .map_err(|e| infernum::Error::Other(format!("Failed to load metallib: {e}")))?;

    let mut pipelines = HashMap::new();
    for name in &library.function_names() {
        let name_str = name.clone();
        let function = library.get_function(&name_str, None).map_err(|e| {
            infernum::Error::Other(format!("Failed to get function '{name_str}': {e}"))
        })?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                infernum::Error::Other(format!("Failed to create pipeline for '{name_str}': {e}"))
            })?;
        pipelines.insert(name_str, pipeline);
    }
    Ok(pipelines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = MetalContext::new();
        // Verify we can create a command buffer without panic
        let _cmd = ctx.command_buffer();
    }

    #[test]
    fn test_context_has_fill_kernel() {
        let ctx = MetalContext::new();
        assert!(
            ctx.has_pipeline("fill_f32"),
            "fill_f32 pipeline should exist"
        );
    }

    #[test]
    fn test_dispatch_1d_fill() {
        let ctx = MetalContext::new();
        let n: usize = 64;
        let buf = ctx.device().new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let value: f32 = 42.0;
        ctx.dispatch_1d("fill_f32", &[(&buf, 0)], bytemuck::bytes_of(&value), n);

        let result: &[f32] = unsafe { std::slice::from_raw_parts(buf.contents().cast::<f32>(), n) };
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 42.0).abs() < 1e-6,
                "fill_f32: element {i} = {v}, expected 42.0"
            );
        }
    }
}
