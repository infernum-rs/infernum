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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use metal::{Device, MTLSize};

/// Per-kernel dispatch timing statistics.
#[derive(Default)]
struct KernelStats {
    count: u64,
    total_time: Duration,
}

/// Aggregate dispatch statistics for profiling.
#[derive(Default)]
struct DispatchStats {
    kernels: HashMap<String, KernelStats>,
}

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
    /// Dispatch profiling stats (behind Mutex for interior mutability).
    stats: Mutex<DispatchStats>,
    /// True when GPU dispatches have been committed but not yet waited on.
    /// `flush()` drains pending work; CPU reads must call `flush()` first.
    has_pending: AtomicBool,
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
                stats: Mutex::new(DispatchStats::default()),
                has_pending: AtomicBool::new(false),
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
                stats: Mutex::new(DispatchStats::default()),
                has_pending: AtomicBool::new(false),
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
    // Dispatch profiling
    // ------------------------------------------------------------------

    /// Record a kernel dispatch timing.
    fn record_dispatch(&self, kernel: &str, elapsed: Duration) {
        let mut stats = self.inner.stats.lock().unwrap();
        let entry = stats
            .kernels
            .entry(kernel.to_string())
            .or_insert_with(KernelStats::default);
        entry.count += 1;
        entry.total_time += elapsed;
    }

    /// Print a summary of dispatch statistics to stderr.
    ///
    /// Shows per-kernel call count, total time, average time per call,
    /// and percentage of total GPU time.
    pub fn print_dispatch_stats(&self) {
        let stats = self.inner.stats.lock().unwrap();
        if stats.kernels.is_empty() {
            eprintln!("[Metal] No dispatch stats recorded.");
            return;
        }

        let total_time: Duration = stats.kernels.values().map(|s| s.total_time).sum();
        let total_count: u64 = stats.kernels.values().map(|s| s.count).sum();
        let total_ms = total_time.as_secs_f64() * 1000.0;

        // Sort by cumulative time descending
        let mut entries: Vec<_> = stats.kernels.iter().collect();
        entries.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

        eprintln!();
        eprintln!(
            "[Metal dispatch stats] {total_count} dispatches, {total_ms:.1}ms total GPU wait"
        );
        eprintln!(
            "{:<40} {:>8} {:>10} {:>10} {:>6}",
            "kernel", "calls", "total(ms)", "avg(μs)", "%"
        );
        eprintln!("{}", "-".repeat(78));

        for (name, ks) in &entries {
            let ms = ks.total_time.as_secs_f64() * 1000.0;
            let avg_us = if ks.count > 0 {
                ks.total_time.as_secs_f64() * 1_000_000.0 / ks.count as f64
            } else {
                0.0
            };
            let pct = if total_ms > 0.0 {
                ms / total_ms * 100.0
            } else {
                0.0
            };
            eprintln!(
                "{name:<40} {c:>8} {ms:>9.1} {avg_us:>9.0} {pct:>5.1}%",
                c = ks.count
            );
        }
        eprintln!();
    }

    /// Reset all dispatch statistics.
    pub fn reset_dispatch_stats(&self) {
        let mut stats = self.inner.stats.lock().unwrap();
        stats.kernels.clear();
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

        let t0 = Instant::now();
        cmd.commit();
        self.inner.has_pending.store(true, Ordering::Release);
        self.record_dispatch(pipeline, t0.elapsed());
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

        let t0 = Instant::now();
        cmd.commit();
        self.inner.has_pending.store(true, Ordering::Release);
        self.record_dispatch(pipeline, t0.elapsed());
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

        let t0 = Instant::now();
        cmd.commit();
        self.inner.has_pending.store(true, Ordering::Release);
        self.record_dispatch(pipeline, t0.elapsed());
    }

    /// Ensure all committed GPU work has completed before CPU reads.
    ///
    /// Submits an empty command buffer and blocks until it completes.
    /// Since command buffers execute in queue order, this guarantees all
    /// prior dispatches have finished.  No-op if no work is pending.
    pub fn flush(&self) {
        if !self.inner.has_pending.load(Ordering::Acquire) {
            return;
        }
        let t0 = Instant::now();
        let cmd = self.command_buffer();
        cmd.commit();
        cmd.wait_until_completed();
        self.inner.has_pending.store(false, Ordering::Release);
        self.record_dispatch("_flush", t0.elapsed());
    }
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------------------------------------------------
// Public helpers
// ------------------------------------------------------------------

/// Choose a threadgroup size for reduction kernels.
/// Must be a power of 2 for the tree reduction, capped at 256.
#[must_use]
pub fn reduction_threadgroup_size(n: usize) -> usize {
    let max_tg: usize = 256;
    let tg = n.min(max_tg);
    // Round down to nearest power of 2
    1 << (usize::BITS - 1 - tg.leading_zeros())
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
        ctx.flush();

        let result: &[f32] = unsafe { std::slice::from_raw_parts(buf.contents().cast::<f32>(), n) };
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 42.0).abs() < 1e-6,
                "fill_f32: element {i} = {v}, expected 42.0"
            );
        }
    }
}
