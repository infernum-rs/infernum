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

use std::cell::UnsafeCell;
use std::collections::HashMap;
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

/// Active command buffer + encoder for batching dispatches.
///
/// All dispatches within a decode step encode onto the same command buffer
/// and compute encoder, eliminating per-dispatch command buffer overhead.
/// `flush()` ends the encoder, commits the buffer, and waits.
struct ActiveEncoder {
    /// Owned (retained) command buffer.
    cmd: metal::CommandBuffer,
    /// Raw pointer to the active compute command encoder.
    /// Valid as long as `cmd` is alive and `end_encoding()` hasn't been called.
    encoder: *mut metal::ComputeCommandEncoderRef,
    /// Number of dispatches encoded (for stats).
    dispatch_count: u32,
}

// SAFETY: Metal command buffers and encoders are accessed from a single
// thread at a time (the model forward-pass thread). The UnsafeCell
// protecting `active_encoder` relies on this single-threaded access
// pattern — no concurrent mutation is possible.
unsafe impl Send for ActiveEncoder {}

struct MetalContextInner {
    device: Device,
    queue: metal::CommandQueue,
    /// Preloaded compute pipeline states, keyed by kernel function name.
    pipelines: HashMap<String, metal::ComputePipelineState>,
    /// Dispatch profiling stats (behind Mutex for interior mutability).
    stats: Mutex<DispatchStats>,
    /// Active command buffer + encoder, or None if no work is pending.
    ///
    /// Uses `UnsafeCell` instead of `Mutex` because all encoding happens
    /// on a single thread (the model forward-pass worker). Eliminating
    /// the per-dispatch mutex lock/unlock overhead.
    active_encoder: UnsafeCell<Option<ActiveEncoder>>,
    /// When true, each dispatch uses its own command buffer for per-kernel
    /// GPU timing. Much slower, but gives accurate per-kernel time breakdown.
    profile_per_kernel: bool,
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
                active_encoder: UnsafeCell::new(None),
                profile_per_kernel: false,
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
                active_encoder: UnsafeCell::new(None),
                profile_per_kernel: false,
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

    /// Enable per-kernel GPU timing profiling.
    ///
    /// When enabled, each dispatch uses its own command buffer and waits
    /// for completion, giving accurate per-kernel GPU time. This is much
    /// slower than normal batched execution — use only for profiling.
    ///
    /// Must be called before any dispatches (typically right after
    /// construction). Requires exclusive access to the `Arc`.
    pub fn set_profile_per_kernel(&mut self, enabled: bool) {
        Arc::get_mut(&mut self.inner)
            .expect("set_profile_per_kernel: MetalContext already shared")
            .profile_per_kernel = enabled;
    }

    // ------------------------------------------------------------------
    // Dispatch profiling
    // ------------------------------------------------------------------

    /// Print a summary of dispatch statistics to stderr.
    ///
    /// In normal mode: shows per-kernel dispatch counts.
    /// In `profile_per_kernel` mode: shows per-kernel GPU time breakdown.
    #[allow(clippy::cast_precision_loss)]
    pub fn print_dispatch_stats(&self) {
        let stats = self.inner.stats.lock().unwrap();
        if stats.kernels.is_empty() {
            eprintln!("[Metal] No dispatch stats recorded.");
            return;
        }

        let has_timing = self.inner.profile_per_kernel;

        // Kernel entries (excluding internal stats entries)
        let mut entries: Vec<_> = stats
            .kernels
            .iter()
            .filter(|(k, _)| !k.starts_with('_'))
            .collect();

        let total_dispatches: u64 = entries.iter().map(|(_, s)| s.count).sum();

        if has_timing {
            // Per-kernel GPU timing mode
            let total_time: Duration = entries.iter().map(|(_, s)| s.total_time).sum();
            let total_ms = total_time.as_secs_f64() * 1000.0;

            entries.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

            eprintln!();
            eprintln!(
                "[Metal per-kernel GPU timing] {total_dispatches} dispatches, \
                 {total_ms:.1}ms total GPU time"
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
        } else {
            // Dispatch count mode (no per-kernel timing)
            let dispatches = stats.kernels.get("_dispatches").map_or(0, |d| d.count);
            let flush = stats.kernels.get("_flush");
            let flush_count = flush.map_or(0, |f| f.count);
            let total_ms = flush.map_or(0.0, |f| f.total_time.as_secs_f64() * 1000.0);

            eprintln!();
            eprintln!(
                "[Metal dispatch stats] {dispatches} dispatches in \
                 {flush_count} flushes, {total_ms:.1}ms total GPU wait"
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
    // Shared encoder management
    // ------------------------------------------------------------------

    /// Get mutable access to the active encoder state.
    ///
    /// # Safety
    /// All encoding must happen on a single thread. This is guaranteed
    /// by the engine's worker thread architecture — only one thread
    /// calls forward passes and dispatch methods at a time.
    #[allow(clippy::mut_from_ref)]
    fn active_encoder_mut(&self) -> &mut Option<ActiveEncoder> {
        unsafe { &mut *self.inner.active_encoder.get() }
    }

    /// Get the active compute command encoder, creating a new command
    /// buffer + encoder pair if none exists.
    ///
    /// Returns a raw pointer to the encoder.
    fn ensure_encoder(
        active: &mut Option<ActiveEncoder>,
        queue: &metal::CommandQueue,
    ) -> *mut metal::ComputeCommandEncoderRef {
        if let Some(ref ae) = active {
            return ae.encoder;
        }
        let cmd_ref = queue.new_command_buffer();
        let cmd = cmd_ref.to_owned();
        let enc = cmd_ref.new_compute_command_encoder();
        let encoder = std::ptr::from_ref(enc).cast_mut();
        *active = Some(ActiveEncoder {
            cmd,
            encoder,
            dispatch_count: 0,
        });
        encoder
    }

    /// After encoding a dispatch, increment the dispatch counter and
    /// optionally flush immediately for per-kernel timing.
    ///
    /// In normal (non-profiling) mode, this is a single u32 increment —
    /// no String allocation, no HashMap lookup, no Mutex lock.
    fn finish_dispatch(&self, active: &mut Option<ActiveEncoder>, kernel: &str) {
        let ae = active.as_mut().unwrap();
        ae.dispatch_count += 1;

        if self.inner.profile_per_kernel {
            // Flush this single dispatch to get isolated GPU timing.
            let ae = active.take().unwrap();
            let enc = unsafe { &*ae.encoder };
            enc.end_encoding();

            let t0 = Instant::now();
            ae.cmd.commit();
            ae.cmd.wait_until_completed();
            let elapsed = t0.elapsed();

            let mut stats = self.inner.stats.lock().unwrap();
            let entry = stats.kernels.entry(kernel.to_owned()).or_default();
            entry.count += 1;
            entry.total_time += elapsed;
        }
    }

    // ------------------------------------------------------------------
    // Dispatch helpers
    // ------------------------------------------------------------------

    /// Dispatch a 1-D compute kernel.
    ///
    /// Encodes onto the shared command buffer without committing.
    /// Call [`flush`](Self::flush) to submit all pending work.
    #[allow(clippy::cast_possible_truncation)]
    pub fn dispatch_1d(
        &self,
        pipeline: &str,
        buffers: &[(&metal::BufferRef, usize)],
        params: &[u8],
        n: usize,
    ) {
        let pso = self.pipeline(pipeline);
        let active = self.active_encoder_mut();
        let enc_ptr = Self::ensure_encoder(active, &self.inner.queue);
        let enc = unsafe { &*enc_ptr };

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

        self.finish_dispatch(active, pipeline);
    }

    /// Dispatch a 2-D compute kernel.
    ///
    /// Encodes onto the shared command buffer without committing.
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
        let active = self.active_encoder_mut();
        let enc_ptr = Self::ensure_encoder(active, &self.inner.queue);
        let enc = unsafe { &*enc_ptr };

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

        self.finish_dispatch(active, pipeline);
    }

    /// Dispatch a compute kernel with a 3D thread grid.
    pub fn dispatch_3d(
        &self,
        pipeline: &str,
        buffers: &[(&metal::BufferRef, usize)],
        params: &[u8],
        width: usize,
        height: usize,
        depth: usize,
    ) {
        let pso = self.pipeline(pipeline);
        let active = self.active_encoder_mut();
        let enc_ptr = Self::ensure_encoder(active, &self.inner.queue);
        let enc = unsafe { &*enc_ptr };

        enc.set_compute_pipeline_state(pso);

        for (idx, &(buf, offset)) in buffers.iter().enumerate() {
            enc.set_buffer(idx as u64, Some(buf), offset as u64);
        }

        if !params.is_empty() {
            let param_idx = buffers.len() as u64;
            enc.set_bytes(param_idx, params.len() as u64, params.as_ptr().cast());
        }

        let max_total = pso.max_total_threads_per_threadgroup();
        let max_w = pso.thread_execution_width();
        let max_h = (max_total / max_w).min(height as u64);
        let max_d = (max_total / (max_w * max_h)).max(1).min(depth as u64);
        let grid = MTLSize::new(width as u64, height as u64, depth as u64);
        let group = MTLSize::new(max_w.min(width as u64), max_h.min(height as u64), max_d);
        enc.dispatch_threads(grid, group);

        self.finish_dispatch(active, pipeline);
    }

    /// Dispatch a compute kernel with explicit threadgroup counts.
    ///
    /// Encodes onto the shared command buffer without committing.
    /// Supports threadgroup shared memory via `threadgroup_mem_len`.
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
        let active = self.active_encoder_mut();
        let enc_ptr = Self::ensure_encoder(active, &self.inner.queue);
        let enc = unsafe { &*enc_ptr };

        enc.set_compute_pipeline_state(pso);

        for (idx, &(buf, offset)) in buffers.iter().enumerate() {
            enc.set_buffer(idx as u64, Some(buf), offset as u64);
        }

        if !params.is_empty() {
            let param_idx = buffers.len() as u64;
            enc.set_bytes(param_idx, params.len() as u64, params.as_ptr().cast());
        }

        if threadgroup_mem_len > 0 {
            enc.set_threadgroup_memory_length(0, threadgroup_mem_len as u64);
        }

        enc.dispatch_thread_groups(threadgroups, threads_per_group);

        self.finish_dispatch(active, pipeline);
    }

    /// Commit all pending GPU work and block until complete.
    ///
    /// Ends the active compute encoder, commits the command buffer, and
    /// waits for completion. No-op if no work is pending.
    pub fn flush(&self) {
        let active = self.active_encoder_mut();
        let Some(ae) = active.take() else {
            return;
        };

        // End encoding and commit
        let enc = unsafe { &*ae.encoder };
        enc.end_encoding();

        let t0 = Instant::now();
        ae.cmd.commit();
        ae.cmd.wait_until_completed();
        let elapsed = t0.elapsed();

        // Record flush stats: count and total GPU wait time.
        {
            let mut stats = self.inner.stats.lock().unwrap();
            let dispatch_entry = stats.kernels.entry("_dispatches".to_owned()).or_default();
            dispatch_entry.count += u64::from(ae.dispatch_count);
            let flush_entry = stats.kernels.entry("_flush".to_owned()).or_default();
            flush_entry.count += 1;
            flush_entry.total_time += elapsed;
        }
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
        let _ctx = MetalContext::new();
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
