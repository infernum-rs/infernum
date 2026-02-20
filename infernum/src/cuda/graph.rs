//! CUDA graph capture and replay for decode acceleration.
//!
//! Captures the GPU work issued by a model's decode step into a CUDA graph,
//! then replays it with near-zero CPU launch overhead on subsequent steps.
//!
//! The graph is re-captured each step (kernel parameters like KV cache length
//! change) but the existing executable is updated in-place via
//! `cuGraphExecUpdate_v2`, which is much cheaper than full re-instantiation
//! when the graph topology is unchanged.

#![allow(
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions
)]

use std::ptr;
use std::sync::Arc;

use cudarc::driver::sys::{
    self, CUgraph, CUgraphExec, CUgraphExecUpdateResult, CUgraphExecUpdateResultInfo,
    CUstreamCaptureMode,
};
use cudarc::driver::{CudaDevice, CudaSlice};

use crate::error::Error;
use crate::Result;

/// Check a raw CUDA driver result, converting non-success to our error type.
fn check(result: sys::CUresult, op: &str) -> Result<()> {
    if result == sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::CudaGraph(format!("{op} failed: {result:?}")))
    }
}

/// Manages CUDA graph capture, instantiation, update, and replay.
///
/// # Lifecycle
///
/// 1. Call [`begin_capture`] to start recording GPU work on the device stream.
/// 2. Run the model's forward pass (all GPU ops are recorded, not executed).
/// 3. Call [`end_capture`] to finalise the graph.
///    - On the first call this instantiates a new executable.
///    - On subsequent calls it updates the existing executable in-place.
/// 4. Call [`launch`] to replay the captured work.
///
/// Between launches, update any input buffers (e.g. write the next token ID
/// into the pre-allocated device buffer) — the graph references their fixed
/// device addresses.
pub struct CudaGraph {
    device: Arc<CudaDevice>,
    /// Instantiated executable graph (None until first `end_capture`).
    exec: Option<CUgraphExec>,
    /// Pre-allocated device buffer holding the current token ID.
    token_input: CudaSlice<u32>,
}

impl CudaGraph {
    /// Create a new graph manager for the given device.
    ///
    /// Allocates a single-element `u32` buffer on the device for the token ID.
    ///
    /// # Errors
    /// Returns an error if the device allocation fails.
    pub fn new(device: &Arc<CudaDevice>) -> Result<Self> {
        let token_input = device.alloc_zeros::<u32>(1)?;
        Ok(Self {
            device: Arc::clone(device),
            exec: None,
            token_input,
        })
    }

    /// GPU buffer holding the current token ID (immutable).
    ///
    /// Pass this to `forward_next_token_device` — the model reads from
    /// this fixed device address during both eager and captured execution.
    #[must_use]
    pub fn token_input(&self) -> &CudaSlice<u32> {
        &self.token_input
    }

    /// GPU buffer holding the current token ID (mutable).
    ///
    /// Use with `device.htod_copy_into()` to write the next token before
    /// each graph capture or replay.
    pub fn token_input_mut(&mut self) -> &mut CudaSlice<u32> {
        &mut self.token_input
    }

    /// Whether an executable graph has been instantiated.
    #[must_use]
    pub fn is_instantiated(&self) -> bool {
        self.exec.is_some()
    }

    /// Begin capturing GPU work on the device's stream.
    ///
    /// All subsequent GPU operations (kernel launches, cuBLAS calls,
    /// device-to-device copies) are recorded into a graph instead of
    /// being executed immediately.
    ///
    /// # Errors
    /// Returns an error if stream capture fails to start.
    pub fn begin_capture(&self) -> Result<()> {
        let stream = *self.device.cu_stream();
        let lib = unsafe { sys::lib() };
        check(
            unsafe {
                lib.cuStreamBeginCapture_v2(
                    stream,
                    CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            },
            "cuStreamBeginCapture_v2",
        )
    }

    /// End stream capture and instantiate or update the executable graph.
    ///
    /// On the first call, a new `CUgraphExec` is created. On subsequent
    /// calls, `cuGraphExecUpdate_v2` attempts an in-place update (cheap
    /// when only kernel parameters changed). If the topology changed,
    /// falls back to full re-instantiation.
    ///
    /// # Errors
    /// Returns an error if capture, instantiation, or update fails.
    pub fn end_capture(&mut self) -> Result<()> {
        let stream = *self.device.cu_stream();
        let lib = unsafe { sys::lib() };

        // Finalise capture → CUgraph
        let mut graph: CUgraph = ptr::null_mut();
        check(
            unsafe { lib.cuStreamEndCapture(stream, &raw mut graph) },
            "cuStreamEndCapture",
        )?;

        if let Some(exec) = self.exec {
            // Try in-place update
            let mut info = CUgraphExecUpdateResultInfo {
                result: CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS,
                errorFromNode: ptr::null_mut(),
                errorNode: ptr::null_mut(),
            };
            let result = unsafe { lib.cuGraphExecUpdate_v2(exec, graph, &raw mut info) };

            if result == sys::cudaError_enum::CUDA_SUCCESS
                && info.result == CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS
            {
                // Update succeeded — destroy the temporary graph
                unsafe { lib.cuGraphDestroy(graph) };
                return Ok(());
            }

            // Topology changed — destroy old exec and re-instantiate
            unsafe { lib.cuGraphExecDestroy(exec) };
            self.exec = None;
        }

        // First capture, or update failed: full instantiation
        let mut exec: CUgraphExec = ptr::null_mut();
        check(
            unsafe { lib.cuGraphInstantiateWithFlags(&raw mut exec, graph, 0) },
            "cuGraphInstantiateWithFlags",
        )?;
        unsafe { lib.cuGraphDestroy(graph) };
        self.exec = Some(exec);
        Ok(())
    }

    /// Replay the captured graph on the device's stream.
    ///
    /// The graph executes asynchronously. Call `device.synchronize()` if you
    /// need results on the host before proceeding.
    ///
    /// # Errors
    /// Returns an error if no graph has been captured or if launch fails.
    pub fn launch(&self) -> Result<()> {
        let exec = self
            .exec
            .ok_or_else(|| Error::CudaGraph("no graph instantiated".into()))?;
        let stream = *self.device.cu_stream();
        let lib = unsafe { sys::lib() };
        check(unsafe { lib.cuGraphLaunch(exec, stream) }, "cuGraphLaunch")
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if let Some(exec) = self.exec.take() {
            let lib = unsafe { sys::lib() };
            unsafe { lib.cuGraphExecDestroy(exec) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::ops::add_inplace;
    use crate::cuda::{CudaContext, CudaTensor};
    use cudarc::driver::DeviceSlice;

    fn make_device() -> Arc<CudaDevice> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        ctx.device().clone()
    }

    #[test]
    fn test_new_initial_state() {
        let device = make_device();
        let graph = CudaGraph::new(&device).unwrap();

        assert!(!graph.is_instantiated());
        assert_eq!(graph.token_input().len(), 1);
    }

    #[test]
    fn test_token_input_roundtrip() {
        let device = make_device();
        let mut graph = CudaGraph::new(&device).unwrap();

        device
            .htod_copy_into(vec![42_u32], graph.token_input_mut())
            .unwrap();
        let host = device.dtoh_sync_copy(graph.token_input()).unwrap();
        assert_eq!(host, vec![42]);
    }

    #[test]
    fn test_launch_before_capture_errors() {
        let device = make_device();
        let graph = CudaGraph::new(&device).unwrap();

        let err = graph.launch().unwrap_err();
        assert!(
            err.to_string().contains("no graph instantiated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_capture_launch_add_kernel() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let device = ctx.device().clone();

        let a = CudaTensor::<f32>::from_slice(&ctx, &[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut b = CudaTensor::<f32>::from_slice(&ctx, &[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();

        // Eagerly run once so the PTX module for `add` is loaded
        add_inplace(&mut b, &a).unwrap();
        let warmup = b.to_vec().unwrap();
        assert_eq!(warmup, vec![11.0, 22.0, 33.0, 44.0]);

        // Reset b for the graph capture
        let mut b = CudaTensor::<f32>::from_slice(&ctx, &[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();

        let mut graph = CudaGraph::new(&device).unwrap();

        // Capture
        graph.begin_capture().unwrap();
        add_inplace(&mut b, &a).unwrap();
        graph.end_capture().unwrap();

        assert!(graph.is_instantiated());

        // Launch (replay the captured add)
        graph.launch().unwrap();
        device.synchronize().unwrap();

        let result = b.to_vec().unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_recapture_updates_exec() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let device = ctx.device().clone();

        let a = CudaTensor::<f32>::from_slice(&ctx, &[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut b = CudaTensor::<f32>::from_slice(&ctx, &[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();

        // Warmup to load PTX
        add_inplace(&mut b, &a).unwrap();

        let mut graph = CudaGraph::new(&device).unwrap();

        // First capture + launch
        let mut b = CudaTensor::<f32>::from_slice(&ctx, &[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        graph.begin_capture().unwrap();
        add_inplace(&mut b, &a).unwrap();
        graph.end_capture().unwrap();

        graph.launch().unwrap();
        device.synchronize().unwrap();
        assert_eq!(b.to_vec().unwrap(), vec![11.0, 22.0, 33.0, 44.0]);

        // Second capture with different data — re-capture + update
        let a2 = CudaTensor::<f32>::from_slice(&ctx, &[4], &[100.0, 200.0, 300.0, 400.0]).unwrap();
        let mut b2 = CudaTensor::<f32>::from_slice(&ctx, &[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();

        graph.begin_capture().unwrap();
        add_inplace(&mut b2, &a2).unwrap();
        graph.end_capture().unwrap();

        // Should still be instantiated (updated in-place or re-instantiated)
        assert!(graph.is_instantiated());

        graph.launch().unwrap();
        device.synchronize().unwrap();
        assert_eq!(b2.to_vec().unwrap(), vec![101.0, 202.0, 303.0, 404.0]);
    }
}
