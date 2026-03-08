//! Metal device context.
//!
//! Holds the Metal device, command queue, and preloaded compute pipeline
//! states. Constructed once at startup and shared (via `Clone`) across
//! model layers and ops.

use std::collections::HashMap;
use std::sync::Arc;

use metal::Device;

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

impl MetalContext {
    /// Create a new Metal context using the system default GPU.
    ///
    /// # Panics
    /// Panics if no Metal device is available.
    #[must_use]
    pub fn new() -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let queue = device.new_command_queue();
        Self {
            inner: Arc::new(MetalContextInner {
                device,
                queue,
                pipelines: HashMap::new(),
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

        let library = device
            .new_library_with_data(metallib_data)
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
                    infernum::Error::Other(format!(
                        "Failed to create pipeline for '{name_str}': {e}"
                    ))
                })?;
            pipelines.insert(name_str, pipeline);
        }

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
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new()
    }
}
