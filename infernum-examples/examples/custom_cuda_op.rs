//! Example: Adding a custom CUDA op to Infernum
//!
//! This shows the full workflow for adding a new GPU operation:
//!
//! 1. Write the CUDA kernel in `kernels/relu.cu`
//! 2. The `build.rs` compiles it to PTX at build time via `nvcc`
//! 3. Load the PTX with `include_str!` and launch via cudarc
//!
//! Run with:
//!   cargo run --example custom_cuda_op --features cuda

#![allow(
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use infernum::cuda::CudaContext;
use infernum::tensor::Tensor;
use infernum::{CudaTensor, DType, Result};

// ---------- Step 1: Load the pre-compiled PTX ----------
//
// build.rs compiles kernels/relu.cu → $OUT_DIR/kernels/relu.ptx
// This embeds the PTX string into the binary at compile time.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/relu.ptx"));

// ---------- Step 2: Implement the op ----------
//
// Follow the same pattern as infernum's built-in ops:
// - Load PTX into the device (once, checked by has_func)
// - Get the function handle
// - Configure grid/block dimensions
// - Launch the kernel

/// Apply ReLU activation element-wise: out[i] = max(0, x[i])
fn relu(x: &CudaTensor) -> Result<CudaTensor> {
    let n = x.numel();
    let mut output = unsafe { CudaTensor::uninit(x.context(), x.shape(), DType::F32)? };

    let device = x.context().device();

    // Load PTX into the CUDA module (only on first call)
    let module_name = "relu";
    if !device.has_func(module_name, "relu_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            module_name,
            &["relu_f32", "relu_inplace_f32"],
        )?;
    }

    let func = device.get_func(module_name, "relu_f32").unwrap();

    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(cfg, (output.cuda_slice_mut(), &x.cuda_slice(), n as i32))?;
    }

    Ok(output)
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    println!("CUDA context initialized");

    // Test data with positive, negative, and zero values
    let data: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 3.0, -0.5, 2.5, -2.0];
    let x = CudaTensor::from_slice(&ctx, &[data.len()], &data)?;

    println!("Input:  {:?}", data);

    let y = relu(&x)?;
    let result = y.to_vec::<f32>()?;

    println!("ReLU:   {result:?}");
    assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 2.5, 0.0]);
    println!("✓ ReLU output matches expected values");

    Ok(())
}
