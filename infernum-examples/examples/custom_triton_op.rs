//! Example: Adding a custom Triton op to Infernum
//!
//! This shows how to integrate a Triton kernel (written in Python) with
//! Infernum's Rust runtime:
//!
//! 1. Write the Triton kernel in `kernels/triton_gelu.py`
//! 2. The `build.rs` compiles it to PTX at build time via `python3`
//! 3. Load the PTX with `include_str!` and launch via cudarc — same as CUDA
//!
//! The Triton kernel uses a compile-time constant `BLOCK_SIZE = 1024`,
//! so the grid launch must match: `grid = ceil(n / 1024)`.
//!
//! Run with:
//!   cargo run --example custom_triton_op --features cuda
//!
//! Requires: `pip install triton`

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
// build.rs runs: python3 kernels/triton_gelu.py $OUT_DIR/kernels/
// which writes triton_gelu.ptx via Triton's ahead-of-time compiler.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/triton_gelu.ptx"));

// The BLOCK_SIZE constexpr is baked into the Triton kernel at compile time.
// The grid launch divides work into chunks of this size.
const BLOCK_SIZE: usize = 1024;

// Triton's compiler sets `.reqntid 128` (4 warps × 32 threads).
// The block_dim must match this, not BLOCK_SIZE.
const TRITON_NUM_THREADS: u32 = 128;

// ---------- Step 2: Implement the op ----------
//
// From Rust's perspective, a Triton-compiled PTX is almost identical to an
// nvcc-compiled PTX. Two Triton-specific details:
//
// - block_dim is the number of threads (from `.reqntid`), NOT the BLOCK_SIZE.
//   Triton maps BLOCK_SIZE elements across threads internally.
//
// - Triton injects two extra u64 pointer parameters after the user-defined
//   ones (launch metadata and global scratch). Pass null (0u64) for both.

/// Apply GELU activation element-wise: out[i] = 0.5 * x * (1 + erf(x / √2))
fn gelu(x: &CudaTensor) -> Result<CudaTensor> {
    let n = x.numel();
    let mut output = unsafe { CudaTensor::uninit(x.context(), x.shape(), DType::F32)? };

    let device = x.context().device();

    let module_name = "triton_gelu";
    if !device.has_func(module_name, "gelu_kernel") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            module_name,
            &["gelu_kernel"],
        )?;
    }

    let func = device.get_func(module_name, "gelu_kernel").unwrap();

    let grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (TRITON_NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    // Triton kernels expect raw device pointers (u64), not CudaSlice wrappers.
    // They also take two extra trailing u64 parameters (null pointers for
    // launch metadata and global scratch buffer).
    let out_ptr = output.as_mut_ptr() as u64;
    let x_ptr = x.as_ptr() as u64;
    let null_ptr: u64 = 0;

    unsafe {
        func.launch(cfg, (x_ptr, out_ptr, n as i32, null_ptr, null_ptr))?;
    }

    Ok(output)
}

/// Reference GELU on the CPU for validation.
///
/// Uses the exact formula: 0.5 * x * (1 + erf(x / sqrt(2))),
/// with an Abramowitz & Stegun polynomial approximation for erf.
fn gelu_reference(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * (1.0 + erf_approx(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

fn erf_approx(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let poly = 0.254_829_592 * t - 0.284_496_736 * t2 + 1.421_413_741 * t3 - 1.453_152_027 * t4
        + 1.061_405_429 * t5;
    sign * (1.0 - poly * (-x * x).exp())
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    println!("CUDA context initialized");

    let data: Vec<f32> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let x = CudaTensor::from_slice(&ctx, &[data.len()], &data)?;

    println!("Input:    {data:?}");

    let y = gelu(&x)?;
    let result = y.to_vec()?;
    let expected: Vec<f32> = data.iter().map(|&v| gelu_reference(v)).collect();

    println!("GELU:     {result:?}");
    println!("Expected: {expected:?}");

    for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "Mismatch at index {i}: got {got}, expected {want}"
        );
    }
    println!("✓ Triton GELU output matches CPU reference");

    Ok(())
}
