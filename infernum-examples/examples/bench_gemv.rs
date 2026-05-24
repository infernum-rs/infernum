//! GEMV micro-benchmark: encode N identical dispatches in one command buffer.
//!
//! By packing thousands of identical Q8_0 GEMV dispatches into a single
//! command buffer and measuring total flush time, we get pure GPU kernel
//! execution time with amortized launch overhead.
//!
//! Also runs a single-dispatch-per-flush test to isolate command-buffer overhead.
//!
//! Usage:
//!   cargo run --release --example bench_gemv --features metal
//!   cargo run --release --example bench_gemv --features metal -- --rows 2560 --cols 960

#![cfg(feature = "metal")]

use std::time::Instant;

use clap::Parser;
use infernum_metal::MetalContext;
use metal::MTLSize;

// Must match infernum-metal/kernels/quantized_matmul.metal
const Q8B_NR: u64 = 2;
const Q8B_NSG: u64 = 4;
const Q8B_ROWS_PER_TG: u64 = Q8B_NR * Q8B_NSG; // 8
const Q8B_THREADS_PER_TG: u64 = Q8B_NSG * 32; // 128

const Q8_BLOCK_SIZE: usize = 32;
const Q8_BLOCK_BYTES: usize = 34; // 2-byte f16 scale + 32 int8 weights

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    k: u32,
}

#[derive(Parser)]
#[command(name = "bench_gemv")]
struct Cli {
    /// Output rows (N)
    #[arg(long, default_value_t = 2560)]
    rows: usize,

    /// Input columns (K, must be divisible by 32)
    #[arg(long, default_value_t = 960)]
    cols: usize,

    /// Dispatches per timed command buffer
    #[arg(long, default_value_t = 5000)]
    iters: usize,

    /// Warmup dispatches (not timed)
    #[arg(long, default_value_t = 500)]
    warmup: usize,

    /// Single-dispatch overhead measurement iterations
    #[arg(long, default_value_t = 200)]
    overhead_iters: usize,
}

fn flush_time(ctx: &MetalContext) -> std::time::Duration {
    let t0 = Instant::now();
    ctx.flush();
    t0.elapsed()
}

fn main() {
    let cli = Cli::parse();
    let n = cli.rows;
    let k = cli.cols;
    assert!(k % Q8_BLOCK_SIZE == 0, "cols must be divisible by 32");

    let ctx = MetalContext::new();
    let device = ctx.device();

    let nb = k / Q8_BLOCK_SIZE;
    let weight_bytes = n * nb * Q8_BLOCK_BYTES;

    // Build Q8_0 weight buffer: scale=f16(1.0), weights=int8(1)
    let mut weight_data = vec![0u8; weight_bytes];
    for row in 0..n {
        for b in 0..nb {
            let off = (row * nb + b) * Q8_BLOCK_BYTES;
            // f16(1.0) = 0x3C00 little-endian
            weight_data[off] = 0x00;
            weight_data[off + 1] = 0x3C;
            for j in 0..Q8_BLOCK_SIZE {
                weight_data[off + 2 + j] = 1; // int8(1)
            }
        }
    }

    let weight_buf = device.new_buffer_with_data(
        weight_data.as_ptr().cast(),
        weight_bytes as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_data = vec![1.0f32; k];
    let input_buf = device.new_buffer_with_data(
        input_data.as_ptr().cast(),
        (k * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buf =
        device.new_buffer((n * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

    let params = Params {
        n: n as u32,
        k: k as u32,
    };
    let params_bytes = bytemuck::bytes_of(&params);
    let tg = MTLSize::new(u64::try_from(n).unwrap().div_ceil(Q8B_ROWS_PER_TG), 1, 1);
    let tpg = MTLSize::new(Q8B_THREADS_PER_TG, 1, 1);

    let dispatch = |iters: usize| {
        for _ in 0..iters {
            ctx.dispatch_threadgroups(
                "gemv_q8_blocks_f32",
                &[
                    (input_buf.as_ref(), 0),
                    (weight_buf.as_ref(), 0),
                    (output_buf.as_ref(), 0),
                ],
                params_bytes,
                tg,
                tpg,
                0,
            );
        }
    };

    let bytes_per_dispatch = (weight_bytes + k * 4) as f64;
    println!("Q8_0 GEMV micro-benchmark  N={n}  K={k}");
    println!("  Weight buffer:      {:.2} MB", weight_bytes as f64 / 1e6);
    println!("  Bytes/dispatch:     {:.2} MB", bytes_per_dispatch / 1e6);
    println!(
        "  Threadgroups:       {}",
        n.div_ceil(Q8B_ROWS_PER_TG as usize)
    );
    println!("  Threads/group:      {Q8B_THREADS_PER_TG}");
    println!();

    print!("Warmup ({} dispatches)... ", cli.warmup);
    dispatch(cli.warmup);
    ctx.flush();
    println!("done");

    // ── Batched timing (amortizes overhead) ────────────────────────────────
    print!("Batched ({} dispatches, 1 flush)... ", cli.iters);
    dispatch(cli.iters);
    let batched = flush_time(&ctx);
    println!("done");

    let secs = batched.as_secs_f64();
    let per_us = secs / cli.iters as f64 * 1e6;
    let bw = bytes_per_dispatch * cli.iters as f64 / secs / 1e9;
    let min_us = bytes_per_dispatch / 150e9 * 1e6;

    println!();
    println!("Batched (GPU kernel time):");
    println!("  Per dispatch:        {per_us:.1} μs  (min @ 150 GB/s: {min_us:.1} μs)");
    println!(
        "  Effective bandwidth: {bw:.1} GB/s  ({:.1}%)",
        bw / 150.0 * 100.0
    );

    // ── Single-dispatch overhead ────────────────────────────────────────────
    println!();
    println!(
        "Single-dispatch overhead ({} flushes)...",
        cli.overhead_iters
    );
    let mut total_single = std::time::Duration::ZERO;
    for _ in 0..cli.overhead_iters {
        ctx.dispatch_threadgroups(
            "gemv_q8_blocks_f32",
            &[
                (input_buf.as_ref(), 0),
                (weight_buf.as_ref(), 0),
                (output_buf.as_ref(), 0),
            ],
            params_bytes,
            tg,
            tpg,
            0,
        );
        total_single += flush_time(&ctx);
    }
    let single_us = total_single.as_secs_f64() / cli.overhead_iters as f64 * 1e6;
    println!("  Per dispatch+flush:  {single_us:.1} μs");
    println!("  Overhead per flush:  {:.1} μs", single_us - per_us);
    println!("  GPU kernel only:     {per_us:.1} μs");
}
