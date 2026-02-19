"""Compile a Triton GELU kernel to PTX.

Usage (called by build.rs):
    python3 kernels/triton_gelu.py <output_dir> [--arch SM]

Writes <output_dir>/triton_gelu.ptx containing the compiled kernel.
The Rust side loads it with `include_str!` and launches `gelu_kernel`.

The --arch flag sets the GPU compute capability (default: auto-detect,
or sm_80 if no GPU is visible at build time).
"""

import sys
from pathlib import Path

import triton
import triton.language as tl
from triton.compiler import ASTSource, compile
from triton.compiler.compiler import GPUTarget


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    out = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, out, mask=mask)


def get_target(arch_override=None):
    """Get the GPU target for compilation.

    If arch_override is given (e.g. 80 for sm_80), use that.
    Otherwise try auto-detection, falling back to sm_80.
    """
    if arch_override is not None:
        return GPUTarget(backend="cuda", arch=arch_override, warp_size=32)

    try:
        from triton.runtime import driver
        return driver.active.get_current_target()
    except RuntimeError:
        # No GPU visible at build time — use sm_80 (Ampere) as a safe baseline
        return GPUTarget(backend="cuda", arch=80, warp_size=32)


def main():
    args = sys.argv[1:]
    arch_override = None

    # Parse optional --arch flag
    if "--arch" in args:
        idx = args.index("--arch")
        arch_override = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    if len(args) != 1:
        print(f"Usage: {sys.argv[0]} <output_dir> [--arch SM]", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    block_size = 1024
    target = get_target(arch_override)

    src = ASTSource(
        fn=gelu_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        constexprs={(3,): block_size},
    )
    compiled = compile(src, target=target)
    ptx = compiled.asm["ptx"]

    ptx_path = output_dir / "triton_gelu.ptx"
    ptx_path.write_text(ptx)
    print(f"Wrote {ptx_path} ({len(ptx)} bytes, target {target})")


if __name__ == "__main__":
    main()
