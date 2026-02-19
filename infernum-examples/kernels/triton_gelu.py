"""Compile a Triton GELU kernel to PTX.

Usage (called by build.rs):
    python3 kernels/triton_gelu.py <output_dir>

Writes <output_dir>/triton_gelu.ptx containing the compiled kernel.
The Rust side loads it with `include_str!` and launches `gelu_kernel`.
"""

import sys
from pathlib import Path

import triton
import triton.language as tl
from triton.compiler import ASTSource, compile


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    out = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, out, mask=mask)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    block_size = 1024

    src = ASTSource(
        fn=gelu_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        constexprs={(3,): block_size},
    )
    compiled = compile(src)
    ptx = compiled.asm["ptx"]

    ptx_path = output_dir / "triton_gelu.ptx"
    ptx_path.write_text(ptx)
    print(f"Wrote {ptx_path} ({len(ptx)} bytes)")


if __name__ == "__main__":
    main()
