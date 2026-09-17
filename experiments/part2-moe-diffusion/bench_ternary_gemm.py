"""
Ternary GEMM Kernel Benchmark: is there ALU headroom left after ternary?
=======================================================================
RESULTS-roofline.md finding 1: on an RTX 2080 Super the verdict on block-parallel
drafting flips entirely on which compute path the ternary GEMM lands in.

    dequantize 2-bit -> fp16 GEMM     ridge  44.96 FLOP/B  -> ternary block-parallel DEAD
    packed int8 tensor-core GEMM      ridge 179.84 FLOP/B  -> ternary block-parallel ALIVE

Those ridges assume vendor peak FLOP/s. This script measures the *achieved* rate
of each path on real shapes, which is the number that actually decides.

Read the output:
  packed int8 >= 2.0x fp16 baseline  ->  arithmetic headroom exists; ternary + Uno compose.
  packed int8 <= fp16 baseline       ->  no headroom; ship ternary alone, drop the Uno layer
                                         on this device and keep it for a server target.

Requires CUDA. Run on the target device, not in WSL without a CUDA driver.
"""

from __future__ import annotations

import argparse
import time

import torch

# (name, M, K, N) -- MoE expert FFN shapes, batch-1 decode and 32-token block.
DEFAULT_SHAPES = [
    ("decode  M=1    (batch-1 AR step)", 1, 4096, 11008),
    ("block   M=32   (Uno draft block)", 32, 4096, 11008),
    ("block   M=512  (tree verify)", 512, 4096, 11008),
]


def synchronize() -> None:
    torch.cuda.synchronize()


def timed(fn, warmup: int, iters: int) -> float:
    """Return median seconds per call."""
    for _ in range(warmup):
        fn()
    synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def pack_ternary(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BitNet b1.58 style: ternarize, then pack 4 values per int8 byte (2 bits each)."""
    gamma = weight.abs().mean().clamp(min=1e-5)
    ternary = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
    codes = (ternary + 1).to(torch.uint8)  # {-1,0,1} -> {0,1,2}
    flat = codes.reshape(-1)
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)])
    groups = flat.reshape(-1, 4)
    packed = (
        groups[:, 0]
        | (groups[:, 1] << 2)
        | (groups[:, 2] << 4)
        | (groups[:, 3] << 6)
    )
    return packed, gamma


def unpack_ternary(packed: torch.Tensor, gamma: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Unpack 2-bit codes back to an fp16 dense matrix."""
    b0 = packed & 0x03
    b1 = (packed >> 2) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 6) & 0x03
    flat = torch.stack([b0, b1, b2, b3], dim=1).reshape(-1)
    flat = flat[: shape.numel()]
    return (flat.to(torch.float16) - 1.0).reshape(shape) * gamma


def bench_shape(name: str, m: int, k: int, n: int, warmup: int, iters: int) -> None:
    device = torch.device("cuda")
    flops = 2.0 * m * k * n

    x16 = torch.randn(m, k, device=device, dtype=torch.float16)
    w16 = torch.randn(n, k, device=device, dtype=torch.float16)

    print(f"\n{name}   [M={m}, K={k}, N={n}]   {flops / 1e9:.2f} GFLOP/call")
    print(f"  {'-' * 78}")
    print(f"  {'kernel':<34} | {'ms':>9} | {'TFLOP/s':>9} | {'vs fp16':>8}")
    print(f"  {'-' * 78}")

    # --- 1. fp16 baseline: the reference ceiling -----------------------------
    baseline_s = timed(lambda: torch.nn.functional.linear(x16, w16), warmup, iters)
    baseline_tflops = flops / baseline_s / 1e12
    print(f"  {'fp16 baseline':<34} | {baseline_s * 1e3:9.3f} | {baseline_tflops:9.2f} | {1.0:7.2f}x")

    # --- 2. pessimistic path: unpack 2-bit -> fp16 GEMM ----------------------
    packed, gamma = pack_ternary(w16.float())
    shape = w16.shape

    def unpack_then_matmul() -> None:
        w = unpack_ternary(packed, gamma, shape)
        torch.nn.functional.linear(x16, w)

    unpack_s = timed(unpack_then_matmul, warmup, iters)
    unpack_tflops = flops / unpack_s / 1e12
    print(
        f"  {'unpack 2-bit -> fp16 matmul':<34} | {unpack_s * 1e3:9.3f} | "
        f"{unpack_tflops:9.2f} | {unpack_tflops / baseline_tflops:7.2f}x"
    )

    # --- 3. optimistic path: packed int8 tensor-core GEMM --------------------
    try:
        x8 = torch.randint(-127, 127, (m, k), device=device, dtype=torch.int8)
        w8 = torch.randint(-1, 2, (n, k), device=device, dtype=torch.int8)
        wt8 = w8.t().contiguous().t()

        def int8_matmul() -> None:
            torch._int_mm(x8, wt8.t())

        if m < 16:
            raise RuntimeError("torch._int_mm requires M >= 16 on most backends")
        int8_s = timed(int8_matmul, warmup, iters)
        int8_tflops = flops / int8_s / 1e12
        print(
            f"  {'packed int8 tensor-core matmul':<34} | {int8_s * 1e3:9.3f} | "
            f"{int8_tflops:9.2f} | {int8_tflops / baseline_tflops:7.2f}x"
        )
    except Exception as exc:  # noqa: BLE001 - report and continue across shapes
        print(f"  {'packed int8 tensor-core matmul':<34} | {'n/a':>9} | {'n/a':>9} | {str(exc)[:30]}")

    print(f"  {'-' * 78}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Run this on the target GPU; a CPU result answers nothing "
            "because the ridge point is a property of the device."
        )

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print("=" * 82)
    print("  TERNARY GEMM KERNEL BENCHMARK")
    print("=" * 82)
    print(f"  device : {name}")
    print(f"  sm     : {props.major}.{props.minor}    memory: {props.total_memory / 1024**3:.1f} GiB")
    print(f"  torch  : {torch.__version__}")

    for shape in DEFAULT_SHAPES:
        bench_shape(*shape, warmup=args.warmup, iters=args.iters)

    print("\n" + "=" * 82)
    print("  DECISION RULE")
    print("=" * 82)
    print("  packed int8 >= 2.0x fp16  ->  ALU headroom exists. Ternary + Uno compose.")
    print("  packed int8 <= 1.0x fp16  ->  No headroom. Ternary alone; move Uno to a server target.")
    print("  between                   ->  Marginal. Decide with a wall-clock end-to-end run.")
    print("=" * 82)


if __name__ == "__main__":
    main()
