"""
Domestic-PC Roofline: CPU + GPU + RAM + VRAM + NVMe hierarchy
=============================================================
RESULTS-roofline.md assumed weights are VRAM-resident and concluded that ternary
block-parallel drafting is DEAD on consumer GPUs. That conclusion only holds while
the model fits in VRAM.

On a domestic PC the binding bus depends on where the weights actually live:

    tier 1  VRAM resident      288 - 1008 GB/s
    tier 2  RAM over PCIe       12 -   52 GB/s      (20-40x slower)
    tier 3  NVMe direct        3.5 -   14 GB/s      (40-150x slower)

Streaming weights across PCIe collapses the ridge point by the same factor, which
pushes every workload deep into the memory-bound regime and makes block-parallel
drafting pay again -- this time by a wide margin.

The decision therefore reduces to one question:

    does the model fit in VRAM at the chosen bit width?

    fits     -> tier 1 -> high ridge -> compute bound   -> block-parallel drafting marginal
    spills   -> tier 2 -> low  ridge -> memory bound    -> block-parallel drafting wins big

Ternary is precisely what moves a model from tier 2 to tier 1. That is the tension:
the compression that makes the model fit is the same compression that removes the
bottleneck block-parallel drafting was built to hide.
"""

from __future__ import annotations

# Arithmetic intensity of each paradigm, from profile_hardware.py.
# Block-parallel: 1.91x fewer bytes for 6.0x more arithmetic => ~11.5x intensity.
AR_INTENSITY = {4.00: 0.50, 0.50: 4.00, 0.25: 8.00}
BLOCK_INTENSITY = {4.00: 5.73, 0.50: 45.88, 0.25: 91.75}

PRECISIONS = [
    ("fp16", 2.00),
    ("int4", 0.50),
    ("ternary 2-bit", 0.25),
]

# (name, VRAM GiB, VRAM GB/s, fp16 TFLOP/s, int8 TOPS)
GPUS = [
    ("RTX 2080 Super  8GB",   8.0,  496.0,  22.3,  89.2),
    ("RTX 3060       12GB",  12.0,  360.0,  25.6, 102.4),
    ("RTX 4070       12GB",  12.0,  504.0,  29.1, 116.4),
    ("RTX 4060 Ti    16GB",  16.0,  288.0,  22.1,  88.4),
    ("RTX 3090       24GB",  24.0,  936.0,  35.6, 142.4),
    ("RTX 4090       24GB",  24.0, 1008.0, 165.2, 660.6),
]

# (name, effective GB/s) -- real-world, roughly 85% of theoretical
BUSES = [
    ("PCIe 3.0 x16 -> RAM", 12.0),
    ("PCIe 4.0 x16 -> RAM", 26.0),
    ("PCIe 5.0 x16 -> RAM", 52.0),
    ("NVMe Gen3 x4 direct",  3.5),
    ("NVMe Gen4 x4 direct",  7.0),
    ("NVMe Gen5 x4 direct", 13.0),
]

# Candidate model sizes in billions of total parameters.
MODEL_SIZES = [0.9, 8.0, 25.0, 50.0, 100.0]

KV_RESERVE_GIB = 2.0  # KV cache + activations + CUDA context headroom
BAR = "=" * 100


def footprint_gib(params_b: float, bytes_per_param: float) -> float:
    return params_b * 1e9 * bytes_per_param / 1024**3


def verdict(intensity: float, ridge: float) -> tuple[str, float]:
    if intensity < ridge:
        return "PAYS", ridge / intensity
    return "DEAD", intensity / ridge


def section_fit() -> None:
    print(BAR)
    print("  1. FOOTPRINT: does it fit in VRAM?   (reserving %.1f GiB for KV cache + context)" % KV_RESERVE_GIB)
    print(BAR)
    header = f"  {'model':>8} | " + " | ".join(f"{label:>13}" for label, _ in PRECISIONS)
    print(header)
    print(f"  {'-' * 96}")
    for params_b in MODEL_SIZES:
        cells = []
        for _, bpp in PRECISIONS:
            cells.append(f"{footprint_gib(params_b, bpp):10.1f} GiB")
        print(f"  {params_b:6.1f}B  | " + " | ".join(cells))
    print(f"  {'-' * 96}")

    print("\n  Largest model that stays VRAM-resident per GPU:")
    print(f"  {'-' * 96}")
    print(f"  {'gpu':<22} | {'usable':>8} | " + " | ".join(f"{label:>13}" for label, _ in PRECISIONS))
    print(f"  {'-' * 96}")
    for name, vram, _, _, _ in GPUS:
        usable = vram - KV_RESERVE_GIB
        cells = []
        for _, bpp in PRECISIONS:
            max_b = usable * 1024**3 / bpp / 1e9
            cells.append(f"{max_b:10.1f}B  ")
        print(f"  {name:<22} | {usable:6.1f} GiB | " + " | ".join(cells))
    print(f"  {'-' * 96}")


def section_ridges() -> None:
    print(f"\n{BAR}")
    print("  2. RIDGE POINTS: VRAM-resident vs. streamed over the bus")
    print(BAR)
    print("  Ridge = peak compute / bandwidth. Computed against an RTX 4070 (29.1 TFLOP/s fp16).")
    print(f"  {'-' * 96}")
    print(f"  {'tier':<26} | {'bandwidth':>12} | {'ridge (FLOP/B)':>15} | {'note':<32}")
    print(f"  {'-' * 96}")
    ref_tflops = 29.1
    print(f"  {'VRAM resident (4070)':<26} | {504.0:9.1f} GB/s | {ref_tflops * 1e3 / 504.0:13.1f}   | {'weights live on the card':<32}")
    for label, bw in BUSES:
        print(f"  {label:<26} | {bw:9.1f} GB/s | {ref_tflops * 1e3 / bw:13.1f}   | {'weights stream every pass':<32}")
    print(f"  {'-' * 96}")


def section_verdicts() -> None:
    print(f"\n{BAR}")
    print("  3. VERDICT: is block-parallel drafting worth its 6.0x arithmetic overhead?")
    print(BAR)
    ref_tflops = 29.1 * 1e3  # GFLOP/s, RTX 4070 fp16 tensor

    tiers = [("VRAM resident (4070)", 504.0)] + BUSES
    print(f"  {'tier':<26} | {'ridge':>8} | " + " | ".join(f"{label:>16}" for label, _ in PRECISIONS))
    print(f"  {'-' * 96}")
    for label, bw in tiers:
        ridge = ref_tflops / bw
        cells = []
        for _, bpp in PRECISIONS:
            key = 2.00 if bpp == 2.00 else bpp
            intensity = BLOCK_INTENSITY.get(key, BLOCK_INTENSITY[4.00] if key == 2.00 else 0.0)
            if key == 2.00:
                intensity = BLOCK_INTENSITY[4.00] * 2  # fp16 halves bytes vs fp32 -> doubles intensity
            tag, factor = verdict(intensity, ridge)
            cells.append(f"{tag} ({factor:5.1f}x)")
        print(f"  {label:<26} | {ridge:7.1f}  | " + " | ".join(f"{c:>16}" for c in cells))
    print(f"  {'-' * 96}")
    print("  PAYS (Nx) = N times under the ridge, still memory bound, drafting amortizes streaming")
    print("  DEAD (Nx) = N times over the ridge, arithmetic binds, drafting only adds work")


def section_rule() -> None:
    print(f"\n{BAR}")
    print("  4. DECISION RULE FOR DOMESTIC PCs")
    print(BAR)
    print("""
  Pick the target model size first, then read off the tier:

    A. Ternary model FITS in VRAM
       -> tier 1, ridge ~58 FLOP/B on a 4070
       -> ternary block-parallel at 91.75 FLOP/B is COMPUTE bound
       -> ship ternary alone; Uno drafting needs a packed-int8 kernel to be worth it

    B. Ternary model SPILLS to RAM over PCIe
       -> tier 2, ridge ~1119 FLOP/B on PCIe 4.0
       -> everything is deeply MEMORY bound
       -> ternary AND block-parallel drafting both pay, and they compound

    C. Model streams from NVMe
       -> tier 3, ridge ~4157 FLOP/B on Gen4
       -> the bus is catastrophically the bottleneck
       -> maximise tokens per weight-load; block-parallel drafting is the whole game

  The trap: ternary is what moves a model from B to A. Compressing hard enough to
  fit in VRAM is exactly what removes the bottleneck Uno exists to hide. The two
  techniques are complements only while the model is too big to fit.

  Corollary: the strongest configuration is the LARGEST ternary MoE that still
  spills, not the largest that fits. Spilling keeps you memory bound, which is the
  regime where sparse routing, ternary packing, and parallel drafting all pay at once.
""")


def main() -> None:
    print(BAR)
    print("  DOMESTIC-PC ROOFLINE: CPU + GPU + RAM + VRAM + NVMe")
    print(BAR)
    section_fit()
    section_ridges()
    section_verdicts()
    section_rule()
    print(BAR)


if __name__ == "__main__":
    main()
