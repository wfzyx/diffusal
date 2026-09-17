"""
Roofline Ridge-Point Analysis: does ternary leave ALU headroom for Uno's verification?
=====================================================================================
profile_hardware.py reports a precision-INVARIANT 1.91x DRAM cut, because the ratio
    (AR bytes) / (block-diffusion bytes)
cancels bytes_per_param. It therefore cannot answer whether the block-parallel win
survives ternary. The binding question is the roofline ridge point:

    ridge = peak_compute (FLOP/s) / peak_bandwidth (byte/s)     [FLOP/byte]

    arithmetic_intensity <  ridge  ->  MEMORY bound  -> parallel drafting pays
    arithmetic_intensity >= ridge  ->  COMPUTE bound -> parallel drafting is dead weight

Intensities are taken directly from profile_hardware.py's own model
(19.6M total / 7.0M active params, top-2 of 8 experts, B=32, 6 steps/block).
"""

# (name, AR intensity FLOP/B, block-diffusion intensity FLOP/B)  <- from profile_hardware.py
REGIMES = [
    ("FP32    (4.00 B/param)",  0.50,   5.73),
    ("INT4    (0.50 B/param)",  4.00,  45.88),
    ("Ternary (0.25 B/param)",  8.00,  91.75),
]

# (target, bandwidth GB/s, peak compute GFLOP/s, kernel note)
TARGETS = [
    ("RTX 2080 Super  fp32 CUDA core",     496.0,  11150.0, "no low-bit path, dequant to fp32"),
    ("RTX 2080 Super  fp16 tensor core",   496.0,  22300.0, "dequant ternary -> fp16 GEMM"),
    ("RTX 2080 Super  int8 tensor core",   496.0,  89200.0, "dp4a / packed int path"),
    ("Laptop CPU      AVX2 DDR4",           60.0,    200.0, "bitnet.cpp LUT kernel"),
    ("Apple M2 Max    unified memory",     400.0,  13600.0, "Metal fp16"),
    ("H100 SXM        fp16 tensor core",  3350.0, 989000.0, "server, large batch"),
]

BAR = "=" * 96


def verdict(intensity: float, ridge: float) -> str:
    if intensity < ridge:
        return f"MEMORY bound  ({ridge / intensity:5.1f}x headroom)"
    return f"COMPUTE bound ({intensity / ridge:5.1f}x over ridge)"


def main() -> None:
    print(BAR)
    print("  ROOFLINE RIDGE-POINT ANALYSIS  --  ternary vs. block-parallel drafting")
    print(BAR)

    for target, bw_gbs, peak_gflops, note in TARGETS:
        ridge = peak_gflops / bw_gbs
        print(f"\n{target}")
        print(f"  {peak_gflops/1000:8.2f} TFLOP/s  /  {bw_gbs:7.1f} GB/s   ->  ridge = {ridge:7.2f} FLOP/byte")
        print(f"  kernel: {note}")
        print(f"  {'-' * 92}")
        print(f"  {'precision':<24} | {'paradigm':<16} | {'intensity':>10} | verdict")
        print(f"  {'-' * 92}")
        for name, ar_ai, bd_ai in REGIMES:
            print(f"  {name:<24} | {'autoregressive':<16} | {ar_ai:7.2f} F/B | {verdict(ar_ai, ridge)}")
            print(f"  {'':<24} | {'block-parallel':<16} | {bd_ai:7.2f} F/B | {verdict(bd_ai, ridge)}")
        print(f"  {'-' * 92}")

    print(f"\n{BAR}")
    print("  CROSSOVER SUMMARY: where block-parallel drafting stops paying")
    print(BAR)
    print(f"  {'target':<36} | {'ridge':>9} | {'fp32':>8} | {'int4':>8} | {'ternary':>8}")
    print(f"  {'-' * 92}")
    for target, bw_gbs, peak_gflops, _ in TARGETS:
        ridge = peak_gflops / bw_gbs
        cells = []
        for _, _, bd_ai in REGIMES:
            cells.append("PAYS" if bd_ai < ridge else "DEAD")
        print(f"  {target:<36} | {ridge:7.2f}  | {cells[0]:>8} | {cells[1]:>8} | {cells[2]:>8}")
    print(f"  {'-' * 92}")
    print("  PAYS = block-parallel drafting still memory bound -> amortizes weight streaming")
    print("  DEAD = ternary already freed the bus; 6.0x arithmetic overhead now binds")
    print(BAR)


if __name__ == "__main__":
    main()
