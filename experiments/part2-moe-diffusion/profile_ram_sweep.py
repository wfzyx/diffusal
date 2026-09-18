"""
RAM Sweep: does Qwen3.8-Flash-Next become usable with more system memory?
========================================================================
profile_qwen_flash_next.py fixed RAM at 32 GiB and found the box permanently
disk-bound. This sweeps the RAM axis to find where, if anywhere, the model stops
thrashing NVMe.

Model of throughput
-------------------
At batch 1 the expert path dominates. Per token the model touches
    top_k experts/layer x layers x params/expert x bytes/param
Those bytes come from two places:
    - RAM page cache      ~50 GB/s (DDR4) or ~80 GB/s (DDR5), CPU executes in place
    - NVMe on a miss      ~7 GB/s (Gen4)

With a fraction `r` of the model resident, expected time per token is

    t = bytes * (  r / BW_ram  +  (1 - r) / BW_nvme )

which is a harmonic blend, not an average. The NVMe term dominates until r is
very close to 1, which is why partial residency helps far less than it looks.

Units: GGUF sizes are decimal GB (bytes / 1e9). Memory capacities are binary GiB.
Both are converted to bytes before any comparison.
"""

from __future__ import annotations

GIB = 1024**3
GB = 1e9

# ---------------------------------------------------------------- model config
NUM_LAYERS = 48
HIDDEN = 2560
NUM_EXPERTS = 512
TOP_K = 10
MOE_INTERMEDIATE = 640
EXPERT_MATRICES = 3

PARAMS_PER_EXPERT = EXPERT_MATRICES * HIDDEN * MOE_INTERMEDIATE
TOTAL_PARAMS = 354.03 * GB / 2.0  # bf16 GGUF total / 2 bytes per param

# (quant, total decimal GB)
QUANTS = [
    ("UD-IQ1_S", 72.55),
    ("UD-IQ1_M", 74.54),
    ("UD-Q2_K_XL", 78.87),
    ("UD-IQ3_XXS", 81.96),
    ("UD-Q3_K_XL", 89.99),
    ("UD-IQ4_XS", 93.68),
    ("UD-Q4_K_XL", 111.33),
    ("UD-Q5_K_XL", 158.29),
]

# --------------------------------------------------------------------- hardware
VRAM_GIB = 8.0
VRAM_RESERVE_GIB = 1.5
OS_RESERVE_GIB = 4.0

BW_RAM_DDR4 = 50.0 * GB    # dual channel DDR4-3200
BW_RAM_DDR5 = 80.0 * GB    # dual channel DDR5-5600
BW_NVME = 7.0 * GB         # Gen4 x4
BW_NVME_GEN5 = 13.0 * GB

RAM_CONFIGS_GIB = [32.0, 64.0, 96.0, 128.0, 192.0]

BAR = "=" * 100


def budget_bytes(ram_gib: float) -> float:
    """Resident budget: usable VRAM plus usable system RAM, in bytes."""
    return ((VRAM_GIB - VRAM_RESERVE_GIB) + (ram_gib - OS_RESERVE_GIB)) * GIB


def per_token_expert_bytes(quant_gb: float) -> float:
    """Bytes of expert weight touched per decoded token at batch 1."""
    bytes_per_param = quant_gb * GB / TOTAL_PARAMS
    return TOP_K * NUM_LAYERS * PARAMS_PER_EXPERT * bytes_per_param


def tokens_per_second(quant_gb: float, budget: float, bw_ram: float, bw_nvme: float) -> tuple[float, float]:
    """Return (resident_fraction, tok/s ceiling) for the expert path."""
    model_bytes = quant_gb * GB
    resident = min(1.0, budget / model_bytes)
    per_token = per_token_expert_bytes(quant_gb)
    seconds = per_token * (resident / bw_ram + (1.0 - resident) / bw_nvme)
    return resident, 1.0 / seconds


def section_fit() -> None:
    print(BAR)
    print("  1. WHAT FITS AT EACH RAM SIZE   (8 GiB card, 1.5 GiB VRAM reserve, 4 GiB OS reserve)")
    print(BAR)
    print(f"  {'RAM':>7} | {'budget':>10} | largest quant that fits entirely")
    print(f"  {'-' * 96}")
    for ram in RAM_CONFIGS_GIB:
        budget = budget_bytes(ram)
        fits = [q for q, gb in QUANTS if gb * GB <= budget]
        best = fits[-1] if fits else "none"
        print(f"  {ram:5.0f} GiB | {budget/GB:7.1f} GB | {best}")
    print(f"  {'-' * 96}")

    print("\n  Residency percentage per quant:")
    print(f"  {'-' * 96}")
    header = f"  {'quant':<14} | {'size':>8} | " + " | ".join(f"{int(r):>6} GiB" for r in RAM_CONFIGS_GIB)
    print(header)
    print(f"  {'-' * 96}")
    for quant, gb in QUANTS:
        cells = []
        for ram in RAM_CONFIGS_GIB:
            r = min(1.0, budget_bytes(ram) / (gb * GB))
            cells.append(f"{r*100:8.0f}%  " if r < 1.0 else f"{'FITS':>8}  ")
        print(f"  {quant:<14} | {gb:6.1f}GB | " + " | ".join(cells))
    print(f"  {'-' * 96}")


def section_throughput(bw_ram: float, ram_label: str, bw_nvme: float, nvme_label: str) -> None:
    print(f"\n{BAR}")
    print(f"  THROUGHPUT CEILING: expert path only, batch 1   [{ram_label} + {nvme_label}]")
    print(BAR)
    print(f"  {'quant':<14} | " + " | ".join(f"{int(r):>10} GiB" for r in RAM_CONFIGS_GIB))
    print(f"  {'-' * 96}")
    for quant, gb in QUANTS:
        cells = []
        for ram in RAM_CONFIGS_GIB:
            _, tps = tokens_per_second(gb, budget_bytes(ram), bw_ram, bw_nvme)
            cells.append(f"{tps:10.1f} t/s")
        print(f"  {quant:<14} | " + " | ".join(cells))
    print(f"  {'-' * 96}")


def section_answer() -> None:
    print(f"\n{BAR}")
    print("  ANSWER: 32 GiB -> 64 GiB")
    print(BAR)
    b32 = budget_bytes(32.0)
    b64 = budget_bytes(64.0)
    for quant, gb in [("UD-IQ1_S", 72.55), ("UD-Q4_K_XL", 111.33)]:
        r32, t32 = tokens_per_second(gb, b32, BW_RAM_DDR4, BW_NVME)
        r64, t64 = tokens_per_second(gb, b64, BW_RAM_DDR4, BW_NVME)
        print(f"\n  {quant}  ({gb:.1f} GB)")
        print(f"    32 GiB : {r32*100:5.1f}% resident -> {t32:6.2f} tok/s")
        print(f"    64 GiB : {r64*100:5.1f}% resident -> {t64:6.2f} tok/s   ({t64/t32:.2f}x)")

    # Where does the target quant become fully resident?
    print("\n  RAM required for full residency (DDR4, expert path at RAM speed):")
    print(f"  {'-' * 96}")
    for quant, gb in QUANTS:
        needed_gib = gb * GB / GIB - (VRAM_GIB - VRAM_RESERVE_GIB) + OS_RESERVE_GIB
        _, tps = tokens_per_second(gb, float("inf"), BW_RAM_DDR4, BW_NVME)
        print(f"  {quant:<14} | needs {needed_gib:6.1f} GiB RAM | then {tps:6.1f} tok/s")
    print(f"  {'-' * 96}")


def main() -> None:
    print(BAR)
    print("  RAM SWEEP: Qwen3.8-Flash-Next (177B, 512 experts, top-10) on an 8 GiB card")
    print(BAR)
    section_fit()
    section_throughput(BW_RAM_DDR4, "DDR4 50 GB/s", BW_NVME, "NVMe Gen4 7 GB/s")
    section_throughput(BW_RAM_DDR5, "DDR5 80 GB/s", BW_NVME_GEN5, "NVMe Gen5 13 GB/s")
    section_answer()
    print(f"\n{BAR}")
    print("""  CAVEATS
    - Expert path only. Attention, dense weights, and kernel overhead are excluded,
      so every number is an upper bound.
    - Assumes uniform expert access. Fine-grained MoE is trained with a router
      balance loss precisely to keep routing uniform, so caching skew is limited.
      Real skew helps somewhat; it does not change the order of magnitude.
    - Assumes a cold NVMe read on every miss. A warm page cache within a single
      token helps, but the per-token working set (1.5 GB at Q4) exceeds any
      realistic prefetch window.""")
    print(BAR)


if __name__ == "__main__":
    main()
