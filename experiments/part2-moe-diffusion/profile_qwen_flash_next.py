"""
Target Profile: Qwen3.8-Flash-Next on a domestic PC
==================================================
Hardware under test
    GPU    RTX 2080 Super, 8 GiB VRAM, 496 GB/s, 22.3 TFLOP/s fp16 tensor
    CPU    Intel i7
    RAM    32 GiB DDR4, ~50 GB/s dual channel
    Disk   NVMe M.2, ~3.5-7.0 GB/s
    Bus    PCIe (host RAM -> VRAM), ~12-26 GB/s effective

Model (Qwen/Qwen3.8-Flash-Next, arch qwen4_exp)
    48 layers, hidden 2560, 512 experts, top-10 routing
    moe_intermediate_size 640, 1 shared expert
    36 linear-attention + 12 full-attention layers
    GGUF footprints measured from unsloth/Qwen3.8-Flash-Next-GGUF

The earlier roofline work assumed a coarse MoE (8 experts, top-2). This model is
FINE-GRAINED: 512 experts, top-10. That changes the block-drafting arithmetic
fundamentally, because the union of experts touched by a block of B tokens grows
far faster than the per-token working set.
"""

from __future__ import annotations

# ---------------------------------------------------------------- model config
NUM_LAYERS = 48
HIDDEN = 2560
NUM_EXPERTS = 512
TOP_K = 10
MOE_INTERMEDIATE = 640
EXPERT_MATRICES = 3  # gate_proj, up_proj, down_proj

PARAMS_PER_EXPERT = EXPERT_MATRICES * HIDDEN * MOE_INTERMEDIATE
EXPERT_PARAMS_PER_LAYER = NUM_EXPERTS * PARAMS_PER_EXPERT
TOTAL_EXPERT_PARAMS = EXPERT_PARAMS_PER_LAYER * NUM_LAYERS

# ------------------------------------------------------- measured GGUF weights
# (quant, total GB) from the unsloth repository
GGUF_SIZES = [
    ("UD-IQ1_S", 72.55),
    ("UD-IQ1_M", 74.54),
    ("UD-Q2_K_XL", 78.87),
    ("UD-IQ3_XXS", 81.96),
    ("UD-Q3_K_XL", 89.99),
    ("UD-IQ4_XS", 93.68),
    ("UD-Q4_K_XL", 111.33),
    ("UD-Q5_K_XL", 158.29),
    ("Q8_0", 188.23),
    ("BF16", 354.03),
]
TARGET_QUANT = "UD-Q4_K_XL"
TARGET_GB = 111.33
BF16_GB = 354.03
TOTAL_PARAMS = BF16_GB * 1e9 / 2.0  # bf16 = 2 bytes/param

# --------------------------------------------------------------- the host box
VRAM_GIB = 8.0
RAM_GIB = 32.0
VRAM_RESERVE_GIB = 1.5   # KV cache, context, compute buffers
RAM_RESERVE_GIB = 4.0    # OS and everything else

VRAM_BW = 496.0   # GB/s
RAM_BW = 50.0     # GB/s, DDR4 dual channel
PCIE_BW = 12.0    # GB/s, PCIe 3.0 x16 effective
NVME_BW = 7.0     # GB/s, Gen4 x4

BAR = "=" * 100


def distinct_experts(block: int) -> float:
    """Expected distinct experts touched by a block of `block` tokens, per layer."""
    miss = (1.0 - TOP_K / NUM_EXPERTS) ** block
    return NUM_EXPERTS * (1.0 - miss)


def expert_bytes(distinct_per_layer: float, bytes_per_param: float) -> float:
    """Bytes of expert weight touched across all layers."""
    return distinct_per_layer * NUM_LAYERS * PARAMS_PER_EXPERT * bytes_per_param


def section_model() -> None:
    print(BAR)
    print("  1. MODEL SHAPE")
    print(BAR)
    print(f"  layers                       {NUM_LAYERS}")
    print(f"  hidden size                  {HIDDEN}")
    print(f"  experts / layer              {NUM_EXPERTS}   (top-{TOP_K} routing = {TOP_K/NUM_EXPERTS*100:.2f}% per token)")
    print(f"  params per expert            {PARAMS_PER_EXPERT/1e6:.2f} M")
    print(f"  expert params / layer        {EXPERT_PARAMS_PER_LAYER/1e9:.2f} B")
    print(f"  expert params total          {TOTAL_EXPERT_PARAMS/1e9:.1f} B")
    print(f"  model params total (bf16)    {TOTAL_PARAMS/1e9:.1f} B")
    print(f"  expert share of model        {TOTAL_EXPERT_PARAMS/TOTAL_PARAMS*100:.0f}%")


def section_fit() -> None:
    usable_vram = VRAM_GIB - VRAM_RESERVE_GIB
    usable_ram = RAM_GIB - RAM_RESERVE_GIB
    combined = usable_vram + usable_ram

    print(f"\n{BAR}")
    print("  2. FOOTPRINT vs. THIS BOX")
    print(BAR)
    print(f"  usable VRAM {usable_vram:.1f} GiB  +  usable RAM {usable_ram:.1f} GiB  =  {combined:.1f} GiB resident budget")
    print(f"  {'-' * 96}")
    print(f"  {'quant':<14} | {'size':>9} | {'resident':>9} | {'from NVMe':>10} | {'% streamed':>10} | fits?")
    print(f"  {'-' * 96}")
    for quant, size in GGUF_SIZES:
        resident = min(size, combined)
        streamed = max(0.0, size - combined)
        pct = streamed / size * 100
        mark = "YES" if streamed == 0 else "NO"
        star = "  <-- target" if quant == TARGET_QUANT else ""
        print(
            f"  {quant:<14} | {size:6.1f} GB | {resident:6.1f} GB | {streamed:7.1f} GB | "
            f"{pct:9.1f}% | {mark}{star}"
        )
    print(f"  {'-' * 96}")
    print(f"  No quant in the repository fits. Even UD-IQ1_S at {GGUF_SIZES[0][1]:.1f} GB needs")
    print(f"  {GGUF_SIZES[0][1] - combined:.1f} GB of NVMe streaming. This box is permanently in the disk-bound tier.")


def section_working_set() -> None:
    bpp = TARGET_GB * 1e9 / TOTAL_PARAMS  # effective bytes/param at the target quant

    print(f"\n{BAR}")
    print(f"  3. EXPERT WORKING SET vs. BLOCK SIZE   ({TARGET_QUANT}, {bpp:.3f} bytes/param effective)")
    print(BAR)
    print("  A block-parallel draft of B tokens must touch the UNION of every expert any")
    print("  token in the block routes to. With 512 fine-grained experts that union grows fast.")
    print(f"  {'-' * 96}")
    print(f"  {'B':>4} | {'experts/layer':>14} | {'expert bytes':>13} | {'bytes/token':>12} | {'amortization':>13}")
    print(f"  {'-' * 96}")

    base = expert_bytes(distinct_experts(1), bpp)
    for block in (1, 2, 4, 8, 16, 32, 64):
        n = distinct_experts(block)
        by = expert_bytes(n, bpp)
        per_token = by / block
        amort = base / per_token
        print(
            f"  {block:4d} | {n:11.1f}    | {by/1e9:10.2f} GB | {per_token/1e9:9.2f} GB | {amort:10.2f}x"
        )
    print(f"  {'-' * 96}")
    print("  amortization = bytes/token at B=1 divided by bytes/token at this B.")
    print("  A coarse MoE (8 experts, top-2) reaches ~8x here. This model saturates far lower")
    print("  because 32 tokens x top-10 already covers a large fraction of all 512 experts.")


def section_uno_tax() -> None:
    bpp = TARGET_GB * 1e9 / TOTAL_PARAMS
    print(f"\n{BAR}")
    print("  4. DOES BLOCK DRAFTING PAY?   (Uno / MTP costs 2 passes per block: draft + verify)")
    print(BAR)
    print(f"  {'-' * 96}")
    print(f"  {'B':>4} | {'AR bytes/tok':>13} | {'draft+verify':>13} | {'per token':>11} | {'net':>9} | verdict")
    print(f"  {'-' * 96}")

    ar_per_token = expert_bytes(distinct_experts(1), bpp)
    for block in (2, 4, 8, 16, 32, 64):
        n = distinct_experts(block)
        two_pass = 2.0 * expert_bytes(n, bpp)
        per_token = two_pass / block
        net = ar_per_token / per_token
        verdict = "PAYS" if net > 1.0 else "LOSES"
        print(
            f"  {block:4d} | {ar_per_token/1e9:10.2f} GB | {two_pass/1e9:10.2f} GB | "
            f"{per_token/1e9:8.2f} GB | {net:8.2f}x | {verdict}"
        )
    print(f"  {'-' * 96}")
    print("  Assumes a cold cache each pass, i.e. the working set exceeds what RAM holds.")
    print("  That assumption is correct here: the resident budget is 34.5 GiB against a 111 GB model.")


def section_throughput() -> None:
    bpp = TARGET_GB * 1e9 / TOTAL_PARAMS
    per_token = expert_bytes(distinct_experts(1), bpp)

    print(f"\n{BAR}")
    print("  5. CEILING: tokens/sec from expert streaming alone (batch 1, AR decode)")
    print(BAR)
    print(f"  Per-token expert traffic at {TARGET_QUANT}: {per_token/1e9:.2f} GB")
    print(f"  {'-' * 96}")
    print(f"  {'path':<44} | {'bandwidth':>11} | {'tok/s ceiling':>14}")
    print(f"  {'-' * 96}")
    for label, bw in [
        ("experts in VRAM (impossible, 8 GiB card)", VRAM_BW),
        ("experts in RAM, CPU executes (--cpu-moe)", RAM_BW),
        ("experts in RAM, streamed to GPU over PCIe", PCIE_BW),
        ("experts from NVMe Gen4 (cold, mmap miss)", NVME_BW),
    ]:
        tps = bw * 1e9 / per_token
        print(f"  {label:<44} | {bw:8.1f} GB/s | {tps:11.2f} tok/s")
    print(f"  {'-' * 96}")
    print("  These are upper bounds on the expert path only: no attention, no dense weights,")
    print("  no kernel overhead. Real throughput lands below every number above.")


def section_conclusion() -> None:
    print(f"\n{BAR}")
    print("  6. CONCLUSION FOR THIS TARGET")
    print(BAR)
    print("""
  a. The box is disk-bound, not VRAM-bound. 111 GB of model against a 34.5 GiB
     resident budget means ~69% streams from NVMe on every pass. Ridge point is
     ~3200 FLOP/B, so arithmetic is irrelevant: this is a pure bandwidth problem.

  b. Fine-grained routing breaks block drafting. 512 experts at top-10 means a
     32-token block touches ~240 experts per layer instead of 10. The union grows
     24x while token count grows 32x, so amortization saturates near 1.3x, and the
     2-pass draft-plus-verify structure of Uno/MTP eats more than that at every
     block size tested. On this model, block drafting LOSES on weight traffic.

  c. The coarse-MoE intuition does not transfer. The 8-expert top-2 configuration
     in profile_hardware.py amortizes ~8x, which is where the 1.91x DRAM figure
     came from. Swapping in a 512-expert model inverts the conclusion.

  d. Uno is not needed here anyway. Qwen ships an MTP module for this model, and
     llama.cpp b10883 already exposes --spec-type draft-mtp, draft-eagle3, and
     draft-dflash. The drafting layer exists; the bottleneck is expert streaming.

  e. What actually raises tok/s on this box, in order of leverage:
       1. Shrink the model until experts fit in 32 GiB RAM (UD-IQ1_S is 72.6 GB,
          still 2x too large; this argues for a smaller base model, not a smaller quant)
       2. --cpu-moe so experts execute where they already live, at 50 GB/s DDR4,
          instead of crossing PCIe 3.0 at 12 GB/s
       3. -ngl to pin attention and dense weights in the 8 GiB card
       4. Expert caching by access frequency, which is the real research contribution
""")


def main() -> None:
    print(BAR)
    print("  QWEN3.8-FLASH-NEXT ON RTX 2080 SUPER 8GB / i7 / 32GB RAM / NVMe")
    print(BAR)
    section_model()
    section_fit()
    section_working_set()
    section_uno_tax()
    section_throughput()
    section_conclusion()
    print(BAR)


if __name__ == "__main__":
    main()
