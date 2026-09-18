"""
Decode Simulation: Qwen3.8-Flash-Next on 128 GiB RAM + RTX 2080 Super 8 GiB
==========================================================================
Corrects profile_ram_sweep.py, which modelled the expert path only and treated
the model as one uniform blob. The real parameter budget has three parts with
completely different access patterns.

Verified parameter count (HF safetensors index): 179,999,981,424 params.

  part          params    share   access pattern per decoded token
  ------------  --------  ------  --------------------------------------------
  MoE experts    120.8 B    67%   top-10 of 512 per layer -> 1.95% streamed
  PLE n-gram      ~55   B    31%   embedding LOOKUP -> a few rows, kilobytes
  dense rest      ~4.5 B     2%   attention + lm_head -> read in FULL every token

Derivation of the split
  experts  = 512 experts x 3 mats x 2560 x 640 x 48 layers     = 120.8 B
  PLE      = ngram_vocab_size_base 20e6 x ple_embed_dim 2560   =  ~51-55 B
             (config: split_ngram_parts 128, ngram_size 3, 128 shard tensors)
  dense    = 180 - 120.8 - 55                                  =   ~4.5 B

Why this matters
  1. PLE is 31% of the model but costs kilobytes per token. It can live on NVMe
     essentially for free, because embedding lookups are small random reads.
  2. The dense 4.5 B is only 2% of the model but is read IN FULL every token.
     At Q4 that is ~2.4 GB/token, which is MORE traffic than the 1.48 GB of
     expert weight. The small part dominates.
  3. Therefore the decisive optimisation is placement, not quantisation:
     dense on the GPU, experts in RAM, PLE on disk.

Estimates are flagged. Expert and dense figures come from config arithmetic;
PLE is inferred by subtraction against the verified total.
"""

from __future__ import annotations

GIB = 1024**3
GB = 1e9

# ----------------------------------------------------------- parameter budget
TOTAL_PARAMS = 179_999_981_424

NUM_LAYERS = 48
HIDDEN = 2560
NUM_EXPERTS = 512
TOP_K = 10
MOE_INTER = 640

PARAMS_PER_EXPERT = 3 * HIDDEN * MOE_INTER
EXPERT_PARAMS = NUM_EXPERTS * PARAMS_PER_EXPERT * NUM_LAYERS

# dense: linear-attn (36) + full-attn (12) + shared experts + lm_head + norms
DENSE_PARAMS = 4.5e9          # estimated from config arithmetic
PLE_PARAMS = TOTAL_PARAMS - EXPERT_PARAMS - DENSE_PARAMS

# embedding lookup traffic per token: ngram_size 3 x a few heads x embed dim
PLE_ROWS_PER_TOKEN = 3 * 8    # ngram_size x heads_per_ngram
PLE_EMBED_DIM = 2560

# --------------------------------------------------------------- quantisation
QUANT_GB = 111.33             # UD-Q4_K_XL total, measured
BYTES_PER_PARAM = QUANT_GB * GB / TOTAL_PARAMS

# --------------------------------------------------------------------- machine
RAM_GIB = 128.0
OS_RESERVE_GIB = 6.0
VRAM_GIB = 8.0
VRAM_RESERVE_GIB = 2.0        # KV cache, context, CUDA buffers

BW_VRAM = 496.0 * GB
BW_RAM = 50.0 * GB            # DDR4-3200 dual channel
BW_PCIE = 12.0 * GB           # PCIe 3.0 x16 effective
BW_NVME = 7.0 * GB            # Gen4 x4 sequential
NVME_RANDOM_IOPS = 600_000    # 4K random read, Gen4

BAR = "=" * 100


def gb(x: float) -> str:
    return f"{x / GB:7.2f} GB"


def section_budget() -> None:
    print(BAR)
    print("  1. PARAMETER BUDGET AND PER-TOKEN TRAFFIC   (UD-Q4_K_XL, %.3f bytes/param)" % BYTES_PER_PARAM)
    print(BAR)
    rows = [
        ("MoE experts", EXPERT_PARAMS, EXPERT_PARAMS * TOP_K / NUM_EXPERTS, "top-10 of 512, streamed"),
        ("PLE n-gram table", PLE_PARAMS, PLE_ROWS_PER_TOKEN * PLE_EMBED_DIM, "embedding lookup, %d rows" % PLE_ROWS_PER_TOKEN),
        ("dense (attn + head)", DENSE_PARAMS, DENSE_PARAMS, "read in full"),
    ]
    print(f"  {'part':<22} | {'params':>10} | {'share':>6} | {'on disk':>10} | {'per token':>11} | pattern")
    print(f"  {'-' * 96}")
    total_per_token = 0.0
    for name, params, per_tok_params, pattern in rows:
        size = params * BYTES_PER_PARAM
        per_tok = per_tok_params * BYTES_PER_PARAM
        total_per_token += per_tok
        print(
            f"  {name:<22} | {params/1e9:7.1f} B | {params/TOTAL_PARAMS*100:5.1f}% | "
            f"{size/GB:7.1f} GB | {per_tok/GB:8.3f} GB | {pattern}"
        )
    print(f"  {'-' * 96}")
    print(f"  {'TOTAL':<22} | {TOTAL_PARAMS/1e9:7.1f} B | 100.0% | {QUANT_GB:7.1f} GB | {total_per_token/GB:8.3f} GB |")
    print(f"  {'-' * 96}")
    print("  The dense part is 2% of the model but produces more traffic per token than")
    print("  the 120.8 B of experts. The PLE table is 31% of the model and produces ~0.")


def placements() -> list[tuple[str, dict]]:
    """Each placement maps part -> bandwidth used for its per-token traffic."""
    return [
        ("everything in RAM, CPU executes", {"dense": BW_RAM, "expert": BW_RAM, "ple": BW_RAM}),
        ("dense on GPU, experts+PLE in RAM", {"dense": BW_VRAM, "expert": BW_RAM, "ple": BW_RAM}),
        ("dense on GPU, experts RAM, PLE NVMe", {"dense": BW_VRAM, "expert": BW_RAM, "ple": BW_NVME}),
        ("dense on GPU, experts over PCIe", {"dense": BW_VRAM, "expert": BW_PCIE, "ple": BW_NVME}),
        ("all in RAM, streamed to GPU (no -cmoe)", {"dense": BW_PCIE, "expert": BW_PCIE, "ple": BW_PCIE}),
        ("experts on NVMe (insufficient RAM)", {"dense": BW_VRAM, "expert": BW_NVME, "ple": BW_NVME}),
    ]


def section_placement() -> None:
    print(f"\n{BAR}")
    print("  2. PLACEMENT SIMULATION AT 128 GiB   (whole model resident: 111.3 GB of %.0f GB budget)" %
          ((RAM_GIB - OS_RESERVE_GIB) * GIB / GB))
    print(BAR)

    dense_b = DENSE_PARAMS * BYTES_PER_PARAM
    expert_b = EXPERT_PARAMS * TOP_K / NUM_EXPERTS * BYTES_PER_PARAM
    ple_b = PLE_ROWS_PER_TOKEN * PLE_EMBED_DIM * BYTES_PER_PARAM

    print(f"  per-token bytes:  dense {dense_b/GB:.3f} GB | experts {expert_b/GB:.3f} GB | PLE {ple_b/1e6:.3f} MB")
    print(f"  {'-' * 96}")
    print(f"  {'placement':<40} | {'dense ms':>9} | {'expert ms':>10} | {'total ms':>9} | {'tok/s':>8}")
    print(f"  {'-' * 96}")

    results = []
    for label, bwmap in placements():
        t_dense = dense_b / bwmap["dense"]
        t_expert = expert_b / bwmap["expert"]
        t_ple = ple_b / bwmap["ple"]
        total = t_dense + t_expert + t_ple
        tps = 1.0 / total
        results.append((label, tps))
        print(
            f"  {label:<40} | {t_dense*1e3:8.2f}  | {t_expert*1e3:9.2f}  | "
            f"{total*1e3:8.2f}  | {tps:7.1f}"
        )
    print(f"  {'-' * 96}")
    best = max(results, key=lambda r: r[1])
    worst = min(results, key=lambda r: r[1])
    print(f"  best : {best[0]}  ->  {best[1]:.1f} tok/s")
    print(f"  worst: {worst[0]}  ->  {worst[1]:.1f} tok/s   ({best[1]/worst[1]:.1f}x spread from placement alone)")


def section_vram() -> None:
    print(f"\n{BAR}")
    print("  3. DOES THE DENSE PART FIT IN 8 GiB VRAM?")
    print(BAR)
    usable = (VRAM_GIB - VRAM_RESERVE_GIB) * GIB
    dense_b = DENSE_PARAMS * BYTES_PER_PARAM
    print(f"  usable VRAM after {VRAM_RESERVE_GIB:.1f} GiB reserve : {usable/GB:6.2f} GB")
    print(f"  dense weights at Q4                  : {dense_b/GB:6.2f} GB")
    headroom = usable - dense_b
    print(f"  headroom                             : {headroom/GB:6.2f} GB")
    if headroom > 0:
        n_experts_cached = headroom / (PARAMS_PER_EXPERT * BYTES_PER_PARAM) / NUM_LAYERS
        print(f"\n  Spare VRAM holds ~{n_experts_cached:.0f} experts per layer ({n_experts_cached/NUM_EXPERTS*100:.1f}% of 512).")
        hit = n_experts_cached / NUM_EXPERTS
        expert_b = EXPERT_PARAMS * TOP_K / NUM_EXPERTS * BYTES_PER_PARAM
        t = dense_b / BW_VRAM + expert_b * (hit / BW_VRAM + (1 - hit) / BW_RAM)
        print(f"  With uniform routing that is a {hit*100:.1f}% cache hit rate -> {1/t:.1f} tok/s")
        print("  Uniform routing is the pessimistic case. Caching the most-used experts by")
        print("  access frequency is where the remaining headroom lives.")
    else:
        print("\n  Does not fit; offload some layers with -ngl.")


def section_commands() -> None:
    print(f"\n{BAR}")
    print("  4. COMMANDS TO RUN")
    print(BAR)
    print("""
  Best placement, 128 GiB build:

    llama-server \\
      -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \\
      --cpu-moe \\
      -ngl 99 \\
      -c 8192 \\
      --mlock

  --cpu-moe   keeps the 76 GB of experts in RAM and executes them on the i7,
              avoiding a PCIe crossing that would cost 4x.
  -ngl 99     pins the dense attention stack in VRAM, where it runs at 496 GB/s
              instead of 50 GB/s. This is the single largest win.
  --mlock     stops the kernel paging expert weights back out to NVMe once the
              working set is hot. Needs 111 GB locked, so 128 GiB is the minimum
              config where this is safe.

  Then measure whether drafting helps, which the expert-union analysis in
  profile_qwen_flash_next.py predicts it will not:

    llama-server ... --spec-type draft-mtp \\
      -md MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
""")


def section_caveats() -> None:
    print(BAR)
    print("  CAVEATS")
    print(BAR)
    print("""  - DENSE_PARAMS is estimated at 4.5 B from config arithmetic over 36 linear-attn
    and 12 full-attn layers plus lm_head. PLE is then inferred by subtraction from
    the verified 180.0 B total. A GGUF tensor dump would replace both with measured
    values and is the right next step before trusting absolute tok/s.
  - Bandwidth figures are peak. Achieved DDR4 rates are typically 60-75% of peak,
    so divide the RAM-bound numbers by roughly 1.4 for a realistic expectation.
  - Compute time is ignored entirely. At these intensities the box is bandwidth
    bound, but the i7 executing 10 experts per layer per token is not free.
  - Assumes uniform expert routing. Real routers are skewed enough that frequency
    based caching helps, which is upside, not downside.""")


def main() -> None:
    print(BAR)
    print("  SIMULATION: 128 GiB RAM + RTX 2080 Super 8 GiB + i7 + NVMe Gen4")
    print(BAR)
    section_budget()
    section_placement()
    section_vram()
    section_commands()
    section_caveats()
    print(BAR)


if __name__ == "__main__":
    main()
