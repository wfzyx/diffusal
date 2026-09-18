"""
Ternary-Bonsai + Uno applied to Qwen3.8-Flash-Next on a domestic PC
==================================================================
simulate_128gb.py established the corrected three-part parameter budget and
showed that placement (dense on GPU, experts in RAM) gives ~28.7 tok/s at Q4.

This applies the two proposed techniques to that corrected model:

  ternary-bonsai : native 1.58-bit expert weights, packed at 2 bits, second-order
                   (Hessian) calibration per expert. Shrinks bytes per parameter
                   from 0.619 (Q4_K_XL) to 0.250.

  Uno / Psi-Spec : draft a block of B tokens in one forward pass, verify in a
                   second. Costs 2 passes per block instead of B passes.

Each technique acts on a different term, so they must be evaluated against the
three-part split rather than against the model as a whole:

  part      per-token cost   ternary helps?        Uno helps?
  --------  ---------------  --------------------  ----------------------------
  dense     read in FULL     yes, 2.5x fewer bytes YES, amortises over B tokens
  experts   top-10 of 512    yes, 2.5x fewer bytes NO, block widens the union
  PLE       lookup only      footprint only        no effect

The dense term is the one Uno actually amortises, because it is read in full
every token. The expert term is the one Uno damages, because a block of B tokens
routes to the union of their experts rather than to 10.
"""

from __future__ import annotations

GIB = 1024**3
GB = 1e9

# ----------------------------------------------------------- parameter budget
TOTAL_PARAMS = 179_999_981_424
NUM_LAYERS, HIDDEN, NUM_EXPERTS, TOP_K, MOE_INTER = 48, 2560, 512, 10, 640
PARAMS_PER_EXPERT = 3 * HIDDEN * MOE_INTER
EXPERT_PARAMS = NUM_EXPERTS * PARAMS_PER_EXPERT * NUM_LAYERS   # 120.8 B
DENSE_PARAMS = 4.5e9
PLE_PARAMS = TOTAL_PARAMS - EXPERT_PARAMS - DENSE_PARAMS       # ~54.7 B

# ------------------------------------------------------------ bytes per param
BPP_Q4 = 111.33 * GB / TOTAL_PARAMS   # 0.619, measured from the GGUF
BPP_TERNARY = 0.250                   # 1.58-bit packed into 2 bits

# --------------------------------------------------------------------- machine
BW_VRAM = 496.0 * GB
BW_RAM = 50.0 * GB
BW_NVME = 7.0 * GB
VRAM_USABLE = 6.44 * GB
DDR4_EFFICIENCY = 0.70                # achieved fraction of peak

BAR = "=" * 100


def distinct_experts(block: int) -> float:
    return NUM_EXPERTS * (1.0 - (1.0 - TOP_K / NUM_EXPERTS) ** block)


class Config:
    def __init__(self, name: str, bpp_dense: float, bpp_expert: float, bpp_ple: float, block: int):
        self.name = name
        self.bpp_dense, self.bpp_expert, self.bpp_ple = bpp_dense, bpp_expert, bpp_ple
        self.block = block  # 1 means plain autoregressive

    # ---- footprint
    def dense_gb(self) -> float:
        return DENSE_PARAMS * self.bpp_dense

    def expert_gb(self) -> float:
        return EXPERT_PARAMS * self.bpp_expert

    def ple_gb(self) -> float:
        return PLE_PARAMS * self.bpp_ple

    def total_gb(self) -> float:
        return self.dense_gb() + self.expert_gb() + self.ple_gb()

    # ---- per-token traffic and time
    def seconds_per_token(self) -> tuple[float, float, float]:
        dense_bytes = DENSE_PARAMS * self.bpp_dense
        if self.block == 1:
            experts = distinct_experts(1)
            expert_bytes = experts * NUM_LAYERS * PARAMS_PER_EXPERT * self.bpp_expert
            passes_per_token_dense = 1.0
            expert_bytes_per_token = expert_bytes
        else:
            experts = distinct_experts(self.block)
            expert_bytes = experts * NUM_LAYERS * PARAMS_PER_EXPERT * self.bpp_expert
            # draft pass + verify pass, amortised over the block
            passes_per_token_dense = 2.0 / self.block
            expert_bytes_per_token = 2.0 * expert_bytes / self.block

        # dense is pinned in VRAM whenever it fits, otherwise RAM
        bw_dense = BW_VRAM if dense_bytes <= VRAM_USABLE else BW_RAM
        t_dense = passes_per_token_dense * dense_bytes / bw_dense
        t_expert = expert_bytes_per_token / (BW_RAM * DDR4_EFFICIENCY)
        return t_dense, t_expert, t_dense + t_expert

    def tokens_per_second(self) -> float:
        return 1.0 / self.seconds_per_token()[2]


def section_footprint() -> None:
    print(BAR)
    print("  1. FOOTPRINT: what ternary does to the 111.3 GB model")
    print(BAR)
    configs = [
        ("Q4_K_XL (baseline)", BPP_Q4, BPP_Q4, BPP_Q4),
        ("ternary experts only", BPP_Q4, BPP_TERNARY, BPP_Q4),
        ("ternary experts + PLE", BPP_Q4, BPP_TERNARY, BPP_TERNARY),
        ("ternary everything", BPP_TERNARY, BPP_TERNARY, BPP_TERNARY),
    ]
    print(f"  {'configuration':<24} | {'dense':>8} | {'experts':>9} | {'PLE':>8} | {'total':>9} | min RAM")
    print(f"  {'-' * 96}")
    for name, bd, be, bp in configs:
        c = Config(name, bd, be, bp, 1)
        total = c.total_gb()
        min_ram = (total - VRAM_USABLE) / GIB + 6.0  # plus OS reserve
        print(
            f"  {name:<24} | {c.dense_gb()/GB:5.2f} GB | {c.expert_gb()/GB:6.1f} GB | "
            f"{c.ple_gb()/GB:5.1f} GB | {total/GB:6.1f} GB | {min_ram:5.0f} GiB"
        )
    print(f"  {'-' * 96}")
    print("  Ternary on experts alone takes the model from 111.3 GB to 66.9 GB, which moves")
    print("  the minimum viable build from 128 GiB down to 64 GiB. That is the RAM purchase,")
    print("  cancelled. Ternary on the PLE table saves another 20 GB but ternarising an")
    print("  embedding table is the highest-risk place to do it.")


def section_throughput() -> None:
    print(f"\n{BAR}")
    print("  2. THROUGHPUT: ternary and Uno, separately and together")
    print(BAR)
    print(f"  Dense pinned in VRAM at {BW_VRAM/GB:.0f} GB/s, experts in RAM at "
          f"{BW_RAM/GB:.0f} GB/s x {DDR4_EFFICIENCY:.0%} achieved.")
    print(f"  {'-' * 96}")
    print(f"  {'configuration':<40} | {'dense ms':>9} | {'expert ms':>10} | {'tok/s':>8} | {'vs base':>8}")
    print(f"  {'-' * 96}")

    rows = [
        ("Q4, autoregressive (baseline)", BPP_Q4, BPP_Q4, 1),
        ("Q4, Uno B=8", BPP_Q4, BPP_Q4, 8),
        ("Q4, Uno B=16", BPP_Q4, BPP_Q4, 16),
        ("Q4, Uno B=32", BPP_Q4, BPP_Q4, 32),
        ("ternary experts, autoregressive", BPP_Q4, BPP_TERNARY, 1),
        ("ternary all, autoregressive", BPP_TERNARY, BPP_TERNARY, 1),
        ("ternary all, Uno B=8", BPP_TERNARY, BPP_TERNARY, 8),
        ("ternary all, Uno B=16", BPP_TERNARY, BPP_TERNARY, 16),
        ("ternary all, Uno B=32", BPP_TERNARY, BPP_TERNARY, 32),
    ]
    base = None
    for name, bd, be, block in rows:
        c = Config(name, bd, be, bd, block)
        t_dense, t_expert, _ = c.seconds_per_token()
        tps = c.tokens_per_second()
        if base is None:
            base = tps
        print(
            f"  {name:<40} | {t_dense*1e3:8.2f}  | {t_expert*1e3:9.2f}  | "
            f"{tps:7.1f}  | {tps/base:7.2f}x"
        )
    print(f"  {'-' * 96}")


def section_why() -> None:
    print(f"\n{BAR}")
    print("  3. WHY UNO LOSES HERE BUT WOULD WIN ON A DENSE MODEL")
    print(BAR)
    print(f"  {'B':>4} | {'dense passes/tok':>17} | {'experts touched':>16} | {'expert bytes/tok':>17}")
    print(f"  {'-' * 96}")
    for block in (1, 2, 4, 8, 16, 32):
        if block == 1:
            dense_passes, experts, mult = 1.0, distinct_experts(1), 1.0
        else:
            dense_passes, experts, mult = 2.0 / block, distinct_experts(block), 2.0 / block
        eb = experts * NUM_LAYERS * PARAMS_PER_EXPERT * BPP_Q4 * mult
        print(f"  {block:4d} | {dense_passes:16.3f}  | {experts:15.1f}  | {eb/GB:14.2f} GB")
    print(f"  {'-' * 96}")
    print("""
  Uno amortises the DENSE term beautifully: 1.000 -> 0.063 passes per token at
  B=32, a 16x reduction. If this were a dense model, that alone would be a large
  win and matches the paper's reported speedups.

  It destroys the EXPERT term: 10 experts per token becomes 240 at B=32, and the
  2-pass structure multiplies that again. Expert bytes per token rise from
  1.46 GB to 2.22 GB.

  Because experts are 67% of the parameters and sit on the slow side of the
  hierarchy, the expert term dominates and Uno is a net loss on this model.""")


def section_verdict() -> None:
    base = Config("", BPP_Q4, BPP_Q4, BPP_Q4, 1).tokens_per_second()
    tern = Config("", BPP_TERNARY, BPP_TERNARY, BPP_TERNARY, 1).tokens_per_second()
    tern_uno = max(
        Config("", BPP_TERNARY, BPP_TERNARY, BPP_TERNARY, b).tokens_per_second()
        for b in (2, 4, 8, 16, 32, 64)
    )
    print(f"\n{BAR}")
    print("  4. VERDICT")
    print(BAR)
    print(f"""
  Baseline    Q4, correct placement, autoregressive        {base:6.1f} tok/s
  + ternary   bonsai ternary on all parts                  {tern:6.1f} tok/s   ({tern/base:.2f}x)
  + Uno       best block size on top of ternary            {tern_uno:6.1f} tok/s   ({tern_uno/base:.2f}x)

  a. TERNARY IS THE WIN. It cuts every streamed byte by 2.5x and the machine is
     purely bandwidth bound, so the speedup passes through almost undiluted. It
     also drops the model from 111.3 GB to 45.0 GB, which means a 64 GiB build
     runs it comfortably. The 128 GiB upgrade becomes unnecessary.

  b. UNO IS A NET LOSS ON THIS MODEL, at every block size, with or without
     ternary. The two techniques do not interact: ternary scales all byte counts
     by a constant, so the ratio that decides Uno is unchanged. My earlier claim
     that they compete for the same bottleneck was wrong in its reasoning -- the
     real reason Uno fails here is fine-grained routing, not the roofline.

  c. UNO WOULD WIN ON A DENSE OR COARSE-MoE MODEL. It reduces dense passes per
     token by up to 16x. Qwen3.8-Flash-Next is 67% fine-grained experts, which is
     the worst possible shape for block drafting. The Uno paper's 8B Qwen3 result
     is a DENSE model, which is exactly why it works there.

  d. BONSAI (second-order Hessian PTQ) matters for quality, not speed. Per-expert
     Hessians from dispatched tokens are what let 512 small experts survive
     ternarisation. Naive RTN on a 640-wide expert will not.

  e. The PLE table is the open question. 54.7 B of embedding parameters, 31% of
     the model, contributing almost no bandwidth. Ternarising it saves 20 GB of
     footprint and zero time, while embeddings are the least robust thing to
     quantise. Leave it at Q4 or higher and ternarise only the experts.""")


def main() -> None:
    print(BAR)
    print("  TERNARY-BONSAI + UNO ON QWEN3.8-FLASH-NEXT  (2080 Super 8 GiB, i7, NVMe)")
    print(BAR)
    section_footprint()
    section_throughput()
    section_why()
    section_verdict()
    print(f"\n{BAR}")
    print("""  CAVEATS
    - DENSE_PARAMS estimated at 4.5 B; PLE inferred by subtraction. A GGUF tensor
      dump would measure both.
    - Ternary at 0.250 bytes/param assumes 2-bit packing with per-group scales
      folded in. Real packed formats carry scale overhead of a few percent.
    - Quality is not modelled. Ternary experts require QAT or strong second-order
      PTQ; the tax measured in exp2 was +4.2% to +15.4% at toy scale and is
      unmeasured at 512-expert granularity.
    - Uno acceptance rate is assumed perfect. Real acceptance below 100% makes the
      Uno rows worse, not better.""")
    print(BAR)


if __name__ == "__main__":
    main()
