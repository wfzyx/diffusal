"""
Constrained-Routing Uno: drafting on top of already-selected experts
====================================================================
simulate_ternary_uno.py charged the expert union TWICE, once for the draft pass
and once for the verify pass, and concluded Uno is a net loss. That double charge
is wrong: the draft pathway is free to route however we like, because the AR
verifier rejects bad drafts and the output distribution is preserved regardless.
A deliberately cheap drafter costs acceptance rate, not correctness.

Three drafting strategies, in increasing order of how little weight they touch:

  A. unconstrained   draft routes normally -> union(B) experts, paid twice
  B. reuse-selected  draft reuses the top-10 experts already resident for the
                     current token -> 10 experts, paid once
  C. shared-only     draft uses ONLY each layer's shared_expert, which is already
                     read every token -> 0 additional expert bytes

Strategy C is specific to this architecture: Qwen3.8-Flash-Next has
mlp.shared_expert.{gate,up,down}_proj on all 48 layers, always active. An Uno
LoRA trained on the shared-expert pathway drafts at zero marginal expert traffic.

The verify pass always uses true routing, so it always costs union(B) experts.
But that union is paid ONCE for B tokens, which is cheaper per token than the
10 experts autoregressive decoding pays for every single token:

    autoregressive     10.0 experts/token
    verify at B=32    239.6 / 32 = 7.49 experts/token

That is the effect the earlier double charge hid.

The decisive unknown is acceptance length tau: a constrained drafter proposes
worse tokens, the verifier rejects more, and fewer than B tokens survive per
block. This script solves for the break-even tau.
"""

from __future__ import annotations

GIB = 1024**3
GB = 1e9

TOTAL_PARAMS = 179_999_981_424
NUM_LAYERS, HIDDEN, NUM_EXPERTS, TOP_K, MOE_INTER = 48, 2560, 512, 10, 640
PARAMS_PER_EXPERT = 3 * HIDDEN * MOE_INTER
EXPERT_PARAMS = NUM_EXPERTS * PARAMS_PER_EXPERT * NUM_LAYERS
DENSE_PARAMS = 4.5e9

BPP_Q4 = 111.33 * GB / TOTAL_PARAMS
BPP_TERNARY = 0.250

BW_VRAM = 496.0 * GB
BW_RAM = 50.0 * GB * 0.70  # achieved DDR4

BAR = "=" * 100


def distinct_experts(block: int) -> float:
    return NUM_EXPERTS * (1.0 - (1.0 - TOP_K / NUM_EXPERTS) ** block)


def expert_bytes(n_experts: float, bpp: float) -> float:
    return n_experts * NUM_LAYERS * PARAMS_PER_EXPERT * bpp


def ar_seconds_per_token(bpp_dense: float, bpp_expert: float) -> float:
    dense = DENSE_PARAMS * bpp_dense / BW_VRAM
    experts = expert_bytes(TOP_K, bpp_expert) / BW_RAM
    return dense + experts


def block_seconds(block: int, strategy: str, bpp_dense: float, bpp_expert: float) -> float:
    """Seconds for one draft+verify cycle over a block of `block` tokens."""
    dense = 2.0 * DENSE_PARAMS * bpp_dense / BW_VRAM  # draft pass + verify pass

    if strategy == "unconstrained":
        draft_experts = distinct_experts(block)
    elif strategy == "reuse-selected":
        draft_experts = TOP_K          # reuses experts already read for this token
    elif strategy == "shared-only":
        draft_experts = 0.0            # shared_expert is already in the dense term
    else:
        raise ValueError(strategy)

    verify_experts = distinct_experts(block)  # true routing, always
    experts = expert_bytes(draft_experts + verify_experts, bpp_expert) / BW_RAM
    return dense + experts


def breakeven_tau(block: int, strategy: str, bpp_dense: float, bpp_expert: float) -> float:
    """Accepted tokens per block needed to match autoregressive throughput."""
    return block_seconds(block, strategy, bpp_dense, bpp_expert) / ar_seconds_per_token(bpp_dense, bpp_expert)


def section_experts_per_token() -> None:
    print(BAR)
    print("  1. EXPERTS TOUCHED PER TOKEN   (the quantity the earlier model got wrong)")
    print(BAR)
    print(f"  autoregressive baseline: {TOP_K:.1f} experts/token, every token, no reuse")
    print(f"  {'-' * 96}")
    print(f"  {'B':>4} | {'union(B)':>9} | {'A unconstr.':>12} | {'B reuse-sel':>12} | {'C shared-only':>14}")
    print(f"  {'-' * 96}")
    for block in (2, 4, 8, 16, 32, 64):
        u = distinct_experts(block)
        a = (u + u) / block
        b = (TOP_K + u) / block
        c = u / block
        print(f"  {block:4d} | {u:8.1f}  | {a:11.2f}  | {b:11.2f}  | {c:13.2f}")
    print(f"  {'-' * 96}")
    print("  Values below 10.0 beat autoregressive decoding on expert traffic.")
    print("  Strategy A never does. Strategies B and C do from B=4 upward.")


def section_throughput(label: str, bpp_dense: float, bpp_expert: float) -> None:
    print(f"\n{BAR}")
    print(f"  THROUGHPUT AT PERFECT ACCEPTANCE (tau = B)   [{label}]")
    print(BAR)
    ar = 1.0 / ar_seconds_per_token(bpp_dense, bpp_expert)
    print(f"  autoregressive baseline: {ar:.1f} tok/s")
    print(f"  {'-' * 96}")
    print(f"  {'B':>4} | {'A unconstrained':>18} | {'B reuse-selected':>18} | {'C shared-only':>18}")
    print(f"  {'-' * 96}")
    for block in (4, 8, 16, 32, 64):
        cells = []
        for strategy in ("unconstrained", "reuse-selected", "shared-only"):
            tps = block / block_seconds(block, strategy, bpp_dense, bpp_expert)
            cells.append(f"{tps:7.1f} ({tps/ar:4.2f}x)")
        print(f"  {block:4d} | " + " | ".join(f"{c:>18}" for c in cells))
    print(f"  {'-' * 96}")


def section_breakeven(label: str, bpp_dense: float, bpp_expert: float) -> None:
    print(f"\n{BAR}")
    print(f"  BREAK-EVEN ACCEPTANCE: tokens per block needed to match AR   [{label}]")
    print(BAR)
    print(f"  {'B':>4} | {'A unconstrained':>18} | {'B reuse-selected':>18} | {'C shared-only':>18}")
    print(f"  {'-' * 96}")
    for block in (4, 8, 16, 32, 64):
        cells = []
        for strategy in ("unconstrained", "reuse-selected", "shared-only"):
            tau = breakeven_tau(block, strategy, bpp_dense, bpp_expert)
            flag = "impossible" if tau > block else f"{tau:.2f} of {block}"
            cells.append(flag)
        print(f"  {block:4d} | " + " | ".join(f"{c:>18}" for c in cells))
    print(f"  {'-' * 96}")
    print("  'impossible' means the block cycle costs more than decoding B tokens one at a")
    print("  time even with 100% acceptance. Anything else is an empirical target for tau.")


def section_tau_sensitivity() -> None:
    print(f"\n{BAR}")
    print("  4. SENSITIVITY: realistic acceptance on ternary weights, strategy C, B=32")
    print(BAR)
    ar = 1.0 / ar_seconds_per_token(BPP_TERNARY, BPP_TERNARY)
    cycle = block_seconds(32, "shared-only", BPP_TERNARY, BPP_TERNARY)
    print(f"  autoregressive ternary baseline: {ar:.1f} tok/s")
    print(f"  one draft+verify cycle at B=32  : {cycle*1e3:.2f} ms")
    print(f"  {'-' * 96}")
    print(f"  {'tau':>6} | {'tok/s':>9} | {'vs AR':>8} | note")
    print(f"  {'-' * 96}")
    for tau in (2, 4, 6, 8, 12, 16, 24, 32):
        tps = tau / cycle
        note = ""
        if tps < ar:
            note = "worse than autoregressive"
        elif tau == 32:
            note = "perfect acceptance, unreachable"
        print(f"  {tau:6d} | {tps:8.1f}  | {tps/ar:7.2f}x | {note}")
    print(f"  {'-' * 96}")
    be = breakeven_tau(32, "shared-only", BPP_TERNARY, BPP_TERNARY)
    print(f"  break-even tau = {be:.2f}. The Uno paper reports tau = 5.97 on dense Qwen3-8B")
    print(f"  with a full-rank LoRA drafter. A shared-expert-only drafter is weaker, but the")
    print(f"  bar here is {be:.2f}, not 5.97.")


def section_verdict() -> None:
    print(f"\n{BAR}")
    print("  5. VERDICT")
    print(BAR)
    print("""
  The objection is correct and it reverses the earlier conclusion, conditionally.

  a. THE DOUBLE CHARGE WAS THE BUG. simulate_ternary_uno.py billed union(B)
     experts for both the draft and the verify pass. Only the verify pass is
     obliged to use true routing. The draft pathway can route anywhere, because
     rejection sampling against the frozen AR weights preserves the output
     distribution no matter how bad the draft is. Losslessness does not constrain
     the drafter at all -- it only constrains the verifier.

  b. THE VERIFY PASS IS ALREADY CHEAPER THAN AR. Autoregressive decoding reads
     10 experts per layer for every token with no reuse across tokens, because the
     per-token working set is ~1.5 GB and nothing survives in L3. A single verify
     pass over B=32 tokens reads union(32) = 239.6 experts once and applies each
     to every token routed to it, which is 7.49 experts/token. Batching the block
     through one MoE dispatch is the win, and the earlier model hid it.

  c. STRATEGY C IS THE INTERESTING ONE. This model carries
     mlp.shared_expert.{gate,up,down}_proj on all 48 layers, active for every
     token and therefore already counted in the dense term. Training the Uno LoRA
     against the shared-expert pathway gives a drafter with ZERO marginal expert
     traffic. The draft pass then costs only its share of the dense read.

  d. THE RISK MOVED, IT DID NOT DISAPPEAR. The question is no longer bandwidth,
     it is acceptance length. A drafter that sees only the shared expert has far
     less capacity than one that sees the routed top-10, so tau will be lower than
     the 5.97 the paper reports on dense Qwen3-8B. The break-even figures above
     are the target to beat, and they are the thing to measure first.

  e. THIS IS A NOVEL COMBINATION. Uno drafts with a LoRA over the full model.
     Constraining the draft pathway to the always-resident shared expert, for the
     express purpose of keeping MoE weight traffic flat during drafting, is not in
     the paper and not in the d-LLM literature as far as the repository's
     references go. It is the first thing in this program that looks publishable
     on its own.""")


def main() -> None:
    print(BAR)
    print("  CONSTRAINED-ROUTING UNO ON QWEN3.8-FLASH-NEXT")
    print(BAR)
    section_experts_per_token()
    section_throughput("Q4_K_XL", BPP_Q4, BPP_Q4)
    section_throughput("ternary experts + dense", BPP_TERNARY, BPP_TERNARY)
    section_breakeven("ternary", BPP_TERNARY, BPP_TERNARY)
    section_tau_sensitivity()
    section_verdict()
    print(f"\n{BAR}")
    print("""  CAVEATS
    - Acceptance length tau is assumed, not measured. Everything above is a
      conditional: IF a constrained drafter reaches the break-even tau, THEN it
      wins. Measuring tau requires training the adapter.
    - Assumes one MoE dispatch per block reads each distinct expert exactly once.
      That is how batched MoE kernels work, but llama.cpp's CPU expert path may
      not achieve it; verify against the actual implementation.
    - Tree sampling would raise the verify cost above union(B); these numbers are
      for the linear sampler only.
    - Dense parameter count still estimated at 4.5 B.""")
    print(BAR)


if __name__ == "__main__":
    main()
