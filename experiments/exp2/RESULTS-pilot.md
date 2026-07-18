# Exp 2 pilot results — native ternary QAT gap-of-gaps (toy scale)

Run 2026-07-16 → 2026-07-18 under the frozen pre-registration
(`prereg-exp2-pilot-2026-07-16.1`, configs/exp2-pilot.yaml). Four matched
runs: {AR, dLLM (masked diffusion)} x {fp16, ternary-QAT}, bd3lms model=tiny
(~7M non-embedding params), wikitext103, ~196M tokens, global batch 64,
seed 1, identical recipe and STE wrapper (experiments/exp2/qat.py) for both
architectures; embeddings/vocab head excluded from ternarization.

## Best validation perplexity (wikitext103; dLLM value is its NELBO bound)

| model | fp16 | ternary | ternary/fp16 gap |
|---|---:|---:|---:|
| AR   | 50.77 | 60.12 | **1.184** (+18.4%) |
| dLLM | 97.97 | 103.05 | **1.052** (+5.2%) |

**Ratio of gaps R = 1.052 / 1.184 = 0.888**

## Verdict against frozen thresholds

- R = 0.888 falls in the pre-registered `comparable` band [0.8, 1.25]:
  **diffusion pays no extra ternary tax at this scale** — and the point
  estimate sits near the `dllm_better` boundary (0.8), directionally
  consistent with Exp 0's INT4 finding that the dLLM absorbs weight noise
  better than its AR twin.
- Both ternary runs trained stably to 6000 steps at the shared recipe
  (no divergence), so the non-claims clause about unstable cells does not
  apply.
- Also notable: the AR ternary gap itself is modest (+18%), not
  catastrophic — consistent with Nielsen et al. (arXiv:2411.05882) that
  small-scale ternary QAT failure should not be assumed.

## Caveats (pre-registered)

- **Single seed.** This is a direction for the seeded Exp 2, not a claim.
  The dLLM's val metric is an MC-estimated bound; between-architecture gap
  comparisons inherit that asymmetry.
- Toy scale (~7M non-embedding), one dataset, one token budget. The
  absolute dLLM-vs-AR difference (97.97 vs 50.77) reflects bound-vs-exact
  and diffusion's known small-scale disadvantage, and is NOT the object —
  only the within-architecture ternary/fp16 ratios are compared.
- Training interrupted/resumed several times (disk pressure, machine
  sharing); resume is checkpoint-faithful (seed/optimizer/EMA restored).

## Routing

Per the frozen config: proceed. The end-goal build (A2D conversion +
ternary QAT distillation at 2-3B) inherits a green light from both Exp 0
(PTQ excess) and this pilot (QAT gap-of-gaps). Next: seeded (>=3) Exp 2
cells and the Exp 1 contractivity program (ReMDM).
