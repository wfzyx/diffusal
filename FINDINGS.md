# Diffusal — findings summary

Status as of 2026-07-20. Full detail: [experiments/exp0/RESULTS.md](experiments/exp0/RESULTS.md),
[experiments/exp2/RESULTS-pilot.md](experiments/exp2/RESULTS-pilot.md). Chart:
[assets/bench-ar-vs-dllm.png](assets/bench-ar-vs-dllm.png). Pre-registrations:
[prereg-exp0-2026-07-16](https://github.com/wfzyx/diffusal/releases/tag/prereg-exp0-2026-07-16),
[prereg-exp2-pilot-2026-07-16.1](https://github.com/wfzyx/diffusal/releases/tag/prereg-exp2-pilot-2026-07-16.1).

## The question

Does extreme weight quantization (INT8/INT4/ternary) hurt a masked diffusion
LLM (dLLM) more than it hurts an autoregressive LLM (AR) of matched scale,
data, tokenizer, and recipe? All three experiments below compare a dLLM
against a *matched* AR control, quantized/trained identically — only the
**excess** (dLLM degradation minus AR degradation) is treated as evidence
about diffusion itself.

## Results

### 1. Post-training quantization — likelihood (Exp 0)

MDLM-owt 130M vs. its matched AR baseline, naive round-to-nearest PTQ,
degradation relative to each model's own FP16.

| precision | AR degradation | dLLM degradation | excess |
|---|---:|---:|---:|
| INT8 | ~0% | +0.3–0.6% | ≤ +0.7pp |
| INT4 | +25–32% | +9–21% | **−11 to −16pp** |
| ternary | ~10⁴× fp16 | ~10¹²× fp16 | uninformative (pre-registered) |

**Verdict: `dllm_more_robust` fired.** At INT4 the dLLM absorbs weight
quantization noise roughly 2× better than its AR twin, unanimous across
PTB / Wikitext103 / LAMBADA.

### 2. Generative anchor (Exp 0)

64 samples/condition, GPT-2-large oracle, unigram-entropy degeneracy guard
(passed everywhere sampling succeeded).

| precision | AR gen-ppl delta | dLLM gen-ppl delta | excess |
|---|---:|---:|---:|
| INT4 | +96.2% | +45.4% | **−50.8pp** |
| ternary | sampler validity guard trips | sampler validity guard trips | uninformative |

**Verdict: corroborates (1).** Not a perplexity artifact — the AR model's
*generated text* degrades roughly 2× more than the dLLM's at INT4.

### 3. Native ternary QAT — gap-of-gaps (Exp 2 pilot)

Four matched ~7M-param models trained from scratch (AR, dLLM × FP16,
ternary-QAT via BitNet-style straight-through estimator), identical recipe,
single seed, wikitext103.

| model | fp16 val ppl | ternary val ppl | ternary tax |
|---|---:|---:|---:|
| AR | 50.77 | 60.12 | +18.4% |
| dLLM | 97.97 | 103.05 | +5.2% |

**Ratio of gaps R = 0.888** → **Verdict: `comparable`** (pre-registered
band [0.8, 1.25]), sitting near the `dllm_better` boundary. Diffusion pays
no extra ternary tax at this scale when trained natively. Single seed —
a direction, not a claim.

## Why this might be happening

Token commitment in masked diffusion is itself a quantizer: a logit
perturbation smaller than the gap to the runner-up token vanishes entirely
at argmax time. Iterative refinement may be absorbing sub-threshold
quantization noise that an autoregressive model must carry forward on every
token, every time. This is a hypothesis, not yet measured directly — see
"what's next."

## What this does not show

- Nothing about 1B+ models — smallest AR/dLLM pair tested is 130M (Exp 0),
  ternary QAT pair is ~7M (Exp 2 pilot).
- Ternary PTQ conclusions — it fails catastrophically on both architectures,
  exactly as BitNet's own premise predicts; that failure is pre-registered
  as uninformative about diffusion specifically.
- A calibrated-PTQ comparison — naive RTN throughout, by design, so the
  matched AR control (not calibration engineering) carries the claim.
- A trained-ternary claim beyond a single seed.

## What's next

1. **Exp 1 (not started):** perturbation-injection contractivity — inject
   controlled errors mid-generation and measure whether the dLLM's own
   refinement dynamics contracts or amplifies them, as a function of mask
   fraction, under genuine remasking (ReMDM) vs. commit-and-freeze. This is
   the experiment that would confirm or refute the argmax-quantizer
   hypothesis directly, rather than inferring it from PTQ/QAT outcomes.
2. **Seeded Exp 2:** ≥3 seeds per cell, ablation grid (per-module
   ternarization, embeddings-included variant per Nielsen et al. and
   QE-CDLM), to turn the pilot's single-seed direction into a claim.
3. **Publicize:** results-first README and chart are ready; r/LocalLLaMA
   post drafted, scheduled for a weekday morning.

## Platform notes (for anyone reproducing this)

Single RTX 2080 Super (8GB, Turing/sm75 — no FlashAttention-2). Eval/training
run through [bd3lms](https://github.com/kuleshov-group/bd3lms) (not the
`mdlm` repo — its zero-shot eval cannot reproduce the paper's own published
numbers; see [mdlm#22](https://github.com/kuleshov-group/mdlm/issues/22)),
with a numerically-validated flash-attn→SDPA shim (`experiments/exp0/shim/`).
Full bring-up log, dependency pins, and environment gotchas:
[experiments/exp0/PREP.md](experiments/exp0/PREP.md).
