# Diffusal

**Do diffusion LLMs tolerate extreme weight quantization better than autoregressive LLMs?**
Measured on matched pairs — same authors, data, tokenizer, scale, recipe — with
pre-registered thresholds ([timestamped](https://github.com/wfzyx/diffusal/releases)
*before* each run). All experiments on a single RTX 2080 Super (8 GB).

## Results so far

The publishable finding, with caveats and limitations: [ARXIV-PAPER.md](ARXIV-PAPER.md).

![AR vs dLLM under extreme quantization — INT8/INT4 likelihood, generative anchor, and native ternary QAT gap-of-gaps](assets/bench-ar-vs-dllm.png)


**Exp 0 — post-training quantization, likelihood** (dLLM: MDLM-owt 130M vs its
matched AR baseline; damage relative to each model's own FP16):

| precision | AR degradation | dLLM degradation | excess (dLLM − AR) |
|---|---:|---:|---:|
| INT8 | ~0% | +0.3–0.6% | ≤ +0.7 pp |
| INT4 | +25–32% | +9–21% | **−11 to −16 pp** |
| ternary | ~10⁴× fp16 | ~10¹²× fp16 | uninformative (pre-registered) |

→ Pre-registered **`dllm_more_robust`** verdict fired: the diffusion model
absorbs INT4 weight noise roughly **2× better** than its AR twin, unanimous
across PTB / Wikitext103 / LAMBADA.

**Exp 0 — generative anchor** (64 samples/condition, GPT-2-large oracle,
entropy guard passing): INT4 costs the AR model **+96%** generative perplexity
vs **+45%** for the dLLM. Ternary-PTQ models fail to produce a single valid
sample. Corroborates the likelihood result — not a perplexity artifact.

**Exp 2 — native ternary QAT replication** (12 matched tiny models trained
from scratch, BitNet-style STE, identical recipe, three seeds):

| metric | result |
|---|---:|
| AR ternary tax | +16.1% to +30.7% |
| dLLM ternary tax | +4.2% to +15.4% |
| gap ratio R (dLLM tax / AR tax) | 0.883 to 0.898 |
| geometric mean R (95% t CI) | **0.890 [0.872, 0.908]** |

→ The pre-registered **no-extra-dLLM-tax** criterion fires (upper CI 0.908
≤ 1.25). The stronger pre-registered `dllm_better` criterion does not: its
required upper CI below 0.80 was not met. This is a three-seed, 7M-scale,
Wikitext103 feasibility result—not evidence of superior diffusion tolerance.

Details, caveats, raw values, and checkpoint hashes:
[experiments/exp0/RESULTS.md](experiments/exp0/RESULTS.md),
[experiments/exp2/RESULTS-replication.md](experiments/exp2/RESULTS-replication.md).

## Why this might happen

At a fixed position and commit set, a sub-margin logit perturbation can leave
an argmax token unchanged. That limited identity preservation is not general
error absorption: in 96 native commit-and-freeze trajectories, changing eight
fixed tokens caused 65.7% FP16 and 66.3% ternary-QAT later-token disagreement;
the QAT-minus-FP16 interval was [-2.2, +3.3] points. A wrong committed token
remains conditioning context. See
[the exploratory result](experiments/exp2/RESULTS-contractivity.md).
This concerns the local commit-and-freeze sampler, not DiffusionGemma: its
entropy-bound sampler re-noises non-accepted canvas tokens and recomputes
acceptance each step. A custom post-hoc revisable-remasking probe also failed
to absorb perturbations (~98% disagreement), but it was not trained under that
sampler and is not evidence about DiffusionGemma. Next: train and evaluate a
matched native revisable model under one frozen corruption/sampler pair.

## The program

Measurement-then-build, defined in the
[whitepaper](whitepaper/diffusal-whitepaper.pdf) ([LaTeX](whitepaper/diffusal-whitepaper.tex))
and [PHD-THESIS.md](PHD-THESIS.md):

1. **Exp 0 (done):** PTQ degradation with matched AR controls — the *excess*
   is the only admissible evidence about diffusion.
2. **Exp 1 (two exploratory sampler conditions):** native commit-and-freeze
   and a post-hoc custom revisable sampler both propagated perturbations, with
   no resolved QAT effect. Next: train a matched native revisable condition.
3. **Exp 2 (three-seed replication done):** native ternary QAT with matched
   AR controls — the gap-of-gaps supports no extra dLLM tax at toy scale, not
   superior tolerance. End goal: a natively ternary (1.58-bit) masked
   diffusion model via AR-to-diffusion conversion at ~2–3B, where BitNet says
   ternary reaches parity.

Every experiment carries pre-registered thresholds and kill criteria
([configs/](configs/)), frozen and tagged before runs; amendments are new
tagged commits with justification.

## What this is not

Not "a 25B ternary dLLM in 6 GB" — activations, logits, and buffers dominate
at small scale; in the current QAT checkpoint only 21.5% of model-state
parameters are ternarized, for an idealized weight-only reduction of 1.23×.
Not a claim about 7B+ models. Not calibrated PTQ — naive RTN by design, so the
AR control carries the comparison.

## Reproduce

`experiments/exp0/` (PTQ grid + generative anchors) and `experiments/exp2/`
(QAT pilot) contain drivers, sanity tests, and resumable grid scripts; both
run against pinned [bd3lms](https://github.com/kuleshov-group/bd3lms) with an
sm75 flash-attn→SDPA shim (`experiments/exp0/shim/`, numerically validated).
See `experiments/exp0/PREP.md` for the full bring-up log, including why
zero-shot numbers must be reproduced through bd3lms (`insert_valid_eos=False`,
per [mdlm#22](https://github.com/kuleshov-group/mdlm/issues/22)).

## Non-obvious facts already established

- LLaDA/Dream "low-confidence remasking" is commit-and-freeze: committed
  tokens are frozen forever. True remasking needs ReMDM-style retrofits.
- The MDLM release ships a matched AR baseline — the confound-free control
  comes for free; its published zero-shot numbers reproduce only under the
  bd3lms eval protocol.
- Ternary PTQ fails on every architecture (BitNet's premise) — only *native*
  ternary training is informative, and no dLLM QAT existed in the literature
  as of 2026-07 (verified sweep in whitepaper §2).
