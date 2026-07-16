# Exp 0 results — naive PTQ with matched AR control

Run 2026-07-16 under the frozen pre-registration (`prereg-exp0-2026-07-16`,
configs/exp0.yaml). Harness: bd3lms @ 1c3e8f4, `+data.insert_valid_eos=False`,
eval_batch_size 2, length 1024, identical command and quantizer code path for
both models. Quantized: all transformer nn.Linear weights (70.4% of linear
params); excluded per config: vocab embedding, output head, layernorms, biases.
Raw logs: results/quant/ (gitignored); FP16 references: results/fp16-bd3lms/.

## Perplexities

| model | precision | PTB | Wikitext103 | LAMBADA |
|---|---|---:|---:|---:|
| AR   | fp16    | 80.75 | 25.61 | 52.99 |
| AR   | int8    | 80.72 | 25.67 | 52.98 |
| AR   | int4    | 106.43 | 32.49 | 66.12 |
| AR   | ternary | 4.5e6 | 1.8e6 | 1.7e6 |
| dLLM | fp16    | 82.40 | 30.67 | 49.12 |
| dLLM | int8    | 82.92 | 30.84 | 49.25 |
| dLLM | int4    | 99.58 | 34.23 | 53.55 |
| dLLM | ternary | 3.8e14 | 5.0e14 | 9.2e13 |

## Pre-registered quantities

delta = ppl_q / ppl_fp16 − 1;  excess = delta(dLLM) − delta(AR)

| precision | dataset | delta AR | delta dLLM | excess |
|---|---|---:|---:|---:|
| int8 | PTB         | −0.03% | +0.62% | +0.65 pp |
| int8 | Wikitext103 | +0.21% | +0.56% | +0.35 pp |
| int8 | LAMBADA     | −0.03% | +0.27% | +0.30 pp |
| int4 | PTB         | +31.81% | +20.85% | **−10.97 pp** |
| int4 | Wikitext103 (primary) | +26.84% | +11.59% | **−15.25 pp** |
| int4 | LAMBADA     | +24.77% | +9.02% | **−15.74 pp** |

## Verdict against frozen thresholds (configs/exp0.yaml)

- **INT8**: |excess| ≤ 0.65 pp everywhere — well inside the ±5 pp band.
- **INT4, primary metric (Wikitext103)**: excess = −15.25 pp ≤ −5 pp →
  **`dllm_more_robust` fires.** Direction is unanimous across all three
  datasets. The dLLM loses 9–21% from INT4 where its matched AR control
  loses 25–32%.
- **Ternary**: both models catastrophically fail (AR ~10^4×, dLLM ~10^12× fp16
  ppl) → pre-registered `expected_uninformative`; per §4.1 no conclusion about
  diffusion is drawn. The trend-signal clause (both excesses positive) does not
  apply. Side observation, hypothesis-generating only: the dLLM's ternary
  collapse is ~8 orders of magnitude deeper than the AR control's.
- Routing per the frozen config: **proceed toward the build; the science
  re-scopes toward the absorption mechanism** (why is iterated denoising
  MORE tolerant of INT4 weight noise than single-pass AR at matched scale,
  data, tokenizer, recipe?).

## Generative anchor (quality guard for the likelihood results)

64 unconditional samples per condition, length 512, nucleus 0.9, seed 1,
scored offline with the pre-registered gpt2-large oracle (score_samples.py).
Deviations from the frozen config, forced by the 8 GB card and applied
identically to both models: length 512 (not 1024), 64 samples x 1 seed
(not 512 x 3). Corpus unigram entropy stays 9.4–9.6 bits everywhere sampling
succeeds — no mode collapse, so gen-ppl comparisons are honest.

| precision | AR gen-ppl | AR delta | dLLM gen-ppl | dLLM delta | excess |
|---|---:|---:|---:|---:|---:|
| fp16    | 14.64 | — | 33.87 | — | — |
| int8    | 14.79 | +1.0% | 37.42 | +10.5% | +9.5 pp |
| int4    | 28.72 | **+96.2%** | 49.25 | **+45.4%** | **−50.8 pp** |
| ternary | sampler guard trips | — | sampler guard trips | — | — |

- **INT4 anchor agrees with (and amplifies) the likelihood finding**: the AR
  control's generation quality roughly halves (+96% gen-ppl) while the dLLM
  degrades +45%. Per the frozen quality-anchor clause, the likelihood result
  is corroborated, not fidelity-only.
- **INT8 nuance**: the dLLM's anchor delta (+10.5%) exceeds its likelihood
  delta (+0.6%); AR is unchanged. Single-seed, 64 samples — noted as a
  question for the seeded rerun, not a finding.
- **Ternary**: both models fail bd3lms' sampling-validity guard outright —
  ternary-PTQ models cannot produce a single valid sample. Consistent with
  the likelihood blowup; uninformative about diffusion per pre-registration.
- AR fp16 gen-ppl 14.64 closely matches the BD3-LM paper's published AR
  value (14.1 at length 1024) — the anchor pipeline reproduces.

## Caveats (stated in advance in exp0.yaml)

- The dLLM likelihood is an MC-estimated NELBO upper bound; same-seed
  same-protocol deltas cancel most scatter, but bound-vs-exact asymmetry
  vs the AR control is inherent to the design.
- Naive RTN only; no calibration, no activation quantization. INT4 numbers
  are a floor, not a statement about good INT4 PTQ.
- Single seed for the likelihood evals (deterministic given harness seed);
  generative-anchor metrics (gen ppl under GPT-2-large + entropy guard) are
  the next required output before publicizing conclusions.

## Peak VRAM (validation, MiB allocated)

From the [vram] audit lines: dLLM ~2.3 GB at batch 2 / length 1024 — the
fp32 logits buffer over the 50k vocab dominates, consistent with §5 of the
whitepaper (weights-only compression is the wrong lever at this scale).
