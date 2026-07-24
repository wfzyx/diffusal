# Exploratory revisable-remasking result

This post-hoc experiment evaluates the frozen protocol in `REVISABLE-SAMPLER-PROBE-PLAN.md`. It uses the existing BD3LMS checkpoints with a **custom revisable-remasking sampler**; it is not a DiffusionGemma experiment and cannot be used to infer DiffusionGemma behavior.

## Design

For each of the six dLLM checkpoints, 16 paired 512-token trajectories start with identical masks and eight differing fixed anchors. At update $k$, the sampler samples a candidate token for each mutable position, retains the $k$ highest-max-probability candidates, and re-masks every other mutable position. Acceptance is recomputed from scratch at every update. The paired branches use identical RNG and 503 model-forward updates. The metric excludes BOS and anchors.

## Results

| training seed | FP16 final disagreement, mean ± SD | ternary final disagreement, mean ± SD | ternary − FP16 |
|---:|---:|---:|---:|
| 1 | 0.978 ± 0.011 | 0.980 ± 0.008 | +0.003 |
| 2 | 0.981 ± 0.008 | 0.981 ± 0.010 | +0.000 |
| 3 | 0.971 ± 0.021 | 0.981 ± 0.007 | +0.010 |
| mean of seed means | 0.976 | 0.981 | +0.004 |

The QAT-minus-FP16 effect is **+0.004**, with a two-sided three-training-seed 95% t-interval of **[-0.008, +0.017]**. There is no resolved QAT effect.

| update | FP16 mean disagreement | ternary mean disagreement |
|---:|---:|---:|
| 64 | 0.209 | 0.205 |
| 128 | 0.402 | 0.391 |
| 256 | 0.712 | 0.706 |
| 384 | 0.912 | 0.924 |
| 503 | 0.976 | 0.981 |

For comparison, the completed native commit-and-freeze condition ended at 0.657 FP16 and 0.663 QAT. This custom revisable condition ends about 0.32 higher.

## Interpretation

Revision/remasking did **not** reduce perturbation propagation here; it increased it to near-complete downstream divergence. This is not evidence that revisable diffusion is intrinsically less stable. The sampler was introduced after training, uses masking rather than DiffusionGemma's random-token re-noising, and forces a changing acceptance set under checkpoints trained/evaluated with the native monotonic sampler.

The reliable conclusion across both sampler conditions is narrower: neither permanent commitment nor this custom revision policy automatically absorbs a fixed-token perturbation, and neither produced a measurable ternary-QAT propagation difference. The lower QAT validation-loss tax therefore remains mechanistically unexplained by these probes.

A claim about revisable diffusion requires a model trained and evaluated with the same revisable sampler, ideally an actual DiffusionGemma-compatible experiment or a matched native-remasking training run.
