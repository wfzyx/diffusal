# Exp 2 replication plan

## Purpose

Turn the native ternary-QAT pilot into a portfolio-grade, scoped result. The claim under test is not that all diffusion models absorb ternary error. It is: **at this matched tiny-model setup, a dLLM pays no larger ternary validation-loss tax than its AR control; a consistently smaller tax would support the stronger claim.**

## Fixed cohort

The cohort is three independent training seeds: `1`, `2`, and `3`.

- Seed `1` is the completed frozen pilot (`prereg-exp2-pilot-2026-07-16.1`) and is reused unchanged.
- Seeds `2` and `3` run the same four cells: `{AR, dLLM} × {FP16, ternary-QAT}`.
- No architecture, data, token budget, QAT wrapper, optimizer, schedule, evaluation cadence, or selection rule may change between seeds.
- Any necessary change creates a new experiment/configuration; it does not silently join this cohort.

## Before training

1. Review `configs/exp2-replication.yaml` and commit/tag it before starting seed 2.
2. Run `experiments/exp2/test_qat.py` in the pinned Exp 0 environment.
3. Record the repository revision, CUDA/PyTorch versions, free disk space, and GPU model in the run log.
4. Confirm each seed has an isolated run directory and log.

## Training

Run `experiments/exp2/run_replication_seeds.sh`. It trains only seeds 2 and 3; seed 1 remains the completed pilot. Each cell is resumable from its own `runs/seed-<n>/` directory. Do not select a checkpoint by visual inspection: use the same best-validation rule already used by the pilot.

## Analysis

For each seed, compute:

- `gap(model) = best_val_ppl(ternary) / best_val_ppl(fp16)`
- `R = gap(dLLM) / gap(AR)`
- raw FP16 and ternary values for both models.

Report all three seed-level values, their arithmetic mean and standard deviation, and a 95% t-interval for `log(R)`. Exponentiate interval endpoints for presentation. Do not pool individual model runs as independent observations: the matched unit is a seed.

Primary interpretation is fixed before the new runs:

- upper 95% CI for `R` ≤ 1.25: no extra dLLM ternary tax at this scale;
- upper 95% CI for `R` < 0.80: evidence favoring lower dLLM ternary tax;
- lower 95% CI for `R` > 1.25: evidence of extra dLLM ternary fragility;
- otherwise: inconclusive.

Run the locked generative-quality evaluation for every final checkpoint before making a quality claim. It is an anchor, not a substitute for the matched validation-loss analysis.

## Publication bar

Update `RESULTS-pilot.md` into a replication results file with raw per-seed values, commands, checkpoint hashes, failure/restart history, and generation outputs. The paper and README must describe the result as a 7M non-embedding-parameter, Wikitext103, three-seed matched experiment. Do not generalize to larger dLLMs or all diffusion models.
