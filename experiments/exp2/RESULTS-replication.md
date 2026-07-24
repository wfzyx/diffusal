# Exp 2 replication results — native ternary QAT

Run under the frozen `prereg-exp2-replication-2026-07-21` tag (`caedf937698c984abf27e1dcb68773ee2363a6ef`). This is the three-seed extension of `RESULTS-pilot.md`: matched tiny bd3lms AR and masked-diffusion models, Wikitext103, 6,000 steps, global batch 64, and BitNet-style weight-only ternary QAT. The dLLM metric is its validation NELBO perplexity bound.

## Raw best validation results

| seed | AR FP16 PPL | AR ternary PPL | AR tax | dLLM FP16 PPL | dLLM ternary PPL | dLLM tax | R |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50.77 | 60.12 | 1.184 (+18.41%) | 97.97 | 103.05 | 1.052 (+5.19%) | 0.888 |
| 2 | 49.97 | 65.31 | 1.307 (+30.71%) | 96.02 | 110.83 | 1.154 (+15.43%) | 0.883 |
| 3 | 51.39 | 59.67 | 1.161 (+16.12%) | 106.87 | 111.39 | 1.042 (+4.23%) | 0.898 |

`tax = PPL(ternary) / PPL(FP16)` and `R = tax(dLLM) / tax(AR)`. Values derive from the final `val/nll` checkpoint-monitor events in the twelve per-cell logs; PPL is `exp(val/nll)`.

## Prespecified aggregate

The matched unit is a training seed. On `log(R)`, the geometric mean is **0.890** (SD 0.00824); the two-sided 95% t-interval (`df=2`, `t*=4.303`) is **[0.872, 0.908]** after exponentiation. The arithmetic mean R is 0.890.

| criterion from `configs/exp2-replication.yaml` | result |
|---|---|
| no extra dLLM ternary tax: upper 95% CI(R) ≤ 1.25 | **passes** (0.908) |
| lower dLLM ternary tax: upper 95% CI(R) < 0.80 | does not pass |
| extra dLLM ternary fragility: lower 95% CI(R) > 1.25 | does not pass |

All four cells completed for all three seeds; neither ternary condition diverged. The result supports the scoped feasibility claim: the dLLM paid no extra ternary tax in this matched 7M-scale Wikitext103 setup. It does **not** meet the preregistered criterion for claiming that diffusion pays a smaller ternary tax.

## Reproducibility record

- Training revision/tag: `caedf937698c984abf27e1dcb68773ee2363a6ef` / `prereg-exp2-replication-2026-07-21`
- Environment: PyTorch 2.13.0+cu130; Lightning 2.6.5; CUDA 13.0; NVIDIA RTX 2080 SUPER; driver 595.80.
- Frozen design: `configs/exp2-replication.yaml`; recipe: `configs/exp2-pilot.yaml`.
- Per-cell commands and monitor events: `experiments/exp2/logs/`.
- SHA-256 hashes of the twelve selected best checkpoints: `CHECKPOINTS-replication.sha256`.
- Seeds 2 and 3 ran via `run_replication_seeds.sh`; seed 1 is the unchanged completed pilot.

## Generative-quality anchor

No final-checkpoint generative evaluation was implemented and frozen before this cohort began. Do not retrospectively describe one as preregistered. If added, it must be a separately versioned, explicitly exploratory protocol and be reported as such; the validation gap-of-gaps result above stands independently.
