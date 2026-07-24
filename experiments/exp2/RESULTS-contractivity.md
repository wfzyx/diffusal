# Exploratory native state-perturbation result

This is exploratory, post hoc, and separate from the preregistered validation-loss result. It tests propagation of synthetic fixed-token perturbations under the existing `semi_ar`/`first_hitting` sampler; it is not a general estimate of dLLM contractivity or generation quality.

## Design

For each matched dLLM FP16/QAT checkpoint pair and each of 16 deterministic sampler seeds, two 512-token states begin fully masked except for BOS and eight valid fixed token IDs. The anchor positions and IDs vary by trajectory but are identical within its FP16/QAT pair; only the eight token IDs differ between the paired states. Each update resets RNG identically for the two branches. The metric is Hamming disagreement outside BOS and the eight injected positions.

Every trajectory runs 503 updates: one of the 503 mutable positions is committed per `first_hitting` update.

## Results

| training seed | FP16 final disagreement, mean ± SD | ternary final disagreement, mean ± SD | ternary − FP16 |
|---:|---:|---:|---:|
| 1 | 0.654 ± 0.088 | 0.665 ± 0.094 | +0.011 |
| 2 | 0.665 ± 0.106 | 0.658 ± 0.117 | -0.007 |
| 3 | 0.653 ± 0.057 | 0.666 ± 0.078 | +0.013 |
| mean of seed means | 0.657 | 0.663 | +0.006 |

The across-training-seed mean QAT-minus-FP16 effect is **+0.006**, with a two-sided 95% t-interval of **[-0.022, +0.033]** (`n=3` matched training seeds). The 16 sampler trajectories quantify within-checkpoint stochasticity; they are not 16 independent trained models.

| update | FP16 mean disagreement | ternary mean disagreement |
|---:|---:|---:|
| 64 | 0.053 | 0.057 |
| 128 | 0.121 | 0.125 |
| 256 | 0.282 | 0.288 |
| 384 | 0.468 | 0.474 |
| 503 | 0.657 | 0.663 |

## Interpretation

The trajectory grows steadily rather than reconverging. Under this native commit-and-freeze sampler, a fixed-token perturbation commonly changes roughly two thirds of subsequent token identities. Therefore the broad mechanism claim that token commitment automatically absorbs such errors is unsupported.

There is no resolved FP16-versus-QAT propagation difference: the training-seed interval spans a 2.2-point QAT reduction through a 3.3-point increase. This does not contradict the separate three-seed validation result that dLLM had a lower relative ternary loss tax than AR; it says that the fixed-token-propagation mechanism does not explain that result.

A genuine-remasking sampler is a separate experimental condition. It may change this behavior, but it must be trained/evaluated under its own frozen protocol rather than retroactively used to explain this result.
