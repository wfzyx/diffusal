# Native state-perturbation smoke result

This is an exploratory validation of the probe in `CONTRACTIVITY-PROBE.md`, not a preregistered result and not a contractivity estimate.

## Setup

For each of the three FP16/QAT dLLM checkpoint pairs, one pair of 512-token masked states was sampled with the native `semi_ar`/`first_hitting` sampler. Eight valid token IDs and positions were deterministically drawn for each seed; only those IDs differ in the paired states. All other initial tokens and the per-update random variates were identical. Hamming disagreement excludes BOS and the eight injected positions.

Each chain completed after 503 updates: the sampler unmasks one of the 503 mutable positions per update.

| seed | FP16 final non-anchor Hamming | ternary final non-anchor Hamming | ternary − FP16 |
|---:|---:|---:|---:|
| 1 | 0.716 | 0.646 | -0.070 |
| 2 | 0.555 | 0.421 | -0.133 |
| 3 | 0.712 | 0.748 | +0.036 |
| mean | 0.661 | 0.605 | -0.056 |

The trajectory rises rather than contracts. Averaged across the six smoke trajectories, non-anchor Hamming is 0.045 at update 64, 0.116 at 128, 0.271 at 256, 0.451 at 384, and 0.633 at completion.

## Interpretation

A small fixed-token perturbation causes substantial downstream divergence under this sampler. This rejects the simplistic mechanism claim that discrete commitment automatically ``absorbs'' a fixed-token error. The result is consistent with commit-and-freeze: an incorrect committed token remains in context and can influence future denoising.

The smoke is not evidence of a QAT propagation effect. Its mean QAT-minus-FP16 difference is -5.6 percentage points, but the three effects range from -13.3 to +3.6 points and each has one trajectory. It establishes that the instrumentation works and that the full run must report propagation, not assume contraction.

## Next

Run 16 independently seeded trajectories for each of the six checkpoints, summarize per-checkpoint means and bootstrap intervals, and compare the QAT-minus-FP16 effect across the three matched seeds. Keep the claim scoped to this synthetic fixed-token perturbation and sampler.
