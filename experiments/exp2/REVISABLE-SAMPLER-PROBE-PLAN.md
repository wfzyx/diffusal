# Next step: revisable-sampler perturbation probe

## Question

Does allowing tokens to be revisited change fixed-token perturbation propagation, and does ternary QAT change that effect?

This follows the completed native `semi_ar`/`first_hitting` commit-and-freeze probe. It is a new exploratory experiment; it cannot reinterpret the completed result.

## Scope boundary

The local model is not DiffusionGemma. A sampler adapted from its entropy-bound re-noising idea must be called **DiffusionGemma-inspired**, not DiffusionGemma. DiffusionGemma's actual sampler re-noises non-accepted canvas tokens and recomputes acceptance each step; completed 256-token blocks are subsequently fixed context.

## Frozen design before execution

1. Keep the six existing dLLM checkpoints and the same 16 paired trajectories per checkpoint.
2. Keep the eight paired anchors, maximum sequence length, per-step paired RNG, and non-anchor Hamming metric from `CONTRACTIVITY-PROBE.md`.
3. Add exactly one revisable sampler condition. At update $k$, score each mutable position by maximum predicted-token probability, sample candidates, retain the top $k$ positions, and re-mask every other non-anchor position. Acceptance is recomputed from scratch each update; no position is permanently accepted before the final step. Masking is the native model's forward-noise distribution, so this is a revisable-remasking condition, not a literal DiffusionGemma reproduction.
4. Match the native probe's 503 model-forward-update budget. Report the actual count if early termination differs.
5. Primary quantities: final Hamming, maximum Hamming, and trajectory area under the disagreement curve. Compare QAT minus FP16 within each training seed, then use the three training-seed means as the inferential units.
6. Report every trajectory, sampler settings, and failures. Do not tune the re-noising schedule after inspecting these outputs.

## Interpretation rules

- Lower propagation than commit-and-freeze suggests revision reduces sensitivity for this model/sampler pair; it does not establish a DiffusionGemma result.
- No reduction or higher propagation means revision is not an automatic error-correction mechanism.
- A QAT difference is unresolved unless its three-seed interval excludes zero.

## Gate

Implement and validate the sampler on two paired trajectories (one FP16, one QAT) before launching the full 96-trajectory run. If the adapted sampler does not preserve anchors, uses a different model-forward budget, or cannot produce deterministic paired RNG, stop and revise the protocol rather than compare it to the completed native result.
