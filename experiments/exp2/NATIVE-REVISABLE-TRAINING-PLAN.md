# Next step: matched native revisable training

## Why

The completed native commit-and-freeze probe (~66% final downstream disagreement) and the post-hoc revisable-remasking retrofit (~98%) show that neither sampler rule automatically absorbs a fixed-token perturbation. The retrofit is not decisive because the checkpoints were trained for the monotonic transition rule.

The next valid mechanism experiment is a dLLM **trained and evaluated under the same revisable corruption and inference transition**.

## Scope

This is a 7M-scale mechanism test, not a DiffusionGemma reproduction and not a new AR-vs-dLLM quality claim. Train dLLM FP16 and dLLM ternary-QAT only, with three matched seeds: six runs. The existing AR controls and completed QAT loss result remain unchanged.

## Freeze before training

1. Implement one categorical corruption process that mixes masking with random-token replacement, and train the denoiser to reconstruct the original token from that exact corruption.
2. Implement its matching reverse sampler: at every update, recompute confidence/acceptance and re-noise non-accepted non-anchor tokens using the same categorical noise family.
3. Fix model size, data, token budget, optimizer, QAT exclusions, seeds (1/2/3), number of updates, acceptance schedule, and random-token distribution in a versioned config.
4. Add CPU tests for: corruption/reverse transition compatibility, fixed-anchor preservation, non-monotonic acceptance, and paired-RNG reproducibility.
5. Run two one-trajectory sampler gates after training one FP16 and one QAT seed-1 checkpoint. Stop if the sampler cannot terminate, changes anchors, or lacks paired reproducibility.

## Measurements

For each trained checkpoint, run the existing 16 paired fixed-token trajectories and report final Hamming, maximum Hamming, trajectory AUC, and raw trajectories. The QAT-minus-FP16 comparison uses the three training-seed means; sampler trajectories quantify only within-checkpoint uncertainty.

## Interpretation

- Lower propagation than the completed monotonic condition is evidence that this trained revisable transition is less sensitive to this perturbation.
- Equal or higher propagation means revision did not create correction under this transition.
- No result may be described as DiffusionGemma behavior without running DiffusionGemma itself.
- This experiment cannot rescue the broad ``diffusion absorbs token errors'' claim; it can only test a specified trained transition mechanism.

## Resource estimate

Six 7M runs at the completed recipe's observed duration are approximately 18--24 single-GPU hours, plus sampler work. Start only after the corruption/reverse transition and tests are frozen; sampler code is the present gating work.
