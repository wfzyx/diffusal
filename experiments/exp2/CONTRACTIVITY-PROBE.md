# Exploratory native state-perturbation probe

This is exploratory and does not alter the preregistered validation result.

For each of the three dLLM FP16/QAT checkpoint pairs, create paired masked states with the same eight fixed-token positions. Positions and valid token IDs are deterministically resampled for each trajectory; the two states differ only in those eight token IDs. Run the existing `semi_ar`/`first_hitting` sampler with identical per-step RNG resets for the pair. At every sampler update, record Hamming disagreement outside the eight injected positions, plus remaining mask counts.

The primary descriptive outputs are the mean non-anchor disagreement trajectory, its maximum, and its final value, separately for FP16 and ternary-QAT. This measures propagation of a fixed-token perturbation within each native model. It is not a cross-model trajectory comparison and does not establish global contractivity.

Initial smoke: one trajectory per checkpoint. Full exploratory run: 16 trajectories per checkpoint. The FP16/QAT runs use the same anchors, initial states, per-step RNG, maximum 512 updates, and checkpoint selection.
