# Masked Diffusion Language Models Absorb Extreme Weight Quantization Better Than Autoregressive Models at Matched Scale

**Working draft — arXiv technical report.** Scope is deliberately narrow: a single, well-controlled empirical finding about *relative* quantization robustness, with matched autoregressive (AR) controls throughout. This paper makes no claim about absolute dLLM quality, scale transfer beyond the tested points, deployment economics, or a mechanism. Those are the subject of a separate program document ([PHD-THESIS.md](PHD-THESIS.md)); nothing here depends on them.

---

## Abstract

We ask whether masked diffusion language models (dLLMs) tolerate extreme weight quantization better than autoregressive (AR) models of matched scale, data, tokenizer, and training recipe. The question is only meaningful against a matched control: extreme post-training quantization (PTQ) fails on *every* architecture, so a dLLM failing in isolation says nothing about diffusion. We therefore report only the **excess** degradation of the dLLM over its AR twin.

Two matched setups. **(1) Post-training quantization at 130M.** MDLM-owt (~130M non-embedding params) and the same release's AR baseline — same authors, data, scale, codebase — are quantized identically. At INT4, the AR model degrades +25–32% in validation likelihood while the dLLM degrades only +9–21%; the excess is **−11 to −16 percentage points**, unanimous across PTB / Wikitext103 / LAMBADA. A generative anchor (64 samples/condition, GPT-2-large oracle) corroborates: AR generated-text perplexity worsens +96.2% at INT4 versus +45.4% for the dLLM (**−50.8pp excess**), ruling out a likelihood-only artifact. **(2) Native ternary QAT at 7M.** Twelve models trained from scratch (AR/dLLM × FP16/ternary-QAT × three seeds) give a geometric-mean gap ratio R = 0.890, 95% CI [0.872, 0.908]: the dLLM pays *no extra* ternary tax, though the stricter "materially better" criterion (upper CI < 0.80) does not fire.

We also report two negative mechanism probes that reject the tempting "token commitment absorbs errors" story: changing eight fixed tokens propagates to ~66% of downstream positions under commit-and-freeze (FP16 and ternary statistically indistinguishable), and a post-hoc revisable retrofit reaches ~98%. The robustness is real and replicated; the obvious explanation for it is not supported.

---

## 1. Introduction

Extreme weight quantization — INT4 and below, down to ternary `{-1, 0, +1}` (BitNet b1.58) — is the main lever for shrinking LLM memory. Autoregressive (AR) decoders are the default target. Masked diffusion language models (MDLM, LLaDA, and block-diffusion variants) generate by iteratively refining a token canvas rather than emitting strictly left to right, and it is not known whether that different computation makes them more or less fragile under aggressive quantization.

This paper isolates one comparison and answers it cleanly:

> At matched scale, data, tokenizer, and recipe, does extreme weight quantization degrade a dLLM *more* than an AR model?

Our answer, at the two scales we can afford to control confound-free, is **no — and at INT4 the dLLM is clearly more robust.** We are explicit about what this does and does not establish (§6).

**Contributions.**
1. A matched 130M INT4 PTQ result showing the dLLM absorbs ~2× less degradation than its AR twin, unanimous across three datasets and corroborated by a generative anchor.
2. A three-seed native ternary-QAT result at 7M showing no extra diffusion tax (R = 0.890).
3. Two honest negative mechanism probes that reject "commitment = error absorption," narrowing the space of viable explanations.
4. A full reproducibility release (configs, checkpoint hashes, pre-registrations).

## 2. Related Work

**Quantization of diffusion.** Timestep-aware quantization for *continuous image* diffusion (PTQ4DM, Q-Diffusion, TDQ) established that quantization error accumulates non-uniformly across denoising steps and that step-dependent precision is the mitigation. We do not re-derive this. Our setting is discrete *text* diffusion, where the per-step operation includes an **argmax commitment** with no continuous analogue. DLLMQuant studies PTQ for diffusion LLMs specifically; we differ by insisting on a matched AR control so the reported quantity is diffusion-specific excess, not dLLM degradation in isolation.

**Low-bit LLMs.** BitNet b1.58 argues native ternary training is competitive from ~3B parameters upward; below that, ternary underperforms FP16 *for reasons unrelated to diffusion*. This is exactly why our QAT result reports a gap-of-gaps, not an absolute number.

**Masked diffusion LMs and samplers.** MDLM and LLaDA make iterative text denoising a real compression target. We note (and use) that the common "remasking"-named samplers of LLaDA/Dream are in fact commit-and-freeze; genuine remasking exists as an inference-time retrofit (ReMDM). Sampler family is tracked with every trajectory claim.

## 3. Setup

**Matched-control principle.** Ternary/INT4 PTQ fails on all architectures, so we never report dLLM degradation alone. The evidence is always **excess = (dLLM quantized − dLLM FP16) − (AR quantized − AR FP16)** at identical precision, scale, data, tokenizer, and recipe.

**PTQ pair (130M).** Primary dLLM: `kuleshov-group/mdlm-owt` (~130M non-embedding, GPT-2 tokenizer, OpenWebText). Matched AR control: the same release's `ar.ckpt`, trained by the same authors on the same data at the same scale in the same codebase. Both evaluated at `model.length=1024` with the repo's `data=openwebtext-split` config (tokenizer verified identical). Naive round-to-nearest PTQ at INT8, INT4, ternary. Eval runs through the bd3lms codebase (the mdlm repo's zero-shot eval cannot reproduce its own published numbers; see mdlm#22) with a numerically-validated flash-attn→SDPA shim.

**QAT pair (7M).** Twelve models trained from scratch on Wikitext103: {AR, dLLM} × {FP16, ternary-QAT} × {seed 1,2,3}. Tiny BD3LMS backbone (~7M non-embedding params), 6,000 steps, global batch 64. Ternarization is BitNet-style absmean with a straight-through estimator; embeddings, output head, norms, and biases are excluded (21.5% of model-state params ternarized — a limitation, §6).

**Metrics.** PTQ: validation likelihood degradation relative to each model's own FP16, plus a generative anchor (64 samples/condition, GPT-2-large oracle perplexity, with a unigram-entropy degeneracy guard). QAT: per-model "tax" = ternary/FP16 relative degradation, summarized by the gap ratio R = dLLM-tax / AR-tax.

**Pre-registration.** Kill criteria and the `dllm_more_robust` / `dllm_better` thresholds were declared before runs (`prereg-exp0-2026-07-16`, `prereg-exp2-pilot-2026-07-16.1`).

## 4. Results

### 4.1 Post-training quantization at 130M (primary)

Degradation relative to each model's own FP16:

| precision | AR degradation | dLLM degradation | excess (dLLM − AR) |
|---|---:|---:|---:|
| INT8 | ~0% | +0.3–0.6% | ≤ +0.7pp |
| INT4 | +25–32% | +9–21% | **−11 to −16pp** |
| ternary | ~10⁴× FP16 | ~10¹²× FP16 | uninformative (pre-registered) |

At INT4 the dLLM absorbs weight-quantization noise roughly 2× better than its AR twin, unanimous across PTB / Wikitext103 / LAMBADA. The pre-registered `dllm_more_robust` verdict fired. Ternary PTQ collapses on both, exactly as BitNet's premise predicts — pre-registered as uninformative.

### 4.2 Generative anchor (corroboration)

64 samples/condition, GPT-2-large oracle, degeneracy guard passed wherever sampling succeeded.

| precision | AR gen-ppl Δ | dLLM gen-ppl Δ | excess |
|---|---:|---:|---:|
| INT4 | +96.2% | +45.4% | **−50.8pp** |
| ternary | sampler-validity guard trips | guard trips | uninformative |

The dLLM's *generated text* degrades ~2× less than the AR model's at INT4. The likelihood result is not a perplexity artifact.

### 4.3 Native ternary QAT at 7M (supporting)

| seed | AR tax | dLLM tax | gap ratio R |
|---:|---:|---:|---:|
| 1 | +18.41% | +5.19% | 0.888 |
| 2 | +30.71% | +15.43% | 0.883 |
| 3 | +16.12% | +4.23% | 0.898 |

Geometric-mean **R = 0.890, 95% t CI [0.872, 0.908]**. The pre-registered no-extra-tax criterion fires (upper CI ≤ 1.25). The stricter `dllm_better` criterion (upper CI < 0.80) does **not**. Reading: diffusion pays no extra ternary tax at this scale, but this is not evidence of *superior* tolerance — and see §6 for the confounds specific to this rung.

### 4.4 Mechanism probes (honest negatives)

The tempting explanation is that argmax commitment absorbs errors: a sub-margin logit perturbation leaves the committed token unchanged. We tested whether that produces *end-to-end* absorption.

- **Commit-and-freeze, 96 trajectories.** Changing eight fixed tokens propagated to **65.7% (FP16) vs 66.3% (ternary)** of downstream positions. Three-seed ternary-minus-FP16 interval [−2.2, +3.3] pp — no resolved precision effect. A wrong frozen token remains conditioning context; perturbations grow, not shrink.
- **Post-hoc revisable retrofit.** ~98% disagreement — revision applied to a model not trained for it is not corrective.

Both reject "commitment automatically erases fixed-token errors." The robustness in §4.1–4.3 is real but is **not** explained by this mechanism.

## 5. Reproducibility

Single RTX 2080 Super (8GB, sm75, no FlashAttention-2). All configs, checkpoint SHA-256 hashes, pre-registration tags, and the flash-attn→SDPA numerical-equivalence shim are released. Full bring-up log and dependency pins in `experiments/exp0/PREP.md`. Raw perplexities and hashes in `experiments/exp2/RESULTS-replication.md`.

## 6. Limitations (what this does *not* show)

- **Scale.** Smallest matched PTQ pair is 130M; QAT pair is ~7M. Nothing here transfers to 1B+ without evidence.
- **QAT rung is confounded three ways.** At 7M, ternary is below BitNet's ~3B competitiveness threshold, only 21.5% of params are ternarized, and the dLLM objective is a NELBO bound rather than exact likelihood. The 130M INT4 PTQ result carries none of these; treat it as the spine and QAT as the riskier supporting evidence.
- **No absolute-quality parity claim.** dLLMs lose to AR on absolute quality at these scales regardless of quantization. This paper is about *relative sensitivity to quantization*, not usefulness.
- **PTQ method.** Naive RTN throughout, by design — the matched control, not calibration engineering, carries the claim. A calibrated-PTQ comparison is out of scope.
- **No mechanism.** §4.4 rejects the obvious one and offers no replacement.
- **No deployment claim.** Only 21.5% of params ternarized; weight-only idealized reduction is ~1.23×. No packed-ternary kernel, latency, or VRAM claim is made.

## 7. Conclusion

At matched scale, data, tokenizer, and recipe, masked diffusion language models absorb extreme weight quantization better than autoregressive models — clearly so at INT4 (130M, ~2×, three datasets, corroborated generatively), and with no extra ternary tax under native QAT (7M, three seeds). The effect is robust and replicated. The intuitive mechanism — token commitment as an error-absorbing quantizer — does not survive testing. We publish the effect, the negative mechanism result, and a full reproducibility release, and leave scale transfer and deployment to future work.

## References

- Ma et al., *The Era of 1-bit LLMs* (BitNet b1.58), arXiv:2402.17764.
- Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (MDLM), arXiv:2406.07524.
- *Large Language Diffusion Models* (LLaDA), arXiv:2502.09992.
- Xu et al., *DLLMQuant*, arXiv:2508.14090.
- Shang et al., *PTQ4DM*, arXiv:2211.15736. Li et al., *Q-Diffusion*, arXiv:2302.04304. So et al., *TDQ*, arXiv:2306.02316.
- Wang, Schiff, Sahoo, Kuleshov, *ReMDM*, arXiv:2503.00307.
- MDLM checkpoints + matched AR baseline: github.com/kuleshov-group/mdlm; huggingface.co/kuleshov-group/mdlm-owt.
