# Diffusal — Building a Natively Ternary Text Diffusion Model in the Gemma Family

**Program document (the "monster").** This is the full de-risking program whose end goal is a **1.58-bit (ternary) masked diffusion language model in the Gemma family** — concretely, an A2D (autoregressive-to-diffusion) conversion of **Gemma 4 E2B** trained with ternary QAT distilled from its FP16 parent. The scoped empirical finding that motivates the program (dLLMs are more robust to extreme weight quantization than matched AR models) is written up separately and independently in [ARXIV-PAPER.md](ARXIV-PAPER.md); that paper stands on its own and does not depend on anything below. This document is everything the paper is *not*: the build, the scale ladder, the systems and memory reality, the fragility map, and the kill gates.

**Separation of concerns (read this first).** The *paper* has one bar: is one measurement clean, matched, replicated, honestly scoped? It already clears that bar. The *program* has a different, receding bar — scale transfer, deployment economics, mechanism — and it is *supposed* to keep receding, because it is a multi-rung ladder. Do not grade the paper against the program's rubric; that is the goalpost treadmill. Publish the paper; run the program.

---

## 0. Status and the two-track plan

**Track A — publish (done evidence, no new compute).** Ship [ARXIV-PAPER.md](ARXIV-PAPER.md) on the 130M INT4 PTQ spine now. Gate the adversarial-venue push (Reddit/HN) on the 130M QAT twin (§5.2), because that is the specific patch for the paper's soft underbelly (the 7M QAT rung).

**Track B — build (the program).** Climb the scale ladder toward Gemma 4 E2B ternary QAT, killing at any gate in §10.

| Scale | Purpose | Approx. requirement | Status |
|---|---|---|---|
| 7M | Feasibility / mechanism evidence | RTX 2080 Super (owned) | **done** (Exp 2) |
| 130M | Native-QAT bridge to the Exp 0 PTQ scale | 24GB GPU (40GB preferred) | **next** (§5.2) |
| 300–500M | First claim-bearing scaling rung | 48–80GB GPU | planned |
| 1B | Scale-transfer test | ~4×80GB | planned |
| 2–3B (E2B) | Product-target validation | ~8×80GB | end goal |

**Funding ask, minimal and staged:** one 48–80GB GPU for a full 300–500M three-seed matched QAT cohort, *after* profiling one 130M run. Do **not** fund custom kernels or FSDP engineering until the larger QAT effect survives.

## 1. The core scientific risk

Diffusion LMs repeatedly refine a token canvas, so the same quantized denoiser is applied to (partially) its own output many times. The program's central question: **across refinement steps, does quantization error contract or expand?** A crossover — absorption at high mask fraction, amplification near convergence — is a *falsifiable working hypothesis*, not the expected answer.

What is already settled (do not re-derive): in continuous image diffusion, quantization error accumulates non-uniformly by timestep, and step-dependent precision is the fix (PTQ4DM, Q-Diffusion, TDQ). What is *different* in masked text diffusion is that **argmax commitment is itself a quantizer**: a sub-threshold logit perturbation can leave token identity unchanged, while a margin crossing flips it discretely and — under commit-and-freeze — irreversibly. This narrow identity-preservation fact is **not** end-to-end absorption; the completed mechanism probes (§4) prove it is not. A wrong committed token still conditions later denoising. The commit/remask policy is therefore load-bearing.

## 2. Main claims of the program

1. **PTQ is a baseline and control, never the path.** Ternary PTQ fails on every architecture; only *excess* over a matched AR control is evidence about diffusion.
2. **The primary observable is the error trajectory and its crossover, in mask-fraction terms** — not a single final score, not raw step index.
3. **Contractivity and per-step distortion are different measurements** (teacher-forced distortion ≠ perturbation contractivity); neither substitutes for the other.
4. **All-ternary behavior matters more than one-module sensitivity;** quantization effects are non-additive.
5. **The go/no-go metric is the gap-of-gaps:** (ternary dLLM − FP16 dLLM) vs (ternary AR − FP16 AR), matched scale/data/tokenizer/recipe.
6. **A 6GB 25B deployment is probably dead on arrival;** weight compression is the wrong lever if activations/embeddings/logits/buffers dominate.

## 3. Experiment 0 — PTQ failure with an AR control (done)

Objective: PTQ degradation of a small dLLM *relative to a matched AR control*, per precision, before building anything.

**Platform, chosen to make matched-control + remasking free:** MDLM-owt (~130M) with the same release's AR baseline (`ar.ckpt`) as the confound-free control; ReMDM as the genuine-remasking condition and the vanilla MDLM confidence sampler as commit-and-freeze, on the same weights. Second scale point available via Tiny-A2D 0.5B (dLLM/ZHZisZZ conversion recipe, each checkpoint has a *literal AR parent*). DiffusionGemma-26B-A4B is a **measurement-only** context model (MoE, out of QAT budget), not a QAT target.

**Result:** at INT4 the dLLM degrades ~2× less than its AR twin (excess −11 to −16pp likelihood, −50.8pp generative), unanimous across three datasets. Ternary PTQ collapses on both (uninformative, pre-registered). Full write-up feeds [ARXIV-PAPER.md](ARXIV-PAPER.md) §4.1–4.2.

## 4. Experiment 1 — error-trajectory dynamics (partially done, exploratory)

Objective: how quantization error evolves under iterative refinement — the per-step distortion curve vs mask fraction, whether the dynamics contracts injected errors, and where absorption turns into amplification.

**Sampler family (design commitment).** The contract/expand question is *sampler-dependent*. With commit-and-freeze unmasking, committed tokens are never revisited, the feedback loop barely exists, and the question collapses to "does quantization change the order and identity of commits?" (§6.1). A code-inspection caution: the well-known "remasking"-named samplers of LLaDA and Dream are in fact commit-and-freeze — LLaDA's `low_confidence` mode recomputes the mask set as `(x == mask_id)` and assigns `-inf` confidence to unmasked positions, so a committed token can never re-enter play. Genuine remasking exists as an inference-time retrofit: **ReMDM** (Wang et al.) re-opens committed tokens on any pretrained masked-diffusion checkpoint without retraining (cap / rescale / conf / loop schedules). Diffusal's **primary** sampler is ReMDM on the chosen base model, where the feedback loop genuinely exists; commit-and-freeze is the **secondary** condition on the *same* checkpoint — a clean two-condition design that isolates the feedback loop with weights held fixed.

**The alignment problem.** Running FP16 and quantized denoisers from the same canvas and diffing intermediate states is confounded: as soon as one commits a different token, the two are denoising different problems. "KL grew" can just mean "one early commit disagreement snowballed" — a commit-decision problem, not a weight-precision one. Three protocols, kept separate:

- **Protocol A — teacher-forced.** Feed the quantized model the exact FP16 canvas each step. Isolates per-step distortion at a fixed state. *Cannot observe amplification by construction* (nothing accumulates); growth with mask fraction is state-dependent sensitivity, not feedback. Off-policy caveat stated.
- **Protocol B — free-running.** Deployment condition, but divergence mixes distortion and trajectory forking; cannot alone prove expansion.
- **Protocol C — perturbation-injection contractivity.** Inject a controlled perturbation into the FP16 model's *own* state at noise level σ; track Hamming distance to the unperturbed run. Lyapunov-style, quantization-independent, applies to the from-scratch ternary line directly. Repeat with perturbation magnitude set to the Protocol A distortion at that σ.

**Metrics and interpretation.** All trajectory metrics are reported as a function of **mask fraction / noise level** (not raw step index, which is sampler-dependent) and as **per-sample distributions**, not only means: token agreement vs FP16 (A, B) or vs the unperturbed run (C); Hamming-distance decay after injection (C — the contractivity estimate); oscillation rate (remasking sampler only, n/a for commit-and-freeze). KL by noise level is *supporting only* — over a large vocab it is tail-dominated and temperature-sensitive, near-degenerate around the mask token. **Quality anchor, required alongside every divergence metric:** divergence measures *fidelity* to FP16, not quality — a diverged completion can be equally good. So every condition also reports oracle perplexity (fixed larger model) and final task score. **A divergence finding with no quality drop is a fidelity result, not a failure**, and is labeled as such. What we read off: (a) the **crossover** mask fraction where Protocol C turns expansive and Protocol A distortion becomes large, reported per-sample — if bimodal (early-fork vs smooth-drift), that *is* the finding, do not average it away; (b) **early-fork vs late-drift** — free-running divergence dominated by a few early commit disagreements implies a schedule/threshold fix (§6.1), smooth late-stage drift corroborated by Protocol C expansion implies genuine precision sensitivity near convergence; (c) **mitigation cost, priced honestly** — the implied fix (higher precision below the crossover) needs a resident higher-precision weight copy or on-the-fly dequant, charged against §8; a crossover at 40% mask fraction with a resident FP16 copy is not a ternary model. Thresholds for "grows / bounded / contracts" are pre-registered before runs.

**Completed exploratory probes (mechanism negatives).** Commit-and-freeze, 96 trajectories: eight fixed-token changes propagate to ~66% of downstream positions (FP16 65.7% vs ternary 66.3%, three-seed interval [−2.2, +3.3]pp — no resolved precision effect). Post-hoc revisable retrofit: ~98%. These **reject** the broad premise that commitment alone erases fixed-token errors, and show no FP16-vs-ternary propagation difference. They are exploratory (not pre-registered) and feed [ARXIV-PAPER.md](ARXIV-PAPER.md) §4.4.

**Next Exp 1 step:** train and evaluate a *matched native revisable* model with one frozen corruption and one inference sampler — the only clean way to ask whether trained revision reduces fixed-token sensitivity without sacrificing quality. Quality gate: lower disagreement is meaningless if validation loss or completion behavior worsens; report propagation, validation loss, and completion together. Only an actual DiffusionGemma run may be labeled "DiffusionGemma."

## 5. Experiment 2 — ternary training as the main line

Objective: train the actual artifact and determine whether ternary hurts diffusion *more than autoregression* — the only question small-scale training can answer.

### 5.1 The end-goal build: Gemma 4 E2B, A2D + ternary QAT

Shipping artifact: an **A2D conversion of Gemma 4 E2B (~2.3B effective params) trained with ternary QAT, distilled from its FP16 parent.**

- **Scale:** BitNet's competitiveness threshold is ~3B; E2B sits near it — the smallest Gemma 4 where ternary has a real chance, largest whose QAT fits a serious-hobbyist cloud budget. E4B is the stretch; 12B/26B-A4B/31B are out of budget (26B is MoE → router fragility, §6.3).
- **Recipe:** the conversion recipe converts any AR model; the FP16 parent doubles as distillation teacher *and* matched FP16 AR control, making the gap-of-gaps structurally free.
- **Pilot ladder:** debug the pipeline on Gemma 3 270M / 1B (cheap, dense, vanilla) before E2B. A pipeline bug found at E2B prices is a process failure.
- **Known risk — Per-Layer Embeddings.** E2B reaches 2.3B *effective* params via PLE, so its embedding share is unusually large and embeddings are excluded from ternarization by default. The non-ternary memory fraction is structurally higher; measured and charged against §8. If PLE tables cannot be compressed safely, reconsider the E4B / dense-Gemma-3 fallback.
- **Double distribution shift, staged.** Convert to diffusion at FP16 first, verify against the AR parent, *then* ternarize via QAT with the FP16 diffusion model as teacher. One-stage is an ablation, not the plan.

**The scale confound, stated up front.** Below ~3B, ternary QAT underperforms FP16 for reasons unrelated to diffusion, so a small ternary dLLM losing to its FP16 twin is expected under every hypothesis and discriminates nothing. The discriminating quantity is the **gap-of-gaps** at matched scale/data/tokenizer/token-budget/recipe.

### 5.2 The 130M QAT bridge (next experiment)

**Done at 7M** (feeds [ARXIV-PAPER.md](ARXIV-PAPER.md) §4.3): three-seed R = 0.890, CI [0.872, 0.908]; no extra dLLM tax, but confounded three ways (sub-3B scale, 21.5% ternarization coverage, NELBO bound).

**130M is the next rung and the paper's insurance.** Its value is *scale symmetry*: you already have the strong PTQ result at 130M (Exp 0); a 130M QAT cohort gives PTQ *and* QAT at the same scale and removes the "only 7M / partial coverage" attack surface before the adversarial-venue push. Cohort: {AR, dLLM} × {FP16, ternary-QAT} × 3 seeds = 12 runs. Requires a rented 24–40GB GPU (does not fit the owned 8GB card). **Frame it as closing the PTQ/QAT scale gap, not as a publication gate** — the arXiv v1 ships without it.

**Ablation grid (dLLM line, later rungs):** all-ternary; all-ternary except embeddings/head; all-ternary except mask embedding + remasking/confidence path; all-ternary except self-conditioning; FP16 control. 5 cells × ≥3 seeds (dLLM) + 2 × ≥3 seeds (AR). Only claim an ablation effect when between-cell difference exceeds the pre-registered multiple of within-cell seed spread. Drop cells for budget only from the bottom, never the all-ternary or AR-control cells.

## 6. Special fragility points

- **6.1 Mask token / remasking schedule.** Under commit-and-freeze this is nearly the *entire* quantization question (committed tokens never revisited). Default: keep mask embedding + confidence/remasking computation higher precision until ablations clear them.
- **6.2 Self-conditioning.** If previous-step state feeds forward, it may amplify error. Default: test separately, keep FP16/INT8 if it diverges.
- **6.3 MoE routing.** Router errors are non-local. Default: routers stay FP16/INT8; do not ternarize routers first. (Directly relevant to the 26B-A4B context model.)

## 7. Systems reality check

Ternary weights reduce storage, but GPUs are optimized for INT4/INT8/FP16 tensor-core paths, not native ternary matmul. Expected: **ternary is slower than a good INT4 baseline until custom kernels prove otherwise.** If Exp 1 recommends mixed precision below a crossover mask fraction, serving must carry a second resident weight copy or on-the-fly dequant — that cost belongs in every memory/latency number. Serving path, lazy and in order: standalone fixed-canvas loop → real latency/memory measurement → custom kernels only if worth it → vLLM/serving integration last. Do **not** start with vLLM (built for AR append + paged KV, not full-canvas refinement).

## 8. Memory reality check

A 25B ternary diffusion model is a *warning, not a target*: **6GB 25B deployment is probably not the thesis.** The "dead on arrival" claim must rest on a **measured peak-VRAM breakdown on the actual small model under test** — weights (incl. any mixed-precision tail), embeddings/head, per-step activations, logits, runtime buffers — as real numbers at real canvas/batch/step. Current coverage audit: only **21.5%** of model-state params ternarized, idealized weight-only reduction **~1.23×**. The useful question is not "can a fictional 25B fit in 6GB?" but: *at a fixed VRAM budget, does a ternary dLLM beat the best smaller INT4/FP16 AR baseline on quality, latency, reliability?* If not, the end-goal model ships as **science, not deployment** — an acceptable outcome, stated in advance.

## 9. Baselines and go/no-go

Every stage compares against: (1) FP16 of the same dLLM; (2) INT4 of the same dLLM where available; (3) matched ternary/FP16 AR controls; (4) a smaller off-the-shelf AR FP16/INT4 model at the same VRAM budget; (5) wall-clock latency on target hardware.

**Primary go/no-go: the gap-of-gaps (comparison 3)** — the only comparison that isolates diffusion confound-free at small scale. **AR frontier (comparison 4) is context, never a kill:** at affordable scales dLLMs lose to AR on quality-per-GB regardless of quantization; if the end-goal model can't beat the frontier even at scale, it ships as science (§8). All thresholds pre-registered before the relevant runs.

## 10. Kill criteria

Stop or re-scope when any fires (each references its pre-registered threshold; divergence-only signals with no quality drop never kill):

- **No diffusion-specific effect at Exp 0** → re-scope to "why is iterated denoising robust," build proceeds on the ternary-AR analogy. *(Did not fire — Exp 0 showed a clear effect.)*
- **Expansive dynamics with quality loss:** Protocol C expands across most of the schedule, Protocol A distortion large there, *and* quality anchor drops past threshold.
- **Mask-path fragility:** quantizing mask embedding / remasking path misses quality thresholds.
- **Gap-of-gaps failure:** ternary dLLM gap exceeds matched ternary AR gap by the pre-registered margin and §6 mitigations don't close it.
- **Mitigation uneconomic:** crossover at high mask fraction → higher-precision tail erases the memory advantage.
- **No kernel win:** ternary inference slower than INT4/FP16 after reasonable effort.
- **Memory miss:** safe activation/embedding/logit compression still misses the VRAM target.

Losing to an *unmatched* off-the-shelf AR model is context, never a kill.

## 11. Conclusion

The end goal is a natively ternary text diffusion model in the Gemma family — an A2D-converted, ternary-QAT Gemma 4 E2B. It must **not** be sold as "ternary makes huge diffusion LLMs fit on tiny GPUs"; the arithmetic does not support that. The sharper scientific thesis on the way is error dynamics: *at which noise level does iterative denoising stop absorbing ternary quantization noise and start amplifying it, and is that worse than autoregression's at the same precision?* The expected answer is a crossover distribution, not a verdict, and the crossover is what prices the mitigation. If diffusion's ternary gap matches autoregression's, the end-goal model inherits BitNet's scaling story and the build is justified; if it is worse, the negative result is still real. Either way the finding is trustworthy only if the trajectory comparison is well-posed (§4), contractivity is measured on the dynamics rather than inferred, and every quantization claim carries its matched AR control.

## References

- BitNet b1.58 (arXiv:2402.17764); MDLM (2406.07524); LLaDA (2502.09992); DLLMQuant (2508.14090); ReMDM (2503.00307); PTQ4DM (2211.15736); Q-Diffusion (2302.04304); TDQ (2306.02316); Constrained Discrete Diffusion (2503.09790).
- MDLM checkpoints + matched AR/SEDD baselines: github.com/kuleshov-group/mdlm.
- Tiny-A2D 0.5B/0.6B AR→diffusion recipes (dLLM): github.com/ZHZisZZ/dllm.
- Sampler-correctness eval (dllm_sampler): github.com/LuhanTang/dllm_sampler.
- DiffusionGemma-26B-A4B: huggingface.co/google/diffusiongemma-26B-A4B-it; ai.google.dev/gemma/docs/diffusiongemma.
- Gemma 4 (E2B/E4B/…): ai.google.dev/gemma/docs/core.
