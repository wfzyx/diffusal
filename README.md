# Diffusal

**End goal:** a natively ternary (1.58-bit, BitNet-style) diffusion language model in the Gemma family — built by converting a small Gemma 4 model to masked diffusion (A2D) and training it with ternary QAT — that wins on **quality-per-VRAM among diffusion LMs** and shows ternary hurts diffusion no more than it hurts autoregression.

What this is *not*: post-training quantization of DiffusionGemma-26B to 1.58 bits. Ternary PTQ fails on every architecture (BitNet's own premise), and 26B-scale QAT is out of budget. See [THESIS.md](THESIS.md) §2 and §8 for why the naive version of this goal is dead on arrival.

## Path

1. **Measure (Experiments 0–1, cheap):** error-trajectory dynamics of quantized masked diffusion on MDLM-owt 130M with true remasking (ReMDM) vs commit-and-freeze, against the same release's matched AR baseline. PTQ ladder (INT8/INT4/ternary) also run on **DiffusionGemma-26B-A4B** (4B active — single-GPU inference) as the real-world context model.
2. **De-risk training (Experiment 2, moderate):** ternary QAT at small scale with matched ternary AR controls — the go/no-go is the *gap-of-gaps* (ternary−FP16 for diffusion vs the same gap for AR), not absolute benchmark wins.
3. **Build (the artifact):** Gemma 4 **E2B** → A2D diffusion conversion (dLLM/Tiny-A2D recipe) → ternary QAT with distillation from the FP16 parent. Pipeline is debugged first on Gemma 3 270M/1B before spending the E2B budget. E2B's Per-Layer Embeddings stay higher precision (embeddings are excluded from ternarization by default), which inflates the non-ternary memory share — tracked as a §8 risk in the thesis.

## Key documents

- [THESIS.md](THESIS.md) — the full measurement program, protocols, fragility points, and kill criteria.
- [whitepaper/diffusal-whitepaper.pdf](whitepaper/diffusal-whitepaper.pdf) — the research program proposal (compiled from [LaTeX source](whitepaper/diffusal-whitepaper.tex)).

## Non-obvious facts already established

- LLaDA/Dream "low-confidence remasking" is commit-and-freeze: committed tokens are frozen forever (`-inf` confidence for unmasked positions). True remasking requires ReMDM-style inference-time retrofitting.
- The MDLM release ships a matched AR baseline (same OpenWebText split, same GPT-2 tokenizer, same 1024 context, same codebase) — the confound-free control comes for free.
- BitNet-scale reality: ternary matches FP16 only from roughly 3B parameters up, so small-model ternary losses discriminate nothing without an AR ternary control at the same scale.
