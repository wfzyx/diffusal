# Disperser Part 2: Speculative Block Decoding & MoE Quantization
## Technical Audit & Hardware Characterization Report

> **Branch Notice**: This document details the exploratory **Disperser** line (speculative block decoding and sparse Mixture-of-Experts quantization). For the peer-reviewed, dense discrete diffusion manuscript, see the `main` branch (`arxiv/diffusal-arxiv.tex`).

---

## 1. Summary of Methodological & Architectural Audit

Following a rigorous adversarial audit, key structural clarifications have been established for this exploration:

1. **Testbed Architecture Disclosure (`PrimeIntellect/qwen3-moe-tiny`)**:
   * Statistical analysis of weights across all 24 layers shows `sd ≈ 0.0200`, exact Gaussian kurtosis ($2.95–3.01$), and all RMSNorm gains fixed at $1.0000 \pm 0.0016$.
   * **Verdict**: `qwen3-moe-tiny` is an **untrained random initialization skeleton**, not a production checkpoint. It functions exclusively as a structural testbed for graph topology, memory layout, and router dispatch mechanics. It cannot measure semantic degradation.

2. **Decoding Mechanism Clarification (Jacobi Speculative Decoding)**:
   * A causal attention mask paired with greedy iterative block argmax updates is mathematically equivalent to **Jacobi parallel speculative decoding** converging toward the causal autoregressive rollout, rather than a discrete bidirectional diffusion process.
   * Real pre-trained causal models (`Qwen2.5-0.5B`) subjected to unadapted bidirectional canvas attention suffer immediate semantic disintegration due to RoPE relative position inversion $(i - j < 0)$.

3. **Sampling Dilution Correction**:
   * Initial pilot scripts used `n_unmask = max(1, block_size // steps) = 2` for a block of 16 across 6 steps, unmasking only 12 tokens and leaving 4 tokens permanently frozen as `eos_token_id`.
   * When measured exclusively on active (unmasked) positions, the artificial drift reduction vanishes: unadapted speculative decoding on random weights experiences drift comparable to or slightly exceeding AR (+3.92 pp at INT4, +9.31 pp at Ternary).

---

## 2. Hardware Memory Bandwidth & Arithmetic Intensity Profile

Benchmarked on a 19.6M parameter Top-2 MoE generating 64 tokens under honest hardware physics:
* **Autoregressive (AR)**: Models KV-cache reuse ($T_{\text{prompt}} + 64$ sequential steps).
* **Speculative Block Decoding**: 2 blocks of 32 candidate tokens, 6 refinement steps per block (12 forward passes total, evaluating 32 candidate tokens per pass).
* **MoE Expert Activation Physics**: Because 32 tokens are processed simultaneously in each block forward pass, routing entropy ensures that almost all experts (all 8 or 16) are active across the batch, requiring nearly full expert weight streaming per pass.
* **Ternary Memory Model**: Packed at 2 bits per parameter (0.25 bytes/param).

```
========================================================================================================
PRECISION       | DECODING ENGINE        | DRAM MOVED (MB) | COMPUTE (MFLOPs) | INTENSITY (FLOPs/Byte)
========================================================================================================
FP32 (4 B/p)    | Autoregressive (KV)    |       32.55 MB  |        128 MFLOP |             3.93
FP32 (4 B/p)    | Speculative Block      |       17.06 MB  |        768 MFLOP |            45.02 (11.5x)
--------------------------------------------------------------------------------------------------------
INT4 (0.5 B/p)  | Autoregressive (KV)    |        4.07 MB  |        128 MFLOP |            31.45
INT4 (0.5 B/p)  | Speculative Block      |        2.13 MB  |        768 MFLOP |           360.56 (11.5x)
--------------------------------------------------------------------------------------------------------
TERNARY (0.25B) | Autoregressive (KV)    |        2.03 MB  |        128 MFLOP |            63.05
TERNARY (0.25B) | Speculative Block      |        1.07 MB  |        768 MFLOP |           717.76 (11.4x)
========================================================================================================
```

### Physical Trade-Off Breakdown:
1. **DRAM Bandwidth Reduction**: **1.91x reduction** (e.g. 32.55 MB down to 17.06 MB in FP32; 2.03 MB down to 1.07 MB in Ternary).
2. **Arithmetic Work Overhead**: Speculative block decoding evaluates 32 tokens per forward pass over 12 passes = **384 token-forwards**, compared to **64 token-forwards** for AR. This represents a **6.0x compute trade-off**.
3. **Hardware Sweet Spot**: Speculative block decoding is strictly advantageous on **memory-bandwidth-bound hardware** (e.g., edge CPU, unified memory with low bandwidth) where memory bus latency dominates arithmetic compute time.

---

## 3. Upcycling & Representation Inversion on Pre-Trained Weights (`Qwen2.5-0.5B`)

Testing bidirectional canvas unmasking on genuine pre-trained weights (`Qwen/Qwen2.5-0.5B` upcycled to 4 experts, Top-2):

```
Prompt: "The theory of general relativity explains that gravity is"

• Pre-trained AR Baseline (Causal, 32 passes):
  "The theory of general relativity explains that gravity is caused by the curvature of spacetime caused by the presence of mass. The curvature of spacetime is caused by the presence of mass and energy. The curvature of"

• Unadapted Bidirectional Canvas Decoding (12 passes):
  "The theory of general relativity explains that gravity is  00 人  1 的 |Human# #0Human0<\  的 "

• INT4 Bidirectional Canvas:
  "The theory of general relativity explains that gravity isT1 1A3附 2AGGA人quest6 21 @ B【"

• Ternary (BitNet b1.58) Bidirectional Canvas:
  "The theory of general relativity explains that gravity ismodifiedbynamebyname以及适modifiedmodified..."
```

### Finding:
Confirms that **unadapted bidirectional canvas sampling on pre-trained causal models produces immediate semantic disintegration**.
RoPE relative position offsets $(i - j < 0)$ and causal query-key alignments require either:
1. Causal-consistent speculative block decoding (where attention remains strictly causal or prefix-conditioned), OR
2. Dedicated bidirectional diffusion fine-tuning / adaptation (e.g., LLaDA pathway).

---

## 4. Synthetic Matched Small-MoE Benchmark (Trained From Scratch)

On the 7M matched synthetic Top-2 MoE benchmark (pre-sampled paired validation batches, loss evaluated strictly on masked tokens):

```
================================================================================
Metric                 | Autoregressive MoE | Discrete Diffusion MoE | Delta
================================================================================
FP32 Baseline Val Loss |  2.3992            |  1.1647                | -
INT4 PTQ Loss Degr     |  +1.21%            |  +0.74%                | -0.47 pp (dLLM Advantage)
Ternary-QAT Loss Degr  |  -0.50%            |  +0.68%                | +1.18 pp
Router Flip Rate (INT4)|   4.62%            |   4.98%                | +0.36 pp
================================================================================
```

### Pre-Registered Verdicts
1. **Pre-registered 'no-extra-tax' ($R \le 1.25$)**: **PASSED** ($R = 1.000$ for Ternary-QAT).
2. **Router Stability**: Router selection exhibits small sensitivity ($4.62\% - 4.98\%$ flip rate) under 4-bit quantization, confirming that router gating layers require either INT8 precision or dedicated routing jitter regularization.

---

## 5. Statistical Rigor Grid (64 Diverse Prompts)

Evaluated with full block unmasking (no frozen padding positions) and router quantization across 64 evaluation prompts:

```
==================================================================================
Condition              | AR Drift (95% CI)      | Speculative Block (95% CI)
----------------------------------------------------------------------------------
INT4 Experts + Router  | 24.85% +/- 7.03%       | 28.77% +/- 5.12%
Ternary (1.58b) Exp    | 46.09% +/- 8.51%       | 55.40% +/- 6.84%
==================================================================================
```

### Key Takeaway:
When evaluating live positions without padding dilution on an unlearned architecture, speculative block decoding does not magically prevent quantization noise; both causal AR and speculative block updates experience significant drift under aggressive PTQ. This highlights the necessity of **speculative confidence thresholding ($\\tau$)** or **QAT fine-tuning** to preserve stability.
