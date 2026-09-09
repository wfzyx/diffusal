# Diffusal Part 2: AR-to-Diffusion Conversion & Extreme MoE Quantization

> **Research Branch**: Expanding *"Masked Diffusion Language Models Absorb Extreme Weight Quantization Better Than Autoregressive Models at Matched Scale"* (Panisa, July 2026).

## Context & Motivation

The original paper established that discrete masked diffusion models (dLLMs) absorb extreme weight quantization significantly better than matched autoregressive (AR) models (~2x degradation advantage at INT4, zero extra ternary tax under native QAT, $R = 0.890$). However, the original work evaluated **dense transformer backbones only**.

To make this discovery impactful for frontier deployment on bandwidth-constrained hardware (e.g. 16GB commodity RAM or single workstations), we must transition from dense models to **Sparse Mixture-of-Experts (MoE)** architectures.

This creates **two distinct, decoupled problems** that must be solved systematically:

```
                          DIFFUSAL PART 2 RESEARCH SCOPE
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
   [ PROBLEM 1 ]                                             [ PROBLEM 2 ]
   AR-to-Diffusion Conversion                                Extreme MoE Quantization
   ──────────────────────────                                ────────────────────────
   Can we convert pre-trained                                Can we aggressively quantize
   causal models into diffusion                              (INT4, Ternary b1.58, Bonsai,
   decoders without multi-million                            DeltaNet) compact production MoEs
   dollar retraining?                                        without catastrophic collapse?
           │                                                         │
   • Causal vs Bidir Conflict                                • Double-Noise Model:
   • Naive Unmasking Breakdown                                 intra-expert (We) vs router (Wg)
   • Block-Diffusion Solution                                • Compounding AR Router Drift
   • Hybrid KV-Cache Architecture                            • dLLM Bidirectional Error Damping
```

---

## Folder Structure

* [`PROBLEM-1-AR-TO-DIFFUSION.md`](./PROBLEM-1-AR-TO-DIFFUSION.md): Architectural analysis of causal mask dropping, representation drift, and the Semi-Autoregressive Block-Diffusion Hybrid solution.
* [`PROBLEM-2-MOE-QUANTIZATION.md`](./PROBLEM-2-MOE-QUANTIZATION.md): Mathematical derivation and analysis of routing gate jitter ($\Delta W_g$), expert quantization ($\Delta W_e$), and why dLLM canvas refinement bounds routing misallocations.
* [`RESULTS-laptop-pilot.md`](./RESULTS-laptop-pilot.md): Complete empirical logs from the $0 laptop experiment on matched Top-2 Sparse MoE pairs across {FP32, INT4, Ternary-QAT} and DRAM hardware profiling.
* [`block_diffusion.py`](./block_diffusion.py): Reference implementation of the hybrid causal-past / bidirectional-candidate-block engine.
* [`moe_quant.py`](./moe_quant.py): Matched Top-2 Sparse MoE benchmark script testing router flip rate and trajectory drift.
* [`profile_hardware.py`](./profile_hardware.py): Arithmetic intensity (FLOPs/Byte) and DRAM streaming memory profiler.

---

## Executive Summary of Pilot Findings

1. **Hardware Physics Inversion**:
   * Pure AR decoding is memory-bound ($\text{Intensity} \approx 0.5 - 4.0\text{ FLOPs/Byte}$).
   * Block-Diffusion with Ternary experts reaches **$53.33\text{ FLOPs/Byte}$** (saturated compute).
   * Generates 64 tokens with **$5.33\times$ fewer forward passes** (4.34x wall-clock speedup) and reduces total DRAM streaming from **1,716 MB to 16 MB** ($106.7\times$ total data reduction).

2. **Empirical MoE Quantization Resilience**:
   * **Ternary QAT Experts**: Trained cleanly from scratch on real text ($R = 1.000$, passing the pre-registered $R \le 1.25$ no-extra-tax criterion).
   * **Router Gate Noise Absorption**: In end-to-end rollout, AR-MoE suffered **$20.31\%$** trajectory corruption due to compounding router misallocations, whereas dLLM-MoE canvas refinement limited corruption to **$12.50\%$** ($R = 0.615 < 0.80$, firing the strict `dllm_more_robust` verdict).
