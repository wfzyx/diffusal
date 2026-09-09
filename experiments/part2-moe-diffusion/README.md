# Disperser: Speculative Block Decoding & Sparse MoE Quantization

> **Research Branch (`disperser`)**: An exploratory investigation into speculative parallel block decoding and native Mixture-of-Experts (MoE) quantization on edge-constrained hardware.
>
> *For the primary, peer-reviewed dense discrete diffusion manuscript, see `main` (`arxiv/diffusal-arxiv.tex`).*

---

## Context & Motivation

Autoregressive (AR) sequence generation on edge devices is fundamentally memory-bandwidth bound: each token generated requires streaming multi-gigabyte weight matrices across the memory bus. Sparse Mixture-of-Experts (MoE) architectures and extreme quantization (INT4, Ternary BitNet b1.58) provide theoretical compression, but introduce two critical challenges:

1. **The Causal-Speculative Decoding Trade-Off**:
   Can pre-trained causal architectures execute parallel block decoding without millions of dollars in continual pre-training?
   * Pre-trained models fail under naive bidirectional unmasking due to RoPE position inversion ($i - j < 0$).
   * Jacobi parallel speculative decoding preserves causal prefix consistency while evaluating candidate blocks in parallel, trading arithmetic compute for memory bandwidth amortization.

2. **MoE Quantization Dynamics**:
   In sparse MoEs, aggressive quantization affects both routing gate layers ($W_g$) and expert parameters ($W_e$). We study router flip sensitivity and trajectory stability under INT4 and ternary representations.

---

## Key Hardware Findings

* **Memory Bandwidth vs Compute Trade-Off**:
  * Block decoding of 32 candidate tokens reduces DRAM traffic by **1.91x** compared to standard AR with KV-cache.
  * However, evaluating 32 candidate tokens across 12 refinement steps requires **384 token-forwards** (versus 64 for AR), incurring a **6.0x arithmetic compute overhead**.
  * **Hardware Implication**: Speculative block decoding is strictly beneficial on compute-dense, memory-bandwidth-choked hardware (e.g. commodity CPUs, memory-bound accelerators).

* **Architecture Disclosure**:
  * Experiments utilizing `PrimeIntellect/qwen3-moe-tiny` serve as a **structural topology testbed** (measuring routing mechanics and parameter layouts on an untrained random skeleton), while real semantic adaptation is profiled via `Qwen/Qwen2.5-0.5B`.

---

## Repository Files

* [`RESULTS-laptop-pilot.md`](./RESULTS-laptop-pilot.md): Methodological audit, physical memory profiles, and empirical stability benchmarks.
* [`PROBLEM-1-AR-TO-DIFFUSION.md`](./PROBLEM-1-AR-TO-DIFFUSION.md): In-depth analysis of causal RoPE inversion and speculative block decoding pathways.
* [`PROBLEM-2-MOE-QUANTIZATION.md`](./PROBLEM-2-MOE-QUANTIZATION.md): Mathematical formulations of expert quantization noise and router gate sensitivity.
* [`profile_hardware.py`](./profile_hardware.py): Rigorous DRAM streaming profiler incorporating KV-cache amortization and batch-wide MoE expert activation.
* [`moe_quant.py`](./moe_quant.py): Matched synthetic MoE training and quantization testbed with masked-token loss evaluation.
* [`speculative_block_diffusion.py`](./speculative_block_diffusion.py): Confidence-gated speculative block decoding engine with verified fluency fallback.
* [`run_64_sample_statistical_grid.py`](./run_64_sample_statistical_grid.py): Multi-prompt evaluation grid with router quantization and paired Student-t statistical tests.
* [`marimo_moe_diffusion.py`](./marimo_moe_diffusion.py): Interactive Marimo notebook visualizing arithmetic intensity and memory bandwidth dynamics.
