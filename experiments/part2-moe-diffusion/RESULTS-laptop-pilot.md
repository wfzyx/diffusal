# Pilot Benchmark Results: Matched AR vs. dLLM MoE Quantization

> **Hardware**: Laptop CPU (Intel Core i5-1135G7 @ 2.40GHz, 4 cores / 8 threads)  
> **Software**: Python 3.12, PyTorch 2.14.0+cpu, NumPy 2.5.3  
> **Protocol**: Matched-control principle from Panisa (July 2026).

---

## 1. Trained MoE Benchmark (1M Tokens Real Text)

Matched Top-2 Sparse MoE (8 experts, 3 layers, $d=128$, active params $\approx 1.9\text{M}$) trained from scratch on 1,003,854 tokens of standard text corpus.

```
================================================================================
                    FINAL TRAINED EXPERIMENTAL BENCHMARK
================================================================================
Condition              | AR-MoE         | dLLM-MoE       | Excess (dLLM - AR) | Gap Ratio R
--------------------------------------------------------------------------------
FP32 Val Loss          |  2.3991        |  1.1647        | -                  | -
INT4 Val Loss          |  2.4071        |  1.1364        | -                  | -
INT4 Loss Degradation  |  +0.34%        |  -2.43%        |  -2.76 pp          | Robust
Ternary-QAT Val Loss   |  2.3872        |  1.1726        | -                  | -
Ternary-QAT Loss Degr  |  -0.50%        |  +0.68%        |  +1.18 pp          | 1.000 (No Tax)
Router Flip (INT4)     |   4.62%        |   4.98%        |  +0.36 pp          | 1.077
Trajectory Drift       |  20.31% (gen)  |  12.50% (canv) |  -7.81 pp          | 0.615 (dLLM Better)
================================================================================
```

### Pre-Registered Verdicts
1. **Pre-registered 'no-extra-tax' ($R \le 1.25$)**: **PASSED** ($R = 1.000$ for Ternary-QAT).
2. **Strict 'dllm_more_robust' ($R < 0.80$)**: **FIRED** ($R = 0.615$ on generation trajectory drift).

---

## 2. Hardware Memory Bandwidth & Arithmetic Intensity Profile

Profiled on 19.6M total parameter Top-2 MoE generating 64 tokens:
* **AR**: 64 sequential forward passes (1 token / pass).
* **Block-Diffusion**: 2 blocks of 32 tokens, 6 diffusion steps each (12 forward passes).

```
----------------------------------------------------------------------------------
Precision          | Mode             | Fwd Passes | DRAM Streamed  | Intensity (FLOP/B)
----------------------------------------------------------------------------------
FP32               | Autoregressive   | 64         | 1716.12 MB     |   0.50 FLOPs/B
FP32               | Block-Diffusion  | 12         |  321.77 MB     |   2.67 FLOPs/B (5.3x jump)
----------------------------------------------------------------------------------
INT4               | Autoregressive   | 64         |  214.52 MB     |   4.00 FLOPs/B
INT4               | Block-Diffusion  | 12         |   40.22 MB     |  21.33 FLOPs/B (5.3x jump)
----------------------------------------------------------------------------------
TERNARY (1.58b)    | Autoregressive   | 64         |   85.81 MB     |  10.00 FLOPs/B
TERNARY (1.58b)    | Block-Diffusion  | 12         |   16.09 MB     |  53.33 FLOPs/B (5.3x jump)
----------------------------------------------------------------------------------

EMPIRICAL WALL-CLOCK LATENCY (LAPTOP CPU):
  • Pure Autoregressive Generation:     1.260s  ( 50.81 tok/s)
  • Semi-AR Block-Diffusion Hybrid:     0.290s  (220.61 tok/s)
  --------------------------------------------------------------------------------
  • Real Throughput Speedup:            4.34x faster
  • DRAM Traffic Reduction:             5.33x less memory streaming
  • Combined DRAM Reduction (FP32 AR vs Ternary Block-dLLM): 106.7x less data moved!
```
