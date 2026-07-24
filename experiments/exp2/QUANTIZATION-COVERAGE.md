# Exp 2 quantization-coverage audit

Ran `audit_quantization_coverage.py` against all three dLLM ternary-QAT best checkpoints. Their model-state structure is identical.

| quantity | value |
|---|---:|
| model-state parameters | 32,941,026 |
| ternary-QAT linear weights | 7,098,368 (21.5%) |
| all-FP16 weight storage | 62.83 MiB |
| ideal packed 2-bit ternary + remaining FP16 | 50.98 MiB |
| idealized weight-only reduction | 1.23× |

The current QAT result is therefore **not** a whole-model 1.58-bit deployment result. It ternarizes the transformer/sigma linear layers while leaving token embeddings, vocabulary/output head, norms, biases, and other excluded parameters in FP. The reported packed figure is a conservative 2-bit representation for ternary values; it excludes scales, metadata, activations, KV/cache-like state, and runtime-kernel overhead.

This audit is a deployment claim boundary, not a quality result. A meaningful memory claim requires separate ablations covering the excluded parameter groups and a packed inference implementation.
