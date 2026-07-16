#!/usr/bin/env bash
# Exp 0 quantized grid (configs/exp0.yaml, frozen prereg-exp0-2026-07-16).
# FP16 references already measured in results/fp16-bd3lms/ with the identical
# harness command; this sweep adds the quantized cells.
set -uo pipefail

EXP0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$EXP0_DIR/results/quant"
mkdir -p "$RESULTS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for data in ptb wikitext103 lambada; do
  for m in ar dllm; do
    for prec in int8 int4 ternary; do
      log="$RESULTS/${m}_${prec}_${data}.log"
      if [[ -s "$log" ]] && grep -q "val/ppl" "$log"; then
        echo "skip ${m} ${prec} ${data} (done)"; continue
      fi
      "$EXP0_DIR/.venv/bin/python" "$EXP0_DIR/run_quant_eval.py" \
        "$m" "$prec" "$data" 2>&1 | grep -vE "it/s\]" > "$log"
      ppl=$(grep -oE 'val/ppl.*[0-9.]+' "$log" | grep -oE '[0-9.]+$' | head -1)
      echo "${m} ${prec} ${data}: ${ppl:-FAILED}"
    done
  done
done
echo "grid complete"
