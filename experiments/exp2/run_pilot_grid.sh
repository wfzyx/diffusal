#!/usr/bin/env bash
# Exp 2 pilot: 4 training runs (configs/exp2-pilot.yaml), sequential on 1 GPU.
# Resumable: lightning resumes from runs/<cell>/checkpoints/last.ckpt.
set -uo pipefail

EXP2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$EXP2_DIR/../exp0/.venv/bin/python"
LOGS="$EXP2_DIR/logs"
mkdir -p "$LOGS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for m in ar dllm; do
  for prec in fp16 ternary; do
    log="$LOGS/${m}_${prec}.log"
    if grep -q "PILOT-RUN-DONE" "$log" 2>/dev/null; then
      echo "skip $m $prec (done)"; continue
    fi
    echo "=== training $m $prec ($(date)) ==="
    "$PYTHON" "$EXP2_DIR/run_pilot.py" "$m" "$prec" 2>&1 \
      | grep -vE "it/s" > "$log" \
      && echo "PILOT-RUN-DONE" >> "$log"
    echo "$m $prec finished ($(date)); last val: $(grep -oE 'val/ppl[^0-9]*[0-9.]+' "$log" | tail -1)"
  done
done
echo "pilot grid complete"
