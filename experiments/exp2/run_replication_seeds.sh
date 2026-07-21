#!/usr/bin/env bash
# Exp 2 replication: run the two missing seeds; seed 1 is the frozen pilot.
# Requires configs/exp2-replication.yaml to be committed/tagged first.
set -euo pipefail

EXP2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$EXP2_DIR/../exp0/.venv/bin/python"
LOGS="$EXP2_DIR/logs"
mkdir -p "$LOGS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for seed in 2 3; do
  for model in ar dllm; do
    for precision in fp16 ternary; do
      log="$LOGS/seed-${seed}_${model}_${precision}.log"
      if grep -q "REPLICATION-RUN-DONE" "$log" 2>/dev/null; then
        echo "skip seed $seed $model $precision (done)"
        continue
      fi

      echo "=== seed $seed: $model $precision ($(date -Is)) ==="
      EXP2_SEED="$seed" "$PYTHON" "$EXP2_DIR/run_pilot.py" "$model" "$precision" \
        2>&1 | grep -vE "it/s" > "$log"
      echo "REPLICATION-RUN-DONE" >> "$log"

      run_dir="$EXP2_DIR/runs/seed-${seed}/${model}_${precision}/checkpoints"
      # Keep best/last only; intermediate checkpoints previously exhausted disk.
      rm -f "$run_dir"/[0-9]*-*.ckpt
      echo "seed $seed $model $precision finished ($(date -Is)); last val: $(grep -oE 'val/ppl[^0-9]*[0-9.]+' "$log" | tail -1)"
    done
  done
done

echo "Exp 2 replication seeds complete"
