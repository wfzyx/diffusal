#!/usr/bin/env bash
# Exp 2 — 130M native ternary QAT cohort (12 runs: 3 seeds x {ar,dllm} x {fp16,ternary}).
# Mirrors run_replication_seeds.sh but drives bd3lms model=small via hydra overrides.
# Designed to run ON THE GCP L4 VM (see infra/gcp/). Idempotent: re-running skips
# cells already marked done, so it survives spot preemption.
#
# Requires: configs/exp2-qat130m.yaml committed+tagged first (pre-registration),
# and $DIFFUSAL_VENV pointing at a torch+lightning env (the VM startup builds it).
set -euo pipefail

EXP2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${DIFFUSAL_VENV:-$EXP2_DIR/../exp0/.venv}/bin/python"
LOGS="$EXP2_DIR/logs/qat130m"
mkdir -p "$LOGS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# model=small overrides. Budget/batch mirror configs/exp2-qat130m.yaml (DRAFT);
# keep this list and the config in lockstep — the config is the pre-registration.
COMMON_OVERRIDES=(
  model=small
  model.length=1024
  loader.global_batch_size=64
  loader.batch_size=8                 # micro-batch; grad-accum makes up global 64
  loader.eval_batch_size=8
  trainer.max_steps=40000
  optim.lr=3e-4
  lr_scheduler.num_warmup_steps=2000
)

for seed in 1 2 3; do
  for model in ar dllm; do
    for precision in fp16 ternary; do
      log="$LOGS/seed-${seed}_${model}_${precision}.log"
      if grep -q "QAT130M-RUN-DONE" "$log" 2>/dev/null; then
        echo "skip seed $seed $model $precision (done)"; continue
      fi
      echo "=== seed $seed: $model $precision ($(date -Is)) ==="
      EXP2_SEED="$seed" EXP2_RUN_TAG="qat130m" \
        "$PYTHON" "$EXP2_DIR/run_pilot.py" "$model" "$precision" \
        "${COMMON_OVERRIDES[@]}" 2>&1 | grep -vE "it/s" > "$log"
      echo "QAT130M-RUN-DONE" >> "$log"

      run_dir="$EXP2_DIR/runs/qat130m/seed-${seed}/${model}_${precision}/checkpoints"
      rm -f "$run_dir"/[0-9]*-*.ckpt   # keep best/last only
      echo "seed $seed $model $precision done ($(date -Is)); last val: $(grep -oE 'val/ppl[^0-9]*[0-9.]+' "$log" | tail -1)"
    done
  done
done

# Checkpoint hashes for the reproducibility record (required_outputs in the config).
find "$EXP2_DIR/runs/qat130m" -name '*.ckpt' -print0 | sort -z \
  | xargs -0 sha256sum > "$EXP2_DIR/CHECKPOINTS-qat130m.sha256" || true

echo "Exp 2 130M QAT cohort complete"
