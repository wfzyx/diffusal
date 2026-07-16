#!/usr/bin/env bash
# Exp 0, step 3-4 of PREP.md: FP16 baseline reproduction (dLLM + matched AR
# control) on the pre-registered likelihood datasets. Must be run and checked
# against published numbers BEFORE the prereg tag is cut. No quantization here.
set -euo pipefail

EXP0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP0_DIR/../.." && pwd)"
MDLM="$REPO_ROOT/third_party/mdlm"
PYTHON="$EXP0_DIR/.venv/bin/python"
RESULTS="$EXP0_DIR/results/fp16"
mkdir -p "$RESULTS"

# flash_attn shim (sm75 has no FlashAttention-2) — see configs/exp0.yaml deviations.
export PYTHONPATH="$EXP0_DIR/shim:$MDLM"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

DATASETS=(wikitext103 ptb lambada)

for data in "${DATASETS[@]}"; do
  echo "=== dLLM (mdlm-owt) ppl_eval on $data ==="
  (cd "$MDLM" && "$PYTHON" main.py \
    mode=ppl_eval \
    loader.batch_size=2 \
    loader.eval_batch_size=2 \
    data="$data" \
    data.cache_dir="$HOME/.cache/mdlm_data" \
    model=small \
    backbone=hf_dit \
    model.length=1024 \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    +wandb.offline=true) 2>&1 | tee "$RESULTS/dllm_$data.log"
done

AR_CKPT="$EXP0_DIR/checkpoints/ar-clean.ckpt"   # sanitized weights-only copy (see PREP.md)
if [[ ! -f "$AR_CKPT" ]]; then
  echo "ar.ckpt missing — download it first (see PREP.md):"
  echo "  $EXP0_DIR/.venv/bin/gdown --folder 'https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau' -O $EXP0_DIR/checkpoints"
  exit 1
fi

for data in "${DATASETS[@]}"; do
  echo "=== AR control ppl_eval on $data ==="
  (cd "$MDLM" && "$PYTHON" main.py \
    mode=ppl_eval \
    loader.batch_size=2 \
    loader.eval_batch_size=2 \
    data="$data" \
    data.cache_dir="$HOME/.cache/mdlm_data" \
    model=small-ar \
    parameterization=ar \
    backbone=ar \
    model.length=1024 \
    eval.checkpoint_path="$AR_CKPT" \
    +wandb.offline=true) 2>&1 | tee "$RESULTS/ar_$data.log"
done

echo "Done. Logs in $RESULTS — compare against MDLM paper Table 4 (zero-shot ppl)."
