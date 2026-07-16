#!/usr/bin/env bash
# Exp 0 generative-anchor sweep (configs/exp0.yaml).
# Documented deviations from the frozen config, forced by the 8 GB card:
#   - sample length 512 (not 1024): nucleus sort over 1024x50k fp32 logits OOMs
#   - 64 samples, seed=1 only (not 512 x 3 seeds)
#   - inline oracle gpt2 for logging; OFFICIAL scores come from score_samples.py
#     with the pre-registered gpt2-large oracle over the saved samples
# Applied identically to both models, so excess comparisons stay well-posed.
set -uo pipefail

EXP0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$EXP0_DIR/results/anchor"
mkdir -p "$RESULTS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for m in ar dllm; do
  for prec in fp16 int8 int4 ternary; do
    log="$RESULTS/${m}_${prec}.log"
    if [[ -s "$log" ]] && grep -qi "generative perplexity" "$log"; then
      echo "skip $m $prec (done)"; continue
    fi
    "$EXP0_DIR/.venv/bin/python" "$EXP0_DIR/run_quant_eval.py" \
      "$m" "$prec" gen model.length=512 \
      "sampling.logdir=$RESULTS/samples_${m}_${prec}" \
      2>&1 | grep -viE "it/s\]" > "$log"
    echo "$m $prec: $(grep -i 'generative perplexity' "$log" | head -1) $(grep -i '^entropy' "$log" | head -1)"
  done
done

echo "=== official scores (gpt2-large) ==="
"$EXP0_DIR/.venv/bin/python" "$EXP0_DIR/score_samples.py" "$RESULTS"/samples_*/
echo "anchor sweep complete"
