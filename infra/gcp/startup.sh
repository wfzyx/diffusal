#!/usr/bin/env bash
# GCP VM startup script for the Exp 2 130M QAT cohort. Runs UNATTENDED on boot
# (passed as --metadata-from-file startup-script=...). Clones the public repo,
# rebuilds the env, runs all 12 cells, syncs artifacts to GCS, then deletes the
# instance so a spot VM never bills idle. Everything logs to the serial console
# and to gs://$BUCKET/boot.log.
#
# NOTE: first bring-up on a fresh image usually needs 1-2 dependency tweaks;
# that is expected. Watch `gcloud compute instances get-serial-port-output`.
set -uxo pipefail

# --- these are templated in by launch.sh via instance metadata ---
BUCKET="$(curl -s -H 'Metadata-Flavor: Google' http://metadata/computeMetadata/v1/instance/attributes/results-bucket)"
BD3LMS_SHA="$(curl -s -H 'Metadata-Flavor: Google' http://metadata/computeMetadata/v1/instance/attributes/bd3lms-sha)"
SELF_NAME="$(curl -s -H 'Metadata-Flavor: Google' http://metadata/computeMetadata/v1/instance/name)"
SELF_ZONE="$(curl -s -H 'Metadata-Flavor: Google' http://metadata/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')"

WORK=/opt/diffusal
exec > >(tee /var/log/diffusal-startup.log) 2>&1

echo "=== [$(date -Is)] startup begin on $SELF_NAME ($SELF_ZONE) ==="
nvidia-smi || { echo "NO GPU VISIBLE — aborting"; }

# 1. Code: public repo + bd3lms pinned at the exp0 SHA (bd3lms is gitignored in
#    diffusal, so it must be cloned separately into third_party/).
git clone --depth 1 https://github.com/wfzyx/diffusal.git "$WORK"
git -C "$WORK" clone https://github.com/kuleshov-group/bd3lms.git third_party/bd3lms
git -C "$WORK/third_party/bd3lms" checkout "$BD3LMS_SHA"

# 2. Env. The Deep Learning VM image ships CUDA + a base torch; layer the repo's
#    known pins on top (see experiments/exp0/PREP.md). Reuse the SDPA shim rather
#    than compiling flash-attn.
python3 -m venv --system-site-packages "$WORK/.venv"
V="$WORK/.venv/bin/pip"
$V install --upgrade pip
$V install "lightning==2.6.5" "setuptools<81" "numpy<2" \
           hydra-core omegaconf datasets transformers einops rich fsspec gcsfs
# torch is inherited from the image; if the image lacks it, uncomment:
# $V install torch --index-url https://download.pytorch.org/whl/cu121

export DIFFUSAL_VENV="$WORK/.venv"

# 3. Run the pre-registered 130M cohort (idempotent; survives preemption reboots).
cd "$WORK"
bash experiments/exp2/run_qat130m_seeds.sh

# 4. Ship every artifact to GCS: logs, checkpoints, hashes, pip freeze.
$DIFFUSAL_VENV/bin/pip freeze > "$WORK/experiments/exp2/ENV-qat130m.txt"
gsutil -m rsync -r "$WORK/experiments/exp2/runs/qat130m"  "gs://$BUCKET/runs/qat130m"
gsutil -m cp -r "$WORK/experiments/exp2/logs/qat130m"     "gs://$BUCKET/logs/"
gsutil cp "$WORK/experiments/exp2/CHECKPOINTS-qat130m.sha256" "gs://$BUCKET/" || true
gsutil cp "$WORK/experiments/exp2/ENV-qat130m.txt"        "gs://$BUCKET/"
gsutil cp /var/log/diffusal-startup.log                   "gs://$BUCKET/boot.log"

echo "=== [$(date -Is)] done; deleting self to stop billing ==="
# 5. Self-terminate. Guarded: only deletes if the cohort marker set completed.
if gsutil -q stat "gs://$BUCKET/runs/qat130m/**/*.ckpt"; then
  gcloud --quiet compute instances delete "$SELF_NAME" --zone "$SELF_ZONE"
else
  echo "No checkpoints synced — leaving VM up for debugging (delete manually)."
fi
