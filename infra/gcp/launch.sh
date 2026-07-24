#!/usr/bin/env bash
# Provision a spot L4 VM that runs the Exp 2 130M QAT cohort unattended and
# self-deletes when done (see startup.sh). Run this LOCALLY once GPU quota is
# granted. It is idempotent-ish: re-running creates a NEW instance name.
#
# PRECONDITIONS (checked below):
#   - GPUS_ALL_REGIONS >= 1 and PREEMPTIBLE_NVIDIA_L4_GPUS >= 1 in $ZONE's region
#   - configs/exp2-qat130m.yaml committed + tagged (pre-registration frozen)
set -euo pipefail

PROJECT="${PROJECT:-hellow-484216}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE="${INSTANCE:-diffusal-qat130m-$(date +%m%d-%H%M)}"
BUCKET="${BUCKET:-${PROJECT}-diffusal-qat130m}"
BD3LMS_SHA="1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7"   # from configs/exp0.yaml
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== preflight: GPU quota ==="
gpu=$(gcloud compute project-info describe --project "$PROJECT" --flatten="quotas[]" \
      --format="value(quotas.limit)" --filter="quotas.metric=GPUS_ALL_REGIONS" 2>/dev/null)
[ "${gpu%.*}" -ge 1 ] 2>/dev/null || { echo "GPUS_ALL_REGIONS is ${gpu:-0} (<1). Request the increase first."; exit 1; }

echo "=== results bucket ==="
gsutil ls -b "gs://$BUCKET" >/dev/null 2>&1 || gsutil mb -p "$PROJECT" -l us-central1 "gs://$BUCKET"

echo "=== creating spot L4 instance: $INSTANCE ==="
gcloud compute instances create "$INSTANCE" \
  --project "$PROJECT" --zone "$ZONE" \
  --machine-type g2-standard-8 \
  --accelerator "type=nvidia-l4,count=1" \
  --provisioning-model SPOT \
  --instance-termination-action DELETE \
  --maintenance-policy TERMINATE \
  --image-family common-cu123-ubuntu-2204-py310 \
  --image-project deeplearning-platform-release \
  --boot-disk-size 100GB --boot-disk-type pd-balanced \
  --scopes cloud-platform \
  --metadata "results-bucket=${BUCKET},bd3lms-sha=${BD3LMS_SHA},install-nvidia-driver=True" \
  --metadata-from-file "startup-script=${HERE}/startup.sh"

cat <<EOF

Launched. The VM will: install driver -> clone repo + bd3lms@${BD3LMS_SHA:0:7}
-> build env -> run 12 QAT cells -> rsync to gs://${BUCKET}/ -> delete itself.

Watch progress:
  gcloud compute instances get-serial-port-output $INSTANCE --zone $ZONE --project $PROJECT | tail -40
Pull results when done:
  gsutil -m cp -r gs://${BUCKET}/runs/qat130m experiments/exp2/runs/
  gsutil cp gs://${BUCKET}/CHECKPOINTS-qat130m.sha256 experiments/exp2/
If the VM is still up after many hours, check gs://${BUCKET}/boot.log for the failing step.
EOF
