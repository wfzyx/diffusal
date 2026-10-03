#!/usr/bin/env bash
# Pointer chasing, depth 2, mixed {1,2} training, comma format. Status in runs/status_chase.
set -uo pipefail
c="--depth 2 --train-depths 1,2 --fmt comma --steps 3000 --batch 32 --lr 3e-3 --n-sym 64 --n-min 2 --n-max 16 --queries 8 --eval-n 8,16,32 --threads 1"
run() { local name=$1; shift; .venv/bin/python train.py "$@" $c --out runs/$name.json > runs/$name.log 2>&1; echo "$name exit=$?" >> runs/status_chase; }
run chase_spotlight --mixer spotlight &
run chase_gdn --mixer gdn &
wait
