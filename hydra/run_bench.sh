#!/usr/bin/env bash
# Main benchmark: 64 symbols, train n in [2,16], eval to n=64. Spotlight runs get 1 thread each,
# attn/gdn run sequentially on 2 threads. Status lines land in runs/status_bench.
set -uo pipefail
c="--steps 3000 --batch 32 --lr 3e-3 --n-sym 64 --n-min 2 --n-max 16 --queries 8 --eval-n 8,16,32,48,64"
run() { local name=$1; shift; .venv/bin/python train.py "$@" $c --out runs/$name.json > runs/$name.log 2>&1; echo "$name exit=$?" >> runs/status_bench; }
run spotlight_d1 --mixer spotlight --depth 1 --threads 1 &
run spotlight_d2 --mixer spotlight --depth 2 --threads 1 &
( for d in 1 2; do for m in attn gdn; do run ${m}_d$d --mixer $m --depth $d --threads 2; done; done ) &
wait
