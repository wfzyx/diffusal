#!/usr/bin/env bash
# exits 2 if any python train.py process is alive (anchored on argv[0], never matches the calling shell)
if pgrep -f '^\S*python\S* train\.py' >/dev/null; then echo "ABORT: training process active"; pgrep -fa '^\S*python\S* train\.py'; exit 2; fi
