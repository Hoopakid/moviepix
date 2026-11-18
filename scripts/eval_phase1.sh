#!/bin/sh
# Phase 1: clean comparison c32v2_new (alpha head + self-cond) vs c32v2_control, same data and budget.
set -e
NEW=data/runs/20261118-1627-c32v2_new
CTL=data/runs/20261118-2227-c32v2_control
for split in validation unseen; do
  for r in $CTL $NEW; do
    echo "=== decode $r $split"
    nice -n 5 uv run python scripts/decode_experiments.py $r --set agg --split $split 2>&1 | grep -v Warn
  done
done
echo "=== out-of-style walk"
nice -n 5 uv run python scripts/oos_walk.py $CTL $NEW --device mps --out docs/research/oos-walk-phase1.json 2>&1 | grep -v Warn
