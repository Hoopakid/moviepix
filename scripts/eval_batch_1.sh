#!/bin/sh
# Wave 3 evaluation batch: colour-aggregated decoding + detail metrics, then out-of-style walk.
set -e
for r in 20261117-1931-copy_animator 20261118-1627-c32v2_new 20261118-0016-copy_animator_64 20261118-1032-copy_animator_64_sc; do
  echo "=== decode $r"
  nice -n 5 uv run python scripts/decode_experiments.py data/runs/$r --set agg 2>&1 | grep -v Warn
done
echo "=== out-of-style walk"
nice -n 5 uv run python scripts/oos_walk.py data/runs/20261117-1931-copy_animator data/runs/20261118-1627-c32v2_new \
  data/runs/20261118-0016-copy_animator_64 data/runs/20261118-1032-copy_animator_64_sc --device mps 2>&1 | grep -v Warn
