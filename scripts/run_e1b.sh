#!/bin/sh
# Phase 2 E1b: train c32v3_gentle (~5 h, low priority, GPU memory capped by the config),
# then evaluate it automatically (~30 min). Results land in docs/research/e1b-*.
# Run from anywhere:  sh scripts/run_e1b.sh
set -e
cd "$(dirname "$0")/.."
NEW=data/runs/20261118-1627-c32v2_new   # Phase 1 winner, the bar to match on LPC
E1=data/runs/20261119-0433-c32v3_aug    # E1, the bar to keep out of style

echo "[$(date '+%H:%M')] Training E1b (c32v3_gentle). Ctrl+C stops it; the last checkpoint is kept."
caffeinate -i -s nice -n 10 uv run moviepix-train configs/model/c32v3_gentle.toml

RUN=$(ls -dt data/runs/*-c32v3_gentle | head -1)
echo "[$(date '+%H:%M')] Training done: $RUN. Evaluating."
{
  for split in validation unseen; do
    echo "=== decode E1b $split"
    caffeinate -i nice -n 5 uv run python scripts/decode_experiments.py "$RUN" --set agg --split $split 2>&1 | grep -v Warn
  done
  echo "=== out-of-style walk"
  caffeinate -i nice -n 5 uv run python scripts/oos_walk.py $NEW $E1 "$RUN" --device mps \
      --out docs/research/oos-walk-e1b.json 2>&1 | grep -v Warn
  echo "=== motion (share of pixels changed from the reference)"
  caffeinate -i nice -n 5 uv run python scripts/oos_motion.py $NEW $E1 "$RUN" --device mps 2>&1 | grep -v Warn
} | tee docs/research/e1b-eval.log
echo "[$(date '+%H:%M')] All done. Results: docs/research/e1b-eval.log"
