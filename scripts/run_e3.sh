#!/bin/sh
# Phase 2 E3: train c32v3_packs for one seed (~5.5 h, low priority, GPU memory capped by the config),
# then evaluate it against E1b with the frozen pass marks (docs/research/phase2-design.md).
# Run from anywhere, one seed at a time:   sh scripts/run_e3.sh 1     then later   sh scripts/run_e3.sh 2
# Ctrl+C stops training; continue with:  uv run moviepix-train --resume data/runs/<run>
set -e
SEED=${1:?give the seed: 1 or 2}
cd "$(dirname "$0")/.."
E1B=data/runs/20261124-1816-c32v3_gentle   # the reference run for every pass mark
[ -f data/built/packs32/manifest.jsonl ] || { echo "data/built/packs32 is missing: run scripts/build_packs32.py"; exit 1; }

echo "[$(date '+%H:%M')] Training E3 seed $SEED (c32v3_packs). Ctrl+C stops it; the last checkpoint is kept."
caffeinate -i -s nice -n 10 uv run moviepix-train configs/model/c32v3_packs.toml --seed "$SEED" --name "c32v3_packs_s$SEED"

RUN=$(ls -dt data/runs/*-c32v3_packs_s"$SEED" | head -1)
echo "[$(date '+%H:%M')] Training done: $RUN. Evaluating."
{
  echo "=== 1. LPC validation (pass: exact >= 0.772)"
  caffeinate -i nice -n 5 uv run python scripts/decode_experiments.py "$RUN" --set agg --split validation 2>&1 | grep -v Warn
  echo "=== 2-3. dev packs, in-order motion and appearance, paired against E1b"
  caffeinate -i nice -n 5 uv run python scripts/oos_score.py "$E1B" "$RUN" --device mps --against "$(basename $E1B)" \
      --out "docs/research/oos-score-e3-s$SEED.json" 2>&1 | grep -v Warn
} | tee "docs/research/e3-s$SEED-eval.log"
echo "[$(date '+%H:%M')] All done. Results: docs/research/e3-s$SEED-eval.log"
