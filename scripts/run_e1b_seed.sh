#!/bin/sh
# Phase 2 seed check: train E1b's config (c32v3_gentle) again with another seed (~5.5 h, low priority,
# GPU memory capped by the config), then score it the same way as E1b so the seed noise is measured.
# Run from anywhere:   sh scripts/run_e1b_seed.sh 2
# Ctrl+C stops training; continue with:  uv run moviepix-train --resume data/runs/<run>
set -e
SEED=${1:?give the seed, e.g. 2 (E1b itself is seed 1)}
cd "$(dirname "$0")/.."
E1B=data/runs/20261124-1816-c32v3_gentle   # E1b, seed 1

echo "[$(date '+%H:%M')] Training E1b config, seed $SEED (c32v3_gentle_s$SEED). Ctrl+C stops it; the last checkpoint is kept."
caffeinate -i -s nice -n 10 uv run moviepix-train configs/model/c32v3_gentle.toml --seed "$SEED" --name "c32v3_gentle_s$SEED"

RUN=$(ls -dt data/runs/*-c32v3_gentle_s"$SEED" | head -1)
echo "[$(date '+%H:%M')] Training done: $RUN. Evaluating."
{
  echo "=== 1. LPC validation (E1b seed 1, 1-shot colour: exact 0.7816)"
  caffeinate -i nice -n 5 uv run python scripts/decode_experiments.py "$RUN" --set agg --split validation 2>&1 | grep -v Warn
  echo "=== 2-3. dev packs, in-order motion and appearance, paired against E1b seed 1"
  caffeinate -i nice -n 5 uv run python scripts/oos_score.py "$E1B" "$RUN" --device mps --against "$(basename $E1B)" \
      --out "docs/research/oos-score-e1b-s$SEED.json" 2>&1 | grep -v Warn
} | tee "docs/research/e1b-s$SEED-eval.log"
echo "[$(date '+%H:%M')] All done. Results: docs/research/e1b-s$SEED-eval.log"
