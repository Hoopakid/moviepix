#!/bin/sh
# Phase 2 E1: wait for dataset v3, then train c32v3_aug (low priority, GPU memory capped by the config).
cd "$(dirname "$0")/.."
echo "Waiting for dataset lpc32v3 to finish building..."
while [ ! -f data/built/lpc32v3/dataset.json ]; do sleep 30; done
echo "Dataset ready. Starting E1 training."
caffeinate -i -s nice -n 10 uv run moviepix-train configs/model/c32v3_aug.toml
