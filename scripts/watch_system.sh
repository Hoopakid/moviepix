#!/bin/sh
# Watch the Mac while a training run is going. Logs a line every few minutes and
# exits (so the caller is alerted) when the run ends or the Mac is under pressure.
#   scripts/watch_system.sh <run-dir> [interval-seconds]
RUN="$1"; EVERY="${2:-180}"; LOG="$RUN/system.log"
while true; do
  free=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  swap=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}')
  therm=$(pmset -g therm | grep -v "^Note" | grep -ci "warning level")  # real warnings only; "No ... recorded" notes are fine
  step=$(tail -1 "$RUN/log.jsonl" 2>/dev/null | sed -E 's/.*"step": ([0-9]+).*/\1/')
  rate=$(grep examples_per_s "$RUN/log.jsonl" 2>/dev/null | tail -1 | sed -E 's/.*"examples_per_s": ([0-9.]+).*/\1/')
  gpu=$(grep gpu_gb "$RUN/log.jsonl" 2>/dev/null | tail -1 | sed -E 's/.*"gpu_gb": ([0-9.]+).*/\1/')
  echo "$(date +%H:%M) free=${free}% swap=${swap}MB step=$step ex/s=$rate gpu=${gpu}GB" >> "$LOG"
  if ! pgrep -f "moviepix-train" >/dev/null; then echo "RUN ENDED at step $step"; tail -3 "$LOG"; exit 0; fi
  if [ "${free:-100}" -lt 15 ]; then echo "ALERT: memory free only ${free}%"; tail -3 "$LOG"; exit 1; fi
  if [ "${swap:-0}" -gt 5000 ]; then echo "ALERT: swap ${swap}MB"; tail -3 "$LOG"; exit 1; fi
  if [ "$therm" -gt 0 ]; then echo "ALERT: thermal warning"; pmset -g therm; exit 1; fi
  sleep "$EVERY"
done
