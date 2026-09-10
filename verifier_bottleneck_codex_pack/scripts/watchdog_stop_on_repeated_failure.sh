#!/usr/bin/env bash
# Stops a running series driver if runs start failing in quick succession.
#
# Motivating case: the laptop suspends with an active CUDA context, the context
# does not survive resume, and every remaining cell then fails within seconds.
# Without this, a 24-cell grid turns into 24 FAILED directories overnight. With
# it, the driver stops after the second failure and the series resumes cleanly
# from its .done markers.
#
# Usage: watchdog_stop_on_repeated_failure.sh <driver-script-name> [max_failures]
set -u
cd "$(dirname "$0")/.."

DRIVER="${1:?driver script name, e.g. run_capacity_series.sh}"
MAX_FAILURES="${2:-2}"
STARTED_AT=$(date +%s)
LOG=artifacts/watchdog.log

echo "$(date -u +%FT%TZ) watchdog armed for ${DRIVER}, threshold ${MAX_FAILURES} failures" >> "${LOG}"

while true; do
  pid=$(pgrep -f "bash .*${DRIVER}" | head -1)
  if [ -z "${pid}" ]; then
    echo "$(date -u +%FT%TZ) ${DRIVER} no longer running, watchdog exiting" >> "${LOG}"
    exit 0
  fi

  # FAILED run directories created since the watchdog started.
  failures=0
  for marker in artifacts/runs/*/FAILED; do
    [ -e "${marker}" ] || continue
    [ "$(stat -c %Y "${marker}")" -ge "${STARTED_AT}" ] && failures=$((failures + 1))
  done

  if [ "${failures}" -ge "${MAX_FAILURES}" ]; then
    echo "$(date -u +%FT%TZ) ${failures} failures since arming; stopping ${DRIVER} (pid ${pid})" >> "${LOG}"
    pkill -f "bash .*${DRIVER}"
    pkill -f "scripts/run_action_search_screen.py"
    pkill -f "scripts/train_sft.py"
    echo "$(date -u +%FT%TZ) stopped. Resume with: bash scripts/${DRIVER}" >> "${LOG}"
    exit 1
  fi

  sleep 60
done
