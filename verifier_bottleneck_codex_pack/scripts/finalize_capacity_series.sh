#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

exec > artifacts/capacity_finalize.stdout.log 2> artifacts/capacity_finalize.stderr.log
trap 'status=$?; [ "$status" -eq 0 ] || touch artifacts/capacity_finalize.failed' EXIT

while [ "$(find artifacts -maxdepth 1 -type f -name 'eval_capacity_*_seed*.done' | wc -l)" -lt 24 ]; do
  tmux has-session -t verifier-capacity 2>/dev/null || {
    echo "capacity driver stopped before 24 complete cells" >&2
    exit 1
  }
  sleep 60
done

.venv/bin/python scripts/analyze_capacity.py
.venv/bin/python scripts/analyze_ranking.py
.venv/bin/python scripts/plot_ranking_results.py
.venv/bin/python scripts/analyze_results.py --runs artifacts/runs --output artifacts/reports
touch artifacts/capacity_finalize.done
