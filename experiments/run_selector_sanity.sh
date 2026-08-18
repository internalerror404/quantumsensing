#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"${PYTHON:-python3}" "$ROOT/src/train_selector.py" \
  --output "$ROOT/../results/selector_checkpoint.pt" \
  --report "$ROOT/../results/selector_report.json" \
  --steps 1200 --batch 8 --objective E --seed 20260818
