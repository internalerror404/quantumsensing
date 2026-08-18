#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
python "$ROOT/src/train_selector.py" \
  --output "$ROOT/../results/selector_checkpoint.pt" \
  --report "$ROOT/../results/selector_report.json" \
  --steps 250 --batch 8 --k 16 --objective E --seed 20260818
