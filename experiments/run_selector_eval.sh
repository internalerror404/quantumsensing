#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"${PYTHON:-python3}" "$ROOT/src/evaluate_selector.py" \
  --checkpoint "$ROOT/../results/selector_checkpoint.pt" \
  --output "$ROOT/../results/selector_evaluation.json" \
  --tasks 100 --seed 2026
