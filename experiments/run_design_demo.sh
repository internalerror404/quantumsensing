#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
python "$ROOT/src/design_experiment.py" --output "$ROOT/../results/design_demo.json" --k 16 --seed 20260818
