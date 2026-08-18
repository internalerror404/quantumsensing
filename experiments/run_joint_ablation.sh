#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"${PYTHON:-python3}" "$ROOT/src/joint_delay_redshift_experiment.py" \
  --output "$ROOT/../results/joint_delay_redshift.json" \
  --order 1024 \
  --svd-tol 1e-10 \
  --seed 2026
