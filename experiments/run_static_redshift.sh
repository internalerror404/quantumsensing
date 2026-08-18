#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"${PYTHON:-python3}" "$ROOT/src/static_redshift_experiment.py" \
  --output "$ROOT/../results/static_redshift_experiment.json" \
  --order 1024 \
  --svd-tol 1e-10
