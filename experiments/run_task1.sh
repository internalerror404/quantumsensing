#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"${PYTHON:-python3}" "$ROOT/src/task1_verify.py" --output "$ROOT/../results/task1_verification.json" --svd-tol 1e-10
