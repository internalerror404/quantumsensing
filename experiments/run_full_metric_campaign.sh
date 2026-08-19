#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"${PYTHON:-python3}" "$ROOT/src/full_metric_campaign.py" \
  --output "$ROOT/../results/full_metric_campaign.json" --order 1024
