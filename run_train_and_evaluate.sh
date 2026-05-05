#!/usr/bin/env bash
# Run main portfolio training + evaluation, then Smart Tangency training + evaluation.
# Usage (from anywhere):
#   ./run_train_and_evaluate.sh
# Or pass extra CLI flags to each script (after --) — not supported; edit the script or run steps manually.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Smart Tangency modules live under envs/, core/, train/; root modules stay on default path.
export PYTHONPATH="${ROOT}:${ROOT}/envs:${ROOT}/core:${ROOT}/train"

PY="${PYTHON:-python3}"

echo "== [1/4] train.py (standard portfolio PPO) =="
"$PY" train.py

echo "== [2/4] train/train_smart.py (Smart Tangency) =="
"$PY" train/train_smart.py

echo "== [3/4] evaluate.py (standard backtest) =="
"$PY" evaluate.py

echo "== [4/4] evaluate/evaluate_smart.py (Smart Tangency backtest) =="
"$PY" evaluate/evaluate_smart.py

echo "Done. Check results/ and models/."
