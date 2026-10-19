#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
CONFIG="${1:-configs/local.yaml}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python scripts/preflight.py --config "$CONFIG"
python -u scripts/train_entry.py --config "$CONFIG"
