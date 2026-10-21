#!/bin/bash
set -euo pipefail
cd /data/run01/scze971/usa_moe_gmm_fresh_ft_v3_20261021
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
PY=/data/run01/scze971/maize_paper_workspace/runs/france_agrifm_b1_fullft_n16r4_20261016/prep-venv/bin/python
case "$1" in
 validate) "$PY" validate_gpu.py ;;
 train)
   CONFIG=$("$PY" -c 'import json,os;print(json.load(open("order.json"))[int(os.environ["SLURM_ARRAY_TASK_ID"])])')
   "$PY" train.py --config "$CONFIG" ;;
 test) "$PY" evaluate.py --index "$SLURM_ARRAY_TASK_ID" ;;
 freeze) "$PY" campaign.py freeze ;;
 summary) "$PY" summarize.py ;;
 *) exit 2 ;;
esac
