#!/bin/bash
set -euo pipefail
cd /data/run01/scze971/usa_moe_gmm_fresh_ft_v3_20261021
mkdir -p logs validation outputs
PY=/data/run01/scze971/maize_paper_workspace/runs/france_agrifm_b1_fullft_n16r4_20261016/prep-venv/bin/python
if [[ "${1:-validate}" == validate ]]; then
 test ! -e validation_job.json
 validate=$(sbatch --parsable --partition=vip_gpu_scze971 --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G --time=01:00:00 --job-name=mgf3_validate --output="$PWD/logs/validate-%j.out" --error="$PWD/logs/validate-%j.err" run_array.sh validate)
 printf '{"validation":"%s","formal_training_submitted":false}\n' "$validate" > validation_job.json
 cat validation_job.json
 exit 0
fi
[[ "$1" == train ]]
test ! -e jobs.json
"$PY" -c 'import json;from common import verify;v=json.load(open("validation/gpu.json"));assert v["passed"] and v["source_manifest"]==verify()'
train=$(sbatch --parsable --array=0-1%2 --partition=vip_gpu_scze971 --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G --time=3-00:00:00 --job-name=mgf3_train --output="$PWD/logs/train-%A_%a.out" --error="$PWD/logs/train-%A_%a.err" run_array.sh train)
freeze=$(sbatch --parsable --dependency="afterok:${train%%;*}" --kill-on-invalid-dep=yes --partition=vip_gpu_scze971 --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G --time=00:30:00 --job-name=mgf3_freeze --output="$PWD/logs/freeze-%j.out" --error="$PWD/logs/freeze-%j.err" run_array.sh freeze)
testjob=$(sbatch --parsable --array=0-5%2 --dependency="afterok:${freeze%%;*}" --kill-on-invalid-dep=yes --partition=vip_gpu_scze971 --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G --time=02:00:00 --job-name=mgf3_test --output="$PWD/logs/test-%A_%a.out" --error="$PWD/logs/test-%A_%a.err" run_array.sh test)
summary=$(sbatch --parsable --dependency="afterok:${testjob%%;*}" --kill-on-invalid-dep=yes --partition=vip_gpu_scze971 --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G --time=00:10:00 --job-name=mgf3_report --output="$PWD/logs/summary-%j.out" --error="$PWD/logs/summary-%j.err" run_array.sh summary)
printf '{"training":"%s","freeze":"%s","testing":"%s","summary":"%s","max_concurrent_gpus":2}\n' "$train" "$freeze" "$testjob" "$summary" > jobs.json
cat jobs.json
