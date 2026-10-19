#!/usr/bin/env bash
#SBATCH --job-name=usa-b1-repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
# Activate the site's CUDA Python environment before submission, or here.
# Supply the partition/account required by the cluster through sbatch flags.
cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
bash scripts/train.sh "${1:-configs/local.yaml}"
