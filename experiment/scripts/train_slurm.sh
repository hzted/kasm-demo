#!/usr/bin/env bash
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=kasm_train
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
DATA_NAME="${DATA_NAME:-trip_advisor}"
PYTHON="${PYTHON:-python}"

cd "$PROJECT_DIR"
mkdir -p logs

echo "Host: $(hostname)"
echo "CWD: $(pwd)"
echo "Start time: $(date)"
nvidia-smi || true

srun "$PYTHON" -u train.py \
  data="$DATA_NAME" \
  train.num_workers=8
