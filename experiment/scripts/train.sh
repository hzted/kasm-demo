#!/usr/bin/env bash
set -euo pipefail

DATA_NAME="${DATA_NAME:-trip_advisor}"
GPU="${GPU:-0}"

python -u train.py \
  data="$DATA_NAME" \
  gpu="$GPU"
