#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CKPT:-}" ]]; then
  echo "Set CKPT=/path/to/checkpoint.ckpt before running evaluation." >&2
  exit 1
fi
GPU="${GPU:-0}"

python -u evaluate.py \
  data=trip_advisor \
  ckpt_path="$CKPT" \
  gpu="$GPU"
