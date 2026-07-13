#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOMAIN="${1:-beer_advocate}"
ASSETS="${KASM_RETRIEVER_ASSETS:-${ROOT}/retriever/assets}"

case "${DOMAIN}" in
  trip|trip_advisor|tripadvisor)
    TRAIN_DOMAIN="trip_advisor"
    ASSET_DOMAIN="trip_advisor"
    ;;
  beer|beer_advocate|ba)
    TRAIN_DOMAIN="beer_advocate"
    ASSET_DOMAIN="beer_advocate"
    ;;
  *)
    echo "Unsupported domain: ${DOMAIN}. Use trip_advisor or beer_advocate." >&2
    exit 2
    ;;
esac

CSV="${CSV:-${ASSETS}/${ASSET_DOMAIN}/new_mixed_constrative_triplets_balanced.csv}"
OUT_DIR="${OUT_DIR:-${ROOT}/retriever/outputs/${ASSET_DOMAIN}/ckpt_colbert_deberta_v3_inbatch_aux}"

if [ ! -e "${CSV}" ]; then
  echo "Missing training CSV: ${CSV}" >&2
  exit 1
fi

python "${ROOT}/retriever/src/train_colbert_contrastive_inbatch_aux.py" \
  --domain "${TRAIN_DOMAIN}" \
  --csv "${CSV}" \
  --out_dir "${OUT_DIR}" \
  "$@"
