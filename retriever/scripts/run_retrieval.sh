#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOMAIN="${1:-trip_advisor}"
SPLIT="${2:-dev}"
ASSETS="${KASM_RETRIEVER_ASSETS:-${ROOT}/retriever/assets}"

case "${DOMAIN}" in
  trip|trip_advisor|tripadvisor)
    ACTIVE_DOMAIN="trip"
    DATASET="trip_advisor"
    ASSET_DOMAIN="trip_advisor"
    ;;
  beer|beer_advocate|ba)
    ACTIVE_DOMAIN="beer"
    DATASET="beer_advocate"
    ASSET_DOMAIN="beer_advocate"
    ;;
  *)
    echo "Unsupported domain: ${DOMAIN}. Use trip_advisor or beer_advocate." >&2
    exit 2
    ;;
esac

ASSET_DIR="${ASSETS}/${ASSET_DOMAIN}"
INPUT_CSV="${INPUT_CSV:-${ROOT}/experiment/dataset/${DATASET}/${SPLIT}.csv}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${ROOT}/experiment/dataset/${DATASET}/${SPLIT}.jsonl}"

for required in "${ASSET_DIR}/kb.jsonl" "${ASSET_DIR}/best_model.pt" "${ASSET_DIR}/tokenizer/tokenizer_config.json" "${INPUT_CSV}"; do
  if [ ! -e "${required}" ]; then
    echo "Missing required file: ${required}" >&2
    echo "If assets are not linked yet, run:" >&2
    echo "  bash retriever/scripts/link_figshare_assets.sh /path/to/figshare_kasm_retriever_32970428" >&2
    exit 1
  fi
done

if [ -e "${OUTPUT_JSONL}" ] && [ "${OVERWRITE:-0}" != "1" ]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_JSONL}" >&2
  echo "Set OVERWRITE=1 to regenerate this JSONL in place, or set OUTPUT_JSONL=/path/to/output.jsonl." >&2
  exit 1
fi

export ACTIVE_DOMAIN
export KB_PATH="${KB_PATH:-${ASSET_DIR}/kb.jsonl}"
export ASPECT_WORDS_PATH="${ASPECT_WORDS_PATH:-${ASSET_DIR}/aspect.words}"
export COLBERT_CKPT_DIR="${COLBERT_CKPT_DIR:-${ASSET_DIR}}"
export INPUT_CSV
export OUTPUT_JSONL

echo "[Retriever] domain=${DATASET} split=${SPLIT}"
echo "[Retriever] input=${INPUT_CSV}"
echo "[Retriever] output=${OUTPUT_JSONL}"
python "${ROOT}/retriever/src/faiss_colbert_confidence.py"
