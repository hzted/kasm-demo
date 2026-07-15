#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS="${KASM_RETRIEVER_ASSETS:-${ROOT}/retriever/assets}"
MODE="${MODE:-symlink}"

install_one() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [ ! -e "${src}" ]; then
    echo "Missing source: ${src}" >&2
    exit 1
  fi
  if [ -e "${dst}" ] || [ -L "${dst}" ]; then
    if [ "${OVERWRITE:-0}" != "1" ]; then
      echo "Refusing to overwrite existing file: ${dst}" >&2
      echo "Set OVERWRITE=1 to replace it." >&2
      exit 1
    fi
    rm -f "${dst}"
  fi
  if [ "${MODE}" = "copy" ]; then
    cp "${src}" "${dst}"
  else
    rel_src="$(python - <<'PY' "${src}" "$(dirname "${dst}")"
import os
import sys
print(os.path.relpath(sys.argv[1], sys.argv[2]))
PY
)"
    ln -s "${rel_src}" "${dst}"
  fi
  echo "${dst} <- ${src}"
}

install_domain() {
  local domain="$1"
  local table_domain json_domain
  case "${domain}" in
    trip|trip_advisor|tripadvisor)
      table_domain="trip_advisor"
      json_domain="trip_advisor"
      install_one "${ASSETS}/tables/${table_domain}/train.csv" "${ROOT}/experiment/dataset/${table_domain}/train.csv"
      install_one "${ASSETS}/tables/${table_domain}/dev.csv" "${ROOT}/experiment/dataset/${table_domain}/dev.csv"
      install_one "${ASSETS}/tables/${table_domain}/test.csv" "${ROOT}/experiment/dataset/${table_domain}/test.csv"
      install_one "${ASSETS}/${json_domain}/train_Faiss_matches_by_trigger_confidence.jsonl" "${ROOT}/experiment/dataset/${table_domain}/train.jsonl"
      install_one "${ASSETS}/${json_domain}/dev_Faiss_matches_by_trigger_confidence.jsonl" "${ROOT}/experiment/dataset/${table_domain}/dev.jsonl"
      install_one "${ASSETS}/${json_domain}/test_Faiss_matches_by_trigger_confidence.jsonl" "${ROOT}/experiment/dataset/${table_domain}/test.jsonl"
      ;;
    beer|beer_advocate|ba)
      table_domain="beer_advocate"
      json_domain="beer_advocate"
      install_one "${ASSETS}/tables/${table_domain}/train.csv" "${ROOT}/experiment/dataset/${table_domain}/train.csv"
      install_one "${ASSETS}/tables/${table_domain}/dev.csv" "${ROOT}/experiment/dataset/${table_domain}/dev.csv"
      install_one "${ASSETS}/tables/${table_domain}/test.csv" "${ROOT}/experiment/dataset/${table_domain}/test.csv"
      install_one "${ASSETS}/${json_domain}/train_Faiss_matches_by_trigger.jsonl" "${ROOT}/experiment/dataset/${table_domain}/train.jsonl"
      install_one "${ASSETS}/${json_domain}/dev_Faiss_matches_by_trigger.jsonl" "${ROOT}/experiment/dataset/${table_domain}/dev.jsonl"
      install_one "${ASSETS}/${json_domain}/test_Faiss_matches_by_trigger.jsonl" "${ROOT}/experiment/dataset/${table_domain}/test.jsonl"
      ;;
    all)
      install_domain trip_advisor
      install_domain beer_advocate
      ;;
    *)
      echo "Usage: bash retriever/scripts/install_released_data.sh [trip_advisor|beer_advocate|all]" >&2
      exit 2
      ;;
  esac
}

if [ ! -d "${ASSETS}/tables" ]; then
  echo "Missing released CSV tables under ${ASSETS}/tables." >&2
  echo "The Figshare bundle must include tables/<dataset>/{train,dev,test}.csv for one-command training." >&2
  exit 1
fi

install_domain "${1:-all}"
