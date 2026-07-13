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
  case "${domain}" in
    trip|trip_advisor|tripadvisor)
      install_one "${ASSETS}/trip_advisor/train_Faiss_matches_by_trigger_confidence.jsonl" "${ROOT}/experiment/dataset/trip_advisor/train.jsonl"
      install_one "${ASSETS}/trip_advisor/dev_Faiss_matches_by_trigger_confidence.jsonl" "${ROOT}/experiment/dataset/trip_advisor/dev.jsonl"
      install_one "${ASSETS}/trip_advisor/test_Faiss_matches_by_trigger_confidence.jsonl" "${ROOT}/experiment/dataset/trip_advisor/test.jsonl"
      ;;
    beer|beer_advocate|ba)
      install_one "${ASSETS}/beer_advocate/train_Faiss_matches_by_trigger.jsonl" "${ROOT}/experiment/dataset/beer_advocate/train.jsonl"
      install_one "${ASSETS}/beer_advocate/dev_Faiss_matches_by_trigger.jsonl" "${ROOT}/experiment/dataset/beer_advocate/dev.jsonl"
      install_one "${ASSETS}/beer_advocate/test_Faiss_matches_by_trigger.jsonl" "${ROOT}/experiment/dataset/beer_advocate/test.jsonl"
      ;;
    all)
      install_domain trip_advisor
      install_domain beer_advocate
      ;;
    *)
      echo "Usage: bash retriever/scripts/install_precomputed_jsonl.sh [trip_advisor|beer_advocate|all]" >&2
      exit 2
      ;;
  esac
}

install_domain "${1:-all}"
