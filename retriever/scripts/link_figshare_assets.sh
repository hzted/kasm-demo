#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${ROOT}/retriever/assets"

if [ "$#" -ne 1 ]; then
  echo "Usage: bash retriever/scripts/link_figshare_assets.sh /path/to/figshare_kasm_retriever_32970428[/organized]" >&2
  exit 2
fi

SOURCE="$1"
if [ -d "${SOURCE}/organized" ]; then
  SOURCE="${SOURCE}/organized"
fi

if [ ! -d "${SOURCE}/trip_advisor" ] || [ ! -d "${SOURCE}/beer_advocate" ]; then
  echo "Expected trip_advisor/ and beer_advocate/ under: ${SOURCE}" >&2
  exit 1
fi

mkdir -p "${TARGET}"
ln -sfn "$(cd "${SOURCE}/trip_advisor" && pwd)" "${TARGET}/trip_advisor"
ln -sfn "$(cd "${SOURCE}/beer_advocate" && pwd)" "${TARGET}/beer_advocate"

if [ -d "${SOURCE}/code" ]; then
  ln -sfn "$(cd "${SOURCE}/code" && pwd)" "${TARGET}/figshare_code"
fi

echo "Linked retriever assets:"
echo "  ${TARGET}/trip_advisor -> $(readlink "${TARGET}/trip_advisor")"
echo "  ${TARGET}/beer_advocate -> $(readlink "${TARGET}/beer_advocate")"
