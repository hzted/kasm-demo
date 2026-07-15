#!/usr/bin/env python3
"""Validate released KASM CSV tables and retrieved JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ("trip_advisor", "beer_advocate")
SPLITS = ("train", "dev", "test")


def csv_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        return [row[0] for index, row in enumerate(csv.reader(handle)) if index > 0]


def jsonl_groups(path: Path) -> tuple[list[str], int]:
    groups: list[str] = []
    current = object()
    sentence_rows = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row_id = json.loads(line)["row"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            sentence_rows += 1
            if row_id != current:
                groups.append(str(row_id))
                current = row_id
    return groups, sentence_rows


def validate_split(dataset_root: Path, dataset: str, split: str) -> None:
    csv_path = dataset_root / dataset / f"{split}.csv"
    jsonl_path = dataset_root / dataset / f"{split}.jsonl"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    ids = csv_ids(csv_path)
    groups, sentence_rows = jsonl_groups(jsonl_path)
    if len(ids) != len(groups):
        raise ValueError(
            f"{dataset}/{split}: CSV docs={len(ids)} but JSONL groups={len(groups)}"
        )
    for index, (csv_id, jsonl_id) in enumerate(zip(ids, groups)):
        if str(csv_id) != str(jsonl_id):
            raise ValueError(
                f"{dataset}/{split}: row mismatch at position {index}: "
                f"csv id={csv_id}, jsonl row={jsonl_id}"
            )
    print(f"{dataset}/{split}: docs={len(ids)} sentence_rows={sentence_rows} OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
    )
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        for split in SPLITS:
            validate_split(args.dataset_root, dataset, split)


if __name__ == "__main__":
    main()
