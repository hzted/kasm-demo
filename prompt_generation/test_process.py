"""Preprocess TripAdvisor raw text into CSV files used by KASM.

The original TripAdvisor files are expected to contain records separated by
`<ssssss>`. Each record begins with eight numeric ratings, followed by extracted
opinion terms and the raw review text. This script keeps the preprocessing light:
it removes noisy markup/characters, normalizes whitespace, improves sentence
splitting, and writes stable CSV columns for prompt generation and KASM training.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List

import pandas as pd


RATING_COLUMNS = [
    "overall",
    "value",
    "room",
    "location",
    "clean",
    "checkin",
    "service",
    "business",
]

OUTPUT_COLUMNS = [
    "id",
    "mode",
    "text",
    "sentences",
    *RATING_COLUMNS,
    "opinion",
    "sub_sentence",
]

QUOTE_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00b4": "'",
    "\u00a8": "'",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
})

BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
SOFT_BOUNDARY_RE = re.compile(r"\s*(?:\.\.\.|…|[!?]{2,})\s*")
TAG_RE = re.compile(r"<\s*br\s*/?\s*>|</?p>|</?div>", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def is_rating_line(text: str) -> bool:
    parts = str(text).strip().split()
    return len(parts) == len(RATING_COLUMNS) and all(part.strip("-").isdigit() for part in parts)


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(QUOTE_TRANSLATION)
    text = TAG_RE.sub(". ", text)
    text = text.replace("-lrb-", "(").replace("-rrb-", ")")
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("\\", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def split_sentences(text: str, min_tokens: int = 1) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    text = SOFT_BOUNDARY_RE.sub(". ", text)
    pieces: List[str] = []
    for chunk in BOUNDARY_RE.split(text):
        pieces.append(chunk)

    sentences = []
    for piece in pieces:
        sent = clean_text(piece).strip(" ;")
        if sent and re.search(r"[A-Za-z0-9]", sent) and len(sent.split()) >= min_tokens:
            sentences.append(sent)
    return sentences or [text]


def split_opinions(text: str) -> str:
    terms = [clean_text(term) for term in str(text).split("\t")]
    terms = [term for term in terms if term]
    return " ; ".join(terms)


def iter_raw_entries(content: str) -> Iterable[str]:
    for entry in content.split("<ssssss>"):
        entry = entry.strip()
        if entry:
            yield entry


def parse_tripadvisor_split(mode: str, txt_path: Path) -> pd.DataFrame:
    entries = list(iter_raw_entries(read_text(txt_path)))
    records = []
    i = 0

    while i < len(entries):
        entry = entries[i].strip()
        first_field = entry.split("\t", 1)[0]
        if not is_rating_line(first_field):
            i += 1
            continue

        ratings = first_field.split()
        rest = entry[len(first_field):].strip()
        matches = list(re.finditer(r"\s{2,}", rest))
        if not matches:
            i += 1
            continue

        split_at = matches[-1]
        opinion_raw = rest[:split_at.start()].strip()
        review_parts = [rest[split_at.end():].strip()]

        j = i + 1
        while j < len(entries):
            next_entry = entries[j].strip()
            next_first = next_entry.split("\t", 1)[0]
            if is_rating_line(next_first):
                break
            review_parts.append(next_entry)
            j += 1

        paragraph_parts = [clean_text(part) for part in review_parts if clean_text(part)]
        full_review = clean_text(" ".join(paragraph_parts))
        sentences = split_sentences(full_review)
        opinion = split_opinions(opinion_raw)
        sub_sentences = sentences

        if full_review:
            record = {
                "mode": mode,
                "text": full_review,
                "sentences": sentences,
                "opinion": opinion,
                "sub_sentence": sub_sentences,
            }
            record.update(dict(zip(RATING_COLUMNS, ratings)))
            records.append(record)

        i = j

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df.insert(0, "id", range(len(df)))
    return df[OUTPUT_COLUMNS]


def write_split(mode: str, input_path: Path, output_path: Path) -> None:
    df = parse_tripadvisor_split(mode, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    print(f"{mode}: {len(df)} rows -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw TripAdvisor text splits for KASM.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "trip_advisor_original",
        help="Directory containing raw train/dev/test files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiment" / "dataset" / "trip_advisor",
        help="Directory where train/dev/test CSV files will be written.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"], help="Splits to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        input_path = args.input_dir / split
        output_path = args.output_dir / f"{split}.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing raw split file: {input_path}")
        write_split(split, input_path, output_path)


if __name__ == "__main__":
    main()
