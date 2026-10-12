#!/usr/bin/env python3
"""Download and organize the anonymous KASM release bundle."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


RELEASE_ARTICLE_ID = "32988389"
RELEASE_SHARE_TOKEN = "132e83cda32c9b18aca7"

PRIVATE_TRIP_ADVISOR_FILES = {
    "retriever/TripAdvisor/trip_data/aspect.words": "aspect.words",
    "retriever/TripAdvisor/trip_data/kb.jsonl": "kb.jsonl",
    "retriever/TripAdvisor/trip_retrieve_results/dev_Faiss_matches_by_trigger_confidence.jsonl": "dev_Faiss_matches_by_trigger_confidence.jsonl",
    "retriever/TripAdvisor/trip_retrieve_results/test_Faiss_matches_by_trigger_confidence.jsonl": "test_Faiss_matches_by_trigger_confidence.jsonl",
    "retriever/TripAdvisor/trip_retrieve_results/train_Faiss_matches_by_trigger_confidence.jsonl": "train_Faiss_matches_by_trigger_confidence.jsonl",
    "retriever/TripAdvisor/trip_contrastive/new_mixed_constrative_triplets_balanced.csv": "new_mixed_constrative_triplets_balanced.csv",
    "retriever/TripAdvisor/trip_ckpt/ckpt_colbert_deberta_v3_inbatch_aux/best_model.pt": "best_model.pt",
    "retriever/TripAdvisor/trip_ckpt/ckpt_colbert_deberta_v3_inbatch_aux/tokenizer": "tokenizer",
}

PRIVATE_BEER_ADVOCATE_FILES = {
    "retriever/beer/aspect.words": "aspect.words",
    "retriever/beer/kb.jsonl": "kb.jsonl",
    "retriever/beer/beer_retrieve_results/dev_Faiss_matches_by_trigger.jsonl": "dev_Faiss_matches_by_trigger.jsonl",
    "retriever/beer/beer_retrieve_results/test_Faiss_matches_by_trigger.jsonl": "test_Faiss_matches_by_trigger.jsonl",
    "retriever/beer/beer_retrieve_results/train_Faiss_matches_by_trigger.jsonl": "train_Faiss_matches_by_trigger.jsonl",
    "retriever/beer/beer_constrative/new_mixed_constrative_triplets_balanced.csv": "new_mixed_constrative_triplets_balanced.csv",
    "retriever/beer/beer_ckpt/ckpt_colbert_deberta_v3_inbatch_aux/best_model.pt": "best_model.pt",
    "retriever/beer/beer_ckpt/ckpt_colbert_deberta_v3_inbatch_aux/tokenizer": "tokenizer",
}

PRIVATE_CODE_FILES = {
    "retriever/faiss_Colbert_confidence.py": "faiss_Colbert_confidence.py",
    "retriever/KB_build.py": "KB_build.py",
    "retriever/train_colbert_contrastive_inbatch_aux.py": "train_colbert_contrastive_inbatch_aux.py",
}

PRIVATE_CHECKPOINTS = {
    "trip_advisor_kasm_checkpoint": "trip_advisor_kasm_checkpoint",
}

PRIVATE_TABLE_FILES = {
    "tables/trip_advisor/train.csv": "trip_advisor/train.csv",
    "tables/trip_advisor/dev.csv": "trip_advisor/dev.csv",
    "tables/trip_advisor/test.csv": "trip_advisor/test.csv",
    "tables/beer_advocate/train.csv": "beer_advocate/train.csv",
    "tables/beer_advocate/dev.csv": "beer_advocate/dev.csv",
    "tables/beer_advocate/test.csv": "beer_advocate/test.csv",
}

def download_file(url: str, output: Path) -> None:
    tmp = output.with_suffix(output.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    tmp.replace(output)


def download_bundle(url: str, output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"skip {output.name}")
        return
    print(f"download {output.name}")
    download_file(url, output)


def extract_zip(zip_path: Path, output: Path, force: bool) -> Path:
    extracted = output / "extracted"
    marker = extracted / ".extract_complete"
    if marker.exists() and not force:
        print(f"skip extraction -> {extracted}")
        return extracted
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True, exist_ok=True)
    print(f"extract {zip_path.name} -> {extracted}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extracted)
    marker.write_text("ok\n", encoding="utf-8")
    return extracted


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        rel_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_src)


def find_private_root(extracted: Path) -> Path:
    expected = extracted / "retriever and  ckpt"
    if expected.exists():
        return expected
    matches = [path for path in extracted.iterdir() if path.is_dir() and "retriever" in path.name.lower()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not find private Figshare root under {extracted}")


def organize_private_group(root: Path, mapping: dict[str, str], target: Path, copy: bool) -> None:
    for rel_src, name in mapping.items():
        src = root / rel_src
        if not src.exists():
            raise FileNotFoundError(src)
        link_or_copy(src, target / name, copy=copy)


def organize_optional_private_group(root: Path, mapping: dict[str, str], target: Path, copy: bool) -> None:
    for rel_src, name in mapping.items():
        src = root / rel_src
        if src.exists():
            link_or_copy(src, target / name, copy=copy)


def organize_private_assets(extracted: Path, output: Path, copy: bool = False) -> Path:
    root = find_private_root(extracted)
    organized = output / "organized"
    if organized.exists():
        shutil.rmtree(organized)

    organize_private_group(root, PRIVATE_TRIP_ADVISOR_FILES, organized / "trip_advisor", copy)
    organize_private_group(root, PRIVATE_BEER_ADVOCATE_FILES, organized / "beer_advocate", copy)
    organize_private_group(root, PRIVATE_CODE_FILES, organized / "code", copy)
    organize_private_group(root, PRIVATE_CHECKPOINTS, organized / "checkpoints", copy)
    organize_optional_private_group(root, PRIVATE_TABLE_FILES, organized / "tables", copy)
    return organized


def link_repo_assets(organized: Path, repo_root: Path) -> None:
    target = repo_root / "retriever" / "assets"
    target.mkdir(parents=True, exist_ok=True)
    for domain in ("trip_advisor", "beer_advocate"):
        dst = target / domain
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.exists():
            shutil.rmtree(dst)
        dst.symlink_to(os.path.relpath(organized / domain, target))
    code_dir = organized / "code"
    if code_dir.exists():
        dst = target / "figshare_code"
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.exists():
            shutil.rmtree(dst)
        dst.symlink_to(os.path.relpath(code_dir, target))
    tables_dir = organized / "tables"
    if tables_dir.exists():
        dst = target / "tables"
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.exists():
            shutil.rmtree(dst)
        dst.symlink_to(os.path.relpath(tables_dir, target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("retriever/assets/release_bundle"))
    parser.add_argument("--article-id", default=RELEASE_ARTICLE_ID, help=argparse.SUPPRESS)
    parser.add_argument(
        "--private-link",
        default=RELEASE_SHARE_TOKEN,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--link", action="store_true", help="Link organized assets into retriever/assets/.")
    parser.add_argument("--copy-organized", action="store_true", help="Copy organized files instead of symlinking.")
    parser.add_argument("--force", action="store_true", help="Redownload files even when local MD5 already matches.")
    args = parser.parse_args()

    output = args.output.resolve()

    bundle_url = f"https://ndownloader.figshare.com/articles/{args.article_id}?private_link={args.private_link}"
    bundle_path = output / "kasm_release_bundle.zip"
    output.mkdir(parents=True, exist_ok=True)
    download_bundle(bundle_url, bundle_path, force=args.force)
    extracted = extract_zip(bundle_path, output, force=args.force)
    organized = organize_private_assets(extracted, output, copy=args.copy_organized)
    print(f"organized assets -> {organized}")

    if args.link:
        repo_root = Path(__file__).resolve().parents[2]
        link_repo_assets(organized, repo_root)
        print(f"linked repo assets -> {repo_root / 'retriever' / 'assets'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
