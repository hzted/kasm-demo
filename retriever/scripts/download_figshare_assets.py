#!/usr/bin/env python3
"""Download and organize the Figshare KASM retriever assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable


ARTICLE_ID = "32970428"
PRIVATE_ARTICLE_ID = "32988389"
PRIVATE_LINK_TOKEN = "132e83cda32c9b18aca7"

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

TRIP_ADVISOR_FILES = {
    66566114: "aspect.words",
    66566117: "kb.jsonl",
    66566120: "dev_Faiss_matches_by_trigger_confidence.jsonl",
    66566123: "test_Faiss_matches_by_trigger_confidence.jsonl",
    66566171: "train_Faiss_matches_by_trigger_confidence.jsonl",
    66566132: "new_mixed_constrative_triplets_balanced.csv",
    66566180: "best_model.pt",
}

TRIP_ADVISOR_TOKENIZER = {
    66566126: "added_tokens.json",
    66566129: "special_tokens_map.json",
    66566135: "spm.model",
    66566147: "tokenizer.json",
    66566138: "tokenizer_config.json",
}

BEER_ADVOCATE_FILES = {
    66566141: "aspect.words",
    66566144: "kb.jsonl",
    66566150: "dev_Faiss_matches_by_trigger.jsonl",
    66566156: "test_Faiss_matches_by_trigger.jsonl",
    66566174: "train_Faiss_matches_by_trigger.jsonl",
    66566153: "new_mixed_constrative_triplets_balanced.csv",
    66566186: "best_model.pt",
}

BEER_ADVOCATE_TOKENIZER = {
    66566159: "added_tokens.json",
    66566162: "special_tokens_map.json",
    66566165: "spm.model",
    66566177: "tokenizer.json",
    66566168: "tokenizer_config.json",
}

CODE_FILES = {
    66566105: "faiss_Colbert_confidence.py",
    66566108: "KB_build.py",
    66566111: "train_colbert_contrastive_inbatch_aux.py",
}


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def iter_files(article: dict) -> Iterable[dict]:
    for item in article.get("files", []):
        if item.get("is_link_only"):
            continue
        yield item


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


def organize_group(raw_by_id: Dict[int, Path], mapping: Dict[int, str], root: Path, copy: bool) -> None:
    for file_id, name in mapping.items():
        link_or_copy(raw_by_id[file_id], root / name, copy=copy)


def organize_assets(raw_by_id: Dict[int, Path], output: Path, copy: bool = False) -> Path:
    organized = output / "organized"
    if organized.exists():
        shutil.rmtree(organized)

    organize_group(raw_by_id, TRIP_ADVISOR_FILES, organized / "trip_advisor", copy)
    organize_group(raw_by_id, TRIP_ADVISOR_TOKENIZER, organized / "trip_advisor" / "tokenizer", copy)
    organize_group(raw_by_id, BEER_ADVOCATE_FILES, organized / "beer_advocate", copy)
    organize_group(raw_by_id, BEER_ADVOCATE_TOKENIZER, organized / "beer_advocate" / "tokenizer", copy)
    organize_group(raw_by_id, CODE_FILES, organized / "code", copy)
    return organized


def find_private_root(extracted: Path) -> Path:
    expected = extracted / "retriever and  ckpt"
    if expected.exists():
        return expected
    matches = [path for path in extracted.iterdir() if path.is_dir() and "retriever" in path.name.lower()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not find private Figshare root under {extracted}")


def organize_private_group(root: Path, mapping: Dict[str, str], target: Path, copy: bool) -> None:
    for rel_src, name in mapping.items():
        src = root / rel_src
        if not src.exists():
            raise FileNotFoundError(src)
        link_or_copy(src, target / name, copy=copy)


def organize_optional_private_group(root: Path, mapping: Dict[str, str], target: Path, copy: bool) -> None:
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
    parser.add_argument("--output", type=Path, default=Path("retriever/assets/figshare_32970428"))
    parser.add_argument("--article-id", default=ARTICLE_ID)
    parser.add_argument(
        "--private-release",
        action="store_true",
        help="Download the private-share KASM bundle with retriever assets and the KASM checkpoint.",
    )
    parser.add_argument(
        "--private-link",
        default=PRIVATE_LINK_TOKEN,
        help="Private Figshare share token used with --private-release.",
    )
    parser.add_argument("--link", action="store_true", help="Link organized assets into retriever/assets/.")
    parser.add_argument("--copy-organized", action="store_true", help="Copy organized files instead of symlinking.")
    parser.add_argument("--force", action="store_true", help="Redownload files even when local MD5 already matches.")
    args = parser.parse_args()

    output = args.output.resolve()

    if args.private_release:
        article_id = args.article_id if args.article_id != ARTICLE_ID else PRIVATE_ARTICLE_ID
        bundle_url = f"https://ndownloader.figshare.com/articles/{article_id}?private_link={args.private_link}"
        bundle_path = output / f"figshare_{article_id}_bundle.zip"
        output.mkdir(parents=True, exist_ok=True)
        download_bundle(bundle_url, bundle_path, force=args.force)
        extracted = extract_zip(bundle_path, output, force=args.force)
        organized = organize_private_assets(extracted, output, copy=args.copy_organized)
        print(f"organized assets -> {organized}")
        if args.link:
            repo_root = Path(__file__).resolve().parents[2]
            link_repo_assets(organized, repo_root)
            print(f"linked repo assets -> {repo_root / 'retriever' / 'assets'}")
        return

    article_api = f"https://api.figshare.com/v2/articles/{args.article_id}"
    raw_dir = output / "files_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    article = fetch_json(article_api)
    (output / "figshare_article_manifest.json").write_text(
        json.dumps(article, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    raw_by_id: Dict[int, Path] = {}
    for item in iter_files(article):
        file_id = int(item["id"])
        raw_path = raw_dir / f"{file_id}_{safe_name(item['name'])}"
        expected_md5 = item.get("computed_md5") or item.get("supplied_md5")
        if not args.force and raw_path.exists() and expected_md5 and md5_file(raw_path) == expected_md5:
            print(f"skip {raw_path.name}")
        else:
            print(f"download {raw_path.name}")
            download_file(item["download_url"], raw_path)
        if expected_md5:
            actual_md5 = md5_file(raw_path)
            if actual_md5 != expected_md5:
                raise RuntimeError(f"MD5 mismatch for {raw_path}: {actual_md5} != {expected_md5}")
        raw_by_id[file_id] = raw_path

    organized = organize_assets(raw_by_id, output, copy=args.copy_organized)
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
