# KASM Retriever

This module contains the KASM retriever code that builds and applies a
relation-aware ColBERT retriever. Its output is the sentence-level JSONL
knowledge file consumed by `experiment/`.

In the full pipeline, prompt-generated trigger/opinion records are first
converted to `kb.jsonl`. A ColBERT-style retriever is then trained with
contrastive objectives over sentence-to-KB and KB-to-KB relation pairs. At
inference time, the retriever writes one JSONL row per sentence with matched
aspect triggers, opinion phrases, polarities, and confidence scores.

Large retriever assets are not committed to git. Download them from Figshare:

https://doi.org/10.6084/m9.figshare.32970428

This Figshare package includes the released retriever checkpoints, tokenizer
files, pseudo knowledge bases (`kb.jsonl`), contrastive training CSVs, and
precomputed retrieval JSONL outputs for TripAdvisor and BeerAdvocate. See
`FIGSHARE_ASSETS.md` for the asset inventory and data formats.

To download through the Figshare API and verify MD5 checksums:

```bash
cd /path/to/kasm-demo
python retriever/scripts/download_figshare_assets.py \
  --output retriever/assets/figshare_32970428 \
  --link
```

Then link the downloaded assets into this repo:

```bash
cd /path/to/kasm-demo
bash retriever/scripts/link_figshare_assets.sh /path/to/figshare_kasm_retriever_32970428
```

Expected linked layout:

```text
retriever/assets/
  trip_advisor/
    best_model.pt
    tokenizer/
    kb.jsonl
    aspect.words
    *_Faiss_matches_by_trigger_confidence.jsonl
  beer_advocate/
    best_model.pt
    tokenizer/
    kb.jsonl
    aspect.words
    *_Faiss_matches_by_trigger.jsonl
```

`link_figshare_assets.sh` accepts either the raw download directory containing
`organized/` or the `organized/` directory itself.

## Generate Retrieval JSONL

If you want to use the precomputed Figshare retrieval outputs directly, link
them into `experiment/dataset/`:

```bash
cd /path/to/kasm-demo
bash retriever/scripts/install_precomputed_jsonl.sh all
```

This creates symlinks by default. Use `MODE=copy` to copy files instead, and
`OVERWRITE=1` to replace existing local JSONL files.

```bash
OVERWRITE=1 MODE=copy bash retriever/scripts/install_precomputed_jsonl.sh trip_advisor
```

The retrieval script reads `experiment/dataset/<dataset>/<split>.csv` and writes
the matching JSONL file back into the same dataset split.

```bash
cd /path/to/kasm-demo
bash retriever/scripts/run_retrieval.sh trip_advisor dev
bash retriever/scripts/run_retrieval.sh beer_advocate dev
```

Use the second argument for `train`, `dev`, or `test`.

To write somewhere else:

```bash
OUTPUT_JSONL=/tmp/trip_dev.jsonl bash retriever/scripts/run_retrieval.sh trip_advisor dev
```

If writing into `experiment/dataset/`, set `OVERWRITE=1` when regenerating an
existing split.

## Build KB From Prompt Outputs

If starting from prompt-generated JSON/JSONL records:

```bash
python retriever/src/kb_build.py \
  --input /path/to/prompt_records.jsonl \
  --output retriever/assets/trip_advisor/kb.jsonl
```

The Figshare package already includes `kb.jsonl` for both TripAdvisor and
BeerAdvocate, so this step is optional for reproducing the released runs.

## Train Retriever

The released Figshare checkpoints are usually enough. To retrain:

```bash
cd /path/to/kasm-demo
bash retriever/scripts/train_retriever.sh beer_advocate
bash retriever/scripts/train_retriever.sh trip_advisor
```

The training script writes checkpoints under `retriever/outputs/`, which is
ignored by git.

The contrastive CSV uses relation types `ST`, `SO`, `TT`, `TO`, and `OO`.
`--relation_mode all` uses all relations, `kb_only` uses `TT/TO/OO`, and
`to_oo_only` uses `TO/OO`.

## Source Files

- `src/faiss_colbert_confidence.py`: retrieval/inference pipeline.
- `src/train_colbert_contrastive_inbatch_aux.py`: contrastive ColBERT retriever training.
- `src/kb_build.py`: converts prompt-generated trigger/opinion records into `kb.jsonl`.
- `scripts/download_figshare_assets.py`: downloads, verifies, and organizes released Figshare assets.

The uploader's raw command note contained a local Windows path and is not
included in this repository. The runnable commands above replace it with
repo-relative paths.
