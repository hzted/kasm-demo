# KASM Demo

This repository contains the KASM project demo and its runnable experiment code.

## Method Flow

KASM follows the paper pipeline:

1. Prompt generation extracts aspect-conditioned triggers, opinions, and
   polarities from reviews. The prompt template will live under
   `prompt_generation/`.
2. The extracted records are normalized into a trigger-opinion knowledge base
   (`kb.jsonl`). Pseudo labels are an intermediate artifact of this step and are
   not required to run the released code.
3. The retriever is trained with contrastive learning over sentence-to-KB and
   KB-to-KB relations, then retrieves sentence-level evidence for each document.
4. The KASM model consumes the review CSV plus retrieved JSONL evidence and uses
   pseudo-label-aware sequence modeling for long-document aspect sentiment.

## Modules

- `index.html` and `assets/`: project demo page and figures.
- `experiment/`: model training, data input pipeline, and evaluation code.
- `prompt_generation/`: prompt template placeholder for knowledge extraction.
- `retriever/`: relation-aware ColBERT retriever code that produces the JSONL knowledge files consumed by `experiment/`.

Large dataset, retriever, and checkpoint files are not committed. Place CSV
tables under `experiment/dataset/`, and download/link the released Figshare
assets before training or evaluation.

## Released Data And Checkpoints

We release the KASM data artifacts and checkpoints on Figshare:

https://figshare.com/s/132e83cda32c9b18aca7

The release includes the model input CSV tables, retrieved JSONL knowledge
files, pseudo knowledge bases, retriever checkpoints, and the TripAdvisor KASM
Lightning checkpoint. Follow the command below to download the release locally
and link the retriever assets into this repository:

```bash
python retriever/scripts/download_figshare_assets.py \
  --private-release \
  --output /path/to/figshare_kasm_private_32988389 \
  --link
```

## Setup

```bash
python -m pip install -r requirements.txt
```

If your CUDA environment needs a specific PyTorch build, install PyTorch first
following the official selector, then run the command above.

## Quick Start With Released Assets

```bash
# 1. Download the released Figshare bundle, then link retriever assets.
python retriever/scripts/download_figshare_assets.py \
  --private-release \
  --output /path/to/figshare_kasm_private_32988389 \
  --link

# If already downloaded elsewhere, link that directory instead:
# bash retriever/scripts/link_figshare_assets.sh /path/to/figshare_kasm_private_32988389/organized

# 2. Link released CSV tables and precomputed retrieval JSONL into experiment/dataset/.
bash retriever/scripts/install_released_data.sh all

# 3. Verify CSV/JSONL row alignment.
cd experiment
python scripts/validate_release_data.py

# 4. Train KASM.
bash scripts/train.sh
DATA_NAME=beer_advocate bash scripts/train.sh

# 5. Evaluate the released TripAdvisor KASM checkpoint.
CKPT=/path/to/figshare_kasm_private_32988389/organized/checkpoints/trip_advisor_kasm_checkpoint/trip_advisor_kasm_epoch14_step101640.ckpt \
  bash scripts/evaluate_tripadvisor_checkpoint.sh
```

To regenerate retrieval outputs instead of using the precomputed JSONL, run from
the repository root:

```bash
bash retriever/scripts/train_retriever.sh trip_advisor
OVERWRITE=1 bash retriever/scripts/run_retrieval.sh trip_advisor dev
```

See `experiment/README.md` for the full training/evaluation documentation.
See `retriever/README.md` for retriever asset setup and JSONL generation.
