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
tables under `experiment/dataset/`, and link/download retriever assets before
training.

## Setup

```bash
python -m pip install -r requirements.txt
```

If your CUDA environment needs a specific PyTorch build, install PyTorch first
following the official selector, then run the command above.

## Quick Start With Released Assets

```bash
# 1. Download the Figshare retriever assets, then link them.
bash retriever/scripts/link_figshare_assets.sh /path/to/figshare_kasm_retriever_32970428

# 2. Link precomputed retrieval JSONL into experiment/dataset/.
bash retriever/scripts/install_precomputed_jsonl.sh all

# 3. Train KASM.
cd experiment
bash scripts/train.sh
DATA_NAME=beer_advocate bash scripts/train.sh

# 4. Evaluate a released KASM checkpoint.
CKPT=/path/to/checkpoint.ckpt bash scripts/evaluate_tripadvisor_checkpoint.sh
```

To regenerate retrieval outputs instead of using the precomputed JSONL, run from
the repository root:

```bash
bash retriever/scripts/train_retriever.sh trip_advisor
OVERWRITE=1 bash retriever/scripts/run_retrieval.sh trip_advisor dev
```

See `experiment/README.md` for the full training/evaluation documentation.
See `retriever/README.md` for retriever asset setup and JSONL generation.
