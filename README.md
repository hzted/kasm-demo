# KASM: Learning Sentiment Dynamics

**Knowledge-Augmented Sequence Modeling for Document-Level Aspect-Based Sentiment Analysis**

KASM predicts one sentiment polarity for each labeled document–aspect pair.
It connects review sentences to aspect–trigger–opinion–polarity (ATOP)
knowledge and integrates local evidence through a polarity sequence model.
The framework is studied on BeerAdvocate and TripAdvisor.

[Anonymous project page](https://anonymous.4open.science/w/kasm-source/)

This repository contains a static project page (`index.html`) and the released
experiment and retrieval code. The page illustrates the method; it does not
run inference in the browser.

## Method Flow

1. **Construct ATOP knowledge.** LLM extraction links aspect categories to
   textual triggers, opinion spans, and local polarities in training reviews.
   The normalized records form `kb.jsonl`. Precomputed KB artifacts let users
   start without rerunning extraction; `prompt_generation/` currently contains
   preprocessing code and a placeholder for the extraction prompt.
2. **Retrieve sentence evidence.** Relation-aware contrastive training covers
   sentence–trigger, sentence–opinion, trigger–trigger, trigger–opinion, and
   opinion–opinion relations. Retrieval supplies sentence-level polarity and
   confidence alongside readable evidence paths.
3. **Model the polarity sequence.** FLAN-T5 sentence encoding, learned sentence
   positions, and DART-style global interaction provide document context.
   Confidence-guided decoding uses a retrieved polarity above the feedback
   threshold and predicted polarity otherwise, during both training and inference.
4. **Aggregate and classify.** The final decoder state queries sentence
   representations; a confidence prior adjusts attention logits before pooling.
   Document cross-entropy and an auxiliary confidence-weighted SmoothL1 sequence
   loss train the model. Mixed local opinions still yield one final polarity
   for each target aspect.

The manuscript uses a feedback threshold of 0.50, sequence-loss weight 1,
and confidence-prior weight 1. Retrieval acceptance and decoder feedback are
separate decisions. See `experiment/README.md` for released configuration
settings and `retriever/README.md` for retrieval details.

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
  --output /path/to/kasm_release \
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
  --output /path/to/kasm_release \
  --link

# If already downloaded elsewhere, link that directory instead:
# bash retriever/scripts/link_figshare_assets.sh /path/to/kasm_release/organized

# 2. Link released CSV tables and precomputed retrieval JSONL into experiment/dataset/.
bash retriever/scripts/install_released_data.sh all

# 3. Verify CSV/JSONL row alignment.
cd experiment
python scripts/validate_release_data.py

# 4. Train KASM.
bash scripts/train.sh
DATA_NAME=beer_advocate bash scripts/train.sh

# 5. Evaluate the released TripAdvisor KASM checkpoint.
CKPT=/path/to/kasm_release/organized/checkpoints/trip_advisor_kasm_checkpoint/trip_advisor_kasm_epoch14_step101640.ckpt \
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
