# KASM: Learning Sentiment Dynamics

**Knowledge-Augmented Sequence Modeling for Document-Level Aspect-Based Sentiment Analysis**

KASM predicts one sentiment polarity for each labeled document–aspect pair.
It connects review sentences to aspect–trigger–opinion–polarity (ATOP)
knowledge and integrates local evidence through a polarity sequence model.
The framework is studied on BeerAdvocate and TripAdvisor.

[Anonymous project page](https://anonymous.4open.science/w/kasm-source/) ·
[Browse source code](https://anonymous.4open.science/r/kasm-source/README.md)

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
- [experiment/](https://anonymous.4open.science/r/kasm-source/experiment/): model training, data input pipeline, and evaluation code.
- `prompt_generation/`: prompt template placeholder for knowledge extraction.
- [retriever/](https://anonymous.4open.science/r/kasm-source/retriever/): relation-aware ColBERT retriever code that produces the JSONL knowledge files consumed by `experiment/`.

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

## Manuscript Results

Values and significance markers below are transcribed from Tables 3–4 of the manuscript (EMNLP_KASM.pdf, October 16, 2026); they are manuscript-reported results, not a new evaluation or independent verification performed for this website.

### Table 3

| Model | Beer All Acc | Beer All F1 | Beer >512 Acc | Beer >512 F1 | Trip All Acc | Trip All F1 | Trip >512 Acc | Trip >512 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D-MILN | 72.59 | 70.67 | 68.40 | 65.54 | 73.16 | 72.95 | 71.82 | 71.71 |
| AB-DMSC | 84.56 | 81.77 | 88.21 | 83.04 | 80.80 | 80.16 | 79.10 | 79.05 |
| DART | 88.13 | 86.19 | 91.04 | 88.41 | 86.25 | 85.65 | 84.69 | 84.56 |
| InstructABSA | 88.29 | 86.61 | 82.08 | 76.95 | 84.43 | 83.98 | 80.18 | 80.09 |
| THOR | 88.45 | 87.07 | 90.09 | 87.61 | 86.60 | 86.06 | 82.63 | 82.46 |
| GPT-5.5 zero-shot | 85.50 | 84.04 | 88.68 | 86.57 | 75.01 | 74.71 | 83.88 | 83.86 |
| GPT-5.5 few-shot | 85.62 | 84.19 | 90.09 | 87.73 | 71.99 | 71.72 | 80.78 | 80.78 |
| Retrieval-IT | 86.84 | 85.05 | 79.25 | 71.21 | 82.84 | 82.44 | 79.91 | 79.87 |
| KASM | 91.02† | 89.80† | 91.04 | 88.66 | 88.13† | 87.63† | 87.35† | 87.26† |

† reproduces the manuscript’s paired-permutation significance markers; tests were not rerun for this page.

### Table 4

| Variant | Beer Acc | Beer F1 | Trip Acc | Trip F1 |
|---|---:|---:|---:|---:|
| KASM | 91.02 | 89.80 | 88.13 | 87.63 |
| w/ mean pooling | 88.49 | 86.80 | 86.78 | 86.31 |
| w/o sequence loss | 88.97 | 87.19 | 86.19 | 85.67 |
| w/o confidence prior | 87.80 | 85.85 | 87.05 | 86.69 |
| w/o pseudo guidance | 88.70 | 87.02 | 86.72 | 86.22 |
| w/o contrastive learning | 85.99 | 83.73 | 84.99 | 84.62 |

## Figures and case studies

The project page includes manuscript Figures 1–4 and Tables 19–20.
All displayed figure assets were exported directly from `EMNLP_KASM.pdf`.
PNG previews and vector PDF downloads are provided in `assets/`;
`assets/paper_figure_sources.json` records the source PDF hash and page regions.
