# KASM

Knowledge-Aware Sentiment Modeling (KASM) is the main model in this repository.
The codebase is cleaned from the old DART workspace: KASM is the default and only
first-class training path, while legacy DART code is kept separately under
`baselines/dart/` as an archived baseline reference.

## Repository Layout

```text
kasm-demo/
  experiment/
  configs/                 Hydra configs for KASM training/evaluation
  dataset/
    trip_advisor/          Place CSV tables + retrieved-knowledge JSONL here
    beer_advocate/         Place CSV tables + retrieved-knowledge JSONL here
  datamodules/             KASM datamodule and sentence collator
  models/                  KASM network and Lightning module
  scripts/                 Local and Slurm launch scripts
  baselines/dart/          Legacy DART baseline archive, not imported by KASM
```

## Data

Each dataset split has a table file and a retrieved-knowledge JSONL file:

```text
dataset/<dataset>/train.csv
dataset/<dataset>/train.jsonl
dataset/<dataset>/dev.csv
dataset/<dataset>/dev.jsonl
dataset/<dataset>/test.csv
dataset/<dataset>/test.jsonl
```

The CSV contains document-level labels. The JSONL contains sentence-level
retrieval results aligned by `row` with the CSV `id`. Generate or link these
JSONL files with the sibling top-level `retriever/` module before training.

Expected full-data sizes:

| Dataset | Split | CSV docs | JSONL doc groups | JSONL sentence rows |
|---|---:|---:|---:|---:|
| TripAdvisor | train | 23,468 | 23,468 | 333,353 |
| TripAdvisor | dev | 2,939 | 2,939 | 40,982 |
| TripAdvisor | test | 2,939 | 2,939 | 39,416 |
| BeerAdvocate | train | 22,067 | 22,067 | 244,554 |
| BeerAdvocate | dev | 2,758 | 2,758 | 30,600 |
| BeerAdvocate | test | 2,758 | 2,758 | 30,320 |

## Model Configuration

The default KASM configuration follows the TripAdvisor KASM run used in our experiments:

- Backbone: `google/flan-t5-base`
- Max sentences/document: `25`
- Max tokens/aspect-sentence pair: `64`
- Training decode mode: `pseudo_conf`
- Eval decode mode: `pseudo_conf`
- Pseudo confidence threshold: `0.5`
- Interaction encoder: 4 layers, hidden size 768, 12 heads
- Refinement encoder: 2 layers, hidden size 768, 12 heads
- Learning rate: `1.5e-6`
- Epochs: `15`

Checkpoints are not bundled in this anonymous code repository. Place a checkpoint
under `checkpoints/` or pass its path through `CKPT=/path/to/checkpoint.ckpt`.

Reported comparison results follow the paper table. Values are percentages on
the full test set (`All`) and the long-document subset (`>512` tokens).

| Category | Model | Beer All Acc | Beer All F1 | Beer >512 Acc | Beer >512 F1 | Trip All Acc | Trip All F1 | Trip >512 Acc | Trip >512 F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Task-specific | D-MILN | 79.86 | 80.12 | 87.85 | 86.09 | 79.52 | 79.03 | 83.61 | 83.13 |
| Doc-level ABSA | AB-DMSC | 85.98 | 85.21 | 88.14 | 86.59 | 81.26 | 80.46 | 84.04 | 83.44 |
| Baselines | DART | 88.25 | 87.33 | **94.44** | **92.86** | 86.38 | 85.79 | 86.48 | 85.96 |
| Prompt-based | InstructABSA | **88.67** | 87.97 | 81.25 | 79.83 | 80.26 | 81.71 | 70.01 | 69.65 |
| Fine-tuning | THOR | 79.85 | 78.63 | 75.49 | 76.15 | 79.83 | 78.45 | 80.06 | 79.43 |
| Direct LLM prompting | GPT5.5-zeroshot | 86.15 | 87.08 | 84.23 | 83.97 | 75.73 | 76.98 | 80.43 | 82.44 |
| Direct LLM prompting | GPT5.5-fewshot | 86.45 | 87.31 | 85.56 | 84.15 | 79.92 | 80.35 | 79.68 | 80.12 |
| Knowledge-enhanced | Retrieval-IT | 77.12 | 78.32 | 80.54 | 82.35 | 68.21 | 67.05 | 63.27 | 60.47 |
| Knowledge-enhanced | **KASM** | 88.36 | **88.53** | 90.31 | 89.26 | **87.84** | **87.11** | **87.14** | **86.52** |

## Setup

```bash
cd experiment
python -m pip install -r requirements.txt
```

The original environment used Python with PyTorch Lightning, Transformers, and
SentencePiece. Install the appropriate PyTorch build for your CUDA version if it
is not already present.

## Train

TripAdvisor:

```bash
cd experiment
bash scripts/train.sh
```

BeerAdvocate:

```bash
cd experiment
DATA_NAME=beer_advocate bash scripts/train.sh
```

Slurm:

```bash
cd experiment
DATA_NAME=trip_advisor sbatch scripts/train_slurm.sh
DATA_NAME=beer_advocate sbatch scripts/train_slurm.sh
```

Hydra overrides work normally:

```bash
python -u train.py data=beer_advocate train.batch_size=8 train.num_workers=8
```

Training writes tokenized cache files to `outputs/cache/` and run outputs to
`logs/runs/<dataset>_KASM_kasm_<timestamp>/`. Large dataset files are intentionally
not committed to this repository; see `dataset/README.md`.

## Evaluate The Released Checkpoint

```bash
cd experiment
CKPT=/path/to/checkpoint.ckpt bash scripts/evaluate_tripadvisor_checkpoint.sh
```

Equivalent explicit command:

```bash
python -u evaluate.py \
  data=trip_advisor \
  ckpt_path=/path/to/checkpoint.ckpt
```

## Retrieval Inputs

KASM training assumes retrieval has already been written to
`experiment/dataset/<dataset>/<split>.jsonl`. The easiest path is to link the
released Figshare retrieval files:

```bash
cd ..
bash retriever/scripts/link_figshare_assets.sh /path/to/figshare_kasm_retriever_32970428
bash retriever/scripts/install_precomputed_jsonl.sh all
```

To regenerate a split from CSV with the retriever:

```bash
cd ..
OVERWRITE=1 bash retriever/scripts/run_retrieval.sh trip_advisor dev
```

As long as the `row` ids match the CSV `id` values and the `matches` object
contains the configured aspects, no KASM model code needs to change.

## What Was Removed From The Main Path

The main training path no longer imports DART, AB-DMSC, DMIL, SocialNews, prompt
generation notebooks, aspect embedding files, old pickle caches, or exploratory
debug assets. Those were either omitted or isolated under `baselines/dart/` when
they were directly relevant as a baseline reference.
