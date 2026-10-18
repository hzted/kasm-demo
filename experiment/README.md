# KASM

Knowledge-Augmented Sequence Modeling (KASM) is the main model in this repository.
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
retrieval results aligned by `row` with the CSV `id`. The released Figshare
bundle should provide both the CSV tables and the precomputed JSONL files.

Expected released-data sizes:

| Dataset | Split | CSV docs | JSONL doc groups | JSONL sentence rows |
|---|---:|---:|---:|---:|
| TripAdvisor | train | 22,665 | 22,665 | 321,583 |
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
`experiment/dataset/<dataset>/<split>.jsonl` and CSV tables have been written to
`experiment/dataset/<dataset>/<split>.csv`. The easiest path is to install the
released Figshare files:

```bash
cd ..
bash retriever/scripts/link_figshare_assets.sh /path/to/kasm_release/organized
bash retriever/scripts/install_released_data.sh all
cd experiment
python scripts/validate_release_data.py
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
