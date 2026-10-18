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

## Evaluation outputs

Evaluate labeled document–aspect pairs using accuracy and binary macro-F1.
The manuscript defines long documents as having more than 512 FLAN-T5 tokens
before model input truncation. Numerical comparison tables are not included in
this documentation revision. Preserve the prediction files and the checkpoint
identity when reporting an evaluation.

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
