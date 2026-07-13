# KASM Demo

This repository contains the KASM project demo and its runnable experiment code.

## Modules

- `index.html` and `assets/`: project demo page and figures.
- `experiment/`: model training, data input pipeline, and evaluation code.
- `prompt_generation/`: reserved for prompt-generation code.
- `retriever/`: reserved for retrieval code that will produce the JSONL knowledge files consumed by `experiment/`.

The current runnable code lives in `experiment/`. Large dataset files are not
committed; place them under `experiment/dataset/` before running training.

## Quick Start

```bash
cd experiment
bash scripts/train.sh
DATA_NAME=beer_advocate bash scripts/train.sh
CKPT=/path/to/checkpoint.ckpt bash scripts/evaluate_tripadvisor_checkpoint.sh
```

See `experiment/README.md` for the full training/evaluation documentation.
