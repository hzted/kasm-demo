# Prompt Generation

This module is reserved for the prompt template used to extract
aspect-conditioned triggers, opinions, and polarities from raw reviews.

In the paper pipeline, prompt generation produces structured records that are
normalized into `kb.jsonl` by `retriever/src/kb_build.py`. Pseudo labels are an
intermediate artifact of this process; the runnable release can start from the
provided Figshare KB/retrieval files instead.

## Text Preprocessing

`test_process.py` is a lightweight TripAdvisor preprocessing utility. It parses
the raw split files, removes noisy markup/control characters, normalizes quotes
and whitespace, improves sentence splitting, and writes CSV files compatible
with `experiment/dataset/trip_advisor/`.

```bash
python prompt_generation/test_process.py \
  --input-dir /path/to/trip_advisor_original \
  --output-dir experiment/dataset/trip_advisor \
  --splits train dev test
```

The output keeps document text, sentence lists, aspect ratings, and extracted
opinion terms for downstream prompt generation, retriever construction, and KASM
training.
