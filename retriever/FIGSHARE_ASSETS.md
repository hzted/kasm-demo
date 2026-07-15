# Figshare Retriever Assets

KASM assets are released separately on Figshare:

- Current private-share bundle: https://figshare.com/s/132e83cda32c9b18aca7
- Article id used by the downloader: `32988389`
- Legacy retriever-only DOI: https://doi.org/10.6084/m9.figshare.32970428

The current Figshare package contains the retriever source snapshot, retriever
checkpoints, pseudo knowledge bases, contrastive training data, tokenizer files,
precomputed retrieval outputs, and the TripAdvisor KASM Lightning checkpoint.
These files are intentionally not committed to git.

For one-command training/evaluation after cloning this repository, the bundle
must also include model input CSV tables under:

```text
tables/
  trip_advisor/{train,dev,test}.csv
  beer_advocate/{train,dev,test}.csv
```

## What The Package Contains

TripAdvisor assets:

- `tables/trip_advisor/train.csv`, `dev.csv`, `test.csv`: model input tables
  with document-level labels.
- `kb.jsonl`: pseudo trigger-opinion knowledge base built from prompt outputs.
- `aspect.words`: aspect-word priors for retrieval.
- `new_mixed_constrative_triplets_balanced.csv`: contrastive retriever training pairs.
- `best_model.pt` plus `tokenizer/`: trained ColBERT-style retriever checkpoint.
- `train/dev/test_Faiss_matches_by_trigger_confidence.jsonl`: precomputed sentence-level retrieval outputs.
- `trip_advisor_kasm_checkpoint/`: released Lightning checkpoint and sanitized
  Hydra/Lightning config for the TripAdvisor KASM model.

BeerAdvocate assets:

- `tables/beer_advocate/train.csv`, `dev.csv`, `test.csv`: model input tables
  with document-level labels.
- `kb.jsonl`: pseudo trigger-opinion knowledge base built from prompt outputs.
- `aspect.words`: aspect-word priors for retrieval.
- `new_mixed_constrative_triplets_balanced.csv`: contrastive retriever training pairs.
- `best_model.pt` plus `tokenizer/`: trained ColBERT-style retriever checkpoint.
- `train/dev/test_Faiss_matches_by_trigger.jsonl`: precomputed sentence-level retrieval outputs.

The retriever uses a DeBERTa-v3 backbone with a ColBERT-style token-level
encoder. At retrieval time it combines FAISS HNSW search, BGE sentence
embeddings, fuzzy matching, and aspect-word priors to match review sentences to
aspect triggers and opinion expressions.

## Recommended Local Layout

After downloading and organizing the Figshare files, link them into this repo as:

```text
retriever/assets/
  tables/
    trip_advisor/
      train.csv
      dev.csv
      test.csv
    beer_advocate/
      train.csv
      dev.csv
      test.csv
  trip_advisor/
    aspect.words
    kb.jsonl
    best_model.pt
    tokenizer/
    train_Faiss_matches_by_trigger_confidence.jsonl
    dev_Faiss_matches_by_trigger_confidence.jsonl
    test_Faiss_matches_by_trigger_confidence.jsonl
  beer_advocate/
    aspect.words
    kb.jsonl
    best_model.pt
    tokenizer/
    train_Faiss_matches_by_trigger.jsonl
    dev_Faiss_matches_by_trigger.jsonl
    test_Faiss_matches_by_trigger.jsonl
  checkpoints/
    trip_advisor_kasm_checkpoint/
      trip_advisor_kasm_epoch14_step101640.ckpt
```

If the package has already been downloaded elsewhere:

```bash
bash retriever/scripts/link_figshare_assets.sh /path/to/figshare_kasm_private_32988389/organized
```

To download the current private-share bundle:

```bash
python retriever/scripts/download_figshare_assets.py \
  --private-release \
  --output /path/to/figshare_kasm_private_32988389 \
  --link
```

The legacy public DOI can still be downloaded through the Figshare API by
omitting `--private-release`, but that package does not include the KASM
checkpoint.

Then link precomputed retrieval outputs into the KASM experiment data directory:

```bash
bash retriever/scripts/install_released_data.sh all
cd experiment
python scripts/validate_release_data.py
```

## Relation Data Format

The contrastive CSV must contain at least:

```text
q,pos,neg
```

The released files also include metadata columns:

```text
type,aspect,polarity,source,id
```

`type` is the relation category:

- `ST`: sentence to trigger
- `SO`: sentence to opinion
- `TT`: trigger to trigger
- `TO`: trigger to opinion
- `OO`: opinion to opinion

`train_retriever.sh` supports the same relation modes as the underlying script:

- `all`: use `ST/SO/TT/TO/OO`
- `kb_only`: use `TT/TO/OO`
- `to_oo_only`: use `TO/OO`

## Retrieval Output Format

Each JSONL row corresponds to one sentence:

```json
{
  "row": 0,
  "sub": 0,
  "sentence": "great location!",
  "matches": {
    "location": {
      "trigger": "great location",
      "score": 0.8841,
      "opinions": {
        "text": "great location",
        "polarity": "positive",
        "score": 0.8812
      }
    }
  }
}
```

`experiment/` groups these rows by `row` and aligns them with the CSV `id`.
