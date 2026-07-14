# Figshare Retriever Assets

Retriever assets are released separately on Figshare:

- DOI: https://doi.org/10.6084/m9.figshare.32970428
- Article page: https://figshare.com/articles/conference_contribution/KASM_Retriever/32970428

The Figshare package contains the retriever checkpoints, pseudo knowledge bases,
contrastive training data, tokenizer files, and precomputed retrieval outputs
used by KASM. These files are intentionally not committed to git.

## What The Package Contains

TripAdvisor assets:

- `kb.jsonl`: pseudo trigger-opinion knowledge base built from prompt outputs.
- `aspect.words`: aspect-word priors for retrieval.
- `new_mixed_constrative_triplets_balanced.csv`: contrastive retriever training pairs.
- `best_model.pt` plus `tokenizer/`: trained ColBERT-style retriever checkpoint.
- `train/dev/test_Faiss_matches_by_trigger_confidence.jsonl`: precomputed sentence-level retrieval outputs.

BeerAdvocate assets:

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
```

If the package has already been downloaded elsewhere:

```bash
bash retriever/scripts/link_figshare_assets.sh /path/to/figshare_kasm_retriever_32970428
```

To download it from Figshare with checksum verification:

```bash
python retriever/scripts/download_figshare_assets.py \
  --output retriever/assets/figshare_32970428 \
  --link
```

Then link precomputed retrieval outputs into the KASM experiment data directory:

```bash
bash retriever/scripts/install_precomputed_jsonl.sh all
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
