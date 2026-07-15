# Dataset Files

Large dataset files are not committed to this anonymous repository.

Install the released CSV tables and retrieved JSONL files from the Figshare
bundle:

```bash
cd ../..
bash retriever/scripts/install_released_data.sh all
cd experiment
python scripts/validate_release_data.py
```

The installed layout should be:

```text
dataset/
  trip_advisor/
    train.csv
    train.jsonl
    dev.csv
    dev.jsonl
    test.csv
    test.jsonl
  beer_advocate/
    train.csv
    train.jsonl
    dev.csv
    dev.jsonl
    test.csv
    test.jsonl
```

The CSV files contain document-level aspect labels. The JSONL files contain
sentence-level retrieved knowledge aligned by `row` with the CSV `id`.
