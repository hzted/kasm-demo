# Dataset Files

Large dataset files are not committed to this anonymous repository.

Place files in this layout before training:

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
