from .registry import register_datamodule
from .components import build_collator
from .misc import binary_rating_label, truncate

import ast
import csv
import json
import os
import pickle

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_jsonl_groups(file_path):
    current_group = []
    current_row_id = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                row_id = data.get("row")
                matches = data["matches"]
            except (json.JSONDecodeError, KeyError, AttributeError):
                continue

            if not current_group:
                current_group = [matches]
                current_row_id = row_id
            elif row_id == current_row_id:
                current_group.append(matches)
            else:
                yield current_row_id, current_group
                current_group = [matches]
                current_row_id = row_id

    if current_group:
        yield current_row_id, current_group


@register_datamodule("trip_advisor_kasm")
@register_datamodule("beer_advocate_kasm")
class KASMDatamodule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        os.makedirs(self.conf.cache_dir, exist_ok=True)
        self.file_path = os.path.join(
            self.conf.cache_dir,
            f"corpus_kasm_pseudo_{self.conf.data.name}_{self.conf.data.max_num_sent}_"
            f"{self.conf.data.max_num_token_per_sent}.pickle",
        )
        if not os.path.exists(self.file_path):
            self.convert_data_to_features()
            self.sentence_process()
        else:
            with open(self.file_path, "rb") as f:
                self.packed_data = pickle.load(f)

    def convert_data_to_features(self):
        dataset_map = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}
        output = {}

        for mode, filename in dataset_map.items():
            csv_path = os.path.join(self.conf.data_dir, filename)
            jsonl_path = os.path.join(self.conf.data_dir, filename.replace(".csv", ".jsonl"))
            jsonl_group_gen = generate_jsonl_groups(jsonl_path)

            with open(csv_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
                csv_reader = csv.reader(f, delimiter=",")
                for idx, row in enumerate(csv_reader):
                    if idx == 0:
                        aspects = row[5:5 + self.conf.data.num_aspect]
                        continue
                    output.setdefault(mode, [])

                    jsonl_row_id, sentence_matches_group = next(jsonl_group_gen)
                    doc_id = row[0]
                    if str(jsonl_row_id) != str(doc_id):
                        raise ValueError(
                            f"{self.conf.data.name} CSV/JSONL mismatch in {mode}: "
                            f"csv id={doc_id}, jsonl row={jsonl_row_id}"
                        )

                    aspect_ratings = [int(x) for x in row[5:5 + self.conf.data.num_aspect]]
                    sentence_column = int(self.conf.data.sentence_column)
                    doc_sentences = row[sentence_column]
                    skip_ratings = set(int(x) for x in self.conf.data.skip_ratings)
                    aliases = dict(self.conf.data.get("jsonl_aspect_aliases", {}))

                    for aspect, rating in zip(aspects, aspect_ratings):
                        if rating in skip_ratings:
                            continue
                        jsonl_aspect = aliases.get(aspect, aspect)
                        output[mode].append({
                            "doc_id": doc_id,
                            "doc_text": row[2],
                            "doc_sentences": doc_sentences,
                            "overall_rating": row[4],
                            "doc_aspect": aspect,
                            "doc_aspect_rating": rating,
                            "sentence_matches": [
                                match.get(jsonl_aspect, "") for match in sentence_matches_group
                            ],
                        })

        output["aspects2id"] = {aspect: idx for idx, aspect in enumerate(aspects)}
        self.packed_data = output

    def sentence_process(self):
        conf = self.conf
        tokenizer = AutoTokenizer.from_pretrained(conf.model.backbone, use_fast=False)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<SENT>"]})
        aspects2id = self.packed_data["aspects2id"]

        for mode in ["train", "dev", "test"]:
            feature_list = []

            for item in self.packed_data[mode]:
                sentences = ast.literal_eval(item["doc_sentences"])
                doc_aspect = item["doc_aspect"]
                aspect_label = binary_rating_label(item["doc_aspect_rating"])

                sent_list = []
                sent_ids_list = []
                segment_ids_list = []
                sent_pos_ids_list = []

                for sent_idx, sent in enumerate(sentences):
                    tok_sent = tokenizer.tokenize(sent)
                    tok_aspect = tokenizer.tokenize(doc_aspect)
                    trunc_tok_sent = truncate(
                        tok_sent,
                        max_len=conf.data.max_num_token_per_sent - 3 - len(tok_aspect),
                    )
                    concat_tok_sent = ["<SENT>"] + tok_aspect + ["</s>"] + trunc_tok_sent + ["</s>"]
                    segment_ids = [0] * (len(tok_aspect) + 2) + [1] * (len(trunc_tok_sent) + 1)
                    sent_ids = tokenizer.convert_tokens_to_ids(concat_tok_sent)

                    sent_list.append(concat_tok_sent)
                    sent_ids_list.append(sent_ids)
                    segment_ids_list.append(segment_ids)
                    sent_pos_ids_list.append(sent_idx + 1)

                sent_list = truncate(sent_list, max_len=conf.data.max_num_sent)
                sent_ids_list = truncate(sent_ids_list, max_len=conf.data.max_num_sent)
                segment_ids_list = truncate(segment_ids_list, max_len=conf.data.max_num_sent)
                sent_pos_ids_list = truncate(sent_pos_ids_list, max_len=conf.data.max_num_sent)

                sentence_matches = truncate(item["sentence_matches"], max_len=conf.data.max_num_sent)
                matched_trigger_score = [
                    match["score"] if match != "" else 0.0 for match in sentence_matches
                ]
                matched_opinion_score = [
                    match["opinions"]["score"]
                    if match != "" and match["opinions"] != "" else 0.0
                    for match in sentence_matches
                ]
                pseudo_polarity = [
                    match["opinions"]["polarity"]
                    if match != "" and match["opinions"] != "" else 0.0
                    for match in sentence_matches
                ]
                pseudo_polarity = [
                    1.0 if value == "positive" else -1.0 if value == "negative" else 0.0
                    for value in pseudo_polarity
                ]

                item.update({
                    "sent_list": sent_list,
                    "sent_ids_list": sent_ids_list,
                    "segment_ids_list": segment_ids_list,
                    "sent_pos_ids_list": sent_pos_ids_list,
                    "doc_aspect_id": aspects2id[doc_aspect],
                    "label_id": aspect_label,
                    "matched_trigger_score": matched_trigger_score,
                    "matched_opinion_score": matched_opinion_score,
                    "pseudo_polarity": pseudo_polarity,
                })
                feature_list.append(item)

            self.packed_data[mode] = feature_list

        with open(self.file_path, "wb") as f:
            pickle.dump(self.packed_data, f)

    def train_dataloader(self):
        return DataLoader(
            self.packed_data["train"],
            pin_memory=True,
            batch_size=self.conf.train.batch_size,
            shuffle=True,
            collate_fn=build_collator(conf=self.conf),
            num_workers=self.conf.train.num_workers,
            prefetch_factor=self.conf.train.prefetch_factor,
            persistent_workers=self.conf.train.persistent_workers,
        )

    def val_dataloader(self):
        loader = DataLoader(
            self.packed_data["dev"],
            pin_memory=True,
            batch_size=self.conf.dev.batch_size,
            collate_fn=build_collator(conf=self.conf),
            num_workers=self.conf.dev.num_workers,
            prefetch_factor=self.conf.dev.prefetch_factor,
            persistent_workers=self.conf.dev.persistent_workers,
        )
        return [loader, self.test_dataloader()]

    def test_dataloader(self):
        return DataLoader(
            self.packed_data["test"],
            pin_memory=True,
            batch_size=self.conf.test.batch_size,
            collate_fn=build_collator(conf=self.conf),
            num_workers=self.conf.test.num_workers,
            prefetch_factor=self.conf.test.prefetch_factor,
            persistent_workers=self.conf.test.persistent_workers,
        )
