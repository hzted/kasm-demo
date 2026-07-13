from .registry import register_collator
import torch


@register_collator("sentence")
class SentenceCollator:
    def __init__(self, conf):
        self.conf = conf

    def __call__(self, batch):
        conf = self.conf
        bsz = len(batch)

        sent_ids = torch.zeros(
            (bsz, conf.data.max_num_sent, conf.data.max_num_token_per_sent),
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(sent_ids, dtype=torch.float)
        segment_ids = torch.zeros_like(sent_ids, dtype=torch.long)
        sent_pos_ids = torch.zeros((bsz, conf.data.max_num_sent), dtype=torch.long)
        label_ids = torch.zeros((bsz,), dtype=torch.long)
        aspect_ids = torch.zeros_like(label_ids, dtype=torch.long)
        matched_trigger = torch.zeros((bsz, conf.data.max_num_sent), dtype=torch.float)
        matched_opinion = torch.zeros((bsz, conf.data.max_num_sent), dtype=torch.float)
        pseudo_polarity = torch.zeros((bsz, conf.data.max_num_sent), dtype=torch.float)

        has_matches = "matched_trigger_score" in batch[0]
        for bsz_idx, item in enumerate(batch):
            sent_ids_list = item["sent_ids_list"]
            segment_ids_list = item["segment_ids_list"]
            sent_pos_ids_list = item["sent_pos_ids_list"]

            for sent_idx, token_ids in enumerate(sent_ids_list):
                sent_pos_ids[bsz_idx, sent_idx] = sent_pos_ids_list[sent_idx]
                if has_matches:
                    matched_trigger[bsz_idx, sent_idx] = item["matched_trigger_score"][sent_idx]
                    matched_opinion[bsz_idx, sent_idx] = item["matched_opinion_score"][sent_idx]
                    pseudo_polarity[bsz_idx, sent_idx] = item["pseudo_polarity"][sent_idx]

                for token_idx, token_id in enumerate(token_ids):
                    sent_ids[bsz_idx, sent_idx, token_idx] = token_id
                    attention_mask[bsz_idx, sent_idx, token_idx] = 1.0
                    segment_ids[bsz_idx, sent_idx, token_idx] = segment_ids_list[sent_idx][token_idx]

            label_ids[bsz_idx] = item["label_id"]
            aspect_ids[bsz_idx] = item["doc_aspect_id"]

        output = {
            "input_ids": sent_ids,
            "attention_mask": attention_mask,
            "token_type_ids": segment_ids,
            "sent_pos_ids": sent_pos_ids,
            "label_ids": label_ids,
            "aspect_ids": aspect_ids,
        }
        if has_matches:
            output["matched_trigger_score"] = matched_trigger
            output["matched_opinion_score"] = matched_opinion
            output["pseudo_polarity"] = pseudo_polarity
        return output
