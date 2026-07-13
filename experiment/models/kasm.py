from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BigBirdConfig, BigBirdModel, T5ForConditionalGeneration

from .attention import LocalAttentiveAggregation
from .registry import register_model


@register_model("kasm")
class KASM(nn.Module):
    """
    Knowledge-Aware Sentiment Modeling.

    The model encodes aspect-sentence pairs with a T5 encoder, refines sentence
    evidence globally, decodes pseudo sentiment trajectories, and uses the final
    decoder hidden state as a query for evidence aggregation.
    """

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        interact_encoder_conf = self.conf.model.interact_encoder
        refine_encoder_conf = self.conf.model.refine_encoder
        d_model = refine_encoder_conf.d_model

        if self.conf.model.backbone:
            self.sent_process = T5ForConditionalGeneration.from_pretrained(self.conf.model.backbone)
            tokenizer = AutoTokenizer.from_pretrained(self.conf.model.backbone, use_fast=True)
            tokenizer.add_special_tokens({"additional_special_tokens": ["<SENT>"]})
            self.sent_process.resize_token_embeddings(len(tokenizer))
        else:
            self.sent_process = BigBirdModel(BigBirdConfig())

        self.pos_emb_layer = nn.Embedding(
            self.conf.data.max_num_sent + 1,
            interact_encoder_conf.d_model,
            padding_idx=0,
        )

        interact_layer = nn.TransformerEncoderLayer(
            d_model=interact_encoder_conf.d_model,
            nhead=interact_encoder_conf.num_head,
            dim_feedforward=interact_encoder_conf.ff_dim,
            dropout=interact_encoder_conf.dropout,
            batch_first=True,
        )
        self.interact_encoder = nn.TransformerEncoder(
            interact_layer,
            num_layers=interact_encoder_conf.num_layers,
        )

        refine_layer = nn.TransformerEncoderLayer(
            d_model=refine_encoder_conf.d_model,
            nhead=refine_encoder_conf.num_head,
            dim_feedforward=refine_encoder_conf.ff_dim,
            dropout=refine_encoder_conf.dropout,
            batch_first=True,
        )
        self.refine_encoder = nn.TransformerEncoder(
            refine_layer,
            num_layers=refine_encoder_conf.num_layers,
        )

        self.query_cls = False if self.conf.data.num_aspect > 0 else True
        self.local_pooling = LocalAttentiveAggregation(
            input_size=refine_encoder_conf.d_model,
            query_cls=self.query_cls,
        )

        self.clf = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(p=self.conf.model.dropout),
            nn.Linear(d_model, self.conf.model.num_class),
        )

        self.p_value_proj = nn.Linear(
            self.conf.model.d_model_in if hasattr(self.conf.model, "d_model_in") else 1,
            self.conf.model.d_model,
        )
        self.reg_head = nn.Linear(self.conf.model.d_model, 1)
        self.reg_act = nn.Tanh()

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.score_scale = d_model ** -0.5
        self.evidence_dropout = nn.Dropout(p=self.conf.model.dropout)

        self.pseudo_bias_alpha = getattr(self.conf.model, "pseudo_bias_alpha", 1.0)
        self.train_decode_mode = getattr(self.conf.model, "train_decode_mode", "pseudo_conf")
        self.eval_decode_mode = getattr(self.conf.model, "eval_decode_mode", "pseudo_conf")
        self.pseudo_conf_threshold = getattr(self.conf.model, "pseudo_conf_threshold", 0.5)

    def _decode_autoregressive(self, bos, local_embs, sent_mask, seq_len):
        dec_in = bos
        p_hats = []
        last_h = None

        for _ in range(seq_len):
            dec_out = self.sent_process.decoder(
                inputs_embeds=dec_in,
                encoder_hidden_states=local_embs,
                encoder_attention_mask=sent_mask.bool(),
                use_cache=False,
                return_dict=True,
            )
            h_t = dec_out.last_hidden_state[:, -1, :]
            p_t = self.reg_act(self.reg_head(h_t)).squeeze(-1)
            p_hats.append(p_t)
            last_h = h_t

            next_emb = self.p_value_proj(p_t.unsqueeze(-1))
            dec_in = torch.cat([dec_in, next_emb.unsqueeze(1)], dim=1)

        return torch.stack(p_hats, dim=1), last_h

    def _decode_with_pseudo_conf(self, bos, pseudo_label, pseudo_conf, local_embs, sent_mask, seq_len):
        dec_in = bos
        p_hats = []
        last_h = None

        dec_out = self.sent_process.decoder(
            inputs_embeds=dec_in,
            encoder_hidden_states=local_embs,
            encoder_attention_mask=sent_mask.bool(),
            use_cache=False,
            return_dict=True,
        )
        h_t = dec_out.last_hidden_state[:, -1, :]
        p_t = self.reg_act(self.reg_head(h_t)).squeeze(-1)
        p_hats.append(p_t)
        last_h = h_t

        for t in range(1, seq_len):
            if pseudo_label is not None:
                prev_gold = pseudo_label[:, t - 1]
                prev_pred = p_hats[-1].detach()
                if pseudo_conf is not None:
                    use_gold = pseudo_conf[:, t - 1] > self.pseudo_conf_threshold
                    prev_val = torch.where(use_gold, prev_gold, prev_pred)
                else:
                    prev_val = prev_gold
            else:
                prev_val = p_hats[-1].detach()

            prev_emb = self.p_value_proj(prev_val.unsqueeze(-1))
            dec_in = torch.cat([dec_in, prev_emb.unsqueeze(1)], dim=1)

            dec_out = self.sent_process.decoder(
                inputs_embeds=dec_in,
                encoder_hidden_states=local_embs,
                encoder_attention_mask=sent_mask.bool(),
                use_cache=False,
                return_dict=True,
            )
            h_t = dec_out.last_hidden_state[:, -1, :]
            p_t = self.reg_act(self.reg_head(h_t)).squeeze(-1)
            p_hats.append(p_t)
            last_h = h_t

        return torch.stack(p_hats, dim=1), last_h

    def _decode_with_pseudo_full(self, bos, pseudo_label, local_embs, sent_mask):
        if pseudo_label is None:
            raise ValueError("pseudo_label is required for pseudo_full decoding")
        dec_in = torch.cat([bos, self.p_value_proj(pseudo_label[:, :-1].unsqueeze(-1))], dim=1)
        dec_out = self.sent_process.decoder(
            inputs_embeds=dec_in,
            encoder_hidden_states=local_embs,
            encoder_attention_mask=sent_mask.bool(),
            use_cache=False,
            return_dict=True,
        )
        all_h = dec_out.last_hidden_state
        p_hat = self.reg_act(self.reg_head(all_h)).squeeze(-1)
        return p_hat, all_h[:, -1, :]

    def _decode_dispatch(self, decode_mode, bos, pseudo_label, pseudo_conf, local_embs, sent_mask, seq_len):
        if decode_mode == "ar":
            return self._decode_autoregressive(bos, local_embs, sent_mask, seq_len)
        if decode_mode == "pseudo_conf":
            return self._decode_with_pseudo_conf(
                bos,
                pseudo_label,
                pseudo_conf,
                local_embs,
                sent_mask,
                seq_len,
            )
        if decode_mode == "pseudo_full":
            return self._decode_with_pseudo_full(bos, pseudo_label, local_embs, sent_mask)
        raise ValueError(f"Unknown decode_mode: {decode_mode}")

    def _query_aggregate(self, dec_last_h, local_embs, sent_mask, pseudo_conf=None):
        q = self.query_proj(dec_last_h).unsqueeze(1)
        k = self.key_proj(local_embs)
        v = self.value_proj(local_embs)

        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) * self.score_scale
        if pseudo_conf is not None:
            scores = scores + self.pseudo_bias_alpha * pseudo_conf
        scores = scores.masked_fill(~sent_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(~sent_mask, 0.0)
        attn = self.evidence_dropout(attn)
        doc_emb = torch.bmm(attn.unsqueeze(1), v).squeeze(1)
        return doc_emb, attn, scores

    def forward(self, input_ids, attention_mask, token_type_ids, sent_pos_ids, aspect_ids, **kwargs):
        bsz, num_sent, num_token = input_ids.shape
        attention_mask = attention_mask.to(torch.bool)
        sent_mask = torch.clone(attention_mask[:, :, 0]).detach()

        flatten_input_ids = input_ids.reshape((bsz * num_sent, num_token))
        flatten_attention_mask = attention_mask.reshape((bsz * num_sent, num_token))
        flatten_attention_mask[:, 0] = True

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with autocast_context:
            sent_embs = self.sent_process.encoder(
                input_ids=flatten_input_ids,
                attention_mask=flatten_attention_mask,
            ).last_hidden_state

        sent_anchor_embs = sent_embs[:, 0, :]
        pos_emb = self.pos_emb_layer(sent_pos_ids)
        sent_anchor_embs = sent_anchor_embs.reshape((bsz, num_sent, -1)) + pos_emb
        sent_anchor_embs = self.interact_encoder(
            src=sent_anchor_embs,
            src_key_padding_mask=~sent_mask,
        )

        sent_embs = torch.cat(
            [sent_anchor_embs.reshape((bsz * num_sent, 1, -1)), sent_embs[:, 1:, :]],
            dim=1,
        )
        sent_embs = self.refine_encoder(
            sent_embs,
            src_key_padding_mask=~flatten_attention_mask.bool(),
        ).reshape((bsz, num_sent, num_token, -1))

        local_embs = self.local_pooling(sent_embs, attention_mask.bool())
        batch_size, seq_len, _ = local_embs.size()

        pseudo_label = kwargs.get("pseudo_polarity", None)
        trig_conf = kwargs.get("matched_trigger_score", None)
        opin_conf = kwargs.get("matched_opinion_score", None)
        pseudo_conf = torch.maximum(trig_conf, opin_conf) if trig_conf is not None and opin_conf is not None else None

        start_id = self.sent_process.config.decoder_start_token_id
        start_ids = torch.full(
            (batch_size, 1),
            start_id,
            device=local_embs.device,
            dtype=torch.long,
        )
        bos = self.sent_process.shared(start_ids)

        decode_mode = self.train_decode_mode if self.training else self.eval_decode_mode
        p_hat, dec_last_h = self._decode_dispatch(
            decode_mode,
            bos,
            pseudo_label,
            pseudo_conf,
            local_embs,
            sent_mask,
            seq_len,
        )

        doc_emb, evidence_attn, evidence_scores = self._query_aggregate(
            dec_last_h=dec_last_h,
            local_embs=local_embs,
            sent_mask=sent_mask,
            pseudo_conf=pseudo_conf,
        )
        doc_logits = self.clf(doc_emb)

        loss_seq = None
        if self.training and pseudo_label is not None:
            if pseudo_conf is None:
                pseudo_conf = torch.ones_like(p_hat)
            valid = sent_mask & (pseudo_conf > 0)
            err = F.smooth_l1_loss(p_hat, pseudo_label, reduction="none")
            loss_seq = ((err * pseudo_conf) * valid).sum() / valid.sum().clamp_min(1.0)

        return {
            "logits": doc_logits,
            "p_hat": p_hat,
            "dec_last_h": dec_last_h,
            "evidence_attn": evidence_attn,
            "evidence_scores": evidence_scores,
            "loss_seq": loss_seq,
        }
