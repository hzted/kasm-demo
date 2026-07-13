# -*- coding: utf-8 -*-
import os
import math
import random
import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from transformers import (
    AutoConfig, AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm



# Utils
DOMAIN_ASPECTS = {
    "beer": ["taste", "look", "feel", "smell"],
    "beer_advocate": ["taste", "look", "feel", "smell"],
    "ba": ["taste", "look", "feel", "smell"],
    "trip": ["location", "clean", "service", "room", "value", "business", "check-in"],
    "tripadvisor": ["location", "clean", "service", "room", "value", "business", "check-in"],
    "trip_advisor": ["location", "clean", "service", "room", "value", "business", "check-in"],
}

ASPECT_LABELS = list(DOMAIN_ASPECTS["beer_advocate"])
ASPECT_TO_ID = {a: i for i, a in enumerate(ASPECT_LABELS)}


def set_aspect_labels(labels: List[str]):
    global ASPECT_LABELS, ASPECT_TO_ID
    ASPECT_LABELS = [normalize_aspect_name(x) for x in labels]
    ASPECT_TO_ID = {a: i for i, a in enumerate(ASPECT_LABELS)}


def normalize_aspect_name(aspect: str) -> str:
    x = str(aspect).strip().lower()
    aliases = {
        "checkin": "check-in",
        "check_in": "check-in",
        "check in": "check-in",
        "check-out": "check-in",
        "check out": "check-in",
    }
    return aliases.get(x, x)


def normalize_polarity_name(pol: str) -> str:
    x = str(pol).strip().lower()
    if x in {"positive", "pos", "+", "+1", "1"}:
        return "positive"
    if x in {"negative", "neg", "-", "-1"}:
        return "negative"
    if x in {"neutral", "neu", "0"}:
        return "neutral"
    return ""


def normalize_text_key(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def relation_family(rel: str) -> str:
    key = str(rel).strip().lower()
    if key in {"st", "so"}:
        return "sentence2kb"
    if key in {"tt", "to", "oo"}:
        return "kb2kb"
    return "other"


def aspect_family(aspect: str) -> str:
    key = normalize_aspect_name(aspect)
    mapping = {label: label for label in ASPECT_LABELS}
    return mapping.get(key, "other")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps))


@torch.no_grad()
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)



# Dataset
class TripletCSV(Dataset):
    """
    Read triplets from CSV and always use the row's own negative.
    """
    def __init__(self, csv_path: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        for need in ["q", "pos", "neg"]:
            if need not in self.df.columns:
                raise ValueError(f"CSV missing required column: {need}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        q = str(r["q"])
        pos = str(r["pos"])
        neg_texts = [str(r["neg"])]

        relation_type = str(r["type"]) if "type" in self.df.columns else "default"
        out = {"q": q, "pos": pos, "neg_texts": neg_texts, "relation_type": relation_type}
        for opt in ["type", "aspect", "polarity", "source"]:
            if opt in self.df.columns:
                out[opt] = str(r[opt])
        return out


class RelationAspectBatchSampler(Sampler[List[int]]):
    """
    Build local batches from buckets keyed by (relation_type, aspect_family).
    This keeps in-batch negatives semantically closer and reduces cross-family noise.
    """
    def __init__(self, df: pd.DataFrame, batch_size: int, shuffle: bool = True, seed: int = 42):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0
        self.buckets: Dict[tuple, List[int]] = {}

        type_series = df["type"].astype(str) if "type" in df.columns else pd.Series(["default"] * len(df))
        aspect_series = df["aspect"].astype(str) if "aspect" in df.columns else pd.Series(["other"] * len(df))
        for idx, (rel, asp) in enumerate(zip(type_series.tolist(), aspect_series.tolist())):
            key = (str(rel).strip().lower(), aspect_family(asp))
            self.buckets.setdefault(key, []).append(idx)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        batches: List[List[int]] = []
        for _, idxs in self.buckets.items():
            items = list(idxs)
            if self.shuffle:
                rng.shuffle(items)
            for start in range(0, len(items), self.batch_size):
                batch = items[start:start + self.batch_size]
                if batch:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return sum(math.ceil(len(v) / self.batch_size) for v in self.buckets.values())


@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_len_q: int = 64
    max_len_d: int = 48

    def _build_colbert_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # ColBERT 打分时只保留真实文本 token，排除 <s> / </s> / [CLS] / [SEP] 等特殊符号。
        special_ids = set(self.tokenizer.all_special_ids or [])
        if not special_ids:
            return attention_mask
        special_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for sid in special_ids:
            special_mask |= (input_ids == sid)
        return attention_mask * (~special_mask).long()

    def __call__(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raw_qs = [r["q"] for r in rows]
        poss = [r["pos"] for r in rows]
        relation_types = [str(r.get("relation_type", r.get("type", "default"))) for r in rows]
        qs = [f"relation: {rel.strip().lower()} [SEP] {q}" for q, rel in zip(raw_qs, relation_types)]
        aspect_labels = [ASPECT_TO_ID.get(normalize_aspect_name(r.get("aspect", "")), -100) for r in rows]
        polarities = [normalize_polarity_name(r.get("polarity", "")) for r in rows]
        sources = [normalize_text_key(r.get("source", "")) for r in rows]

        negs = [r["neg_texts"][0] for r in rows]  # [B]

        q = self.tokenizer(qs, padding=True, truncation=True,
                           max_length=self.max_len_q, return_tensors="pt")
        p = self.tokenizer(poss, padding=True, truncation=True,
                           max_length=self.max_len_d, return_tensors="pt")
        n = self.tokenizer(negs, padding=True, truncation=True,
                           max_length=self.max_len_d, return_tensors="pt")

        # attention_mask 继续给 backbone 用；colbert_mask 专门给 MaxSim 用。
        q_colbert_mask = self._build_colbert_mask(q["input_ids"], q["attention_mask"])
        p_colbert_mask = self._build_colbert_mask(p["input_ids"], p["attention_mask"])
        n_colbert_mask = self._build_colbert_mask(n["input_ids"], n["attention_mask"])

        return {
            "q_ids": q["input_ids"], "q_mask": q["attention_mask"],
            "p_ids": p["input_ids"], "p_mask": p["attention_mask"],
            "n_ids": n["input_ids"], "n_mask": n["attention_mask"],
            "q_colbert_mask": q_colbert_mask,
            "p_colbert_mask": p_colbert_mask,
            "n_colbert_mask": n_colbert_mask,
            "relation_types": relation_types,
            "aspect_labels": torch.tensor(aspect_labels, dtype=torch.long),
            "polarities": polarities,
            "raw_qs": raw_qs,
            "pos_texts": poss,
            "sources": sources,
        }



# Model
class GatedProjectionHead(nn.Module):
    """
    A more retrieval-friendly projection head:
    - a direct linear path preserves backbone information
    - a gated nonlinear delta path adds task-specific adaptation
    - LayerNorm stabilizes the token embedding space before L2 normalization
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.base = nn.Linear(input_dim, output_dim)
        self.delta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gate = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(output_dim)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        delta = self.delta(x)
        gate = torch.sigmoid(self.gate(x))
        out = base + self.dropout(gate * delta)
        return self.norm(out)


class ColBERTEncoder(nn.Module):
    """
    用预训练模型的 token 向量 + 投影头 + L2 归一化
    """
    def __init__(self, model_name: str, proj_mid_dim: int = 256, proj_dim: int = 128, dropout: float = 0.1,
                 num_aspects: int = len(ASPECT_LABELS), q_proj_mid_dim: int | None = None,
                 d_proj_mid_dim: int | None = None, backbone_dropout: float = 0.05):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        if hasattr(cfg, "add_pooling_layer"):
            cfg.add_pooling_layer = False  # 关闭 pooler
        self.backbone = AutoModel.from_pretrained(model_name, config=cfg)

        # BigBird 等模型有 block_sparse，强制成 original_full 以避免警告/错误
        if hasattr(self.backbone.config, "attention_type"):
            try:
                self.backbone.config.attention_type = "original_full"
            except Exception:
                pass

        H = self.backbone.config.hidden_size
        self.q_proj_mid_dim = int(q_proj_mid_dim if q_proj_mid_dim is not None else proj_mid_dim)
        self.d_proj_mid_dim = int(d_proj_mid_dim if d_proj_mid_dim is not None else proj_mid_dim)
        self.proj_dim     = int(proj_dim)       # d_z
        self.backbone_norm = nn.LayerNorm(H)
        self.backbone_dropout = nn.Dropout(p=backbone_dropout)

        self.q_proj = GatedProjectionHead(
            input_dim=H,
            hidden_dim=self.q_proj_mid_dim,
            output_dim=self.proj_dim,
            dropout=dropout,
        )
        self.d_proj = GatedProjectionHead(
            input_dim=H,
            hidden_dim=self.d_proj_mid_dim,
            output_dim=self.proj_dim,
            dropout=dropout,
        )
        self.aspect_attn = nn.Linear(self.proj_dim, 1)
        self.aspect_head = nn.Sequential(
            nn.LayerNorm(self.proj_dim),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.proj_dim, int(num_aspects)),
        )

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, side: str) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        tok = out.last_hidden_state                      # [B,L,H]
        tok = self.backbone_norm(tok)
        tok = self.backbone_dropout(tok)
        proj = self.q_proj if side == "q" else self.d_proj
        tok = proj(tok)                                  # [B,L,H]
        tok = l2_normalize(tok, dim=-1)                  # L2
        return tok

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._encode(input_ids, attention_mask, side="q")

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._encode(input_ids, attention_mask, side="d")

    def aspect_logits(self, tok: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid = attention_mask > 0
        attn_scores = self.aspect_attn(tok).squeeze(-1)             # [B, L]
        attn_scores = attn_scores.masked_fill(~valid, -1e4)
        attn = torch.softmax(attn_scores, dim=1).unsqueeze(-1)      # [B, L, 1]
        attn = attn * valid.unsqueeze(-1).float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (tok * attn).sum(dim=1)
        return self.aspect_head(pooled)



# ColBERT MaxSim score
def maxsim_score(Q: torch.Tensor, Qm: torch.Tensor,
                 D: torch.Tensor, Dm: torch.Tensor,
                 length_norm: bool = False) -> torch.Tensor:
    """
    ColBERT MaxSim:
      Q: [B,Lq,H], Qm: [B,Lq]
      D: [B,Ld,H], Dm: [B,Ld]
    返回: [B]
    """
    sim = torch.einsum("bid,bjd->bij", Q, D)                 # [B, Lq, Ld]
    # 文档侧无效 token（padding / special tokens）不参与每个 query token 的最大匹配。
    sim = sim.masked_fill(~(Dm > 0).unsqueeze(1), float("-inf"))
    m, _ = sim.max(dim=2)                                    # [B, Lq]
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    # 查询侧只聚合有效 token，避免特殊 token 对最终分数产生贡献。
    valid_q = (Qm > 0).float()
    m = m * valid_q
    s = m.sum(dim=1)                                         # [B]
    if length_norm:
        # 可选：按有效 query token 数做平均，减轻 query 长度不同带来的分数尺度偏置。
        denom = valid_q.sum(dim=1).clamp_min(1.0)
        s = s / denom
    return s



# Encode helpers
def encode(mdl: ColBERTEncoder, ids: torch.Tensor, mask: torch.Tensor, side: str) -> torch.Tensor:
    if side == "q":
        return mdl.encode_query(ids, mask)
    return mdl.encode_doc(ids, mask)




# Losses
def triplet_loss(s_pos: torch.Tensor, s_neg: torch.Tensor, margin: float = 0.2,
                 reduction: str = "mean") -> torch.Tensor:
    loss = F.relu(margin + s_neg - s_pos)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def infonce_loss(s_pos: torch.Tensor, s_neg_all: torch.Tensor, tau: float = 0.05,
                 reduction: str = "mean"):
    """
    s_pos: [B]
    s_neg_all: [B, K]
    返回 (loss, acc) 其中 acc=argmax 是否为正样本
    """
    logits = torch.cat([s_pos.unsqueeze(1), s_neg_all], dim=1) / tau  # [B, 1+K]
    target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # 正样本在列0
    loss = F.cross_entropy(logits, target, reduction=reduction)
    pred = logits.argmax(dim=1)
    acc = (pred == 0).float().mean()
    return loss, acc


def maxsim_score_matrix(Q: torch.Tensor, Qm: torch.Tensor,
                        D: torch.Tensor, Dm: torch.Tensor,
                        length_norm: bool = False) -> torch.Tensor:
    """
    All-pairs ColBERT MaxSim:
      Q: [Bq,Lq,H], Qm: [Bq,Lq]
      D: [Bd,Ld,H], Dm: [Bd,Ld]
    return: [Bq,Bd]
    """
    sim = torch.einsum("aid,bjd->abij", Q, D)                 # [Bq, Bd, Lq, Ld]
    sim = sim.masked_fill(~(Dm > 0).unsqueeze(0).unsqueeze(2), float("-inf"))
    m, _ = sim.max(dim=3)                                     # [Bq, Bd, Lq]
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    valid_q = (Qm > 0).float().unsqueeze(1)                   # [Bq,1,Lq]
    m = m * valid_q
    s = m.sum(dim=2)                                          # [Bq,Bd]
    if length_norm:
        denom = valid_q.sum(dim=2).clamp_min(1.0)             # [Bq,1]
        s = s / denom
    return s


def build_inbatch_positive_mask(relation_types: List[str],
                                aspects: torch.Tensor,
                                polarities: List[str],
                                raw_qs: List[str],
                                pos_texts: List[str],
                                sources: List[str],
                                device: torch.device) -> torch.Tensor:
    """
    Build a multi-positive mask over in-batch positive documents.
    Besides the diagonal pair, we also treat likely duplicate / same-source /
    same-aspect compatible examples as additional positives.
    """
    B = len(relation_types)
    mask = torch.eye(B, dtype=torch.bool, device=device)
    rel = [str(r).strip().lower() for r in relation_types]
    fam = [relation_family(r) for r in rel]
    asp = aspects.detach().cpu().tolist()
    pol = [normalize_polarity_name(p) for p in polarities]
    q_keys = [normalize_text_key(t) for t in raw_qs]
    p_keys = [normalize_text_key(t) for t in pos_texts]
    src_keys = [normalize_text_key(s) for s in sources]

    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            duplicate = (
                q_keys[i] == q_keys[j]
                or p_keys[i] == p_keys[j]
                or q_keys[i] == p_keys[j]
                or p_keys[i] == q_keys[j]
            )
            same_relation = rel[i] == rel[j] and rel[i] not in {"", "default"}
            same_family = fam[i] == fam[j] and fam[i] != "other"
            same_aspect = asp[i] >= 0 and asp[j] >= 0 and asp[i] == asp[j]
            same_polarity = pol[i] != "" and pol[j] != "" and pol[i] == pol[j]
            polarity_unknown = pol[i] == "" or pol[j] == ""
            same_source = src_keys[i] != "" and src_keys[i] == src_keys[j]

            mask_pair = False
            if duplicate:
                mask_pair = True
            elif same_source and same_aspect:
                mask_pair = True
            elif same_aspect and same_relation and (same_polarity or polarity_unknown):
                mask_pair = True
            elif same_aspect and same_family and rel[i] in {"st", "so"} and rel[j] in {"st", "so"}:
                mask_pair = True
            elif same_aspect and same_family and rel[i] == "tt" and rel[j] == "tt":
                mask_pair = True
            elif same_aspect and same_family and rel[i] in {"to", "oo"} and rel[j] in {"to", "oo"} and (same_polarity or polarity_unknown):
                mask_pair = True

            if mask_pair:
                mask[i, j] = True
    return mask


def build_inbatch_competition_mask(relation_types: List[str], device: torch.device) -> torch.Tensor:
    """
    Restrict in-batch competition to the same relation type.
    This reduces cross-relation false negatives such as ST competing with OO.
    """
    rel = [str(r).strip().lower() for r in relation_types]
    B = len(rel)
    mask = torch.zeros((B, B), dtype=torch.bool, device=device)
    for i in range(B):
        for j in range(B):
            if rel[i] == rel[j]:
                mask[i, j] = True
    return mask


def multi_positive_infonce_loss(logits: torch.Tensor, positive_mask: torch.Tensor):
    """
    logits: [B, C]
    positive_mask: [B, C], where one row may contain multiple positives.
    Returns per-example loss and correctness under any-positive top-1 match.
    """
    pos_logits = logits.masked_fill(~positive_mask, -1e4)
    pos_logsumexp = torch.logsumexp(pos_logits, dim=1)
    all_logsumexp = torch.logsumexp(logits, dim=1)
    per_example_loss = -(pos_logsumexp - all_logsumexp)
    pred = logits.argmax(dim=1)
    per_example_correct = positive_mask.gather(1, pred.unsqueeze(1)).squeeze(1).float()
    return per_example_loss, per_example_correct


def build_relation_weights(args) -> Dict[str, float]:
    # 文档里的总损失是按关系类型加权求和，这里把命令行参数整理成查表形式。
    return {
        "st": float(args.lambda_st),
        "so": float(args.lambda_so),
        "tt": float(args.lambda_tt),
        "to": float(args.lambda_to),
        "oo": float(args.lambda_oo),
        "default": 1.0,
    }


def relation_weight_tensor(relation_types: List[str], weight_map: Dict[str, float],
                           device: torch.device) -> torch.Tensor:
    # 为 batch 中每条样本生成对应的关系权重，后续用于样本级 loss 加权。
    weights = []
    for rel in relation_types:
        key = str(rel).strip().lower()
        weights.append(weight_map.get(key, weight_map["default"]))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def canonical_relation_name(rel: str) -> str:
    key = str(rel).strip().lower()
    if key in {"st", "so", "tt", "to", "oo"}:
        return key.upper()
    return "DEFAULT"


def init_relation_stats() -> Dict[str, Dict[str, float]]:
    return {
        "ST": {"loss": 0.0, "correct": 0.0, "count": 0},
        "SO": {"loss": 0.0, "correct": 0.0, "count": 0},
        "TT": {"loss": 0.0, "correct": 0.0, "count": 0},
        "TO": {"loss": 0.0, "correct": 0.0, "count": 0},
        "OO": {"loss": 0.0, "correct": 0.0, "count": 0},
        "DEFAULT": {"loss": 0.0, "correct": 0.0, "count": 0},
    }


def update_relation_stats(stats: Dict[str, Dict[str, float]],
                          relation_types: List[str],
                          per_example_loss: torch.Tensor,
                          per_example_correct: torch.Tensor) -> None:
    # 这里统计的是未乘 lambda_r 的原始样本 loss/正确率，便于横向比较各关系本身的学习难度。
    losses = per_example_loss.detach().float().cpu().tolist()
    corrects = per_example_correct.detach().float().cpu().tolist()
    for rel, loss_v, correct_v in zip(relation_types, losses, corrects):
        name = canonical_relation_name(rel)
        stats[name]["loss"] += float(loss_v)
        stats[name]["correct"] += float(correct_v)
        stats[name]["count"] += 1


def format_relation_stats(stats: Dict[str, Dict[str, float]]) -> str:
    parts = []
    for name in ["ST", "SO", "TT", "TO", "OO", "DEFAULT"]:
        count = stats[name]["count"]
        if count <= 0:
            continue
        avg_loss = stats[name]["loss"] / count
        avg_acc = stats[name]["correct"] / count
        parts.append(f"{name}:n={count},loss={avg_loss:.4f},acc={avg_acc:.4f}")
    return " | ".join(parts) if parts else "No relation stats"


def filter_relations(df: pd.DataFrame, relation_mode: str) -> pd.DataFrame:
    if relation_mode == "all":
        return df
    if "type" not in df.columns:
        raise ValueError("Filtered relation modes require a 'type' column in the CSV.")

    rel = df["type"].astype(str).str.strip().str.lower()
    keep_map = {
        "kb_only": {"tt", "to", "oo"},
        # to_oo_only 只保留 TO / OO，用于反极性 opinion-oriented KB 数据集实验。
        "to_oo_only": {"to", "oo"},
    }
    if relation_mode not in keep_map:
        raise ValueError(f"Unsupported relation_mode: {relation_mode}")
    return df.loc[rel.isin(keep_map[relation_mode])].reset_index(drop=True)


def stratified_train_val_split(df: pd.DataFrame, val_ratio: float, seed: int):
    """
    Stratified split by relation type when the CSV provides a `type` column.
    Falls back to a global random split otherwise.
    """
    if "type" not in df.columns:
        idxs = np.arange(len(df))
        np.random.shuffle(idxs)
        n_val = int(len(df) * val_ratio)
        val_idx = idxs[:n_val]
        train_idx = idxs[n_val:]
        return train_idx, val_idx

    rng = np.random.RandomState(seed)
    train_parts = []
    val_parts = []

    rel_series = df["type"].astype(str).str.strip().str.lower()
    for rel_name in sorted(rel_series.unique().tolist()):
        rel_idx = np.where(rel_series.values == rel_name)[0]
        rng.shuffle(rel_idx)

        n_val_rel = int(len(rel_idx) * val_ratio)
        if len(rel_idx) > 1:
            n_val_rel = max(1, n_val_rel)
            n_val_rel = min(len(rel_idx) - 1, n_val_rel)
        else:
            n_val_rel = 0

        val_parts.append(rel_idx[:n_val_rel])
        train_parts.append(rel_idx[n_val_rel:])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _backbone_num_layers(backbone: nn.Module) -> int:
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return len(backbone.encoder.layer)
    return 0


def _backbone_layer_id(name: str, num_layers: int) -> int:
    if name.startswith("backbone.embeddings"):
        return 0
    m = re.search(r"backbone\.encoder\.layer\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    return num_layers + 1


def set_backbone_trainable(mdl: nn.Module, trainable: bool) -> None:
    for p in mdl.backbone.parameters():
        p.requires_grad = trainable


def build_optimizer(args, mdl: nn.Module):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"]
    backbone_lr = float(args.backbone_lr if args.backbone_lr is not None else args.lr)
    head_lr = float(args.head_lr if args.head_lr is not None else args.lr)
    llrd = float(args.layerwise_lr_decay)
    num_layers = _backbone_num_layers(mdl.backbone)

    groups = []
    backbone_groups: Dict[tuple, Dict[str, Any]] = {}
    head_decay_params = []
    head_nodecay_params = []

    for name, p in mdl.named_parameters():
        if not p.requires_grad:
            continue
        use_no_decay = any(nd in name for nd in no_decay)
        if name.startswith("backbone."):
            layer_id = _backbone_layer_id(name, num_layers)
            lr = backbone_lr * (llrd ** max(0, (num_layers + 1) - layer_id))
            key = (layer_id, use_no_decay)
            if key not in backbone_groups:
                backbone_groups[key] = {
                    "params": [],
                    "lr": lr,
                    "weight_decay": 0.0 if use_no_decay else args.weight_decay,
                }
            backbone_groups[key]["params"].append(p)
        else:
            if use_no_decay:
                head_nodecay_params.append(p)
            else:
                head_decay_params.append(p)

    groups.extend(backbone_groups.values())
    if head_decay_params:
        groups.append({"params": head_decay_params, "lr": head_lr, "weight_decay": args.weight_decay})
    if head_nodecay_params:
        groups.append({"params": head_nodecay_params, "lr": head_lr, "weight_decay": 0.0})
    return AdamW(groups)


# Legacy Train / Validate (unused)
def train_one_epoch(args, mdl, tok, dl, optimizer, scheduler, scaler, device):
    mdl.train()
    total_loss, total_acc, total_seen = 0.0, 0.0, 0
    relation_weights = build_relation_weights(args)
    relation_stats = init_relation_stats()

    pbar = tqdm(dl, total=len(dl), ncols=120,
                desc=f"Epoch {args.cur_ep}/{args.epochs}")
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar, 1):
        q_ids = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)
        q_colbert_mask = batch["q_colbert_mask"].to(device)
        p_ids = batch["p_ids"].to(device)
        p_mask = batch["p_mask"].to(device)
        p_colbert_mask = batch["p_colbert_mask"].to(device)
        n_ids = batch["n_ids"].to(device)     # [B,L]
        n_mask = batch["n_mask"].to(device)
        n_colbert_mask = batch["n_colbert_mask"].to(device)
        relation_types = batch["relation_types"]

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda" and args.fp16)):
            Q = encode(mdl, q_ids, q_mask)                 # [B,Lq,H]
            P = encode(mdl, p_ids, p_mask)                 # [B,Ld,H]
            N = encode(mdl, n_ids, n_mask)                 # [B,Ld,H]

            # MaxSim 使用 ColBERT mask，而不是原始 attention_mask。
            s_pos = maxsim_score(Q, q_colbert_mask, P, p_colbert_mask,
                                 length_norm=args.length_norm)     # [B]

            s_neg = maxsim_score(Q, q_colbert_mask, N, n_colbert_mask,
                                 length_norm=args.length_norm)       # [B]
            s_neg_all = s_neg.unsqueeze(1)                            # [B,1]

            if args.loss_type == "triplet":
                per_example_loss = triplet_loss(s_pos, s_neg, args.margin, reduction="none")
                per_example_correct = (s_pos > s_neg).float()
                acc = per_example_correct.mean()
            else:  # InfoNCE
                per_example_loss, acc = infonce_loss(s_pos, s_neg_all, args.tau, reduction="none")
                logits = torch.cat([s_pos.unsqueeze(1), s_neg_all], dim=1) / args.tau
                per_example_correct = (logits.argmax(dim=1) == 0).float()

            # 先得到每条样本自己的 loss，再按关系类型乘上 lambda_r，最后做加权平均。
            sample_weights = relation_weight_tensor(relation_types, relation_weights, device)
            loss = (per_example_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
            update_relation_stats(relation_stats, relation_types, per_example_loss, per_example_correct)

        # 反向 + 优化
        scaler.scale(loss / args.grad_accum).backward()
        if step % args.grad_accum == 0:
            if device.type == "cuda" and args.fp16:
                scaler.unscale_(optimizer)
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), args.max_grad_norm)
                
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # 统计
        bs = q_ids.size(0)
        total_loss += loss.detach().item() * bs
        total_acc += acc.detach().item() * bs
        total_seen += bs

        pbar.set_postfix({
            "loss": f"{total_loss / max(1,total_seen):.4f}",
            "acc": f"{total_acc / max(1,total_seen):.4f}",
        })

    return total_loss / max(1, total_seen), total_acc / max(1, total_seen), relation_stats


@torch.no_grad()
def validate(args, mdl, tok, dl, device):
    mdl.eval()
    total_acc, total_seen = 0.0, 0
    relation_weights = build_relation_weights(args)
    total_loss = 0.0
    relation_stats = init_relation_stats()
    pbar = tqdm(dl, total=len(dl), ncols=120, desc="[Val]")

    for batch in pbar:
        q_ids = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)
        q_colbert_mask = batch["q_colbert_mask"].to(device)
        p_ids = batch["p_ids"].to(device)
        p_mask = batch["p_mask"].to(device)
        p_colbert_mask = batch["p_colbert_mask"].to(device)
        n_ids = batch["n_ids"].to(device)   # [B,L]
        n_mask = batch["n_mask"].to(device)
        n_colbert_mask = batch["n_colbert_mask"].to(device)
        relation_types = batch["relation_types"]

        Q = encode(mdl, q_ids, q_mask)
        P = encode(mdl, p_ids, p_mask)
        N = encode(mdl, n_ids, n_mask)

        # 验证阶段与训练阶段保持完全相同的打分口径。
        s_pos = maxsim_score(Q, q_colbert_mask, P, p_colbert_mask,
                             length_norm=args.length_norm)         # [B]
        s_neg = maxsim_score(Q, q_colbert_mask, N, n_colbert_mask,
                             length_norm=args.length_norm)        # [B]
        s_neg_all = s_neg.unsqueeze(1)                            # [B,1]

        if args.loss_type == "triplet":
            per_example_loss = triplet_loss(s_pos, s_neg, args.margin, reduction="none")
            per_example_correct = (s_pos > s_neg).float()
            acc = per_example_correct.mean()
        else:
            logits = torch.cat([s_pos.unsqueeze(1), s_neg_all], dim=1) / args.tau
            per_example_loss = F.cross_entropy(
                logits,
                torch.zeros(logits.size(0), dtype=torch.long, device=logits.device),
                reduction="none",
            )
            per_example_correct = (logits.argmax(dim=1) == 0).float()
            acc = per_example_correct.mean()

        # 验证阶段沿用与训练完全相同的关系加权方式，保证对比公平。
        sample_weights = relation_weight_tensor(relation_types, relation_weights, device)
        loss = (per_example_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
        update_relation_stats(relation_stats, relation_types, per_example_loss, per_example_correct)

        bs = q_ids.size(0)
        total_loss += loss.item() * bs
        total_acc += acc.item() * bs
        total_seen += bs
        pbar.set_postfix({
            "loss": f"{total_loss / max(1,total_seen):.4f}",
            "acc": f"{total_acc / max(1,total_seen):.4f}",
        })

    return total_loss / max(1, total_seen), total_acc / max(1, total_seen), relation_stats



# Active Train / Validate
def train_one_epoch(args, mdl, tok, dl, optimizer, scheduler, scaler, device):
    mdl.train()
    total_loss, total_acc, total_seen = 0.0, 0.0, 0
    relation_weights = build_relation_weights(args)
    relation_stats = init_relation_stats()

    pbar = tqdm(dl, total=len(dl), ncols=120,
                desc=f"Epoch {args.cur_ep}/{args.epochs}")
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar, 1):
        q_ids = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)
        q_colbert_mask = batch["q_colbert_mask"].to(device)
        p_ids = batch["p_ids"].to(device)
        p_mask = batch["p_mask"].to(device)
        p_colbert_mask = batch["p_colbert_mask"].to(device)
        n_ids = batch["n_ids"].to(device)
        n_mask = batch["n_mask"].to(device)
        n_colbert_mask = batch["n_colbert_mask"].to(device)
        relation_types = batch["relation_types"]
        aspect_labels = batch["aspect_labels"].to(device)
        polarities = batch["polarities"]
        raw_qs = batch["raw_qs"]
        pos_texts = batch["pos_texts"]
        sources = batch["sources"]

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda" and args.fp16)):
            Q = encode(mdl, q_ids, q_mask, side="q")
            P = encode(mdl, p_ids, p_mask, side="d")
            N = encode(mdl, n_ids, n_mask, side="d")

            s_pos = maxsim_score(Q, q_colbert_mask, P, p_colbert_mask,
                                 length_norm=args.length_norm)
            s_neg = maxsim_score(Q, q_colbert_mask, N, n_colbert_mask,
                                 length_norm=args.length_norm)

            if args.loss_type == "triplet":
                per_example_loss = triplet_loss(s_pos, s_neg, args.margin, reduction="none")
                per_example_correct = (s_pos > s_neg).float()
                acc = per_example_correct.mean()
            else:
                pos_logits = maxsim_score_matrix(Q, q_colbert_mask, P, p_colbert_mask,
                                                 length_norm=args.length_norm) / args.tau
                pos_mask = build_inbatch_positive_mask(
                    relation_types, aspect_labels, polarities, raw_qs, pos_texts, sources, device
                )
                comp_mask = build_inbatch_competition_mask(relation_types, device)
                pos_logits = pos_logits.masked_fill(~comp_mask, -1e4)
                logits = torch.cat([pos_logits, (s_neg / args.tau).unsqueeze(1)], dim=1)
                full_pos_mask = torch.cat(
                    [pos_mask, torch.zeros((pos_mask.size(0), 1), dtype=torch.bool, device=device)],
                    dim=1
                )
                per_example_loss, per_example_correct = multi_positive_infonce_loss(logits, full_pos_mask)
                acc = per_example_correct.mean()

            aspect_mask = aspect_labels >= 0
            if aspect_mask.any():
                q_aspect_logits = mdl.aspect_logits(Q, q_mask)
                p_aspect_logits = mdl.aspect_logits(P, p_mask)
                aspect_loss_q = F.cross_entropy(q_aspect_logits[aspect_mask], aspect_labels[aspect_mask])
                aspect_loss_p = F.cross_entropy(p_aspect_logits[aspect_mask], aspect_labels[aspect_mask])
                aspect_loss = 0.5 * (aspect_loss_q + aspect_loss_p)
            else:
                aspect_loss = torch.zeros((), dtype=per_example_loss.dtype, device=device)

            sample_weights = relation_weight_tensor(relation_types, relation_weights, device)
            retrieval_loss = (per_example_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
            aspect_weight = args.lambda_aspect * min(1.0, float(args.cur_ep) / max(1.0, float(args.aspect_warmup_epochs)))
            loss = retrieval_loss + aspect_weight * aspect_loss
            update_relation_stats(relation_stats, relation_types, per_example_loss, per_example_correct)

        scaler.scale(loss / args.grad_accum).backward()
        if step % args.grad_accum == 0:
            if device.type == "cuda" and args.fp16:
                scaler.unscale_(optimizer)
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        bs = q_ids.size(0)
        total_loss += loss.detach().item() * bs
        total_acc += acc.detach().item() * bs
        total_seen += bs

        pbar.set_postfix({
            "loss": f"{total_loss / max(1,total_seen):.4f}",
            "acc": f"{total_acc / max(1,total_seen):.4f}",
        })

    return total_loss / max(1, total_seen), total_acc / max(1, total_seen), relation_stats


@torch.no_grad()
def validate(args, mdl, tok, dl, device):
    mdl.eval()
    total_acc, total_seen = 0.0, 0
    relation_weights = build_relation_weights(args)
    total_loss = 0.0
    relation_stats = init_relation_stats()
    pbar = tqdm(dl, total=len(dl), ncols=120, desc="[Val]")

    for batch in pbar:
        q_ids = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)
        q_colbert_mask = batch["q_colbert_mask"].to(device)
        p_ids = batch["p_ids"].to(device)
        p_mask = batch["p_mask"].to(device)
        p_colbert_mask = batch["p_colbert_mask"].to(device)
        n_ids = batch["n_ids"].to(device)
        n_mask = batch["n_mask"].to(device)
        n_colbert_mask = batch["n_colbert_mask"].to(device)
        relation_types = batch["relation_types"]
        aspect_labels = batch["aspect_labels"].to(device)
        polarities = batch["polarities"]
        raw_qs = batch["raw_qs"]
        pos_texts = batch["pos_texts"]
        sources = batch["sources"]

        Q = encode(mdl, q_ids, q_mask, side="q")
        P = encode(mdl, p_ids, p_mask, side="d")
        N = encode(mdl, n_ids, n_mask, side="d")

        s_pos = maxsim_score(Q, q_colbert_mask, P, p_colbert_mask,
                             length_norm=args.length_norm)
        s_neg = maxsim_score(Q, q_colbert_mask, N, n_colbert_mask,
                             length_norm=args.length_norm)

        if args.loss_type == "triplet":
            per_example_loss = triplet_loss(s_pos, s_neg, args.margin, reduction="none")
            per_example_correct = (s_pos > s_neg).float()
            acc = per_example_correct.mean()
        else:
            pos_logits = maxsim_score_matrix(Q, q_colbert_mask, P, p_colbert_mask,
                                             length_norm=args.length_norm) / args.tau
            pos_mask = build_inbatch_positive_mask(
                relation_types, aspect_labels, polarities, raw_qs, pos_texts, sources, device
            )
            comp_mask = build_inbatch_competition_mask(relation_types, device)
            pos_logits = pos_logits.masked_fill(~comp_mask, -1e4)
            logits = torch.cat([pos_logits, (s_neg / args.tau).unsqueeze(1)], dim=1)
            full_pos_mask = torch.cat(
                [pos_mask, torch.zeros((pos_mask.size(0), 1), dtype=torch.bool, device=device)],
                dim=1
            )
            per_example_loss, per_example_correct = multi_positive_infonce_loss(logits, full_pos_mask)
            acc = per_example_correct.mean()

        aspect_mask = aspect_labels >= 0
        if aspect_mask.any():
            q_aspect_logits = mdl.aspect_logits(Q, q_mask)
            p_aspect_logits = mdl.aspect_logits(P, p_mask)
            aspect_loss_q = F.cross_entropy(q_aspect_logits[aspect_mask], aspect_labels[aspect_mask])
            aspect_loss_p = F.cross_entropy(p_aspect_logits[aspect_mask], aspect_labels[aspect_mask])
            aspect_loss = 0.5 * (aspect_loss_q + aspect_loss_p)
        else:
            aspect_loss = torch.zeros((), dtype=per_example_loss.dtype, device=device)

        sample_weights = relation_weight_tensor(relation_types, relation_weights, device)
        retrieval_loss = (per_example_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
        aspect_weight = args.lambda_aspect * min(1.0, float(getattr(args, "cur_ep", 1)) / max(1.0, float(args.aspect_warmup_epochs)))
        loss = retrieval_loss + aspect_weight * aspect_loss
        update_relation_stats(relation_stats, relation_types, per_example_loss, per_example_correct)

        bs = q_ids.size(0)
        total_loss += loss.item() * bs
        total_acc += acc.item() * bs
        total_seen += bs
        pbar.set_postfix({
            "loss": f"{total_loss / max(1,total_seen):.4f}",
            "acc": f"{total_acc / max(1,total_seen):.4f}",
        })

    return total_loss / max(1, total_seen), total_acc / max(1, total_seen), relation_stats


# Args / Entry
def build_args():
    p = argparse.ArgumentParser("ColBERT Contrastive with fixed triplets (Triplet / InfoNCE)")

    # Data
    p.add_argument("--domain", type=str, default=os.environ.get("DOMAIN", "beer_advocate"),
                   choices=sorted(DOMAIN_ASPECTS),
                   help="Dataset domain used to choose auxiliary aspect labels.")
    p.add_argument("--aspects", type=str, default="",
                   help="Optional comma-separated aspect label override.")
    p.add_argument("--csv", type=str, default="retriever/assets/beer_advocate/new_mixed_constrative_triplets_balanced.csv")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # Model / Tokenizer
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base") # BAAI/bge-m3，microsoft/deberta-v3-base
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--backbone_dropout", type=float, default=0.05)

    # Train hyperparams
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--backbone_lr", type=float, default=8e-6,
                   help="Backbone learning rate. Defaults to --lr.")
    p.add_argument("--head_lr", type=float, default=5e-5,
                   help="Learning rate for projection and auxiliary heads. Defaults to --lr.")
    p.add_argument("--weight_decay", type=float, default=0.03)
    p.add_argument("--layerwise_lr_decay", type=float, default=0.9,
                   help="Layer-wise learning rate decay for the backbone.")
    p.add_argument("--freeze_backbone_epochs", type=int, default=2,
                   help="Number of initial epochs that keep the backbone frozen.")
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--margin", type=float, default=0.15)     # for triplet
    p.add_argument("--tau", type=float, default=0.08)       # for InfoNCE
    p.add_argument("--loss_type", type=str, default="infonce",
                   choices=["triplet", "infonce"])
    p.add_argument("--relation_mode", type=str, default="all",
                   choices=["all", "kb_only", "to_oo_only"],
                   help="all: use every relation type; kb_only: only TT/TO/OO; to_oo_only: only TO/OO.")
    p.add_argument("--lambda_st", type=float, default=1.0)
    p.add_argument("--lambda_so", type=float, default=1.0)
    p.add_argument("--lambda_tt", type=float, default=1.2)
    p.add_argument("--lambda_to", type=float, default=1.2)
    p.add_argument("--lambda_oo", type=float, default=1.5)
    p.add_argument("--lambda_aspect", type=float, default=0.5,
                   help="Weight for the auxiliary aspect classification loss.")
    p.add_argument("--aspect_warmup_epochs", type=int, default=3,
                   help="Number of warmup epochs for the auxiliary aspect loss.")
    p.add_argument("--length_norm", action="store_true",
                   # 默认关闭，保持原始 ColBERT 的 sum_i max_j 打分；开启后改为平均。
                   help="Use length-normalized MaxSim instead of the original sum over valid query tokens.")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--proj_mid_dim", type=int, default=160)
    p.add_argument("--q_proj_mid_dim", type=int, default=192,
                   help="Hidden dimension of the query-side projection MLP. Defaults to --proj_mid_dim.")
    p.add_argument("--d_proj_mid_dim", type=int, default=128,
                   help="Hidden dimension of the document-side projection MLP. Defaults to --proj_mid_dim.")
    p.add_argument("--proj_dim", type=int, default=96)


    # Lengths
    p.add_argument("--max_len_q", type=int, default=96)
    p.add_argument("--max_len_d", type=int, default=64)

    # Misc
    p.add_argument("--out_dir", type=str, default="retriever/outputs/beer_advocate/ckpt_colbert_deberta_v3_inbatch_aux")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=0)

    return p.parse_args()


def save_full_model(args, mdl: ColBERTEncoder, tok: AutoTokenizer, out_dir: str, tag: str, best_val: float):
    """
    保存“整套模型”（Encoder+Proj）到一个文件；同时保存 tokenizer。
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "best_model.pt")
    payload = {
        "model_state_dict": mdl.state_dict(),
        "model_name": args.model_name,          # 便于后续重建 backbone
        "config": mdl.backbone.config.to_dict(),
        "args": vars(args),
        "val_acc": float(best_val),
        "tag": tag,
    }
    torch.save(payload, path)
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    print(f"[Save] best model -> {path} (ValAcc={best_val:.4f})")


def main():
    args = build_args()
    labels = [x.strip() for x in args.aspects.split(",") if x.strip()]
    if not labels:
        labels = DOMAIN_ASPECTS[args.domain]
    set_aspect_labels(labels)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device={device}, fp16={args.fp16}, relation_mode={args.relation_mode}")
    print(f"[Info] domain={args.domain}, aspect_labels={ASPECT_LABELS}")

    # Data split
    full_df = pd.read_csv(args.csv)
    # 两种训练流程共用同一套代码：
    # all     -> 使用 ST/SO/TT/TO/OO 全部关系
    # kb_only -> 只使用 TT/TO/OO，便于做知识库关系的 ablation
    # to_oo_only -> 只使用 TO/OO，便于训练反极性 opinion-oriented KB 数据集
    full_df = filter_relations(full_df, args.relation_mode)
    if len(full_df) == 0:
        raise ValueError(f"No samples available after relation filtering: relation_mode={args.relation_mode}")

    train_idx, val_idx = stratified_train_val_split(full_df, args.val_ratio, args.seed)

    train_csv = "_tmp_train.csv"
    val_csv = "_tmp_val.csv"
    full_df.iloc[train_idx].to_csv(train_csv, index=False)
    full_df.iloc[val_idx].to_csv(val_csv, index=False)
    print(f"[Info] filtered_samples={len(full_df)}, train={len(train_idx)}, val={len(val_idx)}")
    if "type" in full_df.columns:
        train_rel = full_df.iloc[train_idx]["type"].astype(str).str.strip().str.lower().value_counts().to_dict()
        val_rel = full_df.iloc[val_idx]["type"].astype(str).str.strip().str.lower().value_counts().to_dict()
        print(f"[Info] train relation dist={train_rel}")
        print(f"[Info] val relation dist={val_rel}")

    # Tokenizer / Model
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    mdl = ColBERTEncoder(args.model_name,
                         proj_mid_dim=args.proj_mid_dim,
                         q_proj_mid_dim=args.q_proj_mid_dim,
                         d_proj_mid_dim=args.d_proj_mid_dim,
                         proj_dim=args.proj_dim,
                         dropout=args.dropout,
                         backbone_dropout=args.backbone_dropout).to(device)
    print(f"[Info] Trainable params: {count_params(mdl):,}")

    # Datasets / Loaders
    ds_tr = TripletCSV(train_csv)
    ds_va = TripletCSV(val_csv)

    collate = Collator(tokenizer=tok, max_len_q=args.max_len_q, max_len_d=args.max_len_d)
    pin_memory = (device.type == "cuda")
    train_batch_sampler = RelationAspectBatchSampler(ds_tr.df, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_batch_sampler = RelationAspectBatchSampler(ds_va.df, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    print(f"[Info] train local buckets={len(train_batch_sampler.buckets)}, train batches={len(train_batch_sampler)}")
    print(f"[Info] val local buckets={len(val_batch_sampler.buckets)}, val batches={len(val_batch_sampler)}")
    dl_tr = DataLoader(ds_tr, batch_sampler=train_batch_sampler,
                       num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_sampler=val_batch_sampler,
                       num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=collate)

    # Optimizer / Scheduler / Scaler
    optimizer = build_optimizer(args, mdl)

    num_steps_per_epoch = math.ceil(len(dl_tr) / max(1, args.grad_accum))
    num_train_steps = args.epochs * num_steps_per_epoch
    num_warmup = int(args.warmup_ratio * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_train_steps)
    scheduler.last_epoch = -1

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda" and args.fp16))

    # Training
    os.makedirs(args.out_dir, exist_ok=True)
    best_val, bad = -1.0, 0
    last_tr_rel_stats = init_relation_stats()
    last_val_rel_stats = init_relation_stats()

    for ep in range(1, args.epochs + 1):
        args.cur_ep = ep
        freeze_backbone = ep <= max(0, int(args.freeze_backbone_epochs))
        set_backbone_trainable(mdl, not freeze_backbone)
        print(f"[Info] epoch={ep} backbone_trainable={not freeze_backbone}")
        tr_loss, tr_acc, tr_rel_stats = train_one_epoch(args, mdl, tok, dl_tr, optimizer, scheduler, scaler, device)
        val_loss, val_acc, val_rel_stats = validate(args, mdl, tok, dl_va, device)
        last_tr_rel_stats = tr_rel_stats
        last_val_rel_stats = val_rel_stats

        print(f"[Epoch {ep}] Train Loss={tr_loss:.4f}, Train Acc={tr_acc:.4f}")
        print(f"[Val] Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            bad = 0
            save_full_model(args, mdl, tok, args.out_dir, tag=f"epoch{ep}", best_val=best_val)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EarlyStop] no improve {bad} epochs. best={best_val:.4f}")
                break

    # Final epoch-level relation stats.
    print(f"[Final Train Relation Stats] {format_relation_stats(last_tr_rel_stats)}")
    print(f"[Final Val Relation Stats] {format_relation_stats(last_val_rel_stats)}")

    for pth in [train_csv, val_csv]:
        try:
            os.remove(pth)
        except Exception:
            pass

    print("[Done]")


if __name__ == "__main__":
    main()
