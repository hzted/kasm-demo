import os
import json
import ast
import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig


class GatedProjectionHead(nn.Module):
    """Projection head aligned with the DeBERTa contrastive checkpoint."""

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
    """ColBERT encoder compatible with ckpt_colbert_deberta_v3_inbatch_aux."""

    def __init__(
        self,
        model_name: str,
        proj_mid_dim: int = 256,
        proj_dim: int = 128,
        dropout: float = 0.1,
        num_aspects: int = 7,
        q_proj_mid_dim: Optional[int] = None,
        d_proj_mid_dim: Optional[int] = None,
        backbone_dropout: float = 0.05,
    ):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        if hasattr(cfg, "add_pooling_layer"):
            cfg.add_pooling_layer = False
        self.backbone = AutoModel.from_pretrained(model_name, config=cfg)
        if hasattr(self.backbone.config, "attention_type"):
            try:
                self.backbone.config.attention_type = "original_full"
            except Exception:
                pass

        hidden = self.backbone.config.hidden_size
        self.q_proj_mid_dim = int(q_proj_mid_dim if q_proj_mid_dim is not None else proj_mid_dim)
        self.d_proj_mid_dim = int(d_proj_mid_dim if d_proj_mid_dim is not None else proj_mid_dim)
        self.proj_dim = int(proj_dim)
        self.backbone_norm = nn.LayerNorm(hidden)
        self.backbone_dropout = nn.Dropout(p=backbone_dropout)

        self.q_proj = GatedProjectionHead(hidden, self.q_proj_mid_dim, self.proj_dim, float(dropout))
        self.d_proj = GatedProjectionHead(hidden, self.d_proj_mid_dim, self.proj_dim, float(dropout))

        self.aspect_attn = nn.Linear(self.proj_dim, 1)
        self.aspect_head = nn.Sequential(
            nn.LayerNorm(self.proj_dim),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.proj_dim, int(num_aspects)),
        )

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, side: str) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        tok = out.last_hidden_state
        tok = self.backbone_norm(tok)
        tok = self.backbone_dropout(tok)
        proj = self.q_proj if side == "q" else self.d_proj
        tok = proj(tok)
        tok = F.normalize(tok, p=2, dim=-1)
        return tok

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._encode(input_ids, attention_mask, side="q")

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._encode(input_ids, attention_mask, side="d")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode_query(input_ids=input_ids, attention_mask=attention_mask)

    def aspect_logits(self, tok: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid = attention_mask > 0
        attn_scores = self.aspect_attn(tok).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~valid, -1e4)
        attn = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        attn = attn * valid.unsqueeze(-1).float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (tok * attn).sum(dim=1)
        return self.aspect_head(pooled)


ACTIVE_DOMAIN = os.environ.get("ACTIVE_DOMAIN", os.environ.get("DOMAIN", "beer")).strip().lower()

DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "beer": {
        "kb_path": "beer/kb.jsonl",
        "input_csv": "beer/dev.csv",
        "output_jsonl": "beer/beer_retrieve_results/dev_Faiss_matches_by_trigger.jsonl",
        "colbert_ckpt_dir": "beer/beer_ckpt/ckpt_colbert_deberta_v3_inbatch_aux",
        "aspect_words_path": "beer/aspect.words",
        "colbert_maxlen": 96,
        "doc_maxlen": 64,
        "length_norm": True,
        "aspect_order": ["feel", "look", "smell", "taste"],
        "aspect_label_order": ["taste", "look", "feel", "smell"],
        "aspect_to_col": {
            "feel": "feel",
            "look": "look",
            "smell": "smell",
            "taste": "taste",
        },
    },
    "trip": {
        "kb_path": "kb.jsonl",
        "input_csv": "test.csv",
        "output_jsonl": "test_Faiss_matches_by_trigger_confidence.jsonl",
        "colbert_ckpt_dir": "ckpt_colbert_deberta_v3_inbatch_aux",
        "aspect_words_path": "",
        "colbert_maxlen": 128,
        "doc_maxlen": 16,
        "length_norm": True,
        "aspect_order": [
            "location",
            "clean",
            "service",
            "room",
            "value",
            "business",
            "check-in",
        ],
        "aspect_label_order": [
            "location",
            "clean",
            "service",
            "room",
            "value",
            "business",
            "check-in",
        ],
        "aspect_to_col": {
            "location": "location",
            "clean": "clean",
            "service": "service",
            "room": "room",
            "value": "value",
            "business": "business",
            "check-in": "checkin",
        },
    },
}
DOMAIN_CONFIGS["tripadvisor"] = DOMAIN_CONFIGS["trip"]

if ACTIVE_DOMAIN not in DOMAIN_CONFIGS:
    raise ValueError(
        f"Unsupported ACTIVE_DOMAIN={ACTIVE_DOMAIN!r}. "
        f"Available domains: {', '.join(sorted(DOMAIN_CONFIGS))}"
    )

DOMAIN_CONFIG = DOMAIN_CONFIGS[ACTIVE_DOMAIN]

KB_PATH = os.environ.get("KB_PATH", DOMAIN_CONFIG["kb_path"])
INPUT_CSV = os.environ.get("INPUT_CSV", DOMAIN_CONFIG["input_csv"])
OUTPUT_JSONL = os.environ.get("OUTPUT_JSONL", DOMAIN_CONFIG["output_jsonl"])
ASPECT_WORDS_PATH = os.environ.get("ASPECT_WORDS_PATH", DOMAIN_CONFIG["aspect_words_path"])

SENT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
SENT_ATTN_TYPE = "original_full"
SENT_BATCH_SIZE = 16
SENT_MAX_LENGTH = 256
SENT_POOLING = "mean"

HNSW_M = 48
HNSW_EFC = 200
HNSW_EFS = 256

COLBERT_CKPT_DIR = os.environ.get("COLBERT_CKPT_DIR", DOMAIN_CONFIG["colbert_ckpt_dir"])
COLBERT_MAXLEN = int(os.environ.get("COLBERT_MAXLEN", DOMAIN_CONFIG["colbert_maxlen"]))
DOC_MAXLEN = int(os.environ.get("DOC_MAXLEN", DOMAIN_CONFIG["doc_maxlen"]))
LENGTH_NORM = os.environ.get("LENGTH_NORM", str(DOMAIN_CONFIG["length_norm"])).strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
QUERY_BATCH_SIZE = 64

TRIGGER_RECALL_TOPK_PER_ASPECT = 24
TRIGGER_PRELIM_TOPK = 6
OPINION_TOPK = 2
ASPECT_AGG_TOPK = 3
W_COLBERT = 0.75
W_COS = 0.20
W_FUZZY = 0.05
W_ASPECT_MODEL = float(os.environ.get("W_ASPECT_MODEL", "0.30"))
ASPECT_PRIOR_BONUS = float(os.environ.get("ASPECT_PRIOR_BONUS", "0.12"))
ASPECT_PRIOR_PENALTY = float(os.environ.get("ASPECT_PRIOR_PENALTY", "0.35"))
SCORE_THRESHOLD = 0.40

ASPECT_ORDER: List[str] = list(DOMAIN_CONFIG["aspect_order"])
ASPECT_LABEL_ORDER: List[str] = list(DOMAIN_CONFIG["aspect_label_order"])
ASPECT_TO_COL: Dict[str, str] = dict(DOMAIN_CONFIG["aspect_to_col"])

DEVICE: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu"


def read_csv(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            try:
                return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
            except Exception:
                continue
    raise RuntimeError(f"Failed to read CSV {path} with tried encodings.")


def split_on_top_level_commas(list_str: str) -> List[str]:
    s = list_str.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    items: List[str] = []
    cur: List[str] = []
    in_quote = False
    quote_ch = ""
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if in_quote:
            cur.append(ch)
            if ch == quote_ch:
                in_quote = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = True
            quote_ch = ch
            cur.append(ch)
            i += 1
            continue
        if ch == ",":
            items.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    if cur:
        items.append("".join(cur).strip())

    cleaned = []
    for it in items:
        if not it:
            continue
        if (it[0] == it[-1]) and it[0] in ("'", '"'):
            cleaned.append(it[1:-1])
        else:
            cleaned.append(it)
    return cleaned


def parse_sentence_cell(cell_val) -> List[str]:
    if isinstance(cell_val, list):
        return [str(x) for x in cell_val]
    s = str(cell_val).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        items = split_on_top_level_commas(s)
        if items:
            return items
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return [str(x) for x in obj]
    except Exception:
        pass
    return [s]


def extract_sentences_from_df(df: pd.DataFrame) -> Tuple[List[str], List[Tuple[int, int]]]:
    if "sub_sentence" not in df.columns:
        raise ValueError("CSV must contain a 'sub_sentence' column for retrieval input.")
    flat: List[str] = []
    idx_map: List[Tuple[int, int]] = []
    for i, v in enumerate(df["sub_sentence"].tolist()):
        items = parse_sentence_cell(v)
        for j, sent in enumerate(items):
            s = re.sub(r"\s+", " ", str(sent)).strip()
            if s:
                flat.append(s)
                idx_map.append((i, j))
    return flat, idx_map


def tokenize_alpha(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", str(text).lower())


def _expand_aspect_word(word: str) -> set:
    w = str(word).strip().lower()
    if not w:
        return set()
    variants = {w}
    manual = {
        "color": {"colour", "colored", "coloured", "coloring", "colouring"},
        "flavor": {"flavour", "flavors", "flavours", "flavourful", "flavorful"},
        "hop": {"hops", "hoppy", "hoppiness"},
        "malt": {"malts", "malty", "maltiness"},
        "aroma": {"aromas", "aromatic"},
        "nose": {"nosed"},
        "smell": {"smells", "smelled", "smelling"},
        "snif": {"sniff", "sniffs", "sniffing"},
        "head": {"heads", "headed"},
        "lace": {"lacing"},
        "body": {"bodied"},
        "mouthfeel": {"mouth-feel"},
        "bitter": {"bitterness"},
        "sugary": {"sugar", "sweet", "sweetness"},
        "dryness": {"dry"},
        "softness": {"soft"},
        "sharpness": {"sharp"},
    }
    variants.update(manual.get(w, set()))
    if len(w) > 3:
        variants.add(f"{w}s")
    return variants


def load_aspect_words(path: str) -> Dict[str, set]:
    if not path or not os.path.exists(path):
        return {}
    aspect_words: Dict[str, set] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            words = [w.strip().lower() for w in line.split() if w.strip()]
            if not words:
                continue
            asp = words[0]
            expanded = set()
            for word in words:
                expanded.update(_expand_aspect_word(word))
            aspect_words[asp] = expanded
    return aspect_words


def aspect_prior_score(text: str, aspect: str, aspect_words: Dict[str, set]) -> Tuple[float, List[str]]:
    words = aspect_words.get(aspect)
    if not words:
        return 0.0, []
    toks = set(tokenize_alpha(text))
    matched = sorted(toks & words)
    if matched:
        return ASPECT_PRIOR_BONUS * min(2, len(matched)), matched
    return -ASPECT_PRIOR_PENALTY, []


@torch.no_grad()
def colbert_aspect_prob_dicts(
    q_tok_batch: torch.Tensor,
    q_mask_batch: torch.Tensor,
    resources: Dict[str, Any],
) -> List[Dict[str, float]]:
    label_order = resources["aspect_label_order"]
    aspect_order = resources["aspect_order"]
    mdl = resources["col_mdl"]
    device = resources["device"]
    logits = mdl.aspect_logits(q_tok_batch.to(device), q_mask_batch.to(device))
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    out: List[Dict[str, float]] = []
    uniform = 1.0 / max(1, len(label_order))
    for row in probs:
        rec: Dict[str, float] = {}
        for asp in aspect_order:
            rec[asp] = float(row[label_order.index(asp)]) if asp in label_order else uniform
        out.append(rec)
    return out


def format_relation_query(text: str, relation: str) -> str:
    rel = str(relation).strip().lower()
    return f"relation: {rel} [SEP] {str(text).strip()}"


def simple_fuzzy_score(text_a: str, text_b: str) -> float:
    return simple_fuzzy_score_prepared(build_fuzzy_repr(text_a), build_fuzzy_repr(text_b))


def build_fuzzy_repr(text: str) -> Tuple[str, set, str]:
    lower = str(text).strip().lower()
    toks = tokenize_alpha(lower)
    tok_set = set(toks)
    tok_sort = " ".join(sorted(toks)) if toks else ""
    return lower, tok_set, tok_sort


def simple_fuzzy_score_prepared(
    repr_a: Tuple[str, set, str],
    repr_b: Tuple[str, set, str],
) -> float:
    a, toks_a, sort_a = repr_a
    b, toks_b, sort_b = repr_b
    if not a or not b:
        return 0.0
    direct = SequenceMatcher(None, a, b).ratio()
    if not toks_a or not toks_b:
        return float(direct)
    token_sort = SequenceMatcher(None, sort_a, sort_b).ratio()
    overlap = len(toks_a & toks_b) / max(1, len(toks_a | toks_b))
    return float(max(direct, 0.7 * token_sort + 0.3 * overlap))


def aspect_agg_weights(sent: str) -> List[float]:
    sent_len = len(tokenize_alpha(sent))
    if sent_len <= 10:
        return [1.0]
    if sent_len <= 25:
        return [0.60, 0.40]
    return [0.50, 0.30, 0.20]


def propose_spans(sent: str) -> List[str]:
    toks = tokenize_alpha(sent)
    if len(toks) <= 25:
        return [sent]

    raw_parts = [
        re.sub(r"\s+", " ", part).strip(" ,;:.-")
        for part in re.split(r"\s*(?:,|;|\.| but | while | although | however | though | yet )\s*", sent, flags=re.IGNORECASE)
    ]
    parts = [p for p in raw_parts if len(tokenize_alpha(p)) >= 5]
    spans: List[str] = [sent]
    spans.extend(parts)
    for i in range(len(parts) - 1):
        merged = f"{parts[i]}, {parts[i + 1]}".strip()
        if len(tokenize_alpha(merged)) >= 5:
            spans.append(merged)
    for i in range(len(parts) - 2):
        merged = f"{parts[i]}, {parts[i + 1]}, {parts[i + 2]}".strip()
        if len(tokenize_alpha(merged)) >= 7:
            spans.append(merged)

    seen = set()
    uniq: List[str] = []
    for span in spans:
        norm = re.sub(r"\s+", " ", span).strip(" ,;:.-!?").lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        uniq.append(re.sub(r"\s+", " ", span).strip(" ,;:.-!?"))
    return uniq[:8] if uniq else [sent]


def row_has_aspect(row: pd.Series, aspect: str) -> bool:
    col = ASPECT_TO_COL[aspect]
    val = row.get(col, -1)
    if pd.isna(val):
        return False
    try:
        return int(val) != -1
    except Exception:
        sval = str(val).strip().lower()
        return sval not in {"", "-1", "nan", "none"}


def extract_triggers_from_kb(kb_path: str) -> List[Dict[str, Any]]:
    seen, rows = set(), []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if isinstance(obj, dict) and ("trigger" in obj or "Trigger" in obj):
                trig = obj.get("trigger", obj.get("Trigger"))
                asp = obj.get("aspect", obj.get("Aspect"))
                if isinstance(trig, str) and trig.strip():
                    key = (str(asp) if asp is not None else None, trig.strip())
                    if key not in seen:
                        rows.append({"aspect": asp, "trigger": trig.strip()})
                        seen.add(key)

            if isinstance(obj, dict) and ("trigger" not in obj and "Trigger" not in obj):
                for asp, info in obj.items():
                    if not isinstance(info, dict):
                        continue
                    for trig in info.keys():
                        if not isinstance(trig, str):
                            continue
                        key = (asp, trig.strip())
                        if key not in seen:
                            rows.append({"aspect": asp, "trigger": trig.strip()})
                            seen.add(key)
    return rows


def extract_trigger_opinions_from_kb(kb_path: str):
    trigger_rows = extract_triggers_from_kb(kb_path)
    trig2idx = {(r.get("aspect"), r.get("trigger")): i for i, r in enumerate(trigger_rows)}
    trig2opinions: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(trigger_rows))}

    def _add(aspect, trigger, text, pol=None):
        key = (aspect, trigger)
        if key in trig2idx and isinstance(text, str) and text.strip():
            trig2opinions[trig2idx[key]].append({"text": text.strip(), "polarity": pol})

    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if isinstance(obj, dict):
                for asp, info in obj.items():
                    if not isinstance(info, dict):
                        continue
                    for trig, payload in info.items():
                        if not isinstance(trig, str) or not isinstance(payload, dict):
                            continue
                        for pol_key in ["positive", "negative", "pos", "neg"]:
                            if pol_key in payload and isinstance(payload[pol_key], (list, tuple)):
                                pol_norm = "positive" if pol_key in ["positive", "pos"] else "negative"
                                for o in payload[pol_key]:
                                    _add(asp, trig, str(o), pol_norm)
    return trigger_rows, trig2opinions


def load_kb(kb_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    trigger_rows, trig2opinions = extract_trigger_opinions_from_kb(kb_path)
    triggers: List[Dict[str, Any]] = []
    aspects: List[str] = []
    for i, row in enumerate(trigger_rows):
        asp = str(row.get("aspect", "")).strip()
        trig = str(row.get("trigger", "")).strip()
        if not asp or not trig:
            continue
        triggers.append({"aspect": asp, "trigger": trig, "opinions": trig2opinions.get(i, [])})
        aspects.append(asp)
    print(f"[KB] loaded {len(triggers)} triggers, {len(set(aspects))} aspects from {kb_path}")
    return triggers, aspects


def build_sentence_model(model_name: str, attn_type: str, device: Optional[str]):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SentenceModel] loading {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    cfg = AutoConfig.from_pretrained(model_name)
    if hasattr(cfg, "add_pooling_layer"):
        cfg.add_pooling_layer = False
    mdl = AutoModel.from_pretrained(model_name, config=cfg)
    if hasattr(mdl.config, "attention_type"):
        try:
            mdl.config.attention_type = attn_type
        except Exception:
            pass
    mdl.to(device).eval()
    return tok, mdl, device


@torch.no_grad()
def embed_texts(
    texts: List[str],
    tok,
    mdl,
    device: str,
    batch_size: int,
    max_length: int,
    pooling: str,
) -> np.ndarray:
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with amp.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
            out = mdl(**enc).last_hidden_state
            if pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1)
                out = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            elif pooling == "cls":
                out = out[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
        vecs.append(out.detach().cpu().numpy().astype(np.float32))
    if not vecs:
        return np.zeros((0, mdl.config.hidden_size), dtype=np.float32)
    return np.concatenate(vecs, axis=0)


def build_faiss_hnsw(vecs: np.ndarray, m: int, efc: int, efs: int) -> faiss.IndexHNSWFlat:
    d = vecs.shape[1]
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efc
    index.hnsw.efSearch = efs
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return index


def faiss_search(index: faiss.IndexHNSWFlat, queries: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    q = queries.astype(np.float32, copy=False)
    faiss.normalize_L2(q)
    return index.search(q, topk)


def load_contrastive_colbert(ckpt_dir: str, device: str):
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    payload = torch.load(ckpt_path, map_location=device)
    model_name = payload.get("model_name", "microsoft/deberta-v3-base")
    args = payload.get("args", {})
    dropout = args.get("dropout", 0.1)

    tok = AutoTokenizer.from_pretrained(os.path.join(ckpt_dir, "tokenizer"), use_fast=True)
    mdl = ColBERTEncoder(
        model_name,
        proj_mid_dim=args.get("proj_mid_dim", 256),
        proj_dim=args.get("proj_dim", 128),
        dropout=dropout,
        num_aspects=len(ASPECT_ORDER),
        q_proj_mid_dim=args.get("q_proj_mid_dim"),
        d_proj_mid_dim=args.get("d_proj_mid_dim"),
        backbone_dropout=args.get("backbone_dropout", 0.05),
    )
    state_dict = payload.get("model_state_dict", payload.get("state_dict", {}))
    mdl.load_state_dict(state_dict, strict=True)
    mdl.to(device).eval()
    infer_cfg = {
        "length_norm": LENGTH_NORM,
        "max_len_q": int(args.get("max_len_q", COLBERT_MAXLEN)),
        "max_len_d": int(args.get("max_len_d", DOC_MAXLEN)),
    }
    return tok, mdl, infer_cfg


def build_colbert_mask(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    special_ids = set(tokenizer.all_special_ids or [])
    if not special_ids:
        return attention_mask
    special_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for sid in special_ids:
        special_mask |= (input_ids == sid)
    return attention_mask * (~special_mask).long()


@torch.no_grad()
def colbert_encode_texts(
    texts: List[str],
    tok,
    mdl,
    device: str,
    max_length: int,
    batch_size: int = 32,
    return_mask: bool = False,
    side: str = "q",
):
    all_out: List[torch.Tensor] = []
    all_mask: List[torch.Tensor] = []

    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i:i + batch_size],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        if side == "d":
            last = mdl.encode_doc(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        else:
            last = mdl.encode_query(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        all_out.append(last.detach().cpu())
        if return_mask:
            all_mask.append(build_colbert_mask(tok, enc["input_ids"], enc["attention_mask"]).detach().cpu())

    emb = torch.cat(all_out, dim=0) if all_out else torch.zeros((0, max_length, mdl.proj_dim))
    if return_mask:
        mask = torch.cat(all_mask, dim=0) if all_mask else torch.zeros((0, max_length), dtype=torch.long)
        return emb, mask
    return emb


def colbert_maxsim_scores(
    q_tok: torch.Tensor,
    d_tok: torch.Tensor,
    q_mask: Optional[torch.Tensor] = None,
    d_mask: Optional[torch.Tensor] = None,
    length_norm: bool = True,
) -> np.ndarray:
    device = q_tok.device
    if q_mask is not None:
        q_mask = q_mask.to(device)
    if d_mask is not None:
        d_mask = d_mask.to(device)

    sim = torch.matmul(q_tok.unsqueeze(0), d_tok.transpose(1, 2))
    if d_mask is not None:
        sim = sim.masked_fill(d_mask.unsqueeze(1) == 0, -1e4)

    max_sim = sim.max(dim=2).values
    if q_mask is not None:
        max_sim = max_sim * q_mask.unsqueeze(0).to(max_sim.dtype)
        q_len = q_mask.sum().clamp(min=1).to(max_sim.dtype)
    else:
        q_len = torch.tensor(max_sim.shape[1], device=device, dtype=max_sim.dtype)

    scores = max_sim.sum(dim=1)
    if length_norm:
        scores = scores / q_len
    return scores.detach().cpu().numpy().astype(np.float32)


def cosine_scores(query_vec: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    c = np.asarray(cand_vecs, dtype=np.float32)
    qn = np.linalg.norm(q) + 1e-8
    cn = np.linalg.norm(c, axis=1) + 1e-8
    return (c @ q) / (cn * qn)


def is_noise_sentence(sent: str) -> bool:
    return len(tokenize_alpha(sent)) == 0


def build_resources():
    global DEVICE
    if DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Device] {DEVICE}")
    print(f"[Domain] {ACTIVE_DOMAIN}")
    print(f"[Load] KB from {KB_PATH}")
    triggers, kb_aspects = load_kb(KB_PATH)
    aspect_words = load_aspect_words(ASPECT_WORDS_PATH)
    if aspect_words:
        print(f"[AspectWords] loaded {sum(len(v) for v in aspect_words.values())} terms from {ASPECT_WORDS_PATH}")
    aspect_order = [asp for asp in ASPECT_ORDER if asp in set(kb_aspects)] or ASPECT_ORDER
    trigger_texts = [t["trigger"] for t in triggers]
    print("[Aspects order]", aspect_order)
    print(f"[AspectHead] label_order={ASPECT_LABEL_ORDER}, weight={W_ASPECT_MODEL}")

    sent_tok, sent_mdl, DEVICE_used = build_sentence_model(SENT_MODEL_NAME, SENT_ATTN_TYPE, DEVICE)
    DEVICE = DEVICE_used

    print("[Embed] KB triggers (sentence embedding)")
    trig_vecs = embed_texts(
        trigger_texts, sent_tok, sent_mdl, DEVICE, SENT_BATCH_SIZE, SENT_MAX_LENGTH, SENT_POOLING
    )
    for trig in triggers:
        trig["_fuzzy"] = build_fuzzy_repr(trig["trigger"])

    print("[ColBERT] load contrastive model")
    col_tok, col_mdl, col_cfg = load_contrastive_colbert(COLBERT_CKPT_DIR, DEVICE)
    q_maxlen = int(col_cfg.get("max_len_q", COLBERT_MAXLEN))
    d_maxlen = int(col_cfg.get("max_len_d", DOC_MAXLEN))
    length_norm = bool(col_cfg.get("length_norm", False))
    print(f"[ColBERT] length_norm={length_norm}, q_maxlen={q_maxlen}, d_maxlen={d_maxlen}")

    print("[ColBERT] encode all triggers")
    trig_tok, trig_mask = colbert_encode_texts(
        trigger_texts, col_tok, col_mdl, DEVICE, d_maxlen, batch_size=32, return_mask=True, side="d"
    )

    all_op_texts: List[str] = []
    all_op_meta: List[Dict[str, Any]] = []
    for trig in triggers:
        trig["_op_indices"] = []
        for op in trig.get("opinions", []) or []:
            all_op_texts.append(op["text"])
            all_op_meta.append(
                {
                    "text": op["text"],
                    "polarity": op.get("polarity"),
                    "_fuzzy": build_fuzzy_repr(op["text"]),
                }
            )
            trig["_op_indices"].append(len(all_op_texts) - 1)

    if all_op_texts:
        print("[ColBERT] encode all opinions")
        op_tok_all, op_mask_all = colbert_encode_texts(
            all_op_texts, col_tok, col_mdl, DEVICE, d_maxlen, batch_size=64, return_mask=True, side="d"
        )
        print("[Embed] all opinions (sentence embedding)")
        op_vecs_all = embed_texts(
            all_op_texts, sent_tok, sent_mdl, DEVICE, SENT_BATCH_SIZE, SENT_MAX_LENGTH, SENT_POOLING
        )
    else:
        op_tok_all = torch.zeros((0, d_maxlen, col_mdl.proj_dim))
        op_mask_all = torch.zeros((0, d_maxlen), dtype=torch.long)
        op_vecs_all = np.zeros((0, trig_vecs.shape[1]), dtype=np.float32)

    aspect_to_trigger_indices: Dict[str, List[int]] = {}
    for j, rec in enumerate(triggers):
        aspect_to_trigger_indices.setdefault(rec["aspect"], []).append(j)

    aspect_faiss: Dict[str, Tuple[faiss.IndexHNSWFlat, np.ndarray]] = {}
    for asp, idxs in aspect_to_trigger_indices.items():
        idxs_np = np.asarray(idxs, dtype=np.int64)
        aspect_faiss[asp] = (
            build_faiss_hnsw(trig_vecs[idxs_np].copy(), HNSW_M, HNSW_EFC, HNSW_EFS),
            idxs_np,
        )

    return {
        "triggers": triggers,
        "aspect_order": aspect_order,
        "aspect_label_order": ASPECT_LABEL_ORDER,
        "aspect_words": aspect_words,
        "sent_tok": sent_tok,
        "sent_mdl": sent_mdl,
        "trig_vecs": trig_vecs,
        "col_tok": col_tok,
        "col_mdl": col_mdl,
        "q_maxlen": q_maxlen,
        "d_maxlen": d_maxlen,
        "length_norm": length_norm,
        "trig_tok": trig_tok,
        "trig_mask": trig_mask,
        "all_op_meta": all_op_meta,
        "op_tok_all": op_tok_all,
        "op_mask_all": op_mask_all,
        "op_vecs_all": op_vecs_all,
        "aspect_faiss": aspect_faiss,
        "device": DEVICE,
    }


def _score_single_query_candidates(
    sent: str,
    sent_fuzzy: Tuple[str, set, str],
    q_vec: np.ndarray,
    q_trig_tok: torch.Tensor,
    q_trig_mask: torch.Tensor,
    q_op_tok: torch.Tensor,
    q_op_mask: torch.Tensor,
    aspect_probs: Optional[Dict[str, float]],
    resources: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    triggers = resources["triggers"]
    aspect_order = resources["aspect_order"]
    aspect_words = resources["aspect_words"]
    trig_tok = resources["trig_tok"]
    trig_mask = resources["trig_mask"]
    trig_vecs = resources["trig_vecs"]
    op_tok_all = resources["op_tok_all"]
    op_mask_all = resources["op_mask_all"]
    op_vecs_all = resources["op_vecs_all"]
    all_op_meta = resources["all_op_meta"]
    aspect_faiss = resources["aspect_faiss"]
    length_norm = resources["length_norm"]

    aspect_candidates: Dict[str, Dict[str, Any]] = {}
    for asp in aspect_order:
        if asp not in aspect_faiss:
            continue
        asp_index, asp_global_idx = aspect_faiss[asp]
        _, local_idx = faiss_search(asp_index, q_vec, TRIGGER_RECALL_TOPK_PER_ASPECT)
        local_idx = local_idx[0]
        local_idx = local_idx[local_idx >= 0]
        if len(local_idx) == 0:
            continue

        cand_idx_list = asp_global_idx[local_idx].tolist()
        cand_trig_tok = trig_tok[cand_idx_list]
        cand_trig_mask = trig_mask[cand_idx_list]
        colb_scores = colbert_maxsim_scores(
            q_trig_tok, cand_trig_tok, q_mask=q_trig_mask, d_mask=cand_trig_mask, length_norm=length_norm
        )
        cos_scores = cosine_scores(q_vec, trig_vecs[cand_idx_list])
        trig_base_scores = (
            W_COLBERT * colb_scores
            + W_COS * cos_scores
        )
        order = np.argsort(-trig_base_scores)
        prelim = order[: min(TRIGGER_PRELIM_TOPK, len(order))]
        prelim_fuzzy = np.asarray(
            [
                simple_fuzzy_score_prepared(sent_fuzzy, triggers[cand_idx_list[int(pos)]]["_fuzzy"])
                for pos in prelim
            ],
            dtype=np.float32,
        )
        prelim_scores = trig_base_scores[prelim] + W_FUZZY * prelim_fuzzy
        prelim_order = np.argsort(-prelim_scores)
        pair_candidates: List[Dict[str, Any]] = []

        for rank in prelim_order:
            cand_pos = prelim[int(rank)]
            trig_j = cand_idx_list[int(cand_pos)]
            trig_text = triggers[trig_j]["trigger"]
            trig_score = float(prelim_scores[int(rank)])

            op_indices = triggers[trig_j].get("_op_indices", []) or []
            if op_indices:
                op_tok = op_tok_all[op_indices]
                op_mask = op_mask_all[op_indices]
                op_scores = colbert_maxsim_scores(
                    q_op_tok, op_tok, q_mask=q_op_mask, d_mask=op_mask, length_norm=length_norm
                )
                op_cos = cosine_scores(q_vec, op_vecs_all[op_indices])
                op_base_scores = (
                    W_COLBERT * op_scores
                    + W_COS * op_cos
                )
                op_prelim_size = min(len(op_base_scores), max(OPINION_TOPK * 2, OPINION_TOPK))
                op_prelim = np.argsort(-op_base_scores)[:op_prelim_size]
                op_fuzzy = np.asarray(
                    [
                        simple_fuzzy_score_prepared(sent_fuzzy, all_op_meta[op_indices[int(idx)]]["_fuzzy"])
                        for idx in op_prelim
                    ],
                    dtype=np.float32,
                )
                op_prelim_scores = op_base_scores[op_prelim] + W_FUZZY * op_fuzzy
                op_order = np.argsort(-op_prelim_scores)[: min(OPINION_TOPK, len(op_prelim_scores))]
                for op_rank in op_order:
                    op_best = op_prelim[int(op_rank)]
                    raw_opinion_score = float(op_prelim_scores[int(op_rank)])
                    op_meta = all_op_meta[op_indices[int(op_best)]]
                    # Penalize a strong trigger only when the matched opinion
                    # clearly fails to support it.
                    pair_score = trig_score - 0.25 * max(
                        0.0, trig_score - raw_opinion_score
                    )
                    pair_candidates.append(
                        {
                            "trigger": trig_text,
                            "score": float(pair_score),
                            "opinions": {
                                "text": op_meta["text"],
                                "polarity": op_meta.get("polarity"),
                                "score": raw_opinion_score,
                            },
                            "raw_trigger_score": trig_score,
                            "raw_opinion_score": raw_opinion_score,
                        }
                    )
            else:
                pair_candidates.append(
                    {
                        "trigger": trig_text,
                        "score": float(trig_score),
                        "opinions": "",
                        "raw_trigger_score": trig_score,
                        "raw_opinion_score": 0.0,
                    }
                )

        pair_candidates.sort(key=lambda x: x["score"], reverse=True)
        best = pair_candidates[0]
        agg_weights = aspect_agg_weights(sent)
        top_scores = [float(x["score"]) for x in pair_candidates[: min(len(agg_weights), len(pair_candidates))]]
        agg_weights = agg_weights[: len(top_scores)]
        weight_sum = sum(agg_weights) if agg_weights else 1.0
        raw_aspect_score = sum(w * s for w, s in zip(agg_weights, top_scores)) / weight_sum
        prior_score, prior_terms = aspect_prior_score(sent, asp, aspect_words)
        aspect_prob = float(aspect_probs.get(asp, 1.0 / max(1, len(aspect_order)))) if aspect_probs else 0.0
        aspect_model_score = W_ASPECT_MODEL * (aspect_prob - (1.0 / max(1, len(aspect_order))))
        aspect_score = raw_aspect_score + prior_score + aspect_model_score
        aspect_candidates[asp] = {
            "trigger": best["trigger"],
            "score": float(aspect_score),
            "raw_score": float(raw_aspect_score),
            "aspect_prior": float(prior_score),
            "aspect_terms": prior_terms,
            "aspect_model_prob": aspect_prob,
            "aspect_model_score": float(aspect_model_score),
            "opinions": best["opinions"],
        }
    return aspect_candidates


def score_sentence_candidates(
    sent: str,
    q_vec: np.ndarray,
    q_trig_tok: torch.Tensor,
    q_trig_mask: torch.Tensor,
    q_op_tok: torch.Tensor,
    q_op_mask: torch.Tensor,
    aspect_probs: Optional[Dict[str, float]],
    resources: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    spans = propose_spans(sent)
    if len(spans) == 1:
        sent_fuzzy = build_fuzzy_repr(sent)
        return _score_single_query_candidates(
            sent=sent,
            sent_fuzzy=sent_fuzzy,
            q_vec=q_vec,
            q_trig_tok=q_trig_tok,
            q_trig_mask=q_trig_mask,
            q_op_tok=q_op_tok,
            q_op_mask=q_op_mask,
            aspect_probs=aspect_probs,
            resources=resources,
        )

    span_queries: List[Tuple[str, Tuple[str, set, str], np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]] = []

    span_pool = [span for span in spans if span != sent] or [sent]
    if span_pool:
        span_vecs = embed_texts(
            span_pool,
            resources["sent_tok"],
            resources["sent_mdl"],
            resources["device"],
            batch_size=min(len(span_pool), SENT_BATCH_SIZE),
            max_length=SENT_MAX_LENGTH,
            pooling=SENT_POOLING,
        )
        span_q_trig_texts = [format_relation_query(span, "st") for span in span_pool]
        span_q_op_texts = [format_relation_query(span, "so") for span in span_pool]
        span_q_trig_tok, span_q_trig_mask = colbert_encode_texts(
            span_q_trig_texts,
            resources["col_tok"],
            resources["col_mdl"],
            resources["device"],
            resources["q_maxlen"],
            batch_size=min(len(span_pool), 16),
            return_mask=True,
            side="q",
        )
        span_q_op_tok, span_q_op_mask = colbert_encode_texts(
            span_q_op_texts,
            resources["col_tok"],
            resources["col_mdl"],
            resources["device"],
            resources["q_maxlen"],
            batch_size=min(len(span_pool), 16),
            return_mask=True,
            side="q",
        )
        span_aspect_probs = colbert_aspect_prob_dicts(span_q_trig_tok, span_q_trig_mask, resources)
        for i, span in enumerate(span_pool):
            span_queries.append(
                (
                    span,
                    build_fuzzy_repr(span),
                    span_vecs[i:i + 1],
                    span_q_trig_tok[i],
                    span_q_trig_mask[i],
                    span_q_op_tok[i],
                    span_q_op_mask[i],
                    span_aspect_probs[i],
                )
            )

    by_aspect: Dict[str, List[Dict[str, Any]]] = {}
    for span_text, span_fuzzy, span_vec, span_trig_tok, span_trig_mask, span_op_tok, span_op_mask, span_probs in span_queries:
        span_candidates = _score_single_query_candidates(
            sent=span_text,
            sent_fuzzy=span_fuzzy,
            q_vec=span_vec,
            q_trig_tok=span_trig_tok,
            q_trig_mask=span_trig_mask,
            q_op_tok=span_op_tok,
            q_op_mask=span_op_mask,
            aspect_probs=span_probs,
            resources=resources,
        )
        for asp, info in span_candidates.items():
            by_aspect.setdefault(asp, []).append(
                {
                    "trigger": info["trigger"],
                    "score": float(info["score"]),
                    "raw_score": float(info.get("raw_score", info["score"])),
                    "aspect_prior": float(info.get("aspect_prior", 0.0)),
                    "aspect_terms": info.get("aspect_terms", []),
                    "aspect_model_prob": float(info.get("aspect_model_prob", 0.0)),
                    "aspect_model_score": float(info.get("aspect_model_score", 0.0)),
                    "opinions": info["opinions"],
                    "_span": span_text,
                }
            )

    merged: Dict[str, Dict[str, Any]] = {}
    for asp, infos in by_aspect.items():
        infos = sorted(infos, key=lambda x: float(x["score"]), reverse=True)
        best = infos[0]
        top_scores = [float(x["score"]) for x in infos[:2]]
        span_weights = [0.70, 0.30][: len(top_scores)]
        denom = sum(span_weights) if span_weights else 1.0
        span_agg_score = sum(w * s for w, s in zip(span_weights, top_scores)) / denom
        merged[asp] = {
            "trigger": best["trigger"],
            "score": float(span_agg_score),
            "raw_score": float(best.get("raw_score", best["score"])),
            "aspect_prior": float(best.get("aspect_prior", 0.0)),
            "aspect_terms": best.get("aspect_terms", []),
            "aspect_model_prob": float(best.get("aspect_model_prob", 0.0)),
            "aspect_model_score": float(best.get("aspect_model_score", 0.0)),
            "opinions": best["opinions"],
            "_span": best["_span"],
        }
    return merged


def score_dataframe(
    df: pd.DataFrame,
    resources: Dict[str, Any],
    score_threshold: float = SCORE_THRESHOLD,
    emit_matches: bool = False,
):
    sentences, idx_map = extract_sentences_from_df(df)
    print(f"[Sentences] total {len(sentences)} sub-sentences")
    sent_vecs = embed_texts(
        sentences,
        resources["sent_tok"],
        resources["sent_mdl"],
        resources["device"],
        SENT_BATCH_SIZE,
        SENT_MAX_LENGTH,
        SENT_POOLING,
    )

    row_best: Dict[int, Dict[str, Dict[str, Any]]] = {}
    out_rows: List[Dict[str, Any]] = []
    col_tok = resources["col_tok"]
    col_mdl = resources["col_mdl"]
    q_maxlen = resources["q_maxlen"]
    device = resources["device"]

    for start in tqdm(range(0, len(sentences), QUERY_BATCH_SIZE), desc="[Score]"):
        end = min(start + QUERY_BATCH_SIZE, len(sentences))
        batch_sents = sentences[start:end]
        batch_vecs = sent_vecs[start:end]

        q_trig_texts = [format_relation_query(s, "st") for s in batch_sents]
        q_op_texts = [format_relation_query(s, "so") for s in batch_sents]
        q_trig_tok_batch, q_trig_mask_batch = colbert_encode_texts(
            q_trig_texts, col_tok, col_mdl, device, q_maxlen, batch_size=32, return_mask=True, side="q"
        )
        q_op_tok_batch, q_op_mask_batch = colbert_encode_texts(
            q_op_texts, col_tok, col_mdl, device, q_maxlen, batch_size=32, return_mask=True, side="q"
        )
        aspect_probs_batch = colbert_aspect_prob_dicts(q_trig_tok_batch, q_trig_mask_batch, resources)

        for off, sent in enumerate(batch_sents):
            qi = start + off
            row_idx, sub_idx = idx_map[qi]

            if is_noise_sentence(sent):
                if emit_matches:
                    out_rows.append(
                        {
                            "row": int(row_idx),
                            "sub": int(sub_idx),
                            "sentence": sent,
                            "matches": {asp: "" for asp in resources["aspect_order"]},
                        }
                    )
                continue

            candidates = score_sentence_candidates(
                sent=sent,
                q_vec=batch_vecs[off:off + 1],
                q_trig_tok=q_trig_tok_batch[off],
                q_trig_mask=q_trig_mask_batch[off],
                q_op_tok=q_op_tok_batch[off],
                q_op_mask=q_op_mask_batch[off],
                aspect_probs=aspect_probs_batch[off],
                resources=resources,
            )

            row_best.setdefault(int(row_idx), {})
            for asp, info in candidates.items():
                prev = row_best[int(row_idx)].get(asp)
                if prev is None or float(info["score"]) > float(prev["score"]):
                    row_best[int(row_idx)][asp] = {
                        "score": float(info["score"]),
                    }

            if emit_matches:
                matches: Dict[str, Any] = {}
                for asp in resources["aspect_order"]:
                    info = candidates.get(asp)
                    if info is None:
                        matches[asp] = ""
                        continue
                    if float(info["score"]) >= score_threshold:
                        matches[asp] = {
                            "trigger": info["trigger"],
                            "score": float(info["score"]),
                            "opinions": info["opinions"],
                        }
                    else:
                        matches[asp] = ""
                out_rows.append(
                    {
                        "row": int(row_idx),
                        "sub": int(sub_idx),
                        "sentence": sent,
                        "matches": matches,
                    }
                )

    return row_best, out_rows


def main():
    resources = build_resources()

    print(f"[Load] input CSV from {INPUT_CSV}")
    input_df = read_csv(INPUT_CSV)
    _, out_rows = score_dataframe(
        input_df,
        resources,
        score_threshold=SCORE_THRESHOLD,
        emit_matches=True,
    )

    output_dir = os.path.dirname(OUTPUT_JSONL)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[Done] wrote {len(out_rows)} rows -> {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
