import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Paths
INPUT_JSONL = Path("beer/chatgpt5_train_auth.json")
OUTPUT_JSONL = Path("beer/kb.jsonl")

# Set to a collection such as {"location", "clean", "service"} to export
# only selected aspects. None means export all aspects.
ASPECT_WHITELIST = None

# True keeps original case. False lowercases text to merge duplicates.
KEEP_CASE = False


def norm(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not KEEP_CASE:
        s = s.lower()
    s = s.replace("\uff0c", ",").replace("\uff08", "(").replace("\uff09", ")")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" '\"\t,.;:")
    return s


def to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [y for y in x if isinstance(y, str)]
    if isinstance(x, str):
        return [x]
    return []


def normalize_polarities(raw):
    POS = {"positive", "pos", "p", "+", "1", "true", "yes"}
    NEG = {"negative", "neg", "n", "-", "-1", "false", "no"}
    NEU = {"neutral", "neu", "0", "none", ""}

    out = set()

    def push_one(v):
        if v is None:
            return
        s = str(v).strip().lower()
        if s in POS:
            out.add("positive")
        elif s in NEG:
            out.add("negative")
        elif s in NEU:
            pass

    if isinstance(raw, list):
        for v in raw:
            push_one(v)
    else:
        push_one(raw)

    return sorted(out)


def iter_input_objects(in_path: Path):
    """Yield dict records from either a JSON file or a JSONL file."""
    text = in_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj
        return

    if isinstance(parsed, list):
        for obj in parsed:
            if isinstance(obj, dict):
                yield obj
    elif isinstance(parsed, dict):
        yield parsed


def is_aspect_map(obj):
    return isinstance(obj, dict) and any(isinstance(v, list) for v in obj.values())


def iter_aspect_maps(obj):
    if not isinstance(obj, dict):
        return

    data = obj.get("data")
    if isinstance(data, dict):
        yield data
        return

    if is_aspect_map(obj):
        yield obj
        return

    for value in obj.values():
        if isinstance(value, dict):
            yield from iter_aspect_maps(value)


def build_aspect_trigger_kb(in_path: Path, out_path: Path):
    # kb[aspect][trigger] = {"positive": set(), "negative": set()}
    kb = defaultdict(lambda: defaultdict(lambda: {"positive": set(), "negative": set()}))
    n_records = 0
    n_aspect_maps = 0
    n_items = 0
    n_skipped_no_opinion = 0
    n_skipped_neutral = 0
    n_skipped_unknown_pol = 0

    for obj in iter_input_objects(in_path):
        n_records += 1

        for data in iter_aspect_maps(obj):
            n_aspect_maps += 1
            # data keys are aspects, values are list[dict(trigger/opinion/polarity)].
            for asp, items in data.items():
                if ASPECT_WHITELIST and asp not in ASPECT_WHITELIST:
                    continue
                if not isinstance(items, list):
                    continue

                asp_key = norm(asp)

                for it in items:
                    if not isinstance(it, dict):
                        continue
                    trig = norm(it.get("trigger", ""))
                    if not trig:
                        continue

                    pol_list = normalize_polarities(it.get("polarity", None))
                    opinions = [norm(x) for x in to_list(it.get("opinion"))]
                    opinions = [x for x in opinions if x]

                    if not opinions:
                        n_skipped_no_opinion += 1
                        continue

                    if not pol_list:
                        raw_pol = it.get("polarity", None)
                        raw_str = str(raw_pol).lower()
                        if "neutral" in raw_str or raw_str.strip() in {"0", "none"}:
                            n_skipped_neutral += 1
                        else:
                            n_skipped_unknown_pol += 1
                        continue

                    for p in pol_list:
                        kb[asp_key][trig][p].update(opinions)
                        n_items += 1

    with out_path.open("w", encoding="utf-8") as fout:
        for asp, trig_map in kb.items():
            out_trig_map = {}
            for trig, buckets in sorted(trig_map.items()):
                out_trig_map[trig] = {
                    "positive": sorted(buckets["positive"]),
                    "negative": sorted(buckets["negative"]),
                }
            record = {asp: out_trig_map}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Read JSON/JSONL records: {n_records}")
    print(f"Found aspect maps: {n_aspect_maps}")
    print(f"Written trigger-opinion entries: {n_items}")
    print(f"Exported aspects: {len(kb)}")
    print(f"Skipped without opinion: {n_skipped_no_opinion}")
    print(f"Skipped neutral: {n_skipped_neutral}")
    print(f"Skipped unknown polarity: {n_skipped_unknown_pol}")
    print(f"Output file: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build an aspect-trigger-opinion KB from JSON/JSONL records.")
    parser.add_argument("--input", type=Path, default=INPUT_JSONL, help="Input JSON or JSONL file.")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSONL, help="Output KB JSONL file.")
    parser.add_argument(
        "--aspect-whitelist",
        type=str,
        default="",
        help="Optional comma-separated aspect whitelist, e.g. location,clean,service.",
    )
    parser.add_argument("--keep-case", action="store_true", help="Keep original text case instead of lowercasing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.aspect_whitelist.strip():
        ASPECT_WHITELIST = {x.strip() for x in args.aspect_whitelist.split(",") if x.strip()}
    KEEP_CASE = bool(args.keep_case)
    build_aspect_trigger_kb(args.input, args.output)
