"""Weak labeling: normalized judgment text -> BIO-tagged JSONL.

Tokens are Vietnamese syllables (whitespace/punctuation split) — no word
segmentation needed for xlm-roberta-base.

CLI:
    python -m labeling.weak_label --in data/text --out data/labeled
    python -m labeling.weak_label --text "/path/to/one.txt" --print-entities
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CHUNK_TOKENS, LABELED_DIR, TEXT_DIR  # noqa: E402
from corpus.normalize import flatten_for_matching  # noqa: E402
from labeling.patterns import Span, derive_doc_meta, find_entities  # noqa: E402

# Syllable tokens: runs of word chars (unicode) or single punctuation marks.
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def spans_to_bio(tokens: list[tuple[str, int, int]], spans: list[Span]) -> list[str]:
    tags = ["O"] * len(tokens)
    for span in spans:
        inside = [
            i for i, (_, ts, te) in enumerate(tokens)
            if ts < span.end and te > span.start  # any overlap
        ]
        if not inside:
            continue
        tags[inside[0]] = f"B-{span.label}"
        for i in inside[1:]:
            tags[i] = f"I-{span.label}"
    return tags


def chunk_indices(tags: list[str], chunk_size: int) -> list[tuple[int, int]]:
    """Split token range into ~chunk_size windows without cutting entities."""
    chunks = []
    start = 0
    n = len(tags)
    while start < n:
        end = min(start + chunk_size, n)
        # never split inside an entity: extend until an O or B- boundary
        while end < n and tags[end].startswith("I-"):
            end += 1
        chunks.append((start, end))
        start = end
    return chunks


def label_document(doc_id: str, text: str) -> tuple[list[dict], list[Span], dict]:
    flat = flatten_for_matching(text)
    spans = find_entities(flat)
    meta = derive_doc_meta(spans)  # case_type/stage from CASE_NUMBER suffix
    tokens = tokenize_with_offsets(flat)
    tags = spans_to_bio(tokens, spans)
    records = []
    for ci, (s, e) in enumerate(chunk_indices(tags, CHUNK_TOKENS)):
        records.append({
            "id": f"{doc_id}#{ci}",
            "tokens": [t for t, _, _ in tokens[s:e]],
            "tags": tags[s:e],
        })
    return records, spans, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Weak-label judgment text into BIO JSONL")
    parser.add_argument("--in", dest="in_dir", default=str(TEXT_DIR),
                        help="directory of normalized .txt files (default: data/text)")
    parser.add_argument("--text", default=None, help="label a single .txt file instead")
    parser.add_argument("--out", default=str(LABELED_DIR), help="output dir (default: data/labeled)")
    parser.add_argument("--print-entities", action="store_true",
                        help="print every matched entity span")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = [Path(args.text)] if args.text else sorted(Path(args.in_dir).glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {args.in_dir}")
        return

    all_records: list[dict] = []
    span_dump: list[dict] = []
    type_counts: Counter = Counter()
    docs_with_entities = 0

    for path in txt_files:
        records, spans, meta = label_document(path.stem, path.read_text(encoding="utf-8"))
        all_records.extend(records)
        if spans:
            docs_with_entities += 1
        for s in spans:
            type_counts[s.label] += 1
        # one record per document: doc-level metadata + all spans
        span_dump.append({
            "doc": path.stem,
            "meta": meta,  # case_number / case_type / procedure_stage
            "spans": [{"label": s.label, "start": s.start, "end": s.end,
                       "text": s.text} for s in spans],
        })
        if args.print_entities:
            print(f"\n=== {path.name} ===  meta={meta}")
            for s in spans:
                print(f"  [{s.label:13s}] {s.text}")

    bio_path = out_dir / "weak_labels.jsonl"
    with bio_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    spans_path = out_dir / "weak_spans.jsonl"
    with spans_path.open("w", encoding="utf-8") as f:
        for row in span_dump:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDocuments labeled : {len(txt_files)} ({docs_with_entities} with >=1 entity)")
    print(f"Chunks written    : {len(all_records)} -> {bio_path}")
    print("Entity counts:")
    for label in sorted(type_counts):
        print(f"  {label:14s} {type_counts[label]}")


if __name__ == "__main__":
    main()
