"""Annotation helper: resolve human-chosen substrings to char offsets into the
flattened gold text and write/append gold.jsonl records.

Annotation is INDEPENDENT of the regex labeler: the spans here come from a
human (model) reading the judgment and choosing the exact substring + which
occurrence. This file only turns those decisions into verified char offsets so
they line up EXACTLY with the flattened stream the model/labeler use.
"""

import json
import sys
from pathlib import Path

GOLD_DIR = Path(__file__).resolve().parent
TEXTS = GOLD_DIR / "texts"


def find_offset(text: str, sub: str, occ: int = 1) -> tuple[int, int]:
    """Return (start, end) of the `occ`-th occurrence (1-based) of `sub`."""
    idx = -1
    for _ in range(occ):
        idx = text.find(sub, idx + 1)
        if idx == -1:
            raise ValueError(f"substring not found (occ={occ}): {sub!r}")
    return idx, idx + len(sub)


def build_record(doc_id: str, source: str, annotations: list) -> dict:
    """annotations: list of (label, substring, occ) tuples (occ optional=1)."""
    fname = f"{doc_id}.txt"
    text = (TEXTS / fname).read_text(encoding="utf-8")
    spans = []
    for ann in annotations:
        if len(ann) == 3:
            label, sub, occ = ann
        else:
            label, sub = ann
            occ = 1
        s, e = find_offset(text, sub, occ)
        assert text[s:e] == sub, f"offset mismatch for {sub!r}"
        spans.append({"start": s, "end": e, "label": label, "text": sub})
    spans.sort(key=lambda x: x["start"])
    return {"id": doc_id, "source": source, "text_file": f"texts/{fname}",
            "spans": spans}


def write_gold(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
