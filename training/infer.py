"""Inference: judgment PDF or text file -> extracted entities JSON.

CLI:
    python -m training.infer --model data/models/legal-ner/final --pdf "/path/x.pdf"
    python -m training.infer --model ... --text data/text/100123.txt --out result.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    INFER_SECTION_AWARE,
    INFER_STRIDE,
    INFER_WINDOW,
    MAX_SEQ_LENGTH,
    MODELS_DIR,
)
from corpus.extract import extract_pdf_text  # noqa: E402
from corpus.normalize import flatten_for_matching, normalize_text  # noqa: E402
from corpus.sectioner import segment_sections  # noqa: E402
from labeling.weak_label import tokenize_with_offsets  # noqa: E402

# Back-compat aliases (some callers / tests import WINDOW/STRIDE by name).
WINDOW = INFER_WINDOW
STRIDE = INFER_STRIDE

# Sentence terminators used to break a long section at a natural boundary
# instead of cutting mid-token. A break is taken AFTER one of these tokens:
#   "."  ";"  ":"  — clause / sentence terminators in judgment prose.
# (Number-internal '.' like "200.000" is a single syllable run under the
# weak-label tokenizer, so it is never a standalone "." token here.)
_SENT_BREAK_TOKENS = frozenset({".", ";", ":", "!", "?"})


def _run_window(model, tokenizer, words: list[str], device):
    """One forward pass over ``words``; return (pred_ids, word_ids)."""
    enc = tokenizer(words, is_split_into_words=True, truncation=True,
                    max_length=MAX_SEQ_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    return logits.argmax(dim=-1).tolist(), enc.word_ids(0)


def _sentence_chunks(syllables: list[str], lo: int, hi: int) -> list[tuple[int, int]]:
    """Sub-chunk ``syllables[lo:hi]`` at sentence boundaries near INFER_WINDOW.

    Returns absolute ``(start, end)`` ranges. Each chunk targets ~INFER_WINDOW
    syllables but is EXTENDED to the next sentence terminator (so entities,
    which rarely cross sentences, stay whole) up to a hard cap, then the next
    chunk OVERLAPS by (WINDOW - STRIDE) to protect any entity sitting on the
    chosen cut. A section short enough for one window yields a single chunk.
    """
    if hi - lo <= INFER_WINDOW:
        return [(lo, hi)]
    # Hard cap so a terminator-less run can't grow past the subword budget.
    hard_cap = INFER_WINDOW + (INFER_WINDOW - INFER_STRIDE)
    chunks: list[tuple[int, int]] = []
    start = lo
    while start < hi:
        target = min(start + INFER_WINDOW, hi)
        end = target
        # Extend to the next sentence terminator, but not past the hard cap.
        limit = min(start + hard_cap, hi)
        while end < limit and syllables[end - 1] not in _SENT_BREAK_TOKENS:
            end += 1
        if end >= hi:  # last chunk
            chunks.append((start, hi))
            break
        chunks.append((start, end))
        # Overlap the next chunk back by the configured margin, but never
        # rewind to/behind the current chunk start (guarantees progress).
        overlap = INFER_WINDOW - INFER_STRIDE
        start = max(start + 1, end - overlap)
    return chunks


def _predict_range(model, tokenizer, syllables: list[str], lo: int, hi: int,
                   tags: list[str], decided: list[bool], device) -> int:
    """Predict tags for ``syllables[lo:hi]`` as an independent sequence.

    Windows are confined to ``[lo, hi)`` (never cross a section boundary) and
    cut at sentence boundaries. Writes into ``tags``/``decided`` in place with
    "first-decided-wins" at overlaps. Returns the number of forward passes.
    """
    passes = 0
    for cs, ce in _sentence_chunks(syllables, lo, hi):
        pred_ids, word_ids = _run_window(model, tokenizer, syllables[cs:ce], device)
        passes += 1
        seen = set()
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            gi = cs + wid
            if not decided[gi]:
                tags[gi] = model.config.id2label[pred_ids[pos]]
                decided[gi] = True
    return passes


def predict_tags(model, tokenizer, syllables: list[str], device,
                 *, section_aware: bool = INFER_SECTION_AWARE,
                 sections: list[tuple[int, int]] | None = None) -> list[str]:
    """Predict a BIO tag for every syllable.

    section_aware (default): each major judgment section is decoded as an
    INDEPENDENT sequence so a window never bleeds context across a section
    boundary; sections longer than the window are sub-chunked at sentence
    boundaries. ``sections`` is an optional list of ``(lo, hi)`` SYLLABLE-index
    ranges covering ``[0, len(syllables))``; when omitted the whole stream is
    treated as one section (caller normally passes section ranges derived from
    char offsets — see ``infer_entities``).

    section_aware=False: legacy blind sliding window (WINDOW / STRIDE), kept
    for before/after comparison.
    """
    tags = ["O"] * len(syllables)

    if not section_aware:
        decided = [False] * len(syllables)
        start = 0
        while start < len(syllables):
            window = syllables[start: start + INFER_WINDOW]
            pred_ids, word_ids = _run_window(model, tokenizer, window, device)
            seen = set()
            for pos, wid in enumerate(word_ids):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                gi = start + wid
                if not decided[gi]:
                    tags[gi] = model.config.id2label[pred_ids[pos]]
                    decided[gi] = True
            if start + INFER_WINDOW >= len(syllables):
                break
            start += INFER_STRIDE
        return tags

    decided = [False] * len(syllables)
    if not sections:
        sections = [(0, len(syllables))]
    for lo, hi in sections:
        _predict_range(model, tokenizer, syllables, lo, hi, tags, decided, device)
    return tags


def tags_to_entities(tokens: list[tuple[str, int, int]], tags: list[str],
                     text: str) -> list[dict]:
    entities = []
    current = None  # (label, start_tok, end_tok)
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current:
                entities.append(current)
            current = [tag[2:], i, i]
        elif tag.startswith("I-") and current and tag[2:] == current[0]:
            current[2] = i
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)
    return [
        {
            "label": label,
            "start": tokens[s][1],
            "end": tokens[e][2],
            "text": text[tokens[s][1]: tokens[e][2]],
        }
        for label, s, e in entities
    ]


def load_model(model_path: str, device: str):
    """Load tokenizer + token-classification model onto ``device`` (eval mode).

    Returned ``(tokenizer, model)`` are reusable across many inference calls;
    callers (e.g. the API singleton) load once and reuse.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device).eval()
    return tokenizer, model


def _section_token_ranges(tokens: list[tuple[str, int, int]],
                          text: str) -> list[tuple[int, int]]:
    """Map char-offset section boundaries to SYLLABLE-index ``(lo, hi)`` ranges.

    Each token is assigned to the section containing its START char offset, so
    the resulting ranges are contiguous, gap-free and cover all tokens.
    """
    sections = segment_sections(text)
    ranges: list[tuple[int, int]] = []
    ti = 0
    n = len(tokens)
    for _name, _s, e in sections:
        lo = ti
        while ti < n and tokens[ti][1] < e:
            ti += 1
        if ti > lo:
            ranges.append((lo, ti))
    if ti < n:                       # any trailing tokens -> last range
        if ranges:
            ranges[-1] = (ranges[-1][0], n)
        else:
            ranges.append((0, n))
    return ranges or [(0, n)]


def infer_entities(text: str, model, tokenizer, device,
                   *, section_aware: bool = INFER_SECTION_AWARE) -> list[dict]:
    """Run NER over already-flattened text and return entity dicts.

    ``text`` must be the flattened-for-matching stream so that the returned
    ``start``/``end`` offsets index into it. Each entity:
    ``{"label", "start", "end", "text"}``.

    When ``section_aware`` (default, from ``config.INFER_SECTION_AWARE``), the
    document is split into major judgment sections and each is decoded as an
    independent sequence so windows never bleed context across section
    boundaries and long-span entities aren't cut at blind window edges. Pass
    ``section_aware=False`` for the legacy sliding window.

    Signature/output contract is unchanged for positional callers (the API
    passes positionally), so no API change is required.
    """
    tokens = tokenize_with_offsets(text)
    syllables = [t for t, _, _ in tokens]
    sections = _section_token_ranges(tokens, text) if section_aware else None
    tags = predict_tags(model, tokenizer, syllables, device,
                        section_aware=section_aware, sections=sections)
    return tags_to_entities(tokens, tags, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract legal entities from a judgment")
    parser.add_argument("--model", default=str(MODELS_DIR / "legal-ner" / "final"),
                        help="fine-tuned checkpoint dir")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", help="judgment PDF (text layer required)")
    source.add_argument("--text", help="plain-text judgment file")
    parser.add_argument("--out", default=None, help="write JSON here (default: stdout)")
    args = parser.parse_args()

    if args.pdf:
        raw = extract_pdf_text(Path(args.pdf))
        if raw is None:
            sys.exit("ERROR: PDF has no text layer (scan?) — not supported.")
        text = flatten_for_matching(normalize_text(raw))
    else:
        text = flatten_for_matching(Path(args.text).read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model, device)

    entities = infer_entities(text, model, tokenizer, device)

    result = json.dumps({"source": args.pdf or args.text, "entities": entities},
                        ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(result, encoding="utf-8")
        print(f"Wrote {len(entities)} entities to {args.out}")
    else:
        print(result)


if __name__ == "__main__":
    main()
