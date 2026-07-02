# Independent GOLD evaluation set — Vietnamese legal NER

> **STATUS: model-annotated DRAFT GOLD — pending human review.**
> Spans were chosen by carefully READING each judgment, INDEPENDENTLY of the
> regex weak-labeler (the regex was looked at only AFTER, to compute agreement,
> never as the source of truth). No network LLM / vLLM was used. This is a
> strong independent reference, but the user may correct it.

## Why this exists
Until now the model was trained AND evaluated on the SAME regex labels
(self-referential): seqeval F1 measured *label-learnability*, not *correctness*.
This gold set, annotated by reading the text, breaks that loop and lets us ask
whether the tighter "v2" regex is actually MORE CORRECT than v1 for the three
weak entities (DECISION, LEGAL_BASIS, VIOLATION_ACT).

## Offset convention (must match the model/labeler EXACTLY)
`gold/texts/<id>.txt` is the output of `corpus.normalize.flatten_for_matching`
applied to the raw judgment — the SAME stream the labeler and `training/infer`
index into. All `start`/`end` in `gold.jsonl` are char offsets into that file,
so they line up with both regex output and model inference. Verified: preserved
criminal `weak_spans*.jsonl` offsets reproduce against these texts with 0
mismatch.

## Structure
```
gold/
  texts/            30 flattened judgment texts (stable reference)
  gold.jsonl        one record/doc: {id, source, text_file, spans:[{start,end,label,text}]}
  annotate.py       substring -> verified-offset helper (independent annotation)
  build_gold.py     the actual human annotations (substring + occurrence) -> gold.jsonl
  selection.json    the 30 chosen ids + type
  README.md         this file
```

## Entity definitions used
- **DECISION**: each INDIVIDUAL operative ruling in the QUYẾT ĐỊNH section
  (one "Tuyên bố/Xử phạt/Buộc/Tịch thu/Trả lại/Không chấp nhận/Chấp nhận/
  Đình chỉ/Hủy/Giữ nguyên/Công nhận…" = one span), bounded to that ruling
  clause; trailing time-credit/procedural tails ("Thời hạn tù tính từ…") are
  EXCLUDED.
- **LEGAL_BASIS**: each "Căn cứ …"/"Áp dụng …" statutory-citation clause,
  bounded to the citation only (stops before the operative ruling that
  follows). "Áp dụng biện pháp …" is a DECISION, not LEGAL_BASIS.
- **VIOLATION_ACT**: the concrete violating-act verb-phrase, tightly bounded
  (excludes procedural "hành vi, quyết định tố tụng …" meta-mentions).
- Anchors also annotated: **CASE_NUMBER** (header number only),
  **CRIME** (the charged offense name).

## The 30 documents
18 criminal (`data/text/`) + 12 civil (`data/text_civil/`); mixed lengths and
case types. `flat_len` = chars in the flattened text.

| id | source | case type | flat_len |
|----|--------|-----------|---------:|
| 106588 | criminal | ma túy (vận chuyển) | 1230 |
| 102375 | criminal | ma túy (mua bán) | 10890 |
| 104263 | criminal | ma túy (tàng trữ) | 13644 |
| 104071 | criminal | ma túy (mua bán/tàng trữ) | 24696 |
| 102835 | criminal | trộm cắp (nhiều bị cáo) | 54886 |
| 104476 | criminal | trộm cắp | 1169 |
| 101464 | criminal | trộm cắp | 11932 |
| 101260 | criminal | trộm cắp | 59891 |
| 100203 | criminal | cố ý gây thương tích | 1304 |
| 101476 | criminal | giết người / cố ý gây thương tích | 35353 |
| 102186 | criminal | giao thông | 7021 |
| 104821 | criminal | tham ô / giao thông | 52061 |
| 102641 | criminal | đánh bạc | 1283 |
| 100640 | criminal | tổ chức đánh bạc / đánh bạc | 65238 |
| 104295 | criminal | cướp tài sản | 1103 |
| 101809 | criminal | cưỡng đoạt tài sản | 27398 |
| 100201 | criminal | lừa đảo | 1190 |
| 103640 | criminal | lừa đảo (lớn) | 98709 |
| civil_100585 | civil | ly hôn (đình chỉ) | short |
| civil_100618 | civil | ly hôn (thuận tình) | ~2.3k |
| civil_100645 | civil | ly hôn (thuận tình) | ~2.8k |
| civil_100883 | civil | ly hôn (thuận tình) | ~3.5k |
| civil_100556 | civil | ly hôn + chia tài sản (phúc thẩm) | 32283 |
| civil_100221 | civil | tranh chấp HNGĐ (đình chỉ) | short |
| civil_100070 | civil | tranh chấp đất (phúc thẩm) | ~5k |
| civil_100043 | civil | tranh chấp đất / thừa kế | 40993 |
| civil_101005 | civil | hợp đồng tín dụng | ~2.9k |
| civil_100062 | civil | hợp đồng / hành chính (phúc thẩm) | 15595 |
| civil_100052 | civil | hành chính (đình chỉ) | short |
| civil_100586 | civil | thừa kế / hành chính (đình chỉ) | ~2k |

## Spans annotated (158 total)
| label | count |
|-------|------:|
| DECISION | 73 |
| LEGAL_BASIS | 45 |
| VIOLATION_ACT | 8 |
| CASE_NUMBER | 18 |
| CRIME | 14 |

VIOLATION_ACT is intentionally sparse: most of the 30 docs are appeal-
discontinuation / family-consent / procedural decisions whose body has no
concrete "có hành vi …" act description. This faithfully reflects the corpus.

## How to score
```bash
python -m training.eval_gold --gold gold/gold.jsonl --pred regex                      # v1 (live patterns.py)
python -m training.eval_gold --gold gold/gold.jsonl --pred regex_v2  --criminal-only  # v2 regex (preserved)
python -m training.eval_gold --gold gold/gold.jsonl --pred model --model data/models/legal-ner/final
```
Both EXACT-match and OVERLAP-match (any char overlap, greedy 1-1) F1 are
printed per entity. Scoring is restricted to the label set present in gold
(so predictors are not penalized for entities we did not annotate).

## CAVEATS
- Draft gold pending human review; 30 docs / 158 spans is a small set —
  per-entity numbers (esp. VIOLATION_ACT, n=8) are indicative, not definitive.
- The **v2 regex** output (`data/labeled/weak_spans_v2.jsonl`) was generated for
  CRIMINAL docs only. v2-regex numbers are therefore reported on the criminal
  subset only (`--criminal-only`); the current `patterns.py` == v1 (verified by
  exact reproduction of `weak_spans.jsonl`). Civil weak labels in
  `weak_spans_combined.jsonl` also use v1 (verified).
- CASE_NUMBER / CRIME P is low because the regex/model emit ALL such mentions in
  the doc (dockets, cross-references) whereas gold annotates only the header /
  charged-offense anchor. Their RECALL is the meaningful signal there.
