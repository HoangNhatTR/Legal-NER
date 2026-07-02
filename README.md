# Legal NER — Vietnamese Court Judgments

Pipeline that crawls judgments from `congbobanan.toaan.gov.vn`, extracts and
normalizes the text, weak-labels 20 entity types with regexes, and fine-tunes
a token-classification model.

**Model choice:** `xlm-roberta-base` — works directly on raw syllables (no
Vietnamese word segmentation needed, unlike PhoBERT which expects
VnCoreNLP-segmented input), with full fast-tokenizer offset support.

## Entities (20 types, BIO scheme = 41 labels)

| Group | Label | Meaning | Example |
|---|---|---|---|
| A | `CASE_NUMBER` | Judgment/decision number (number token only) | 17/2018/HS-ST |
| A | `COURT` | Court name | Tòa án nhân dân thị xã Dĩ An, tỉnh Bình Dương |
| A | `JUDGMENT_DATE` | Header / hearing / pronouncement date | Ngày 26-01-2018 |
| A | `CASE_TYPE` | Explicit case-type mention | "vụ án **hình sự** sơ thẩm" → hình sự |
| B | `DEFENDANT` | Name after "bị cáo / các bị cáo" | Nguyễn Văn T |
| B | `PLAINTIFF` | Name after "nguyên đơn" | Nguyễn Thị H |
| B | `VICTIM` | Name after "(người) bị hại" | Huỳnh Thị Kim T |
| B | `RELATED_PARTY` | Name after "người có quyền lợi, nghĩa vụ liên quan" | Lê Tấn V |
| C | `LAW_NAME` | Statute name | Bộ luật Hình sự năm 2015 (sửa đổi, bổ sung năm 2017) |
| C | `ARTICLE` | Article | Điều 250 |
| C | `CLAUSE` | Clause | khoản 1 |
| C | `POINT` | Point | điểm c |
| C | `LEGAL_BASIS` | Full "Căn cứ ... / Áp dụng ..." citation sentence | Căn cứ điểm c khoản 1 Điều 250 ... |
| D | `CRIME` | Offense name (quoted or unquoted) | Vận chuyển trái phép chất ma túy |
| D | `VIOLATION_ACT` | Act phrase after "có hành vi ..." (best-effort, fuzzy) | bán trái phép 01 gói chất bột màu trắng |
| E | `PENALTY` | Sentence terms | 02 (hai) năm tù, án treo, tù chung thân |
| E | `MONEY_AMOUNT` | Money outside fee/compensation context | 5.000.000 đồng |
| E | `COMPENSATION` | Money in "bồi thường" context | bồi thường ... **5.000.000 đồng** |
| E | `COURT_FEE` | Money in "án phí / lệ phí" context | án phí hình sự sơ thẩm **200.000 đồng** |
| E | `DECISION` | Verdict-sentence remainder in QUYẾT ĐỊNH section | Xử phạt bị cáo ... |

Overlap policy (flat BIO, one label per token — see `labeling/patterns.py`):
`LEGAL_BASIS` swallows nested `LAW_NAME`/`ARTICLE`/`CLAUSE`/`POINT`;
`DECISION` is **lowest** priority — fine-grained entities win inside verdict
sentences and `DECISION` gap-fills only the remaining tokens;
`COMPENSATION`/`COURT_FEE` beat `MONEY_AMOUNT` via a context window around
the amount; remaining ties: priority rank → longer match → left-to-right.

Document-level metadata (not token labels): `case_type` and
`procedure_stage` are derived from the `CASE_NUMBER` suffix
(HS/DS/HC/HNGĐ/KDTM/LĐ + ST/PT) and written per document into
`data/labeled/weak_spans.jsonl` (`{"doc", "meta", "spans"}` records).

## Setup

```bash
source /home/tts/AI/AIHoang/HoangEnv/bin/activate   # or use its python directly
pip install -r requirements.txt
cd /home/tts/AI/AIHoang/Legal/legal_ner
```

## Pipeline (run each step with `--help` for options)

```bash
# 1. Crawl judgments (resumable; state in data/raw/crawl_state.json)
python -m crawler.crawl --target-count 1000 --start-id 100000 --criminal-only

# 2. Extract + normalize text (skips scanned PDFs automatically)
python -m corpus.extract --in data/raw --out data/text

# 3. Weak-label into BIO JSONL
python -m labeling.weak_label --in data/text --out data/labeled

# 4. Train
python -m training.train --data data/labeled/weak_labels.jsonl --epochs 5

# 5. Inference on a new judgment
python -m training.infer --model data/models/legal-ner/final --pdf /path/to/judgment.pdf
```

## Implementation notes

* **Legacy-font normalization** (`corpus/normalize.py`): portal PDFs carry
  TCVN3-era artifacts — `ƣ` (U+01A3) -> `ư`, `Ƣ` -> `Ư`, and combining
  diacritics detached by a stray space ("Thi ̣" -> "Thị"); then Unicode NFC.
* **Portal access** (`crawler/portal_client.py`): TLS chain is incomplete,
  so requests use `verify=False`. Detail page `2ta{id}t1cvn/chi-tiet-ban-an`
  exposes a direct `/5ta{id}.../*.pdf` link; `3ta{id}t1cvn` serves PDF bytes
  as fallback. Polite delay >= 1 s between requests.
* **Happy path only**: text-layer PDFs, standard judgment layout. Scanned
  PDFs are detected (< 200 extracted chars) and skipped — no OCR.
* `data/` is git-ignored (raw PDFs, text, labels, checkpoints).
