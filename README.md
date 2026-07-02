# Legal NER — Vietnamese Judgment Entity Extraction (inference bundle)

Named-entity recognition for **Vietnamese court judgments**. A fine-tuned
`xlm-roberta-base` model tags **31 entity types** (parties, court, case number,
articles/clauses, penalties, money amounts, sentencing factors, …) over the
text of a judgment PDF.

- **Model**: `xlm-roberta-base` fine-tuned on Vietnamese judgments, 31-label
  BIO scheme (63 tags). Self-eval micro-F1 ≈ **0.97**.
- **Input**: a judgment **PDF with a text layer** (or a plain-text file).
- **Output**: a list of entities, each `{type, text, start, end, score}` (char
  offsets into the normalized text; `score` = mean per-token softmax confidence
  of the span, handy for thresholding), plus document-level `case_meta`
  (`case_number`, `case_type`, `procedure_stage`).
- **Interfaces**: a CLI (`python -m training.infer …`) and a small HTTP API
  (`/extract`, `/health`, async `/jobs/extract`).

> **Scope:** this is an **NER-only** bundle. The verification layers from the
> larger project (law-DB citation check, PDF forgery analysis, publication-portal
> lookup) are **not** included.

---

## Requirements

- **Python 3.10+**
- **Git LFS** — the 1.1 GB model weights (`model.safetensors`) are stored via
  Git LFS. You must `git lfs install` *before* cloning, or the file will be a
  small text pointer and the model will fail to load.
- **PyTorch** — installs CUDA-enabled on Linux by default; the bundle also runs
  on **CPU** (slower). For a specific CPU/CUDA build, install `torch` from
  <https://pytorch.org> first, then `pip install -r requirements.txt`.
- **No Java required** for the default extractor (`pymupdf`).

### Clone (with the model)

```bash
git lfs install
git clone https://github.com/HoangNhatTR/Legal-NER.git
cd Legal-NER
# verify the weights actually downloaded (should be ~1.1 GB, not a few KB):
ls -lh model/legal-ner-v3r-full/final/model.safetensors
```

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run — CLI

```bash
python -m training.infer \
  --model model/legal-ner-v3r-full/final \
  --pdf path/to/judgment.pdf
```

- `--pdf FILE` — judgment PDF (must have a text layer).
- `--text FILE` — plain-text judgment instead of a PDF.
- `--out FILE` — write JSON to a file instead of stdout.

Example output (truncated):

```json
{
  "source": "judgment.pdf",
  "entities": [
    { "label": "COURT",         "start": 0,   "end": 15,  "text": "TÒA ÁN NHÂN DÂN" },
    { "label": "CASE_NUMBER",   "start": 119, "end": 132, "text": "17/2018/HS-ST" },
    { "label": "JUDGMENT_DATE", "start": 133, "end": 148, "text": "Ngày 26-01-2018" },
    { "label": "DEFENDANT",     "start": 891, "end": 904, "text": "Nguyễn Vĩnh H" }
  ]
}
```

---

## Run — HTTP API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8100
```

The model is loaded **once** at startup (lifespan). Endpoints:

| Method | Path                 | Purpose                                        |
| ------ | -------------------- | ---------------------------------------------- |
| GET    | `/health`            | model-loaded status + device                   |
| POST   | `/extract`           | synchronous PDF → entities                     |
| POST   | `/jobs/extract`      | async submit → `job_id` (202)                  |
| GET    | `/jobs/{job_id}`     | poll an async job                              |
| GET    | `/jobs`              | list recent jobs                               |
| GET    | `/docs`              | interactive OpenAPI docs                       |

```bash
curl -F 'file=@judgment.pdf' 'http://localhost:8100/extract'
```

`/extract` returns:

```json
{
  "filename": "judgment.pdf",
  "case_meta": { "case_number": "17/2018/HS-ST", "case_type": "hình sự", "procedure_stage": "sơ thẩm" },
  "entities": [ { "type": "DEFENDANT", "text": "Nguyễn Vĩnh H", "start": 891, "end": 904, "score": 0.99 } ],
  "entities_grouped": { "DEFENDANT": [ ... ], "ARTICLE": [ ... ] },
  "num_pages": 5,
  "char_count": 12345,
  "warnings": [],
  "ocr": null,
  "extractor": "pymupdf"
}
```

### Extractor backends

- **`pymupdf`** (default) — native PDF text layer via PyMuPDF. Pure-Python,
  **no Java**, runs anywhere. Recommended.
- **`opendataloader`** (optional) — structured extraction via `opendataloader-pdf`.
  Requires a **JDK 11+** (`LEGAL_NER_ODL_JAVA_HOME=/path/to/jdk`) and, for
  scans, an external hybrid-OCR server. Select per request with
  `?extractor=opendataloader`. Not needed for normal text-layer PDFs.

> **Scanned PDFs (no text layer):** OCR is **not bundled**. The default
> `pymupdf` path will return HTTP 422 for a scan (`allow_ocr=false` gives a
> clear message). Use a text-layer PDF, or wire up your own OCR.

---

## The 31 entity labels

```
CASE_NUMBER  COURT  JUDGMENT_DATE  CASE_TYPE
DEFENDANT  PLAINTIFF  VICTIM  RELATED_PARTY
LAW_NAME  ARTICLE  CLAUSE  POINT  LEGAL_BASIS
CRIME
PENALTY  MONEY_AMOUNT  COMPENSATION  COURT_FEE  DECISION
JUDGE  ASSESSOR  PROSECUTOR  CLERK  LAWYER  WITNESS
COURT_BEHAVIOR
MITIGATING_FACTOR  AGGRAVATING_FACTOR
CRIMINAL_ACT
QUANTITY  EVIDENCE_ITEM
```

---

## Quickstart (Tiếng Việt)

Bộ này trích xuất **thực thể từ bản án Việt Nam** (mô hình NER 31 nhãn,
`xlm-roberta-base`, micro-F1 ≈ 0.97). Đây là bản **chỉ NER** — không kèm các lớp
kiểm chứng (đối chiếu luật, phát hiện chỉnh sửa PDF, tra cứu cổng công bố).

1. Cần **Git LFS** trước khi clone (file model ~1.1 GB):
   ```bash
   git lfs install
   git clone https://github.com/HoangNhatTR/Legal-NER.git && cd Legal-NER
   ```
2. Cài đặt (Python 3.10+, chạy được trên CPU hoặc GPU):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Chạy CLI:
   ```bash
   python -m training.infer --model model/legal-ner-v3r-full/final --pdf ban_an.pdf
   ```
4. Chạy API:
   ```bash
   uvicorn api.main:app --port 8100
   curl -F 'file=@ban_an.pdf' localhost:8100/extract
   ```

> PDF phải có **lớp văn bản** (text layer). Bản scan cần OCR — không có sẵn trong
> bộ này.

---

## Project layout

```
Legal-NER/
├── config.py                      # label inventory + inference window settings
├── corpus/                        # PDF text extraction + normalization + sectioning
├── labeling/                      # tokenizer (weak_label) + regex patterns (case-meta)
├── training/infer.py              # the inference engine (CLI entry point)
├── api/                           # FastAPI app: /extract, /health, /jobs/extract
└── model/legal-ner-v3r-full/final # the production checkpoint (safetensors via LFS)
```
