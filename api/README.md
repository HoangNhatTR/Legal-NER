# Legal NER API (v1)

FastAPI service wrapping the trained Model 1 NER. Upload a judgment PDF, get
structured entities + case metadata as JSON. Reuses the existing pipeline
(`corpus.extract`, `corpus.normalize`, `training.infer`, `labeling.patterns`) —
nothing is reimplemented here.

## Run

The default extraction backend is **opendataloader-pdf** (docling/EasyOCR) for
**both** digital and scanned PDFs. This is a **Java** pipeline (needs JDK 11+;
system `java` is 8, too old), so the API must run with `JAVA_HOME` pointing at
the portable JDK. The simplest way is the launcher script:

```bash
/home/tts/AI/AIHoang/Legal/legal_ner/scripts/start_api.sh
# -> exports JAVA_HOME=/home/tts/jdk/jdk-21.0.11+10, then
#    uvicorn api.main:app --host 0.0.0.0 --port 8100
```

Or manually (run from `legal_ner/` so package imports resolve):

```bash
cd /home/tts/AI/AIHoang/Legal/legal_ner
export JAVA_HOME=/home/tts/jdk/jdk-21.0.11+10
export PATH=$JAVA_HOME/bin:$PATH
/home/tts/AI/AIHoang/HoangEnv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8100
```

Port **8100** (8000 is taken by another service). The model is loaded **once**
at startup (FastAPI lifespan) and reused across requests. The lifespan also sets
`JAVA_HOME` in-process (so digital ODL works from a cold start) and tears down
the auto-started OCR server (below) on shutdown.

### Two-process setup (API + OCR server)

opendataloader's **digital** path (text-layer PDFs) needs only Java — the API
process handles it. opendataloader's **scanned** path needs a separate **hybrid
OCR server** (`opendataloader-pdf-hybrid`, docling/EasyOCR) running on
`LEGAL_NER_ODL_HYBRID_PORT` (default **5002**). Two options:

1. **Auto-start (default, dev/single-host):** the API auto-starts the OCR server
   as a managed subprocess on the **first scan request** (CPU OCR via
   `CUDA_VISIBLE_DEVICES=""`, `--force-ocr --ocr-lang vi,en`), polls its
   `/health` until ready (first-run model load ~1-2 min), reuses it across
   requests, and terminates it on API shutdown. Nothing extra to run.
2. **Supervised (recommended for production):** run the OCR server yourself and
   disable auto-start with `LEGAL_NER_ODL_HYBRID_AUTOSTART=0`:

   ```bash
   /home/tts/AI/AIHoang/Legal/legal_ner/scripts/start_ocr_server.sh
   #  exports JAVA_HOME + CUDA_VISIBLE_DEVICES="" and runs
   #  opendataloader-pdf-hybrid --host 127.0.0.1 --port 5002 \
   #     --force-ocr --ocr-lang vi,en --device cpu
   ```

   Run it under **systemd** so it survives restarts and is independent of the
   API lifecycle (see **Production notes** at the bottom).

**Env vars** (shared by both processes; defaults in parentheses):
| Var | Default | Meaning |
|-----|---------|---------|
| `LEGAL_NER_ODL_JAVA_HOME` | `/home/tts/jdk/jdk-21.0.11+10` | JDK 11+ home (system java 8 too old) |
| `LEGAL_NER_ODL_HYBRID_HOST` | `127.0.0.1` | OCR server host |
| `LEGAL_NER_ODL_HYBRID_PORT` | `5002` | OCR server port |
| `LEGAL_NER_ODL_HYBRID_AUTOSTART` | `1` | `1` auto-start OCR server on first scan; `0` require it pre-started (else 422-style error) |
| `LEGAL_NER_ODL_HYBRID_OCR_LANG` | `vi,en` | OCR languages for the auto-started server |
| `LEGAL_NER_ODL_HYBRID_START_TIMEOUT_S` | `180` | how long to wait for first-run model load before failing |

**Latency expectations:**
- Digital (text-layer) PDF via ODL: **<1s** (~0.5–1s).
- Scanned PDF via ODL hybrid OCR: **~2 min** for ~10 pages on CPU, **plus** a
  one-time first-run model load (~1-2 min) on the very first scan after the OCR
  server starts. Use a long client timeout (e.g. `curl -m 600`).

### Fallback: PyMuPDF / VietOCR (`?extractor=pymupdf`)

The previous default (PyMuPDF text layer + Phase-1 **VietOCR** for scans) is
kept as a still-selectable fallback. Pass `?extractor=pymupdf` to either
endpoint. It needs no Java and no OCR server — see **Extraction backends** below.

## Endpoints

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "model_path": ".../data/models/legal-ner-combined/final",
  "odl": {
    "java_ok": true,
    "java_home": "/home/tts/jdk/jdk-21.0.11+10",
    "hybrid_reachable": false,
    "hybrid_url": "http://127.0.0.1:5002",
    "default_extractor": "opendataloader"
  }
}
```
`device` is `cuda` when the GPU load succeeds, else `cpu` (graceful fallback on
CUDA OOM — the GPU is shared and near-full). The `odl` block is **best-effort**
(never fails the health check): `java_ok` confirms the portable JDK is runnable
(required since ODL is the default), and `hybrid_reachable` reports whether the
OCR server is up (needed for scans; `false` is normal until the first scan
auto-starts it).

### `POST /extract` (multipart, field `file`)
```bash
# text-layer PDF — DEFAULT opendataloader digital extract (fast, <1s)
curl -F file=@"../bản án demo/document (1).pdf" localhost:8100/extract

# scanned PDF — DEFAULT opendataloader hybrid OCR (auto-starts OCR server; ~2 min/10pg + first-run model load)
curl -m 600 -F file=@"../bản án demo/document.pdf" "localhost:8100/extract"

# scanned PDF, fast-only: skip OCR, get a 422
curl -F file=@"../bản án demo/document.pdf" "localhost:8100/extract?allow_ocr=false"

# fallback: previous PyMuPDF + VietOCR backend
curl -F file=@"../bản án demo/document (1).pdf" "localhost:8100/extract?extractor=pymupdf"
```

Query params:
- `allow_ocr` (bool, default `true`) — when a scanned PDF (no usable text
  layer) is uploaded, route it through the OCR pipeline. Set `false` for
  fast-only behavior (scanned PDFs then return 422).
- `extractor` (`opendataloader` | `pymupdf`, default **`opendataloader`**) —
  selects the extraction backend (see **Extraction backends** below). The
  default uses opendataloader for both digital and scanned PDFs; pass
  `pymupdf` to use the legacy PyMuPDF + Phase-1 VietOCR path.

Validation / boundaries:
- Not a PDF (no `%PDF` magic bytes and no PDF content-type) → **400**.
- Scanned PDF + `allow_ocr=false` → **422** (fast-only).
- Scanned PDF + OCR failed (broken venv / OOM after CPU fallback / no text) →
  **422** with the reason.
- Empty upload → **400**.

#### Scanned-PDF / OCR behavior
A PDF whose native text layer is below `MIN_TEXT_CHARS` is treated as a scan.
With `allow_ocr=true` (default) it is handed to the Phase-1 OCR pipeline
(`OCR/phase1/`) as a **subprocess** (separate venv + cwd — its heavy deps are
not in HoangEnv and it uses relative paths). The OCR text is then run through
the SAME normalize → infer → group → derive_meta flow as native text.

- The response gains an `ocr` block (engine, device, page_count,
  mean_confidence, per-page summary) and a `warnings` entry:
  `"text extracted via OCR (engine=…, device=…); accuracy lower than a native
  text layer — verify citations"`.
- **Synchronous (v1):** an OCR request can take **several minutes** (model load
  ~20-40s + ~1-2 min/page on this shared GPU). No job queue yet — the HTTP
  request blocks. Use a long client timeout (e.g. `curl -m 1800`).
- **Accuracy caveat:** Vietnamese OCR NED was ~0.72 in earlier evals, so exact
  citation wording (case numbers, article references) from scans may be
  unreliable — always verify against the source image.

Tunables (env vars, read by `api/ocr_adapter.py`):
| Var | Default | Meaning |
|-----|---------|---------|
| `LEGAL_NER_OCR_ENGINE` | `layout` | `paddle` \| `vl` \| `layout`. **`paddle` segfaults in the current OCR venv (PaddleOCR predictor init) — do not use.** `layout` (PP-DocLayoutV3 + VietOCR) is the working default. |
| `LEGAL_NER_OCR_TIMEOUT` | `1800` | subprocess timeout (seconds) |
| `LEGAL_NER_OCR_USE_GPU` | `1` | try GPU first |
| `LEGAL_NER_OCR_CPU_FALLBACK` | `1` | retry on CPU after a GPU OOM/segfault (a GPU *timeout* is not retried) |

## Extraction backends (`extractor=`)

Both `/extract` and `/verify` accept `?extractor=` to choose how the PDF is
turned into text. The `normalize → infer → group → derive_meta` flow afterward
is **identical** for both — only the text source changes.

| `extractor` | Digital (text-layer) PDFs | Scanned PDFs |
|-------------|---------------------------|--------------|
| `opendataloader` *(default)* | [opendataloader-pdf](https://github.com/opendataloader-project/opendataloader-pdf) structured digital extract (headings/paragraphs/lists in reading order from its JSON tree) | opendataloader **hybrid** mode → docling/**EasyOCR** via the hybrid OCR server (auto-started; see **Run**) |
| `pymupdf` *(fallback)* | PyMuPDF `get_text()` | Phase-1 `OCR/phase1` pipeline (PP-DocLayoutV3 + **VietOCR**, Vietnamese-specialized), as a subprocess |

Adapter: `api/odl_adapter.py` → `extract_with_odl(data, ocr=False|True)`. The
hybrid OCR server lifecycle (health-check + auto-start + reuse + shutdown) lives
in the same module (`ensure_hybrid_server` / `hybrid_healthy` /
`shutdown_hybrid_server`).

### opendataloader prerequisites (the default)

opendataloader-pdf is a **Java** pipeline (needs JDK 11+; system `java` is 8,
too old). The API points it at the portable JDK via `JAVA_HOME` — set by
`scripts/start_api.sh` and re-asserted in the FastAPI lifespan and per-call in
the adapter (`LEGAL_NER_ODL_JAVA_HOME`). **Digital** extraction needs only Java.
**Scanned** extraction needs the hybrid OCR server, which the API **auto-starts**
on the first scan request (or run it yourself with `scripts/start_ocr_server.sh`
and set `LEGAL_NER_ODL_HYBRID_AUTOSTART=0`). See the **Two-process setup** under
**Run** for the full env-var table.

```bash
# digital text-layer PDF (default; <1s)
curl -F file=@"../bản án demo/document (1).pdf" "localhost:8100/extract"
# scanned PDF via opendataloader hybrid OCR (auto-starts the server; ~2 min/10pg CPU + first-run model load)
curl -m 600 -F file=@"../bản án demo/document.pdf" "localhost:8100/extract"
```

Other ODL tunables (env vars, read by `api/odl_adapter.py`):
| Var | Default | Meaning |
|-----|---------|---------|
| `LEGAL_NER_ODL_HYBRID_TIMEOUT_MS` | `0` | per-request hybrid timeout ms (0 = none) |
| `LEGAL_NER_ODL_HYBRID_BACKEND` | `docling-fast` | hybrid backend id passed to `convert()` |

If a scan is uploaded while autostart is **disabled** and no OCR server is
running, the request fails with a clear error (translated to **422**) telling
you to start `scripts/start_ocr_server.sh`. Digital extraction needs only Java.

### When to use which (measured on the demo judgments)

- **Digital text-layer PDFs:** opendataloader (now the **default**) and PyMuPDF
  produce near-identical normalized text (10852 vs 10853 chars on the demo) and
  entity counts (ODL 61 vs PyMuPDF 59); ODL adds slightly cleaner list
  reading-order. ODL costs a Java dependency but is now the default for a single
  consistent pipeline across digital + scanned inputs.
- **Vietnamese scans:** ODL-hybrid/EasyOCR is the default and competitive on the
  demo scan (140 vs 149 NER entities, identical case number, *better* on some
  diacritics like `TỈNH`) but recovers fewer numeric entities (MONEY_AMOUNT 4 vs
  12) and is English/general-purpose, not Vietnamese-specialized. The Phase-1
  **VietOCR** pipeline (`?extractor=pymupdf`) remains available as a fallback and
  is still preferable when numeric-entity recall matters. Both made the same
  `HS-ST`→`HS-SI` glyph error (ambiguous in the source scan).
- The legacy-font artifact `ƣ` persists in **both** ODL and PyMuPDF raw output
  (it is a source-font issue), so `corpus.normalize` runs on ODL output too.

### `POST /verify` (multipart, field `file`)

Full verification chain in one call: **Layer 4 forgery** + **extract**
(NER) + **Layer 2 existence** + **Layer 1 / Layer 3 citation**. Returns **200**
with a combined report. This is **decision support, not a binary real/fake legal
verdict** — see the honest disclaimer below.

```bash
# full chain (forgery + extract + live portal existence + citation, ~3-18s)
curl -m 60 -F file=@"../bản án demo/document (1).pdf" "localhost:8100/verify"

# fast / offline: skip the live portal lookup (forgery + extract + citation, <1s on text PDFs)
curl -m 60 -F file=@"../bản án demo/document (1).pdf" "localhost:8100/verify?check_existence=false"

# crawled civil PDF — existence lookup uses its real case number
curl -m 90 -F file=@"data/raw_civil/100006.pdf" "localhost:8100/verify"
```

Query params:
- `allow_ocr` (bool, default `true`) — same as `/extract`: route scanned PDFs
  through OCR for the extract step (slow: minutes). `false` → scanned PDFs 422.
- `extractor` (`opendataloader` | `pymupdf`, default **`opendataloader`**) —
  same as `/extract`: selects the extraction backend for the extract step.
- `check_existence` (bool, default `true`) — run the Layer-2 portal lookup
  (network, ~1-3s). Set `false` to skip it; `existence` then reports
  `{"status": "skipped", "reason": "check_existence=false"}`.
- `check_citations` (bool, default `true`) — run **Layer 1** (cited-law
  existence against the BLHS 2015 DB) + **Layer 3** (sentencing-frame check).
  **Offline / fast** (DB lookup only, no network). Set `false` to skip;
  `citation` then reports `{"status": "skipped", "reason": "check_citations=false"}`.

Behavior / boundaries:
- L4 forgery runs on the **raw bytes** — works on scans too (no text needed).
- Extract is **reused** from the `/extract` pipeline (text-layer fast path or
  OCR) — not reimplemented.
- L2 existence is **best-effort**: it pulls `case_number` from `case_meta`, the
  first `COURT` entity, and the first `JUDGMENT_DATE` entity (parsed to
  `dd/mm/yyyy` when possible). If **no case number** was extracted → existence
  is `{"status": "skipped", "reason": "..."}`. If the **portal errors / times
  out** → existence carries `status: "error"` but the endpoint still returns
  200 with the rest of the report (timeout ~8s, 1 retry).
- L1/L3 citation is **best-effort & offline**: it parses citations out of the
  `ARTICLE`/`CLAUSE`/`POINT`/`LEGAL_BASIS` entities, verifies each **BLHS 2015**
  citation against the local DB, matches the `CRIME` entity to the cited
  offence's title, and checks the pronounced `PENALTY` against the clause's
  statutory frame. Only BLHS 2015 is in scope — citations to BLTTHS / Nghị
  quyết / Luật THADS etc. are reported as `out_of_scope` (not an error). Any DB
  error degrades to `{"status": "skipped", "reason": "..."}`; the endpoint still
  returns 200.
- Same upload validation as `/extract` (400 not-a-PDF / empty; 422 scanned +
  `allow_ocr=false`).

#### Response schema (`/verify`)

```jsonc
{
  "filename": "document (1).pdf",
  "extract": {
    // the /extract fields minus "filename":
    "case_meta": {"case_number": "17/2018/HS-ST", "case_type": "hình sự", "procedure_stage": "sơ thẩm"},
    "entities": [ {"type": "DEFENDANT", "text": "...", "start": 1234, "end": 1240, "score": null} ],
    "entities_grouped": {"DEFENDANT": [ ... ], "COURT": [ ... ]},
    "num_pages": 5,
    "char_count": 18342,
    "warnings": [],
    "ocr": null
  },
  "forgery": {                       // verify.forgery.ForgeryReport (Layer 4)
    "risk_level": "low",             // low | medium | high
    "risk_score": 0,                 // 0..100
    "is_scanned": false,
    "findings": [ {"signal": "...", "severity": "...", "detail_vi": "...", "evidence": {}} ],
    "metadata": {"producer": "...", "creator": "...", "creation_date": "...", "mod_date": "...", "num_pages": 5, "num_eof": 1, "has_incremental": false},
    "summary_vi": "Rủi ro THẤP: ..."
  },
  "existence": {                     // verify.existence.ExistenceReport (Layer 2) ...
    "status": "found_partial",       // found | found_partial | not_found | error
    "confidence": 0.6,
    "queried": {"case_number": "17/2018/HS-ST", "court": "TÒA ÁN NHÂN DÂN", "judgment_date": "26/01/2018"},
    "matches": [ {"case_number": "17/2018/HS-ST", "court": "TAND Quận 11, TP. Hồ Chí Minh", "date": "2019-03-20", "detail_url": "...", "score": 0.6} ],
    "caveat_vi": "Không tìm thấy KHÔNG đồng nghĩa với giả mạo — ...",
    "summary_vi": "Tìm thấy bản án có số trùng khớp ..."
  },                                 // ... OR {"status": "skipped", "reason": "..."} when L2 not run
  "citation": {                      // verify.citation.CitationReport (Layer 1 + Layer 3)
    "law_scope": "BLHS 2015 only (MVP)",
    "citations": [
      {
        "raw": "điểm c khoản 1 Điều 250 Bộ luật Hình sự",
        "so_dieu": 250, "so_khoan": 1, "diem": "c",
        "law": "BLHS",                // BLHS | BLTTHS | OTHER | UNKNOWN
        "status": "valid",            // valid | not_found | out_of_scope | unparseable
        "article_title": "Tội vận chuyển trái phép chất ma túy",
        "message_vi": "Điều 250 khoản 1 điểm c) tồn tại.",
        "charge_match": "true",       // true | false | uncertain | null (offence vs CRIME)
        "sentencing": {               // Layer 3, null when not applicable
          "status": "within_frame",  // within_frame | out_of_frame | note | uncertain
          "reason_vi": "mức tù tuyên 3 năm nằm trong khung [2–7] năm",
          "penalty_raw": "03 năm tù",
          "frame_vi": "khung tù Điều 250 khoản 1: 2.0–7.0 năm"
        }
      },
      {"raw": "...Điều 106 Bộ luật Tố tụng hình sự...", "so_dieu": 106, "law": "BLTTHS", "status": "out_of_scope", "article_title": null}
    ],
    "summary": {"total": 11, "valid": 6, "not_found": 2, "out_of_scope": 3, "unparseable": 0, "charge_mismatches": 0, "sentencing_flags": 0},
    "flags_vi": ["Điều 999 không tồn tại trong BLHS 2015", "Hình phạt tuyên vượt khung Điều 173 khoản 2: ..."],
    "note_vi": "Chỉ kiểm chứng các viện dẫn thuộc BLHS 2015; các luật khác (Tố tụng hình sự, Nghị quyết, Luật THADS, ...) nằm ngoài phạm vi MVP."
  },                                 // ... OR {"status": "skipped", "reason": "..."} when L1/L3 not run
  "overall": {
    "verdict_vi": "Kết quả thẩm định sơ bộ: ...; ...; kiểm chứng 11 viện dẫn (BLHS 2015): 6 hợp lệ, 2 không tồn tại, 3 ngoài phạm vi. Các dấu hiệu (nếu có) chỉ là điểm CẦN KIỂM TRA, không phải bằng chứng giả mạo; việc không tìm thấy trên cổng công bố KHÔNG kết luận bản án là giả.",
    "flags": ["Điều 250 khoản 1 tồn tại nhưng KHÔNG có điểm s)."],
    "confidence_note_vi": "Đây là công cụ HỖ TRỢ thẩm định, không phải kết luận pháp lý. ..."
  }
}
```

**Layer 1 / Layer 3 details (`citation` block):**
- **Layer 1** — each citation gets a `status`: `valid` (Điều/khoản/điểm exists in
  BLHS 2015), `not_found` (does not exist — a strong signal, surfaced as a flag),
  `out_of_scope` (a non-BLHS law, skipped: out-of-db), `unparseable` (no Điều
  number recovered). Valid citations carry the DB `article_title`.
- **Layer 1+** — `charge_match` compares the `CRIME` entity to the cited
  **offence** article's title (`true`/`false`/`uncertain`). String/token match
  first, optional sentence-transformers cosine as a tiebreaker.
- **Layer 3** — `sentencing` parses the pronounced `PENALTY` (e.g. "03 năm tù")
  into years and checks it against the cited clause's statutory min/max:
  `within_frame`, `out_of_frame` (OVER the max → the real red flag, surfaced as
  a flag), `note` (UNDER the min — legally possible via Điều 54 mitigation, NOT
  alarming), `uncertain` (clause has no comparable custodial frame, e.g. a fine).

#### Honest disclaimer (built into `overall`)

`overall.verdict_vi` is deliberately **non-alarmist** and never a binary
real/fake judgement:
- Forgery findings are framed as **"dấu hiệu cần kiểm tra"** (signals to check),
  not proof of tampering.
- Existence `not_found` is explicitly **"KHÔNG kết luận giả mạo"** — the portal
  has publication lag and many judgments are never published (excluded
  categories per Nghị quyết 03/2017/NQ-HĐTP).
- `overall.confidence_note_vi` reminds that this is **decision support, not a
  legal verdict** — every signal must be checked manually against the original.

- Citation `not_found` and sentencing `out_of_frame` are likewise framed as
  **"cần kiểm tra"** and folded into `overall.flags`; an `out_of_scope` citation
  is never a flag (it is simply outside the MVP DB).

Latency (this shared GPU, native text-layer PDFs):
- `check_existence=true`: ~3-18s (dominated by the live portal round-trip).
- `check_existence=false`: <1s (forgery + extract + citation).
- `check_citations` adds **negligible** cost (offline DB lookups); the optional
  sentence-transformers tiebreaker (CPU) only loads on the first inconclusive
  charge match.
- Scanned PDFs add the OCR cost (minutes) to the extract step — combine with
  `check_existence=false` to avoid stacking the network call on top.

### Async job queue (`POST /jobs/extract`, `POST /jobs/verify`, `GET /jobs/{id}`)

The synchronous `/extract` and `/verify` endpoints block for the whole request.
For text-layer PDFs that is fine (<1s), but a **scanned PDF** runs OCR that takes
**~2-23 min** — far too long to hold an HTTP connection open. The async job queue
lets you submit work, get a `job_id` back immediately, and poll for the result.

```bash
# 1. submit (returns 202 + job_id INSTANTLY, even for a 10-page scan)
curl -X POST "http://localhost:8100/jobs/extract" -F "file=@scan.pdf"
# -> {"job_id":"da35...","status":"queued","poll_url":"http://localhost:8100/jobs/da35..."}

# 2. poll until status == "done" (or "error")
curl "http://localhost:8100/jobs/da35..."
# -> {"job_id":"da35...","status":"running", ...}
# -> {"job_id":"da35...","status":"done","result":{ <full ExtractResponse> }, ...}

# verify variant — same params as sync /verify
curl -X POST "http://localhost:8100/jobs/verify?check_existence=false" -F "file=@scan.pdf"

# recent jobs (metadata only, newest first)
curl "http://localhost:8100/jobs?limit=20"
```

Query params match the sync endpoints: `allow_ocr`, `extractor` (both submit
endpoints) plus `check_existence`, `check_citations` (verify only).

**Job lifecycle:** `queued -> running -> done` (or `-> error`). `GET /jobs/{id}`
returns the full record:

```jsonc
{
  "job_id": "da357be51906427d82f092cb2d3ce889",
  "status": "done",                 // queued | running | done | error
  "kind": "extract",                // extract | verify
  "filename": "document (1).pdf",
  "params": {"allow_ocr": true, "extractor": "opendataloader"},
  "created_at": "2026-06-12T04:45:23+00:00",
  "started_at": "2026-06-12T04:45:23+00:00",
  "finished_at": "2026-06-12T04:45:24+00:00",
  "result": { /* full ExtractResponse / VerifyResponse when status=="done" */ },
  "error": null                     // message string when status=="error"
}
```

**Backend — in-process worker + SQLite (NOT Redis/arq).** A single background
worker thread (`ThreadPoolExecutor(max_workers=1)`) pulls jobs and runs the same
`run_extract` / `run_verify` against the shared GPU model. One worker means OCR
is **serialized** — only one job runs at a time (the GPU and the single hybrid
OCR server are one shared resource), and submit a second job and it waits in
`queued` until the first finishes. Jobs are persisted to `data/jobs.db` (SQLite,
WAL mode) so they survive across requests and restarts. We chose this over
arq+Redis because **this host has no Redis, no arq, and a permission-denied
docker socket** — a hard Redis dependency would stop the API from starting. See
the module docstring in `api/jobs.py` for the full rationale.

**Restart semantics:** on startup the queue reconciles the DB — any job left
`running` (process died mid-OCR) is marked `error` with *"the in-flight OCR did
not survive the restart. Please resubmit."* (the OCR is **not** auto-retried);
any job still `queued` is re-enqueued. On `SIGTERM` the worker drains between
jobs with a bounded wait, so shutdown stays prompt even if a long OCR is running
(that job is reconciled to `error` on the next start).

**Limitations:** single-worker throughput (no parallelism — by design, the GPU
is the bottleneck); an in-flight OCR cannot be cancelled mid-run and does not
survive a restart; this is a single-host design (no horizontal scaling — that
would require swapping the in-process queue for arq+Redis, which the endpoints
are decoupled from).

### `GET /`
Redirects to `/docs` (interactive Swagger UI).

## Response schema (`/extract`)

```jsonc
{
  "filename": "document (1).pdf",
  "case_meta": {
    "case_number": "17/2018/HS-ST",
    "case_type": "hình sự",
    "procedure_stage": "sơ thẩm"
  },
  "entities": [
    {"type": "DEFENDANT", "text": "...", "start": 1234, "end": 1240, "score": null}
  ],
  "entities_grouped": {
    "DEFENDANT": [ ... ],
    "ARTICLE":   [ ... ]
  },
  "num_pages": 7,
  "char_count": 18342,
  "warnings": ["text extracted via opendataloader-digital"],
  "ocr": null,
  "extractor": "opendataloader"
}
```

`extractor` echoes the backend used (`opendataloader` (default) or `pymupdf`).

For a scanned PDF, `ocr` is populated instead of `null`:
```jsonc
"ocr": {
  "engine": "layout",
  "device": "cuda",
  "page_count": 10,
  "mean_confidence": 0.91,
  "quality_warnings": 0,
  "skipped_pages": 0,
  "per_page": [ {"page_index": 0, "engine": "pp-doclayoutv3+paddleocr", "n_blocks": 14, "mean_confidence": 0.93, "skipped": false, "quality": "good"} ]
}
```

- `start` / `end` are character offsets into the **flattened** normalized text
  stream (line-wraps collapsed), matching the model's tokenization.
- `score` is reserved (always `null` in v1).
- `case_meta` is derived from the predicted `CASE_NUMBER` suffix
  (`HS/DS/HC/HNGĐ/KDTM/LĐ` + `ST/PT`) via `labeling.patterns.derive_doc_meta`.
- `warnings` notes missing `CASE_NUMBER` or empty extraction.

## Notes
- No auth (later phase). Digital PDFs stay fast (<1s) on the sync endpoints;
  for scanned PDFs (OCR ~2-23 min) use the **async job queue**
  (`POST /jobs/extract` / `POST /jobs/verify` -> poll `GET /jobs/{id}`) so the
  HTTP request returns immediately. See the *Async job queue* section above.
- The CLI in `training/infer.py` still works unchanged
  (`python -m training.infer --model ... --pdf ...`).

## Production notes (OCR server)

For dev/single-host, the API auto-starting the hybrid OCR server is convenient,
but it has known fragilities:
- **Cold start:** the very first scan after the OCR server launches pays a
  one-time EasyOCR model-load cost (~1-2 min) on top of OCR time. The API polls
  `/health` for up to `LEGAL_NER_ODL_HYBRID_START_TIMEOUT_S` (default 180s).
- **Lifecycle coupling:** an auto-started server is a child of the API process
  and is terminated when the API shuts down (lifespan + `atexit`). If the API
  crashes hard (SIGKILL), the child may be orphaned — check `lsof -ti tcp:5002`.
- **Concurrency:** auto-start is serialised by a lock so only one server is
  spawned, but the OCR server itself processes scans sequentially (CPU-bound).

**Recommended for production:** run the OCR server under **systemd** as an
independent unit and set `LEGAL_NER_ODL_HYBRID_AUTOSTART=0` on the API so it
health-checks but never spawns. Example unit:

```ini
# /etc/systemd/system/legal-ner-ocr.service
[Unit]
Description=Legal-NER opendataloader hybrid OCR server
After=network.target

[Service]
User=tts
Environment=JAVA_HOME=/home/tts/jdk/jdk-21.0.11+10
Environment=CUDA_VISIBLE_DEVICES=
ExecStart=/home/tts/AI/AIHoang/Legal/legal_ner/scripts/start_ocr_server.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then `systemctl enable --now legal-ner-ocr` and start the API with
`LEGAL_NER_ODL_HYBRID_AUTOSTART=0`. This decouples the slow model load from
request handling and keeps the OCR server warm across API restarts.
```
