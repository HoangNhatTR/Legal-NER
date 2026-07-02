# `verify/` — Layer 4: Forgery / Tampering Detection

Standalone module that analyses a judgment PDF for **tampering / forgery
signals** and returns a structured, conservative risk report. **Pure file
analysis** — no ML training, no network. Designed to be cleanly importable so a
later step can wire it into `api/` without changes here.

```python
from verify.forgery import analyze_pdf, ForgeryReport

report: ForgeryReport = analyze_pdf("/path/to/judgment.pdf")   # str | Path | bytes
print(report.risk_level, report.risk_score)
```

CLI:

```bash
python -m verify.forgery --pdf /path/to/judgment.pdf          # human-readable
python -m verify.forgery --pdf /path/to/judgment.pdf --json   # raw JSON report
```

Dependencies: `pikepdf` (metadata / xref / incremental analysis) and `PyMuPDF`
(`fitz`; per-page fonts, dimensions, annotations, color heuristic).

---

## Design philosophy — be conservative and honest

False positives are worse than misses here. We surface **"dấu hiệu cần kiểm
tra"** (signals to investigate), never a verdict of "đã giả mạo" (proven
forgery). **A normal scanned court PDF (image-only, single producer, single
`%%EOF`) must come out LOW** — scanning is not forgery. This was validated on
100+ real crawled court PDFs: all scored LOW.

All finding explanations (`detail_vi`) and the `summary_vi` are written in
Vietnamese for the final report; all code/comments are English.

---

## Output schema (`ForgeryReport`, pydantic v2)

```jsonc
{
  "risk_level": "low|medium|high",   // aggregate
  "risk_score": 0-100,
  "is_scanned": true,                // whole doc is image-only (no text layer)
  "findings": [
    {
      "signal": "fonts.page_outlier",
      "severity": "low|medium|high",
      "detail_vi": "explanation in Vietnamese",
      "evidence": { /* structured backing data */ }
    }
  ],
  "metadata": {
    "producer": "Microsoft® Word 2016",
    "creator": "...",
    "creation_date": "D:20230115...",
    "mod_date": "D:20230115...",
    "num_pages": 6,
    "num_eof": 2,
    "has_incremental": false
  },
  "summary_vi": "one-sentence Vietnamese summary"
}
```

---

## Signals

| `signal` | Severity | What it means |
|---|---|---|
| `metadata.producer_mismatch` | medium | `Producer` in DocInfo ≠ `Producer` in XMP — file may have been re-opened/edited by a different tool. |
| `metadata.creator_mismatch` | low | `Creator` DocInfo ≠ XMP. Weak signal. |
| `metadata.dates_missing` | low | Neither `CreationDate` nor `ModDate` present. Common & benign on old scans. |
| `metadata.mod_before_creation` | medium | `ModDate` earlier than `CreationDate` — impossible timeline. |
| `metadata.large_date_gap` | low | ≥1 year between creation and modification — re-saved long after creation. |
| `metadata.suspicious_producer` | high / medium | Producer/Creator is an **image editor** (Photoshop, GIMP, Photopea → high) or an **online PDF editor** (iLovePDF, Smallpdf, Sejda, PDF-XChange Editor, Foxit PhantomPDF, Nitro Pro, Microsoft Print to PDF, …→ medium). Strings matched are *editor* products, not benign readers/printers. |
| `incremental.prev_xref` | high / medium | A **genuine incremental-update generation** exists (an xref `/Prev` that is *not* the hybrid-reference `/Prev …/XRefStm` pairing). The file was appended to (edited/signed) after creation. ≥2 extra generations → high. |
| `incremental.appended_content` | medium | Multiple `%%EOF` with a **substantial** (≥2 KB) appended block but no `/Prev`. Content may have been concatenated after creation. |
| `fonts.page_outlier` | medium | A page uses a font set **disjoint** from the document's dominant fonts. Spliced/edited pages often introduce a new font. Reports which pages. |
| `annotation.editing_artifacts` | high / medium | Editing-tool annotations: `Redact` (→ high), `FreeText`/`Stamp`/`Caret`/`StrikeOut` (→ medium). Page list in evidence. |
| `annotation.text_over_image` | low | A small amount of vector text drawn over a large scan image — could be a legitimate OCR layer or pasted-on text. |
| `dimensions.size_outlier` | medium | A few pages have a `MediaBox` size different from the dominant size — possible inserted page. |
| `dimensions.rotation_outlier` | low | A few pages have a different rotation than the rest. |
| `seal.absent_low_confidence` | low | **Low-confidence color heuristic**: no predominantly-red region (likely the red court seal "mộc đỏ") found on any page of an image-bearing doc. Black-and-white / grayscale scans legitimately have no red. Presence of a seal produces *no* finding (it is normal). |
| `file.pikepdf_open_failed` | medium | pikepdf could not parse the structure (damaged/encrypted/abnormal). |
| `file.fitz_open_failed` | medium | PyMuPDF could not open the file. |

### Why "multiple `%%EOF`" alone is **not** flagged

MS Word and many PDF libraries write **hybrid-reference** files: a single save
that contains both a classic xref table and a cross-reference stream, with a
trailer `/Prev <offset> /XRefStm <offset>`. This produces **two `%%EOF`
markers** in a perfectly clean, un-edited file. We therefore do **not** flag on
`%%EOF` count. We only flag a true incremental generation, detected as a
`/Prev` that is *not* paired with `/XRefStm` (see `_count_incremental_generations`).
This eliminated a near-universal false positive on legitimate court PDFs.

---

## Risk scoring

Each finding contributes a base weight by severity; the score is summed and
capped at 100:

| Severity | Weight |
|---|---|
| low | 5 |
| medium | 20 |
| high | 40 |

The **aggregate `risk_level`** is then:

- **high** — there is a high-severity finding **and** (`score ≥ 50` **or** at least one medium finding corroborates it).
- **medium** — any high-severity finding, **or** ≥2 medium findings, **or** `score ≥ 35`.
- **low** — otherwise.

So an isolated low finding (e.g. missing dates on a scan, or a low-confidence
seal-absence) stays **LOW**. A single inserted page (font + dimension outliers)
lands **MEDIUM**. A real incremental edit by an image editor lands **HIGH**.

---

## How to read a report

1. Start with `risk_level` / `risk_score` and `summary_vi`.
2. `is_scanned: true` means the doc is image-only — most metadata/font/xref
   signals don't apply; weight visual + structural signals more.
3. Read each `finding.detail_vi` and its `evidence`. Treat every finding as a
   **lead to verify manually**, not a conclusion.
4. `metadata.num_eof` / `has_incremental` tell you whether the file was appended
   to after creation (only `has_incremental: true` is meaningful — see above).

---

## Honest limitations

- **Seal heuristic is low-confidence.** It is a coarse color histogram (counts
  strongly-red pixels). A grayscale/B&W scan has no red regardless of being a
  valid sealed document, so absence is only a *mild* flag, and presence is not
  proof of a real seal. A trained VN-seal detector is future work and is **not**
  attempted here.
- **Scans carry little metadata.** Image-only PDFs often lack producer/dates and
  have a single xref, so most metadata/font signals are inert — there is simply
  less to analyse, not "less suspicious".
- **`text_over_image` is ambiguous** with legitimate OCR text layers; kept at low
  severity on purpose.
- The module reports *structural and metadata anomalies*. It cannot detect a
  forgery that was crafted to be byte-clean (e.g. a fully re-typeset fake printed
  fresh), nor semantic forgery (wrong names/amounts) — those belong to other
  verification layers.
