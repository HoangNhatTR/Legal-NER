"""Orchestration: PDF bytes -> entities + case metadata.

Reuses the existing pipeline:
  * corpus.extract        — text-layer threshold (MIN_TEXT_CHARS)
  * corpus.normalize      — legacy-font/NFC normalize + flatten_for_matching
  * training.infer        — load_model / infer_entities (overlapping windows)
  * labeling.patterns     — derive_doc_meta (case_type / procedure_stage)

The model is loaded ONCE by the FastAPI lifespan and handed to ``run_extract``;
this module never reloads it per request.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus.extract import MIN_TEXT_CHARS  # noqa: E402
from corpus.normalize import flatten_for_matching, normalize_text  # noqa: E402
from labeling.patterns import Span, derive_doc_meta  # noqa: E402
from training.infer import infer_entities, load_model  # noqa: E402

from api.ocr_adapter import OcrError, OcrResult, _venv_ok, run_ocr_on_pdf  # noqa: E402
from api.odl_adapter import OdlError, extract_with_odl  # noqa: E402

# Production NER checkpoint shipped with this bundle: xlm-roberta-base
# fine-tuned on Vietnamese judgments, 31-label schema (v3r-full, self-eval
# micro-F1 ~0.970). The model lives inside the bundle at
# ``<package-root>/model/legal-ner-v3r-full/final`` so the API runs out of the
# box on any machine after `git lfs` pulls the safetensors weights.
_BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = str(_BUNDLE_ROOT / "model" / "legal-ner-v3r-full" / "final")
PDF_MAGIC = b"%PDF"

# Extraction backends selectable per request.
#   "pymupdf"        — native text-layer via PyMuPDF (fast path). This is the
#                      DEFAULT in this bundle: pure-Python, no Java required, so
#                      it runs anywhere out of the box for text-layer PDFs.
#   "opendataloader" — opendataloader-pdf (docling/EasyOCR): structured digital
#                      extraction + hybrid OCR for scans. OPTIONAL — requires a
#                      JDK 11+ (LEGAL_NER_ODL_JAVA_HOME) and, for scans, a hybrid
#                      OCR server. Select per request via ?extractor=opendataloader.
EXTRACTORS = ("pymupdf", "opendataloader")
DEFAULT_EXTRACTOR = "pymupdf"


class NotAPdfError(Exception):
    """Uploaded bytes are not a PDF -> HTTP 400."""


class ScannedPdfError(Exception):
    """PDF has no usable text layer -> HTTP 422 (OCR pipeline pending)."""


class ModelHolder:
    """Singleton-style holder for the loaded model, filled at app startup."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device: str = "cpu"
        self.model_path: str = DEFAULT_MODEL_PATH
        self.loaded: bool = False

    def load(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        """Load once. Prefer GPU; fall back to CPU on CUDA OOM/error.

        GPU here is shared and near-full (~4GB free); the NER checkpoint is
        ~1GB so it usually fits, but we degrade gracefully rather than crash.
        """
        self.model_path = model_path
        want_cuda = torch.cuda.is_available()
        if want_cuda:
            try:
                self.tokenizer, self.model = load_model(model_path, "cuda")
                self.device = "cuda"
                self.loaded = True
                return
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:  # type: ignore[attr-defined]
                # free whatever partially allocated, then retry on CPU
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                print(f"[api] CUDA load failed ({exc}); falling back to CPU", flush=True)
        self.tokenizer, self.model = load_model(model_path, "cpu")
        self.device = "cpu"
        self.loaded = True


def _read_pdf(data: bytes) -> tuple[str, int, bool]:
    """Return (raw_text, num_pages, is_scanned) from PDF bytes, or raise.

    NotAPdfError — bytes don't open as a PDF.
    ``is_scanned`` is True when the native text layer is below MIN_TEXT_CHARS;
    the caller decides whether to OCR (allow_ocr) or raise ScannedPdfError.
    """
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # corrupt / not a PDF
        raise NotAPdfError(f"cannot open as PDF: {exc}") from exc
    try:
        num_pages = doc.page_count
        raw = "".join(page.get_text() for page in doc)
    finally:
        doc.close()
    is_scanned = len(raw.strip()) < MIN_TEXT_CHARS
    return raw, num_pages, is_scanned


def looks_like_pdf(data: bytes, content_type: str | None) -> bool:
    """Accept by magic bytes (authoritative) or declared content-type."""
    if data[:4] == PDF_MAGIC:
        return True
    return bool(content_type and "pdf" in content_type.lower())


def _infer_and_group(
    text: str, holder: ModelHolder
) -> tuple[dict, list[dict], dict, list[str], str]:
    """Shared normalize -> infer -> group -> derive_meta flow.

    ``text`` is the already-extracted raw text (native layer OR OCR). Returns
    (case_meta, entities, entities_grouped, warnings, normalized_text).
    """
    warnings: list[str] = []

    # normalize legacy-font artifacts + NFC, then flatten line-wraps so the
    # model sees the same stream the offsets index into.
    norm = flatten_for_matching(normalize_text(text))

    raw_entities = infer_entities(norm, holder.model, holder.tokenizer, holder.device)

    # case_meta from the predicted CASE_NUMBER span(s): rebuild Span objects so
    # we can reuse derive_doc_meta unchanged.
    case_spans = [
        Span(e["start"], e["end"], e["label"], e["text"])
        for e in raw_entities
        if e["label"] == "CASE_NUMBER"
    ]
    case_meta = derive_doc_meta(case_spans)
    if not case_spans:
        warnings.append("no CASE_NUMBER entity detected; case_meta is empty")

    entities = [
        {"type": e["label"], "text": e["text"], "start": e["start"],
         "end": e["end"], "score": e.get("score")}
        for e in raw_entities
    ]
    grouped: dict[str, list[dict]] = {}
    for ent in entities:
        grouped.setdefault(ent["type"], []).append(ent)

    if not entities:
        warnings.append("no entities extracted")

    return case_meta, entities, grouped, warnings, norm


def _extract_pymupdf(
    data: bytes, *, allow_ocr: bool
) -> tuple[str, int, dict | None, list[str]]:
    """Default backend: PyMuPDF text layer + Phase-1 VietOCR for scans.

    Returns (raw_text, num_pages, ocr_block, extra_warnings). Raises
    ScannedPdfError when a scan cannot be OCR'd.
    """
    raw, num_pages, is_scanned = _read_pdf(data)
    ocr_block: dict | None = None
    extra_warnings: list[str] = []

    if is_scanned:
        if not allow_ocr:
            raise ScannedPdfError(
                "scanned PDF: text-layer extraction failed and allow_ocr=false"
            )
        # This is an NER-only inference bundle: no OCR engine ships with it.
        # The Phase-1 OCR pipeline (api/ocr_adapter.py) is an OPTIONAL hook that
        # expects a separate OCR venv; when it isn't wired up, give the caller
        # the documented, actionable message instead of leaking an absolute path
        # to a venv that only existed on the author's machine.
        ocr_ready, _ = _venv_ok()
        if not ocr_ready:
            raise ScannedPdfError(
                "scanned PDF (no text layer): OCR is not bundled in this "
                "inference package. Supply a text-layer PDF, use "
                "?extractor=opendataloader with a hybrid OCR server, or wire up "
                "your own OCR (see api/ocr_adapter.py)."
            )
        try:
            ocr = run_ocr_on_pdf(data)
        except OcrError as exc:
            raise ScannedPdfError(f"scanned PDF: OCR failed ({exc})") from exc

        if not ocr.text.strip():
            raise ScannedPdfError(
                "scanned PDF: OCR produced no text (page(s) may be too blurry)"
            )

        raw = ocr.text
        num_pages = ocr.page_count or num_pages
        ocr_block = {
            "engine": ocr.engine,
            "device": ocr.device,
            "page_count": ocr.page_count,
            "mean_confidence": ocr.mean_confidence,
            "quality_warnings": ocr.quality_warnings,
            "skipped_pages": ocr.skipped_pages,
            "per_page": ocr.per_page,
        }
        extra_warnings.append(
            f"text extracted via OCR (engine={ocr.engine}, device={ocr.device}); "
            f"accuracy lower than a native text layer — verify citations"
        )
        if ocr.skipped_pages:
            extra_warnings.append(
                f"{ocr.skipped_pages} page(s) skipped by OCR (too blurry)"
            )
    return raw, num_pages, ocr_block, extra_warnings


def _extract_opendataloader(
    data: bytes, *, allow_ocr: bool
) -> tuple[str, int, dict | None, list[str]]:
    """opendataloader-pdf backend: structured digital, hybrid OCR for scans.

    Scan detection still uses the PyMuPDF text-layer threshold (cheap, shared);
    a text layer -> ODL digital; no text layer -> ODL hybrid OCR (when
    ``allow_ocr``). Returns the same 4-tuple as ``_extract_pymupdf``.
    """
    _, pm_pages, is_scanned = _read_pdf(data)
    ocr_block: dict | None = None
    extra_warnings: list[str] = []

    use_ocr = is_scanned
    if is_scanned and not allow_ocr:
        raise ScannedPdfError(
            "scanned PDF: text-layer extraction failed and allow_ocr=false"
        )

    try:
        res = extract_with_odl(data, ocr=use_ocr)
    except OdlError as exc:
        if use_ocr:
            raise ScannedPdfError(
                f"scanned PDF: opendataloader hybrid OCR failed ({exc})"
            ) from exc
        # digital failure on a text-layer PDF -> not recoverable here
        raise NotAPdfError(f"opendataloader extraction failed ({exc})") from exc

    raw = res["text"]
    if not raw.strip():
        if use_ocr:
            raise ScannedPdfError(
                "scanned PDF: opendataloader hybrid OCR produced no text"
            )
        raise NotAPdfError("opendataloader extracted no text from the PDF")

    num_pages = res.get("num_pages") or pm_pages
    extra_warnings.append(f"text extracted via {res['source']}")
    if use_ocr:
        ocr_block = {
            "engine": "opendataloader-hybrid (docling/easyocr)",
            "device": "cpu",
            "page_count": num_pages,
            "mean_confidence": None,
            "quality_warnings": 0,
            "skipped_pages": 0,
            "per_page": [],
        }
        extra_warnings.append(
            "text extracted via opendataloader hybrid OCR (docling/EasyOCR); "
            "accuracy lower than a native text layer — verify citations"
        )
    return raw, num_pages, ocr_block, extra_warnings


def run_extract(
    filename: str,
    data: bytes,
    holder: ModelHolder,
    *,
    allow_ocr: bool = True,
    extractor: str = DEFAULT_EXTRACTOR,
) -> dict:
    """Full pipeline for one uploaded PDF -> response dict.

    ``extractor`` selects the extraction backend:
      * "pymupdf"        (default) — PyMuPDF text layer + Phase-1 VietOCR scans.
      * "opendataloader" — opendataloader-pdf structured digital + hybrid OCR.

    Text-layer PDFs take the fast path. Scanned PDFs (no usable text layer) are
    routed through the backend's OCR when ``allow_ocr`` is True; otherwise
    ScannedPdfError (422) is raised. The normalize -> infer flow afterward is
    IDENTICAL for both backends.

    Raises NotAPdfError (400) or ScannedPdfError (422); callers translate these
    into HTTP responses.
    """
    if extractor not in EXTRACTORS:
        raise NotAPdfError(
            f"unknown extractor {extractor!r}; expected one of {EXTRACTORS}"
        )

    if extractor == "opendataloader":
        raw, num_pages, ocr_block, extra_warnings = _extract_opendataloader(
            data, allow_ocr=allow_ocr
        )
    else:
        raw, num_pages, ocr_block, extra_warnings = _extract_pymupdf(
            data, allow_ocr=allow_ocr
        )

    case_meta, entities, grouped, warnings, norm = _infer_and_group(raw, holder)
    warnings = extra_warnings + warnings

    return {
        "filename": filename,
        "case_meta": case_meta,
        "entities": entities,
        "entities_grouped": grouped,
        "num_pages": num_pages,
        "char_count": len(norm),
        "warnings": warnings,
        "ocr": ocr_block,
        "extractor": extractor,
    }

