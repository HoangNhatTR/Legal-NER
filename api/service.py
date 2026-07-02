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

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODELS_DIR  # noqa: E402
from corpus.extract import MIN_TEXT_CHARS  # noqa: E402
from corpus.normalize import flatten_for_matching, normalize_text  # noqa: E402
from labeling.patterns import Span, derive_doc_meta  # noqa: E402
from training.infer import infer_entities, load_model  # noqa: E402

from api.ocr_adapter import OcrError, OcrResult, run_ocr_on_pdf  # noqa: E402
from api.odl_adapter import OdlError, extract_with_odl  # noqa: E402
from verify.citation import CitationReport, check_citations  # noqa: E402
from verify.existence import ExistenceReport, check_existence  # noqa: E402
from verify.forgery import ForgeryReport, analyze_pdf  # noqa: E402

# Default to the v3r-full checkpoint: full fine-tune, 31-entity schema
# (adds procedural people, sentencing factors, quantities/evidence, COURT_BEHAVIOR,
# CRIMINAL_ACT), self-eval micro-F1 ~0.970 — every legacy label beats the old
# 20-label "combined" model (which was 0.925). See NER_MODEL_REPORT.md.
# The label set is read from the checkpoint's own config.json id2label at load
# time, so decoding stays correct regardless of config.ENTITY_TYPES.
DEFAULT_MODEL_PATH = str(MODELS_DIR / "legal-ner-v3r-full" / "final")
PDF_MAGIC = b"%PDF"

# Extraction backends selectable per request.
#   "opendataloader" — opendataloader-pdf (docling/EasyOCR): structured digital
#                      extraction for text-layer PDFs, hybrid OCR for scans.
#                      This is the DEFAULT (both digital and scanned PDFs).
#   "pymupdf"        — native text-layer via PyMuPDF (fast path) + Phase-1
#                      VietOCR pipeline for scans. Kept as a selectable
#                      fallback via ?extractor=pymupdf.
EXTRACTORS = ("opendataloader", "pymupdf")
DEFAULT_EXTRACTOR = "opendataloader"


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


def _ocr_scan(data: bytes) -> tuple[str, int, dict | None, list[str]]:
    """OCR một bản án SCAN -> (raw_text, num_pages, ocr_block, warnings).

    Chọn backend tự động:
      * Có env LEGAL_NER_OCRSPACE_KEY -> OCR.space (đám mây, NHANH, song song).
        Lỗi mạng/API -> tự lùi về EasyOCR offline.
      * Không có key -> EasyOCR (offline, chậm nhưng không cần dịch vụ ngoài).
    Raises ScannedPdfError khi cả 2 đều thất bại.
    """
    from api.ocr_cloud import OcrError as CloudOcrError, ocrspace_available, run_ocrspace_on_pdf
    from api.ocr_easyocr import OcrError as EasyOcrError, run_easyocr_on_pdf

    ocr = None
    if ocrspace_available():
        try:
            ocr = run_ocrspace_on_pdf(data)
        except CloudOcrError:
            ocr = None  # lùi về EasyOCR
    if ocr is None:
        try:
            ocr = run_easyocr_on_pdf(data)
        except EasyOcrError as exc:
            raise ScannedPdfError(f"scanned PDF: OCR failed ({exc})") from exc

    ocr_block = {
        "engine": ocr.engine,
        "device": ocr.device,
        "page_count": ocr.page_count,
        "mean_confidence": ocr.mean_confidence,
        "quality_warnings": ocr.quality_warnings,
        "skipped_pages": ocr.skipped_pages,
        "per_page": ocr.per_page,
    }
    warnings = [
        f"text extracted via OCR (engine={ocr.engine}, device={ocr.device}); "
        f"accuracy lower than a native text layer — verify citations"
    ]
    if ocr.skipped_pages:
        warnings.append(f"{ocr.skipped_pages} page(s) skipped by OCR (too blurry)")
    return ocr.text, ocr.page_count, ocr_block, warnings


def _extract_pymupdf(
    data: bytes, *, allow_ocr: bool
) -> tuple[str, int, dict | None, list[str]]:
    """PyMuPDF text layer cho PDF có chữ; EasyOCR cho bản scan (khi allow_ocr)."""
    raw, num_pages, is_scanned = _read_pdf(data)
    if is_scanned:
        if not allow_ocr:
            raise ScannedPdfError(
                "scanned PDF: text-layer extraction failed and allow_ocr=false"
            )
        return _ocr_scan(data)
    return raw, num_pages, None, []


def _extract_opendataloader(
    data: bytes, *, allow_ocr: bool
) -> tuple[str, int, dict | None, list[str]]:
    """opendataloader-pdf cho PDF có chữ (structured digital); EasyOCR cho scan.

    Scan detection dùng ngưỡng text-layer của PyMuPDF (rẻ, chung). Bản scan ->
    EasyOCR (thay cho hybrid server Java vốn không có trên Windows).
    """
    _, pm_pages, is_scanned = _read_pdf(data)
    if is_scanned:
        if not allow_ocr:
            raise ScannedPdfError(
                "scanned PDF: text-layer extraction failed and allow_ocr=false"
            )
        return _ocr_scan(data)

    try:
        res = extract_with_odl(data, ocr=False)
    except OdlError as exc:
        raise NotAPdfError(f"opendataloader extraction failed ({exc})") from exc

    raw = res["text"]
    if not raw.strip():
        raise NotAPdfError("opendataloader extracted no text from the PDF")
    num_pages = res.get("num_pages") or pm_pages
    return raw, num_pages, None, [f"text extracted via {res['source']}"]


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
        # Toàn văn đã chuẩn hoá (để client gửi sang RAG/validate Module 1).
        "text": norm,
    }


# ---------------------------------------------------------------------------
# Layer-2 field selection + overall verdict synthesis
# ---------------------------------------------------------------------------

_DATE_DDMMYYYY_RE = re.compile(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})")
_DATE_VI_RE = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.I)


def _first_entity_text(grouped: dict, label: str) -> str | None:
    items = grouped.get(label) or []
    for ent in items:
        text = (ent.get("text") or "").strip()
        if text:
            return text
    return None


def _parse_judgment_date(raw: str | None) -> str | None:
    """Best-effort: turn a JUDGMENT_DATE entity into ``dd/mm/yyyy``.

    check_existence accepts dd/mm/yyyy or ISO and normalizes internally, so we
    only need to surface a parseable date when one is present; otherwise we pass
    the raw string through (existence still matches on case number)."""
    if not raw:
        return None
    m = _DATE_VI_RE.search(raw)
    if m:
        d, mo, y = m.groups()
        return f"{int(d):02d}/{int(mo):02d}/{int(y):04d}"
    m = _DATE_DDMMYYYY_RE.search(raw)
    if m:
        d, mo, y = m.groups()
        return f"{int(d):02d}/{int(mo):02d}/{int(y):04d}"
    return raw.strip() or None


def _build_overall(
    forgery: ForgeryReport,
    existence: ExistenceReport | None,
    existence_skipped_reason: str | None,
    citation: CitationReport | None = None,
    citation_skipped_reason: str | None = None,
) -> dict:
    """Honest, non-alarmist synthesis. Never a binary real/fake verdict."""
    flags: list[str] = []

    # --- forgery flags ---
    if forgery.risk_level == "low":
        forgery_vi = "không phát hiện dấu hiệu chỉnh sửa đáng kể trong tệp PDF"
    else:
        top = "; ".join(f.signal for f in forgery.findings[:3]) or "không rõ"
        level_vi = "trung bình" if forgery.risk_level == "medium" else "cao"
        forgery_vi = (
            f"có dấu hiệu cần kiểm tra trong tệp PDF (mức {level_vi}, "
            f"điểm {forgery.risk_score}/100)"
        )
        flags.append(f"Có dấu hiệu chỉnh sửa PDF cần kiểm tra: {top}")

    # --- existence flags ---
    if existence is None:
        existence_vi = (
            f"bỏ qua tra cứu cổng công bố ({existence_skipped_reason})"
        )
    elif existence.status == "found":
        existence_vi = "tìm thấy bản án khớp trên cổng công bố"
    elif existence.status == "found_partial":
        existence_vi = "tìm thấy bản án gần khớp trên cổng công bố (cần đối chiếu thủ công)"
        flags.append("Cổng công bố chỉ khớp một phần (số trùng, tòa/ngày cần đối chiếu)")
    elif existence.status == "not_found":
        existence_vi = "KHÔNG tìm thấy trên cổng công bố"
        flags.append(
            "Không tìm thấy trên cổng công bố (KHÔNG kết luận giả mạo: cổng có "
            "độ trễ và nhiều bản án không được công bố)"
        )
    else:  # error
        existence_vi = "không tra cứu được cổng công bố (lỗi mạng/cổng)"
        flags.append("Tra cứu cổng công bố thất bại (không kết luận)")

    # --- citation flags (L1 cited-law existence + L3 sentencing frame) ---
    if citation is None:
        citation_vi = f"bỏ qua kiểm chứng viện dẫn pháp luật ({citation_skipped_reason})"
    else:
        s = citation.summary
        if s.total == 0:
            citation_vi = "không tìm thấy viện dẫn BLHS 2015 nào để kiểm chứng"
        else:
            citation_vi = (
                f"kiểm chứng {s.total} viện dẫn (BLHS 2015): {s.valid} hợp lệ, "
                f"{s.not_found} không tồn tại, {s.out_of_scope} ngoài phạm vi"
            )
            # surface each citation finding as a non-alarmist flag (cần kiểm tra)
            for f in citation.flags_vi:
                flags.append(f)

    if not flags:
        flags.append("Không phát hiện dấu hiệu bất thường nào đáng kể")

    verdict_vi = (
        f"Kết quả thẩm định sơ bộ: {forgery_vi}; {existence_vi}; {citation_vi}. "
        "Các dấu hiệu (nếu có) chỉ là điểm CẦN KIỂM TRA, không phải bằng chứng "
        "giả mạo; việc không tìm thấy trên cổng công bố KHÔNG kết luận bản án là giả."
    )

    confidence_note_vi = (
        "Đây là công cụ HỖ TRỢ thẩm định, không phải kết luận pháp lý. "
        "Mọi dấu hiệu cần được thẩm định viên kiểm tra thủ công với bản gốc."
    )

    return {
        "verdict_vi": verdict_vi,
        "flags": flags,
        "confidence_note_vi": confidence_note_vi,
    }


def run_verify(
    filename: str,
    data: bytes,
    holder: ModelHolder,
    *,
    allow_ocr: bool = True,
    extractor: str = DEFAULT_EXTRACTOR,
    check_existence_online: bool = True,
    existence_timeout: int = 8,
    check_citations_offline: bool = True,
) -> dict:
    """Full happy-path verification chain for one uploaded PDF.

    L4 forgery runs on raw bytes (works on scans). Then the existing extract
    flow produces entities + case_meta (reused, not duplicated). L2 existence is
    best-effort: a missing case number -> skipped; any portal error -> the
    ExistenceReport carries status 'error' but the endpoint still succeeds.

    L1 (cited-law existence) + L3 (sentencing frame) run offline on the
    extracted entities against the BLHS 2015 DB; controlled by
    ``check_citations_offline`` (default True; DB lookup only, no network).

    Raises NotAPdfError (400) / ScannedPdfError (422); callers translate these.
    """
    # --- Layer 4: forgery (no network, works on scans) ---
    forgery = analyze_pdf(data)

    # --- Extract (reuses run_extract — text-layer fast path or OCR) ---
    extract = run_extract(
        filename, data, holder, allow_ocr=allow_ocr, extractor=extractor
    )

    # --- Layer 2: existence (best-effort, network) ---
    existence: ExistenceReport | None = None
    existence_skipped_reason: str | None = None

    case_number = (extract.get("case_meta") or {}).get("case_number")
    # Judgment date is extracted unconditionally (used by both existence lookup
    # and citation validity-at-date), so it is defined on every code path.
    _grouped_all = extract.get("entities_grouped") or {}
    judgment_date = _parse_judgment_date(
        _first_entity_text(_grouped_all, "JUDGMENT_DATE")
    )
    if not check_existence_online:
        existence_skipped_reason = "check_existence=false"
    elif not case_number:
        existence_skipped_reason = (
            "không trích xuất được số bản án (CASE_NUMBER) để tra cứu"
        )
    else:
        grouped = extract.get("entities_grouped") or {}
        court = _first_entity_text(grouped, "COURT")
        try:
            existence = check_existence(
                case_number,
                court=court,
                judgment_date=judgment_date,
                timeout=existence_timeout,
                retries=1,
            )
        except Exception as exc:  # network/parse — degrade, never crash
            existence = ExistenceReport(
                status="error",
                confidence=0.0,
                queried={
                    "case_number": case_number,
                    "court": court,
                    "judgment_date": judgment_date,
                },
                matches=[],
                summary_vi=(
                    "Lỗi khi tra cứu cổng công bố "
                    f"({type(exc).__name__}). Kết quả KHÔNG kết luận."
                ),
            )

    # --- Layer 1 + Layer 3: cited-law verification (offline, DB lookup only) ---
    citation: CitationReport | None = None
    citation_skipped_reason: str | None = None
    if not check_citations_offline:
        citation_skipped_reason = "check_citations=false"
    else:
        try:
            # Thread the judgment date so the citation checker can flag temporal
            # anomalies (e.g. a code cited before it took effect). dd/mm/yyyy -> ISO.
            _on_date = None
            if judgment_date and "/" in judgment_date:
                _d, _m, _y = (judgment_date.split("/") + ["", "", ""])[:3]
                if _d.isdigit() and _m.isdigit() and _y.isdigit():
                    _on_date = f"{int(_y):04d}-{int(_m):02d}-{int(_d):02d}"
            citation = check_citations(
                extract.get("entities") or [], on_date=_on_date
            )
        except Exception as exc:  # DB/parse — degrade, never crash the request
            citation_skipped_reason = (
                f"lỗi khi kiểm chứng viện dẫn ({type(exc).__name__})"
            )

    overall = _build_overall(
        forgery,
        existence,
        existence_skipped_reason,
        citation=citation,
        citation_skipped_reason=citation_skipped_reason,
    )

    extract_block = {k: v for k, v in extract.items() if k != "filename"}

    if existence is not None:
        existence_out = existence
    else:
        existence_out = {"status": "skipped", "reason": existence_skipped_reason}

    if citation is not None:
        citation_out = citation
    else:
        citation_out = {"status": "skipped", "reason": citation_skipped_reason}

    return {
        "filename": filename,
        "extract": extract_block,
        "forgery": forgery,
        "existence": existence_out,
        "citation": citation_out,
        "overall": overall,
    }
