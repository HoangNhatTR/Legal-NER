"""Forgery / tampering signal analysis for Vietnamese judgment PDFs.

This is Layer 4 of "Model 2" verification. It performs *pure file analysis*
(no ML training, no network) and returns a structured, conservative risk
report. The guiding principle is: a normal scanned court PDF (image-only,
single producer, single %%EOF) must come out LOW risk — scanning is not
forgery. We surface "dấu hiệu cần kiểm tra" (signals to check), never a
verdict of "đã giả mạo" (proven forgery).

Dependencies: pikepdf (metadata / xref / incremental updates) and PyMuPDF
(fitz; per-page fonts, dimensions, annotations, pixmap color heuristic).

Public entry point:
    analyze_pdf(path_or_bytes) -> ForgeryReport
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import fitz  # PyMuPDF
import pikepdf
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Severity / weighting
# ---------------------------------------------------------------------------

Severity = Literal["low", "medium", "high"]
RiskLevel = Literal["low", "medium", "high"]

# Each severity contributes a base weight to the aggregate score. The score is
# capped at 100. See README for the rationale and full scoring table.
SEVERITY_WEIGHT: dict[Severity, int] = {
    "low": 5,
    "medium": 20,
    "high": 40,
}

# Producer/creator substrings that are unusual on an authentic court document.
# These are *signals to check*, not proof. Matched case-insensitively.
SUSPICIOUS_PRODUCER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("photoshop", "Adobe Photoshop"),
    ("gimp", "GIMP (trình chỉnh sửa ảnh)"),
    ("photopea", "Photopea (trình sửa ảnh online)"),
    ("ilovepdf", "iLovePDF (công cụ sửa PDF online)"),
    ("smallpdf", "Smallpdf (công cụ sửa PDF online)"),
    ("pdfescape", "PDFescape (trình sửa PDF online)"),
    ("sejda", "Sejda (trình sửa PDF online)"),
    ("pdf-xchange editor", "PDF-XChange Editor"),
    ("foxit phantompdf", "Foxit PhantomPDF (trình sửa PDF)"),
    ("foxit editor", "Foxit Editor (trình sửa PDF)"),
    ("nitro pro", "Nitro Pro (trình sửa PDF)"),
    ("microsoft: print to pdf", "Microsoft Print to PDF (in lại từ nguồn khác)"),
    ("microsoft print to pdf", "Microsoft Print to PDF (in lại từ nguồn khác)"),
    ("canva", "Canva"),
    ("inkscape", "Inkscape"),
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """A single tampering signal that fired."""

    signal: str = Field(description="machine-readable signal id")
    severity: Severity
    detail_vi: str = Field(description="human-readable explanation in Vietnamese")
    evidence: dict[str, Any] = Field(
        default_factory=dict, description="structured evidence backing the finding"
    )


class PdfMetadata(BaseModel):
    producer: str | None = None
    creator: str | None = None
    creation_date: str | None = None
    mod_date: str | None = None
    num_pages: int = 0
    num_eof: int = 0
    has_incremental: bool = False


class ForgeryReport(BaseModel):
    """Aggregate forgery-risk report for one PDF."""

    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100)
    is_scanned: bool = Field(
        description="whole document is image-only (no extractable text layer)"
    )
    findings: list[Finding] = Field(default_factory=list)
    metadata: PdfMetadata
    summary_vi: str

    model_config = {"protected_namespaces": ()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_bytes(path_or_bytes: str | Path | bytes) -> bytes:
    if isinstance(path_or_bytes, bytes):
        return path_or_bytes
    return Path(path_or_bytes).read_bytes()


_PDF_DATE_RE = re.compile(
    r"D?:?(?P<y>\d{4})(?P<mo>\d{2})?(?P<d>\d{2})?"
    r"(?P<h>\d{2})?(?P<mi>\d{2})?(?P<s>\d{2})?"
)


def _parse_pdf_date(raw: str | None) -> datetime | None:
    """Best-effort parse of a PDF date string (e.g. ``D:20230115101530+07'00'``)."""
    if not raw:
        return None
    m = _PDF_DATE_RE.match(str(raw).strip())
    if not m:
        return None
    try:
        return datetime(
            int(m.group("y")),
            int(m.group("mo") or 1),
            int(m.group("d") or 1),
            int(m.group("h") or 0),
            int(m.group("mi") or 0),
            int(m.group("s") or 0),
        )
    except (ValueError, TypeError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


# ---------------------------------------------------------------------------
# Signal 1: metadata anomalies (DocInfo vs XMP, dates, producer)
# ---------------------------------------------------------------------------


def _analyze_metadata(
    pdf: pikepdf.Pdf,
) -> tuple[list[Finding], dict[str, str | None]]:
    findings: list[Finding] = []

    docinfo = pdf.docinfo if pdf.docinfo is not None else {}
    producer = _str_or_none(docinfo.get("/Producer"))
    creator = _str_or_none(docinfo.get("/Creator"))
    creation_raw = _str_or_none(docinfo.get("/CreationDate"))
    mod_raw = _str_or_none(docinfo.get("/ModDate"))

    # --- XMP metadata (may disagree with DocInfo) ---
    xmp: dict[str, Any] = {}
    try:
        with pdf.open_metadata() as meta:
            for key in (
                "pdf:Producer",
                "xmp:CreatorTool",
                "dc:creator",
                "xmp:CreateDate",
                "xmp:ModifyDate",
            ):
                try:
                    if key in meta:
                        xmp[key] = meta[key]
                except (KeyError, TypeError):
                    pass
    except Exception:
        # Many scanned PDFs simply have no XMP packet; that is not a signal.
        xmp = {}

    # DocInfo vs XMP producer/creator mismatch.
    xmp_producer = _str_or_none(xmp.get("pdf:Producer"))
    xmp_creator = _str_or_none(xmp.get("xmp:CreatorTool"))
    if producer and xmp_producer and producer != xmp_producer:
        findings.append(
            Finding(
                signal="metadata.producer_mismatch",
                severity="medium",
                detail_vi=(
                    "Producer trong DocInfo và trong XMP không khớp nhau "
                    "(có thể tài liệu đã được mở/chỉnh lại bằng công cụ khác). "
                    "Đây là dấu hiệu cần kiểm tra."
                ),
                evidence={"docinfo_producer": producer, "xmp_producer": xmp_producer},
            )
        )
    if creator and xmp_creator and creator != xmp_creator:
        findings.append(
            Finding(
                signal="metadata.creator_mismatch",
                severity="low",
                detail_vi=(
                    "Creator trong DocInfo và XMP không khớp — dấu hiệu nhẹ "
                    "cần kiểm tra."
                ),
                evidence={"docinfo_creator": creator, "xmp_creator": xmp_creator},
            )
        )

    # --- Date logic ---
    cdate = _parse_pdf_date(creation_raw)
    mdate = _parse_pdf_date(mod_raw)

    if creation_raw is None and mod_raw is None:
        findings.append(
            Finding(
                signal="metadata.dates_missing",
                severity="low",
                detail_vi=(
                    "Thiếu cả CreationDate lẫn ModDate. Bản scan cũ thường thiếu "
                    "metadata nên đây chỉ là dấu hiệu nhẹ, không phải bằng chứng "
                    "giả mạo."
                ),
                evidence={},
            )
        )
    elif cdate and mdate:
        if mdate < cdate:
            findings.append(
                Finding(
                    signal="metadata.mod_before_creation",
                    severity="medium",
                    detail_vi=(
                        "Ngày sửa (ModDate) sớm hơn ngày tạo (CreationDate) — "
                        "mốc thời gian bất thường, cần kiểm tra."
                    ),
                    evidence={
                        "creation_date": creation_raw,
                        "mod_date": mod_raw,
                    },
                )
            )
        else:
            gap_days = (mdate - cdate).days
            # A large gap means the file was re-saved long after creation.
            if gap_days >= 365:
                findings.append(
                    Finding(
                        signal="metadata.large_date_gap",
                        severity="low",
                        detail_vi=(
                            f"Khoảng cách giữa ngày tạo và ngày sửa khá lớn "
                            f"(~{gap_days} ngày) — tài liệu đã được lưu lại lâu "
                            f"sau khi tạo. Dấu hiệu nhẹ cần kiểm tra."
                        ),
                        evidence={
                            "creation_date": creation_raw,
                            "mod_date": mod_raw,
                            "gap_days": gap_days,
                        },
                    )
                )

    # --- Suspicious producer / creator strings ---
    haystack = " ".join(
        filter(None, [producer, creator, xmp_producer, xmp_creator])
    ).lower()
    for needle, label in SUSPICIOUS_PRODUCER_PATTERNS:
        if needle in haystack:
            # Image editors over a court doc are the strongest of these.
            sev: Severity = (
                "high" if needle in ("photoshop", "gimp", "photopea") else "medium"
            )
            findings.append(
                Finding(
                    signal="metadata.suspicious_producer",
                    severity=sev,
                    detail_vi=(
                        f"Phần mềm tạo/sửa tài liệu là '{label}', không phải công "
                        f"cụ thông thường cho văn bản của tòa án. Cần kiểm tra "
                        f"liệu tài liệu có bị chỉnh sửa bằng phần mềm này không."
                    ),
                    evidence={"matched": label, "producer": producer, "creator": creator},
                )
            )
            break  # one suspicious-tool finding is enough

    meta_dict = {
        "producer": producer,
        "creator": creator,
        "creation_date": creation_raw,
        "mod_date": mod_raw,
    }
    return findings, meta_dict


# ---------------------------------------------------------------------------
# Signal 2: incremental updates / xref analysis
# ---------------------------------------------------------------------------


def _count_eof(raw: bytes) -> int:
    return len(re.findall(rb"%%EOF", raw))


def _count_incremental_generations(raw: bytes) -> int:
    """Count genuine incremental-update generations beyond the first save.

    A ``/Prev`` in a trailer/xref points the current cross-reference section at
    a previous one. However a single, un-edited save can still contain ``/Prev``
    in two benign cases that we must NOT count as edits:

      * *Hybrid-reference* files (MS Word et al.) place ``/Prev`` next to
        ``/XRefStm`` in the trailer — that is one generation offering two xref
        representations, not an append.

    We therefore count a ``/Prev`` only when it is a real "go to the previous
    generation" pointer, i.e. it sits near a ``trailer``/``startxref`` /xref
    stream AND is not the hybrid ``/Prev .../XRefStm`` pairing.
    """
    generations = 0
    for m in re.finditer(rb"/Prev\b", raw):
        before = raw[max(0, m.start() - 400) : m.start()]
        # Must be an xref/trailer-related /Prev, not one inside page content.
        if not (b"trailer" in before or b"/XRef" in before or b"startxref" in before):
            continue
        after = raw[m.start() : m.start() + 60]
        # Hybrid-reference trailer: "/Prev <offset> /XRefStm <offset>" -> benign.
        if b"/XRefStm" in after:
            continue
        generations += 1
    return generations


def _has_prev_xref(raw: bytes) -> bool:
    """True if the file has at least one genuine incremental-update generation."""
    return _count_incremental_generations(raw) > 0


def _trailing_eof_payload(raw: bytes) -> int:
    """Bytes of content between the second-to-last %%EOF and end of file.

    A *real* incremental update appends new objects (often KB+) after the
    previous %%EOF. A tiny payload (a few hundred bytes) is just the writer's
    finalising trailer and is not evidence of editing.
    """
    eofs = [m.start() for m in re.finditer(rb"%%EOF", raw)]
    if len(eofs) < 2:
        return 0
    return len(raw) - (eofs[-2] + len(b"%%EOF"))


def _analyze_incremental(raw: bytes) -> tuple[list[Finding], int, bool]:
    findings: list[Finding] = []
    num_eof = _count_eof(raw)
    has_prev = _has_prev_xref(raw)

    # A genuine incremental update is characterised by /Prev linking a new xref
    # to the previous one. Multiple %%EOF *without* /Prev and with only a tiny
    # trailing payload is a benign writer artifact (very common in MS Word /
    # library-generated PDFs) and must NOT inflate the risk score.
    extra_eofs = max(0, num_eof - 1)
    trailing = _trailing_eof_payload(raw)
    SUBSTANTIAL = 2048  # bytes; below this the extra section adds no real objects

    has_incremental = has_prev or (extra_eofs > 0 and trailing >= SUBSTANTIAL)

    if has_prev:
        # /Prev present -> a real incremental-update / signing chain exists.
        sev: Severity = "medium" if num_eof <= 2 else "high"
        findings.append(
            Finding(
                signal="incremental.prev_xref",
                severity=sev,
                detail_vi=(
                    f"Tệp có chuỗi cập nhật gia tăng thực sự (xref chứa /Prev, "
                    f"{num_eof} điểm %%EOF): tài liệu đã được lưu/chỉnh sửa lại "
                    f"sau khi tạo. Cần kiểm tra nội dung được thêm/sửa ở lần lưu "
                    f"sau."
                ),
                evidence={"num_eof": num_eof, "has_prev": True, "trailing_bytes": trailing},
            )
        )
    elif extra_eofs > 0 and trailing >= SUBSTANTIAL:
        # Multiple EOFs with a substantial appended block but no /Prev: unusual
        # structure worth a look, but weaker than a proper /Prev chain.
        findings.append(
            Finding(
                signal="incremental.appended_content",
                severity="medium",
                detail_vi=(
                    f"Sau điểm %%EOF có thêm khối dữ liệu đáng kể "
                    f"(~{trailing} byte) dù không thấy /Prev. Có thể nội dung đã "
                    f"được nối thêm sau khi tạo. Cần kiểm tra."
                ),
                evidence={"num_eof": num_eof, "trailing_bytes": trailing},
            )
        )
    # Multiple EOFs with tiny trailing payload and no /Prev => benign, no finding.
    return findings, num_eof, has_incremental


# ---------------------------------------------------------------------------
# Per-page PyMuPDF analysis (fonts, dimensions, annotations, scanned, seal)
# ---------------------------------------------------------------------------


def _page_dim_key(page: fitz.Page) -> tuple[int, int]:
    r = page.rect
    # Round to whole points to absorb sub-pixel differences.
    return (round(r.width), round(r.height))


def _analyze_pages(
    doc: fitz.Document,
) -> tuple[list[Finding], bool]:
    findings: list[Finding] = []
    num_pages = doc.page_count

    page_fonts: list[set[str]] = []
    page_has_text: list[bool] = []
    page_has_image: list[bool] = []
    dim_keys: list[tuple[int, int]] = []
    rotations: list[int] = []
    annot_types: dict[int, list[str]] = {}
    text_over_image_pages: list[int] = []
    seal_pages: list[int] = []

    for pno in range(num_pages):
        page = doc[pno]

        # --- fonts ---
        fonts = {f[3] for f in page.get_fonts(full=True)}  # f[3] = base font name
        page_fonts.append(fonts)

        # --- text vs image ---
        text = page.get_text("text").strip()
        has_text = len(text) >= 10  # tiny stray text doesn't count as a text layer
        page_has_text.append(has_text)

        images = page.get_images(full=True)
        has_image = len(images) > 0
        page_has_image.append(has_image)

        # text drawn on top of a full-page image -> possible overlay/doctoring
        if has_text and has_image:
            # Heuristic: a scanned page may also carry an OCR text layer, which
            # is legitimate. We only flag when there is a *large* image (likely
            # the scan background) AND a modest amount of text -> looks like a
            # patch of text laid over the scan. Keep severity low.
            big_image = any(
                (img[2] * img[3]) > 500_000 for img in images if len(img) > 3
            )
            if big_image and 0 < len(text) < 400:
                text_over_image_pages.append(pno + 1)

        # --- dimensions & rotation ---
        dim_keys.append(_page_dim_key(page))
        rotations.append(page.rotation)

        # --- annotations ---
        annots = list(page.annots() or [])
        if annots:
            types = []
            for a in annots:
                try:
                    types.append(a.type[1])  # e.g. 'FreeText', 'Redact', 'Highlight'
                except Exception:
                    types.append("Unknown")
            annot_types[pno + 1] = types

        # --- red-seal heuristic (low confidence) ---
        if _page_has_red_region(page):
            seal_pages.append(pno + 1)

    # === scanned detection ===
    any_text = any(page_has_text)
    is_scanned = (not any_text) and any(page_has_image)

    # === Signal 3: per-page font inconsistency ===
    if not is_scanned and num_pages >= 2:
        findings.extend(_font_outlier_findings(page_fonts, page_has_text))

    # === Signal 4: annotation / overlay tampering ===
    findings.extend(_annotation_findings(annot_types, text_over_image_pages))

    # === Signal 5: dimension / rotation outliers ===
    findings.extend(_dimension_findings(dim_keys, rotations))

    # === Signal 6: red-seal presence (best-effort) ===
    findings.extend(
        _seal_findings(seal_pages, num_pages, is_scanned, any(page_has_image))
    )

    return findings, is_scanned


def _font_outlier_findings(
    page_fonts: list[set[str]], page_has_text: list[bool]
) -> list[Finding]:
    findings: list[Finding] = []

    # Build document-wide font frequency, considering only text pages.
    counter: collections.Counter[str] = collections.Counter()
    text_page_indices = [i for i, ht in enumerate(page_has_text) if ht]
    if len(text_page_indices) < 2:
        return findings
    for i in text_page_indices:
        counter.update(page_fonts[i])
    if not counter:
        return findings

    # Dominant fonts = those appearing on >= 30% of text pages.
    n_text_pages = len(text_page_indices)
    dominant = {
        font for font, c in counter.items() if c >= max(2, 0.30 * n_text_pages)
    }
    if not dominant:
        return findings

    outlier_pages: list[dict[str, Any]] = []
    for i in text_page_indices:
        fonts = page_fonts[i]
        if not fonts:
            continue
        # A page that introduces font(s) not in the dominant set AND shares
        # none of the dominant fonts is the strongest outlier signal.
        novel = fonts - dominant
        shared = fonts & dominant
        if novel and not shared:
            outlier_pages.append(
                {"page": i + 1, "fonts": sorted(fonts), "novel": sorted(novel)}
            )

    if outlier_pages:
        pages = [o["page"] for o in outlier_pages]
        findings.append(
            Finding(
                signal="fonts.page_outlier",
                severity="medium",
                detail_vi=(
                    f"Trang {pages} dùng bộ phông chữ hoàn toàn khác với phần còn "
                    f"lại của tài liệu. Trang được chèn/sửa thường mang phông chữ "
                    f"mới. Cần kiểm tra các trang này."
                ),
                evidence={"outlier_pages": outlier_pages, "dominant_fonts": sorted(dominant)},
            )
        )
    return findings


def _annotation_findings(
    annot_types: dict[int, list[str]], text_over_image_pages: list[int]
) -> list[Finding]:
    findings: list[Finding] = []

    editing_annots = {"FreeText", "Redact", "Stamp", "Caret", "StrikeOut"}
    flagged: dict[int, list[str]] = {}
    for page, types in annot_types.items():
        hits = [t for t in types if t in editing_annots]
        if hits:
            flagged[page] = hits

    if flagged:
        has_redact = any("Redact" in v for v in flagged.values())
        has_freetext = any("FreeText" in v for v in flagged.values())
        sev: Severity = "high" if has_redact else "medium"
        parts = []
        if has_redact:
            parts.append("dấu vết che/xóa (Redact)")
        if has_freetext:
            parts.append("ô văn bản tự do (FreeText) chèn thêm")
        if not parts:
            parts.append("chú thích từ công cụ chỉnh sửa")
        findings.append(
            Finding(
                signal="annotation.editing_artifacts",
                severity=sev,
                detail_vi=(
                    f"Phát hiện {', '.join(parts)} trên các trang {sorted(flagged)}. "
                    f"Đây là dấu hiệu tài liệu đã được mở bằng công cụ chỉnh sửa "
                    f"và cần kiểm tra nội dung tại các trang đó."
                ),
                evidence={"annotations_by_page": flagged},
            )
        )

    if text_over_image_pages:
        findings.append(
            Finding(
                signal="annotation.text_over_image",
                severity="low",
                detail_vi=(
                    f"Trang {text_over_image_pages} có một lượng nhỏ văn bản được "
                    f"vẽ đè lên ảnh scan lớn. Điều này có thể là lớp OCR hợp lệ, "
                    f"nhưng cũng có thể là chữ dán đè — dấu hiệu nhẹ cần kiểm tra."
                ),
                evidence={"pages": text_over_image_pages},
            )
        )
    return findings


def _dimension_findings(
    dim_keys: list[tuple[int, int]], rotations: list[int]
) -> list[Finding]:
    findings: list[Finding] = []
    n = len(dim_keys)
    if n < 3:
        return findings

    # --- size outliers ---
    dim_counter = collections.Counter(dim_keys)
    common_dim, common_count = dim_counter.most_common(1)[0]
    if common_count >= max(2, 0.6 * n):  # there is a clear dominant size
        odd_pages = [i + 1 for i, d in enumerate(dim_keys) if d != common_dim]
        if 0 < len(odd_pages) <= max(1, n // 3):  # a few odd pages, not half the doc
            findings.append(
                Finding(
                    signal="dimensions.size_outlier",
                    severity="medium",
                    detail_vi=(
                        f"Trang {odd_pages} có kích thước khác với phần còn lại "
                        f"của tài liệu (kích thước phổ biến: {common_dim[0]}x"
                        f"{common_dim[1]} điểm). Trang chèn thêm thường có kích "
                        f"thước khác. Cần kiểm tra."
                    ),
                    evidence={
                        "common_dim": list(common_dim),
                        "outlier_pages": odd_pages,
                        "all_dims": [list(d) for d in dim_keys],
                    },
                )
            )

    # --- rotation outliers ---
    rot_counter = collections.Counter(rotations)
    common_rot, rcount = rot_counter.most_common(1)[0]
    if rcount >= max(2, 0.6 * n):
        odd_rot_pages = [i + 1 for i, r in enumerate(rotations) if r != common_rot]
        if 0 < len(odd_rot_pages) <= max(1, n // 3):
            findings.append(
                Finding(
                    signal="dimensions.rotation_outlier",
                    severity="low",
                    detail_vi=(
                        f"Trang {odd_rot_pages} có góc xoay khác phần còn lại "
                        f"(góc phổ biến: {common_rot}°). Dấu hiệu nhẹ cần kiểm tra."
                    ),
                    evidence={"common_rotation": common_rot, "outlier_pages": odd_rot_pages},
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Signal 6: red-seal / signature presence (heuristic, low confidence)
# ---------------------------------------------------------------------------


def _page_has_red_region(page: fitz.Page) -> bool:
    """Detect a predominantly-red region (court seal 'mộc đỏ') on a page.

    Low-confidence color-histogram heuristic only. We render a small pixmap
    and count pixels where R is clearly dominant over G and B. A real seal is
    a sizeable cluster of such pixels.
    """
    try:
        # Render at low DPI for speed; we only need a coarse color histogram.
        pix = page.get_pixmap(matrix=fitz.Matrix(0.4, 0.4), colorspace=fitz.csRGB)
    except Exception:
        return False

    n = pix.width * pix.height
    if n == 0:
        return False
    samples = pix.samples  # bytes, len = n * pix.n
    stride = pix.n
    red_pixels = 0
    # Sample every Nth pixel to keep this cheap on large pages.
    step = max(1, n // 20000)
    counted = 0
    for idx in range(0, n, step):
        base = idx * stride
        r = samples[base]
        g = samples[base + 1]
        b = samples[base + 2]
        counted += 1
        # "Court-seal red": strong red, weak green/blue, not near-black.
        if r > 120 and r - g > 60 and r - b > 60:
            red_pixels += 1
    if counted == 0:
        return False
    return (red_pixels / counted) > 0.004  # >0.4% red pixels = plausible seal


def _seal_findings(
    seal_pages: list[int], num_pages: int, is_scanned: bool, has_any_image: bool
) -> list[Finding]:
    findings: list[Finding] = []

    # Only meaningful when the document carries images at all (a seal is visual).
    if not has_any_image:
        return findings

    if not seal_pages:
        findings.append(
            Finding(
                signal="seal.absent_low_confidence",
                severity="low",
                detail_vi=(
                    "Không phát hiện vùng màu đỏ nổi bật (có thể là mộc đỏ/con "
                    "dấu) trên bất kỳ trang nào. Văn bản tòa án thường có mộc đỏ; "
                    "việc thiếu là dấu hiệu nhẹ cần kiểm tra. LƯU Ý: đây chỉ là "
                    "ước lượng theo màu sắc, độ tin cậy thấp — bản đen trắng/"
                    "scan xám sẽ không có màu đỏ dù vẫn hợp lệ."
                ),
                evidence={"confidence": "low", "seal_pages": []},
            )
        )
    # Presence of a seal is normal -> no finding (kept silent on purpose).
    return findings


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(findings: list[Finding]) -> tuple[RiskLevel, int]:
    score = 0
    for f in findings:
        score += SEVERITY_WEIGHT[f.severity]
    score = min(score, 100)

    has_high = any(f.severity == "high" for f in findings)
    n_medium = sum(1 for f in findings if f.severity == "medium")

    # Risk level: a single high signal -> at least medium; high requires either
    # a high-severity signal combined with corroboration, or a high score.
    if has_high and (score >= 50 or n_medium >= 1):
        level: RiskLevel = "high"
    elif has_high or n_medium >= 2 or score >= 35:
        level = "medium"
    else:
        level = "low"
    return level, score


def _build_summary(
    level: RiskLevel, findings: list[Finding], is_scanned: bool, num_pages: int
) -> str:
    scan_note = "tài liệu là bản scan ảnh, " if is_scanned else ""
    if level == "low":
        return (
            f"Rủi ro THẤP: {scan_note}không phát hiện dấu hiệu giả mạo đáng kể "
            f"trên {num_pages} trang (các dấu hiệu nhẹ nếu có là bình thường với "
            f"loại tài liệu này)."
        )
    n = len(findings)
    top = "; ".join(
        f.signal for f in sorted(findings, key=lambda x: x.severity, reverse=True)[:3]
    )
    label = "TRUNG BÌNH" if level == "medium" else "CAO"
    return (
        f"Rủi ro {label}: {scan_note}phát hiện {n} dấu hiệu cần kiểm tra "
        f"(nổi bật: {top}). Cần thẩm định thủ công, đây không phải kết luận giả mạo."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_pdf(path_or_bytes: str | Path | bytes) -> ForgeryReport:
    """Analyze a judgment PDF for tampering/forgery signals.

    Args:
        path_or_bytes: filesystem path (str/Path) or the raw PDF bytes.

    Returns:
        A :class:`ForgeryReport`. The function is conservative: a normal scanned
        court PDF should yield ``risk_level == "low"``.
    """
    raw = _to_bytes(path_or_bytes)

    findings: list[Finding] = []
    meta_strings: dict[str, str | None] = {
        "producer": None,
        "creator": None,
        "creation_date": None,
        "mod_date": None,
    }

    # --- pikepdf: metadata + (raw bytes) incremental ---
    try:
        with pikepdf.open(io.BytesIO(raw)) as pdf:
            meta_findings, meta_strings = _analyze_metadata(pdf)
            findings.extend(meta_findings)
    except Exception as exc:  # corrupt/encrypted -> a signal in itself
        findings.append(
            Finding(
                signal="file.pikepdf_open_failed",
                severity="medium",
                detail_vi=(
                    "Không mở được tệp bằng pikepdf để đọc metadata (tệp có thể "
                    f"hỏng, mã hóa hoặc cấu trúc bất thường: {type(exc).__name__}). "
                    "Cần kiểm tra."
                ),
                evidence={"error": str(exc)[:200]},
            )
        )

    inc_findings, num_eof, has_incremental = _analyze_incremental(raw)
    findings.extend(inc_findings)

    # --- PyMuPDF: per-page analysis ---
    is_scanned = False
    num_pages = 0
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            num_pages = doc.page_count
            page_findings, is_scanned = _analyze_pages(doc)
            findings.extend(page_findings)
    except Exception as exc:
        findings.append(
            Finding(
                signal="file.fitz_open_failed",
                severity="medium",
                detail_vi=(
                    "Không mở được tệp bằng PyMuPDF để phân tích trang "
                    f"({type(exc).__name__}). Cần kiểm tra."
                ),
                evidence={"error": str(exc)[:200]},
            )
        )

    metadata = PdfMetadata(
        producer=meta_strings["producer"],
        creator=meta_strings["creator"],
        creation_date=meta_strings["creation_date"],
        mod_date=meta_strings["mod_date"],
        num_pages=num_pages,
        num_eof=num_eof,
        has_incremental=has_incremental,
    )

    level, score = _aggregate(findings)
    summary = _build_summary(level, findings, is_scanned, num_pages)

    return ForgeryReport(
        risk_level=level,
        risk_score=score,
        is_scanned=is_scanned,
        findings=findings,
        metadata=metadata,
        summary_vi=summary,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _render_human(report: ForgeryReport, path: str) -> str:
    lines = [
        f"=== Forgery report: {path} ===",
        f"risk_level : {report.risk_level.upper()}  (score {report.risk_score}/100)",
        f"is_scanned : {report.is_scanned}",
        f"pages      : {report.metadata.num_pages}  |  %%EOF: {report.metadata.num_eof}"
        f"  |  incremental: {report.metadata.has_incremental}",
        f"producer   : {report.metadata.producer}",
        f"creator    : {report.metadata.creator}",
        f"created    : {report.metadata.creation_date}",
        f"modified   : {report.metadata.mod_date}",
        f"findings   : {len(report.findings)}",
    ]
    for f in report.findings:
        lines.append(f"  - [{f.severity.upper():6}] {f.signal}")
        lines.append(f"      {f.detail_vi}")
    lines.append(f"summary    : {report.summary_vi}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Layer-4 forgery/tampering analysis for judgment PDFs."
    )
    parser.add_argument("--pdf", required=True, help="path to the PDF to analyze")
    parser.add_argument(
        "--json", action="store_true", help="emit the raw JSON report only"
    )
    args = parser.parse_args(argv)

    report = analyze_pdf(args.pdf)
    if args.json:
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    else:
        print(_render_human(report, args.pdf))
    return 0


if __name__ == "__main__":
    sys.exit(main())
