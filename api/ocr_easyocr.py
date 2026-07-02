"""OCR cho bản án SCAN bằng EasyOCR — chạy thuần Python, hợp Windows.

Vì sao adapter này (thay vì opendataloader-hybrid / Phase-1 OCR):
  * opendataloader-hybrid là pipeline Java + docling + 1 server riêng (cổng 5002),
    cấu hình kiểu Linux — phức tạp trên Windows.
  * Phase-1 (`OCR/phase1`) không có trong repo đã clone.
EasyOCR chỉ cần `pip install easyocr` (kéo theo torch đã có); render trang PDF
bằng PyMuPDF rồi nhận chữ trực tiếp.

Luồng: PDF bytes -> fitz render từng trang thành ảnh (DPI cao) -> EasyOCR đọc ->
ghép text theo thứ tự trên-xuống. Text trả về SẼ được normalize y như các backend
khác (corpus.normalize) trước khi đưa vào NER.

Lưu ý: chậm trên CPU (~1-2 phút/trang lần đầu do tải model), độ chính xác thấp hơn
bản có lớp chữ — luôn kèm cảnh báo "verify citations".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import fitz  # PyMuPDF

# Model EasyOCR nạp MỘT lần rồi tái dùng (nặng).
_READER = None
_READER_LANGS: tuple[str, ...] = ()


@dataclass
class OcrResult:
    text: str
    engine: str
    device: str  # "cuda" | "cpu"
    page_count: int
    mean_confidence: float | None
    quality_warnings: int = 0
    skipped_pages: int = 0
    per_page: list[dict] = field(default_factory=list)


class OcrError(Exception):
    """OCR thất bại không thể phục hồi."""


def _get_reader(langs: tuple[str, ...], use_gpu: bool):
    """Khởi tạo (hoặc tái dùng) EasyOCR Reader. Lần đầu sẽ TẢI MODEL (~vài trăm MB)."""
    global _READER, _READER_LANGS
    if _READER is not None and _READER_LANGS == langs:
        return _READER
    try:
        import easyocr
    except ImportError as exc:  # pragma: no cover
        raise OcrError(
            "Chưa cài EasyOCR. Chạy: pip install easyocr"
        ) from exc
    try:
        _READER = easyocr.Reader(list(langs), gpu=use_gpu)
    except Exception:  # GPU lỗi -> CPU
        _READER = easyocr.Reader(list(langs), gpu=False)
    _READER_LANGS = langs
    return _READER


def run_easyocr_on_pdf(
    data: bytes,
    *,
    langs: tuple[str, ...] = ("vi", "en"),
    dpi: int = 220,
    use_gpu: bool | None = None,
    max_pages: int = 40,
) -> OcrResult:
    """OCR toàn bộ trang PDF bằng EasyOCR -> text + tóm tắt. Raises OcrError khi lỗi."""
    if use_gpu is None:
        use_gpu = os.environ.get("LEGAL_NER_OCR_USE_GPU", "0") == "1"

    reader = _get_reader(langs, use_gpu)
    device = "cuda" if use_gpu else "cpu"

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise OcrError(f"không mở được PDF: {exc}") from exc

    n = min(doc.page_count, max_pages)
    parts: list[str] = []
    confs: list[float] = []
    per_page: list[dict] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i in range(n):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png = pix.tobytes("png")
        try:
            # detail=1 -> [(bbox, text, conf)]; paragraph=False để giữ dòng.
            results = reader.readtext(png, detail=1, paragraph=False)
        except Exception as exc:
            raise OcrError(f"EasyOCR lỗi ở trang {i + 1}: {exc}") from exc

        # sắp theo y (trên -> dưới) rồi x (trái -> phải) để giữ thứ tự đọc
        def _key(r):
            box = r[0]
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            return (round(min(ys) / 18), min(xs))  # gom dòng theo ~18px

        results = sorted(results, key=_key)
        page_lines = [r[1] for r in results if (r[1] or "").strip()]
        page_confs = [float(r[2]) for r in results if len(r) > 2 and isinstance(r[2], (int, float))]
        confs.extend(page_confs)
        text = "\n".join(page_lines)
        if text.strip():
            parts.append(text)
        per_page.append(
            {
                "page_index": i,
                "engine": "easyocr",
                "n_blocks": len(results),
                "mean_confidence": round(sum(page_confs) / len(page_confs), 4) if page_confs else None,
                "skipped": not bool(text.strip()),
                "quality": "good" if (page_confs and sum(page_confs) / len(page_confs) > 0.5) else "low",
            }
        )

    doc.close()
    full = "\n\n".join(parts)
    if not full.strip():
        raise OcrError("EasyOCR không nhận được chữ nào (ảnh quá mờ?)")

    return OcrResult(
        text=full,
        engine="easyocr",
        device=device,
        page_count=n,
        mean_confidence=round(sum(confs) / len(confs), 4) if confs else None,
        quality_warnings=sum(1 for p in per_page if p["quality"] == "low"),
        skipped_pages=sum(1 for p in per_page if p["skipped"]),
        per_page=per_page,
    )
