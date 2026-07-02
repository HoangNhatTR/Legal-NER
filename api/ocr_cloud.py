"""OCR đám mây qua OCR.space API — NHANH hơn EasyOCR CPU nhiều (xử lý song song).

Vì sao: EasyOCR trên CPU ~50s/trang (8 phút/10 trang). OCR.space xử lý mỗi trang
~3-8s và ta gửi SONG SONG các trang -> tổng ~10-60s.

Cách dùng: đặt env ``LEGAL_NER_OCRSPACE_KEY`` (lấy free tại https://ocr.space/ocrapi
— đăng ký 1 phút, 25.000 lượt/tháng). Không có key -> service tự dùng EasyOCR offline.

Luồng: render từng trang PDF bằng PyMuPDF -> JPEG nén nhỏ (<1MB/trang) -> POST lên
OCR.space (language=vie) -> ghép ParsedText theo thứ tự trang.

⚠ Gửi ảnh trang lên dịch vụ ngoài (OCR.space) — chỉ nên dùng cho bản án ĐÃ CÔNG BỐ
(công khai). Độ chính xác tiếng Việt có thể khác EasyOCR; đây là đánh đổi tốc độ.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import requests

OCRSPACE_URL = "https://api.ocr.space/parse/image"


@dataclass
class OcrResult:
    text: str
    engine: str
    device: str
    page_count: int
    mean_confidence: float | None = None
    quality_warnings: int = 0
    skipped_pages: int = 0
    per_page: list[dict] = field(default_factory=list)


class OcrError(Exception):
    """OCR đám mây thất bại."""


def ocrspace_available() -> bool:
    return bool(os.environ.get("LEGAL_NER_OCRSPACE_KEY"))


def _render_jpeg(page: fitz.Page, dpi: int) -> bytes:
    """Render 1 trang thành JPEG (nén để < ~1MB cho giới hạn free tier)."""
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    try:
        return pix.tobytes("jpeg", jpg_quality=70)
    except TypeError:  # bản PyMuPDF cũ không nhận jpg_quality
        return pix.tobytes("jpeg")


def _ocr_one_page(idx: int, img: bytes, api_key: str, language: str, engine: int) -> tuple[int, str]:
    """Gọi OCR.space cho 1 trang. Trả về (idx, text).

    Engine 2 (auto-detect) đọc tiếng Việt rất tốt (chuẩn dấu thanh) -> mặc định.
    Engine 1 KHÔNG hỗ trợ tiếng Việt. Chỉ gửi 'language' khi được chỉ định.
    """
    form = {
        "apikey": api_key,
        "OCREngine": str(engine),
        "isOverlayRequired": "false",
        "scale": "true",
        "detectOrientation": "true",
    }
    if language:
        form["language"] = language
    try:
        r = requests.post(
            OCRSPACE_URL,
            files={"file": (f"page{idx}.jpg", img, "image/jpeg")},
            data=form,
            timeout=90,
        )
        r.raise_for_status()
        j = r.json()
        if j.get("IsErroredOnProcessing"):
            msg = j.get("ErrorMessage") or j.get("ErrorDetails") or "unknown"
            if isinstance(msg, list):
                msg = "; ".join(msg)
            raise OcrError(f"OCR.space lỗi (trang {idx + 1}): {msg}")
        results = j.get("ParsedResults") or []
        text = results[0].get("ParsedText", "") if results else ""
        return idx, text
    except requests.RequestException as exc:
        raise OcrError(f"không gọi được OCR.space (trang {idx + 1}): {exc}") from exc


def run_ocrspace_on_pdf(
    data: bytes,
    *,
    dpi: int | None = None,
    language: str | None = None,
    engine: int | None = None,
    max_pages: int = 40,
    workers: int = 4,
) -> OcrResult:
    """OCR toàn bộ trang PDF qua OCR.space (song song). Raises OcrError khi lỗi."""
    api_key = os.environ.get("LEGAL_NER_OCRSPACE_KEY", "")
    if not api_key:
        raise OcrError("chưa đặt LEGAL_NER_OCRSPACE_KEY")
    dpi = dpi or int(os.environ.get("LEGAL_NER_OCRSPACE_DPI", "150"))
    # Engine 2 = auto-detect, đọc tiếng Việt tốt -> không cần language. (Engine 1
    # không hỗ trợ tiếng Việt.) Đặt LEGAL_NER_OCRSPACE_LANG nếu muốn ép mã cụ thể.
    if language is None:
        language = os.environ.get("LEGAL_NER_OCRSPACE_LANG", "")
    engine = engine or int(os.environ.get("LEGAL_NER_OCRSPACE_ENGINE", "2"))

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise OcrError(f"không mở được PDF: {exc}") from exc

    n = min(doc.page_count, max_pages)
    images = [(i, _render_jpeg(doc.load_page(i), dpi)) for i in range(n)]
    doc.close()

    # Gửi song song (giới hạn workers để không vượt rate-limit free tier).
    texts: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(_ocr_one_page, i, img, api_key, language, engine) for i, img in images]
        for f in futures:
            idx, text = f.result()  # lỗi -> raise OcrError
            texts[idx] = text

    per_page = []
    parts = []
    skipped = 0
    for i in range(n):
        t = (texts.get(i) or "").strip()
        if t:
            parts.append(t)
        else:
            skipped += 1
        per_page.append({"page_index": i, "engine": "ocr.space", "skipped": not bool(t)})

    full = "\n\n".join(parts)
    if not full.strip():
        raise OcrError("OCR.space không nhận được chữ nào")
    return OcrResult(
        text=full,
        engine=f"ocr.space (engine {engine})",
        device="cloud",
        page_count=n,
        skipped_pages=skipped,
        per_page=per_page,
    )
