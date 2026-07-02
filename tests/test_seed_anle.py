"""Test parser + trích text án lệ (scripts/seed_anle.py) — offline, không mạng."""
import io
import zipfile

from scripts.seed_anle import (
    _docx_text,
    _extract_text,
    _related_articles,
    parse_anle,
    record_from_meta,
)

SAMPLE = """ÁN LỆ SỐ 30/2020/AL
Về hành vi cố ý điều khiển phương tiện giao thông chèn lên bị hại

Nguồn án lệ: Bản án hình sự phúc thẩm số 280/2019/HS-PT.
Khái quát nội dung án lệ: Sau khi gây tai nạn, bị cáo cố ý điều khiển xe chèn lên bị hại.
Quy định của pháp luật liên quan đến án lệ: Điều 123 và Điều 134 Bộ luật Hình sự năm 2015.
Từ khóa của án lệ: "Giết người"; "Cố ý"

NỘI DUNG VỤ ÁN:
Khoảng 22 giờ, sau khi va chạm, bị cáo đã lùi xe và tiếp tục điều khiển xe chèn qua người bị hại.
NHẬN ĐỊNH CỦA TÒA ÁN:
Hành vi của bị cáo cấu thành tội Giết người.
"""


def test_parse_anle_basic():
    c = parse_anle(SAMPLE, url="http://x")
    assert c is not None
    assert c["case_number"] == "30/2020/AL"
    assert "Giết người" in c["crimes"]
    assert "Điều 123" in c["articles"] and "Điều 134" in c["articles"]
    assert c["case_type"] == "hình sự"
    assert "chèn" in c["facts_summary"].lower()
    assert c["_title"].startswith("Án lệ số 30/2020/AL")
    assert c["court"] == "Tòa án nhân dân tối cao"


def test_parse_anle_keywords_fallback_semicolon():
    text = SAMPLE.replace('Từ khóa của án lệ: "Giết người"; "Cố ý"',
                          "Từ khóa của án lệ: Giết người; Cố ý điều khiển")
    c = parse_anle(text)
    assert "Giết người" in c["crimes"]


def test_parse_non_anle_returns_none():
    assert parse_anle("Thông tư 02/2019/TT-BCT quy định về điện gió.") is None
    assert parse_anle("") is None


# ── trích text file gốc (DOCX / phân loại) ──────────────────────────────────

def _make_docx(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "word/document.xml",
            f"<w:document><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )
    return buf.getvalue()


def test_docx_text_extracts_paragraph():
    data = _make_docx("Án lệ số 50/2021/AL về tranh chấp hợp đồng")
    assert "Án lệ số 50/2021/AL" in _docx_text(data)


def test_extract_text_classifies_docx():
    data = _make_docx("Nội dung án lệ ABC")
    text, kind = _extract_text(data, "application/vnd...wordprocessingml", do_ocr=False)
    assert kind == "docx" and "Nội dung án lệ ABC" in text


def test_extract_text_ole2_doc_is_unknown():
    # .doc cũ (OLE2 magic) -> 'unknown', KHÔNG cố mở như docx
    text, kind = _extract_text(b"\xd0\xcf\x11\xe0" + b"junk" * 20, "wordprocessingml", do_ocr=False)
    assert kind == "unknown" and text == ""


def test_extract_text_bad_zip_is_docx_fail():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("theme/theme1.xml", "<x/>")  # zip nhưng thiếu word/document.xml
    text, kind = _extract_text(buf.getvalue(), "", do_ocr=False)
    assert kind == "docx-fail" and text == ""


def test_related_articles_prefers_section():
    full = (
        "Khái quát nội dung án lệ: ...\n"
        "Quy định của pháp luật liên quan đến án lệ: Điều 468 và Điều 463 "
        "Bộ luật Dân sự năm 2015.\n"
        "NỘI DUNG VỤ ÁN: Tòa viện dẫn lan man Điều 999 và Điều 888.\n"
    )
    arts = _related_articles(full, "Về tranh chấp hợp đồng vay")
    assert arts == ["Điều 468", "Điều 463"]  # chỉ điều CHÍNH, bỏ 999/888 rải rác


def test_related_articles_fallback_to_title():
    assert _related_articles("không có mục liên quan", "Về abc Điều 5") == ["Điều 5"]


def test_record_from_meta_combines_title_and_facts():
    full = "NỘI DUNG VỤ ÁN: Bị đơn vay 500 triệu không trả. NHẬN ĐỊNH: ..."
    rec = record_from_meta("50/2021/AL", "Về tranh chấp hợp đồng vay tài sản", full_text=full)
    # khoá so khớp gồm cả nguyên tắc (tiêu đề) lẫn diễn biến
    assert "tranh chấp hợp đồng vay" in rec["facts_summary"].lower()
    assert "vay 500 triệu" in rec["facts_summary"].lower()
