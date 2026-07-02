"""Sectioner invariants: sections always tile [0, len(text)) with no gaps."""

from corpus.normalize import flatten_for_matching, normalize_text
from corpus.sectioner import segment_sections

_JUDGMENT = (
    "TÒA ÁN NHÂN DÂN THỊ XÃ DĨ AN, TỈNH BÌNH DƯƠNG\n"
    "Bản án số: 17/2018/HS-ST Ngày 26-01-2018\n"
    "NHÂN DANH NƯỚC CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
    "NỘI DUNG VỤ ÁN\n"
    "bị cáo Nguyễn Vĩnh H đã thực hiện hành vi trộm cắp tài sản.\n"
    "NHẬN ĐỊNH CỦA TÒA ÁN\n"
    "xét thấy hành vi của bị cáo là nguy hiểm cho xã hội.\n"
    "QUYẾT ĐỊNH\n"
    "Xử phạt bị cáo Nguyễn Vĩnh H 02 (hai) năm tù."
)


def _assert_tiling(text, sections):
    assert sections, "must return at least one section"
    assert sections[0][1] == 0, "first section starts at 0"
    assert sections[-1][2] == len(text), "last section ends at len(text)"
    for a, b in zip(sections, sections[1:]):
        assert a[2] == b[1], f"gap/overlap between {a} and {b}"


def test_full_judgment_tiles_contiguously():
    flat = flatten_for_matching(normalize_text(_JUDGMENT))
    sections = segment_sections(flat)
    _assert_tiling(flat, sections)
    names = [n for n, _, _ in sections]
    assert names == ["PREAMBLE", "NHAN_DANH", "NOI_DUNG", "NHAN_DINH", "QUYET_DINH"]


def test_no_anchor_is_single_section():
    text = "một đoạn văn bản không có tiêu đề mục nào cả"
    sections = segment_sections(text)
    assert sections == [("PREAMBLE", 0, len(text))]


def test_missing_middle_anchor_still_tiles():
    # no NHẬN ĐỊNH section
    text = ("TÒA ÁN NHÂN DÂN A NHÂN DANH NƯỚC ... NỘI DUNG VỤ ÁN diễn biến "
            "QUYẾT ĐỊNH Xử phạt.")
    sections = segment_sections(text)
    _assert_tiling(text, sections)
    assert "NHAN_DINH" not in {n for n, _, _ in sections}


def test_empty_text():
    assert segment_sections("") == [("PREAMBLE", 0, 0)]
