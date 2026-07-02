"""Text normalization: legacy-font fixes, detached marks, flattening."""

import unicodedata

from corpus.normalize import (
    fix_legacy_chars,
    flatten_for_matching,
    normalize_text,
    reattach_combining_marks,
)


def test_legacy_oi_to_u_horn():
    assert fix_legacy_chars("Dƣơng") == "Dương"
    assert fix_legacy_chars("ƢU") == "ƯU"


def test_reattach_detached_combining_mark():
    # "Thi" + space + U+0323 (dot below) -> glue the mark back (NFC done later).
    out = reattach_combining_marks("Thi ̣ Kim")
    assert unicodedata.normalize("NFC", out) == "Thị Kim"
    # full pipeline (with NFC) yields the precomposed form
    assert "ị" in normalize_text("Thi ̣ Kim")


def test_normalize_collapses_spaces_and_blank_lines():
    out = normalize_text("a    b\n\n\n\nc   ")
    assert "a b" in out
    assert "\n\n\n" not in out


def test_flatten_drops_page_number_lines():
    text = "dòng một\n12\ndòng hai"
    flat = flatten_for_matching(text)
    assert flat == "dòng một dòng hai"


def test_flatten_is_single_line():
    flat = flatten_for_matching("a\nb\nc")
    assert "\n" not in flat
