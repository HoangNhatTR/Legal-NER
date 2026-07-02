"""Test toggle bảo mật OCR đám mây — LEGAL_NER_OCR_CLOUD=0 ép offline."""
from __future__ import annotations

from api.ocr_cloud import ocrspace_available


def test_co_key_thi_bat(monkeypatch):
    monkeypatch.setenv("LEGAL_NER_OCRSPACE_KEY", "k123")
    monkeypatch.delenv("LEGAL_NER_OCR_CLOUD", raising=False)
    assert ocrspace_available()


def test_khong_key_thi_tat(monkeypatch):
    monkeypatch.delenv("LEGAL_NER_OCRSPACE_KEY", raising=False)
    monkeypatch.delenv("LEGAL_NER_OCR_CLOUD", raising=False)
    assert not ocrspace_available()


def test_tat_tuong_minh_du_co_key(monkeypatch):
    """Tài liệu nhạy cảm: LEGAL_NER_OCR_CLOUD=0 ép offline mà không cần xóa key."""
    monkeypatch.setenv("LEGAL_NER_OCRSPACE_KEY", "k123")
    for off in ("0", "false", "off"):
        monkeypatch.setenv("LEGAL_NER_OCR_CLOUD", off)
        assert not ocrspace_available()
