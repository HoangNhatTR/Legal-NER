"""Unit tests cho api/corpus_check.py — kiểm chứng viện dẫn trên kho luật lớn.

Không gọi mạng: stub requests.post.
"""
from __future__ import annotations

import api.corpus_check as cc
from api.corpus_check import (
    _law_matches_title,
    corpus_check,
    extract_citation_pairs,
)


def _ents(*texts: str) -> list[dict]:
    return [{"text": t} for t in texts]


GROUPED = {
    "LEGAL_BASIS": _ents(
        "điểm c khoản 1 Điều 250 của Bộ luật Hình sự năm 2015",
        "Điều 468 Bộ luật Dân sự năm 2015",
    ),
    "ARTICLE": _ents("Điều 250"),
    "LAW_NAME": _ents("Bộ luật Hình sự"),
}


# ── extract_citation_pairs ─────────────────────────────────────────────────────

def test_boc_cap_dieu_luat_tu_legal_basis():
    pairs = extract_citation_pairs(GROUPED)
    assert {"article": 250, "law": "Bộ luật Hình sự"} in [
        {"article": p["article"], "law": p["law"].split(" năm")[0].strip()} for p in pairs
    ]
    assert any(p["article"] == 468 and "Dân sự" in p["law"] for p in pairs)


def test_dieu_tran_ghep_law_name_mac_dinh():
    grouped = {"ARTICLE": _ents("Điều 51"), "LAW_NAME": _ents("Bộ luật Hình sự")}
    pairs = extract_citation_pairs(grouped)
    assert pairs == [{"article": 51, "law": "Bộ luật Hình sự"}]


def test_khong_co_gi_tra_rong():
    assert extract_citation_pairs({}) == []


def test_khu_trung_cap():
    grouped = {"LEGAL_BASIS": _ents(
        "Điều 250 Bộ luật Hình sự", "khoản 1 Điều 250 Bộ luật Hình sự năm 2015",
    )}
    pairs = extract_citation_pairs(grouped)
    assert len([p for p in pairs if p["article"] == 250]) == 1


# ── _law_matches_title ─────────────────────────────────────────────────────────

def test_khop_ban_goc_va_van_ban_hop_nhat():
    assert _law_matches_title("Bộ luật Hình sự năm 2015", "Bộ luật Hình sự")
    assert _law_matches_title(
        "Bộ luật Hình sự", "Văn bản hợp nhất 11/VBHN-VPQH hợp nhất Bộ luật Hình sự"
    )
    assert not _law_matches_title("Bộ luật Dân sự", "Bộ luật Hình sự")


def test_khong_nham_luat_voi_luat_to_tung():
    """Token 'bộ luật hình sự' là tập con của 'bộ luật tố tụng hình sự' —
    discriminator 'tố tụng' phải chặn cả 2 chiều (bug thật từ e2e đợt 3)."""
    assert not _law_matches_title(
        "Bộ luật Hình sự năm 1999",
        "Văn bản hợp nhất 104/VBHN-VPQH năm 2025 hợp nhất Bộ luật Tố tụng hình sự",
    )
    assert not _law_matches_title(
        "Bộ luật Tố tụng Hình sự", "Bộ luật Hình sự",
    )
    assert _law_matches_title(
        "Bộ luật Tố tụng Hình sự năm 2015",
        "Văn bản hợp nhất 46/VBHN-VPQH hợp nhất Bộ Luật Tố tụng hình sự",
    )


# ── corpus_check với stub HTTP ─────────────────────────────────────────────────

class _Resp:
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        pass

    def json(self):
        return {"chunks": self._chunks}


def test_corpus_check_found_va_not_found(monkeypatch):
    def _fake_post(url, headers=None, json=None, timeout=None):
        if "Điều 250" in json["query"]:
            return _Resp([{
                "article": "Điều 250", "title": "Bộ luật Hình sự",
                "doc_number": "100/2015/QH13",
            }])
        return _Resp([{"article": "Điều 1", "title": "Luật gì đó khác"}])

    monkeypatch.setattr(cc.requests, "post", _fake_post)
    report = corpus_check(GROUPED)

    assert report["status"] == "ok"
    assert report["found"] >= 1
    assert report["not_found"] >= 1
    st = {r["article"]: r["status"] for r in report["results"]}
    assert st[250] == "found"
    assert st[468] == "not_found"


def test_corpus_check_module1_tat_thi_skipped(monkeypatch):
    def _fake_post(*a, **kw):
        raise cc.requests.ConnectionError("connection refused")

    monkeypatch.setattr(cc.requests, "post", _fake_post)
    report = corpus_check(GROUPED)

    assert report["status"] == "skipped"
    assert "Module 1" in report["note"]
    assert all(r["status"] == "error" for r in report["results"])


def test_loi_ben_vung_mot_vien_dan_khong_huy_ca_loat(monkeypatch):
    """Timeout LẶP LẠI (cả retry) ở 1 viện dẫn → item đó 'error', viện dẫn khác vẫn kiểm."""

    def _fake_post(url, headers=None, json=None, timeout=None):
        if "Điều 250" in json["query"]:
            raise cc.requests.ReadTimeout("quá 120s")  # lỗi cả 2 lần (retry vẫn fail)
        return _Resp([{
            "article": f"Điều {json['query'].split()[-1]}",
            "title": "Bộ luật Dân sự", "doc_number": "91/2015/QH13",
        }])

    monkeypatch.setattr(cc.requests, "post", _fake_post)
    report = corpus_check(GROUPED)

    assert report["status"] == "ok"
    by_art = {r["article"]: r["status"] for r in report["results"]}
    assert by_art[250] == "error"
    assert by_art[468] in ("found", "not_found")


def test_timeout_thoang_qua_duoc_retry_cuu(monkeypatch):
    """Lỗi 1 lần đầu → retry lần 2 thành công → item vẫn được kiểm bình thường."""
    calls = {"n": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise cc.requests.ReadTimeout("lượt đầu chậm")
        return _Resp([{
            "article": f"Điều {json['query'].split()[-1]}",
            "title": "Bộ luật Hình sự", "doc_number": "100/2015/QH13",
        }])

    monkeypatch.setattr(cc.requests, "post", _fake_post)
    report = corpus_check(GROUPED)

    assert report["status"] == "ok"
    assert all(r["status"] != "error" for r in report["results"])


def test_corpus_check_khong_vien_dan():
    report = corpus_check({})
    assert report["status"] == "ok"
    assert report["checked"] == 0
