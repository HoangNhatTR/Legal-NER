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


def test_corpus_check_khong_vien_dan():
    report = corpus_check({})
    assert report["status"] == "ok"
    assert report["checked"] == 0
