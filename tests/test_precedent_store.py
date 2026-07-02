"""Test kho tiền lệ + engine so khớp (api/precedent_store.py) — offline.

Không cần mạng: thêm tiền lệ với vector tường minh, kiểm tra lọc tội danh/điều
luật + xếp hạng cosine. embed_case test riêng bằng monkeypatch _embed.
"""
import numpy as np
import pytest

from api import precedent_store as ps
from api.precedent_store import (
    PrecedentStore,
    build_case,
    crime_tokens,
    matching_key,
    normalize_articles,
)


# ── trích đặc trưng ──────────────────────────────────────────────────────────

def test_normalize_articles_dedup():
    arts = normalize_articles("áp dụng Điều 250 và Điều 51", "Điều 250 BLHS")
    assert arts == ["Điều 250", "Điều 51"]


def test_build_case_from_ner():
    grouped = {
        "CRIME": [{"text": "Vận chuyển trái phép chất ma túy"}],
        "ARTICLE": [{"text": "Điều 250"}, {"text": "Điều 250"}],
        "LEGAL_BASIS": [{"text": "Áp dụng điểm c khoản 1 Điều 250 Bộ luật Hình sự"}],
        "PENALTY": [{"text": "02 năm tù"}],
        "COURT": [{"text": "TAND huyện X"}],
    }
    text = "NỘI DUNG VỤ ÁN: Bị cáo vận chuyển. NHẬN ĐỊNH: ..."
    case = build_case({"case_number": "17/2018/HS-ST", "case_type": "hình sự"}, grouped, text)
    assert case["crimes"] == ["Vận chuyển trái phép chất ma túy"]
    assert "Điều 250" in case["articles"]
    assert case["penalties"] == ["02 năm tù"]
    assert case["court"] == "TAND huyện X"
    assert "vận chuyển" in case["facts_summary"].lower()


def test_crime_tokens_overlap():
    a = crime_tokens(["Trộm cắp tài sản"])
    b = crime_tokens(["Tội trộm cắp tài sản"])  # 'tội' bị loại
    assert a & b  # vẫn trùng ở 'trộm','cắp','tài','sản'


# ── store + match ────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    return PrecedentStore(tmp_path / "prec.db")


def _add(store, num, articles, vec, source="confirmed", crimes=None):
    case = {
        "case_number": num, "court": "", "case_type": "hình sự",
        "crimes": crimes or [], "articles": articles, "penalties": [],
        "facts_summary": "", "full_text": "",
    }
    return store.add(case, np.asarray(vec, dtype=np.float32), source=source)


def test_add_and_count(store):
    _add(store, "01/2020/HS-ST", ["Điều 100"], [1, 0, 0])
    _add(store, "02/2020/HS-ST", ["Điều 250"], [0, 1, 0], source="an_le")
    assert store.count() == 2
    assert store.count(source="an_le") == 1
    assert store.count(source="confirmed") == 1


def test_match_filters_by_law_and_ranks_cosine(store):
    _add(store, "A", ["Điều 100"], [1, 0, 0])           # không liên quan
    _add(store, "B", ["Điều 250"], [0, 1, 0])           # liên quan, cosine thấp
    _add(store, "C", ["Điều 250"], [0, 0, 1])           # liên quan, cosine cao
    case = {"articles": ["Điều 250"], "crimes": []}
    q = np.array([0, 0.4, 0.9], dtype=np.float32)        # gần C hơn B
    hits = store.match(case, q, top_k=2)
    nums = [h["case_number"] for h in hits]
    assert nums == ["C", "B"]                            # A bị loại (khác điều luật)
    assert all(h["related_by_law"] for h in hits)


def test_match_widens_when_too_few_related(store):
    _add(store, "A", ["Điều 100"], [1, 0, 0])
    _add(store, "B", ["Điều 250"], [0, 1, 0])
    case = {"articles": ["Điều 250"], "crimes": []}
    q = np.array([0.95, 0.1, 0], dtype=np.float32)       # gần A nhất
    # chỉ 1 vụ liên quan (B) < top_k=3 -> nới ra toàn kho, A lọt vào theo cosine
    hits = store.match(case, q, top_k=3)
    nums = {h["case_number"] for h in hits}
    assert "A" in nums and "B" in nums
    a = next(h for h in hits if h["case_number"] == "A")
    assert a["related_by_law"] is False


def test_match_empty_store(store):
    assert store.match({"articles": [], "crimes": []}, np.array([1, 0, 0], np.float32)) == []


def test_search_cosine_plus_keyword_boost(store):
    # A: vector xa query nhưng tiêu đề chứa từ khoá; B: vector gần query.
    store.add({"case_number": "A", "crimes": [], "articles": ["Điều 250"],
               "penalties": [], "facts_summary": "vận chuyển ma túy", "full_text": ""},
              np.array([1, 0, 0], np.float32), source="an_le", title="Điều 250 ma túy")
    store.add({"case_number": "B", "crimes": [], "articles": [],
               "penalties": [], "facts_summary": "", "full_text": ""},
              np.array([0, 1, 0], np.float32), source="an_le", title="khác")
    # query vec gần B, nhưng text 'Điều 250 ma túy' trùng từ khoá A
    hits = store.search("Điều 250 ma túy", np.array([0.2, 0.95, 0], np.float32), top_k=2)
    assert {h["case_number"] for h in hits} == {"A", "B"}
    # A được boost từ khoá đủ để không bị bỏ (cùng có mặt)
    assert any(h["case_number"] == "A" for h in hits)


def test_search_empty_store(store):
    assert store.search("bất kỳ", np.array([1, 0, 0], np.float32)) == []


def test_add_upsert_same_id(store):
    i1 = _add(store, "DUP/2020", ["Điều 1"], [1, 0, 0])
    i2 = _add(store, "DUP/2020", ["Điều 2"], [0, 1, 0])  # cùng case_number+source
    assert i1 == i2 and store.count() == 1


# ── embed_case (monkeypatch _embed) ──────────────────────────────────────────

def test_embed_case_uses_matching_key(monkeypatch):
    seen = {}

    def fake_embed(texts, timeout=120):
        seen["text"] = texts[0]
        return np.array([[0.6, 0.8]], dtype=np.float32)

    monkeypatch.setattr(ps, "_embed", fake_embed)
    case = {"crimes": ["Trộm cắp"], "articles": ["Điều 173"], "facts_summary": "lấy xe máy"}
    v = ps.embed_case(case)
    assert v.shape == (1, 2)
    assert "Trộm cắp" in seen["text"] and "Điều 173" in seen["text"]
    assert "lấy xe máy" in seen["text"]


def test_matching_key_handles_empty():
    k = matching_key({})
    assert "Tội danh:" in k and "Điều luật:" in k
