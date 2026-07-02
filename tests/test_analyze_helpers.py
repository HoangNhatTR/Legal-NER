"""Unit tests cho các helper bóc cấu trúc bản án trong api/analyze.py.

Đây là code dày đặc regex — dễ vỡ âm thầm khi bản án format khác đi.
Không gọi mạng: chỉ test hàm thuần.
"""
from __future__ import annotations

from api.analyze import (
    _applied_citations,
    _build_scenario,
    _chat_body,
    _facts_section,
    _looks_useless,
    _pronounced_penalty,
    _retrieval_query,
)


def _ents(*texts: str) -> list[dict]:
    return [{"text": t} for t in texts]


GROUPED = {
    "CRIME": _ents("Tàng trữ trái phép chất ma túy"),
    "PENALTY": _ents("03 năm tù (đề nghị)", "02 (hai) năm tù"),
    "LEGAL_BASIS": _ents(
        "điểm c khoản 1 Điều 250; điểm s khoản 1 Điều 51 của Bộ luật Hình sự năm 2015",
        "Điều 331 Bộ luật Tố tụng hình sự",
    ),
    "ARTICLE": _ents("Điều 250", "Điều 51"),
}


# ── _chat_body: hợp đồng máy-gọi-máy với Module 1 ─────────────────────────────

def test_chat_body_luon_skip_router():
    body = _chat_body("câu hỏi", "legal-ai-full", "", stream=False)
    assert body["skip_router"] is True


def test_chat_body_search_query_rieng():
    body = _chat_body("prompt dài", "legal-ai-full", "", stream=True,
                      search_query="khung hình phạt Điều 250")
    assert body["search_query"] == "khung hình phạt Điều 250"
    assert body["stream"] is True


def test_chat_body_khong_search_query_thi_khong_gui_field():
    body = _chat_body("câu hỏi", "legal-ai-graph", "", stream=False)
    assert "search_query" not in body


# ── Bóc trích dẫn / hình phạt ─────────────────────────────────────────────────

def test_applied_citations_uu_tien_basis_hinh_su():
    applied, law = _applied_citations(GROUPED)
    assert "điểm c khoản 1 Điều 250" in applied
    assert "Hình sự" in law
    assert "Tố tụng" not in law  # luật tố tụng phải bị loại


def test_pronounced_penalty_lay_muc_toa_tuyen():
    """Ưu tiên mức có số-bằng-chữ trong ngoặc (format phần QUYẾT ĐỊNH của Tòa)."""
    assert _pronounced_penalty(GROUPED) == "02 (hai) năm tù"


def test_pronounced_penalty_thieu_du_lieu():
    assert "không rõ" in _pronounced_penalty({})


def test_build_scenario_chua_du_du_kien():
    scenario, arts = _build_scenario({}, GROUPED)
    assert "Tàng trữ trái phép chất ma túy" in scenario
    assert "02 (hai) năm tù" in scenario
    assert "điểm c khoản 1 Điều 250" in scenario
    assert "250" in arts and "51" in arts


# ── Guard + truy vấn nối tiếp ─────────────────────────────────────────────────

def test_looks_useless_bat_template_validate_va_chao():
    assert _looks_useless("Xin chào! Tôi là trợ lý...")
    assert _looks_useless("Báo cáo kiểm tra văn bản: thiếu Tòa án...")
    assert not _looks_useless("1) Khung hình phạt của điểm c khoản 1 Điều 250 là 2-7 năm tù.")


def test_retrieval_query_gop_ngu_canh_luot_truoc():
    history = [
        {"role": "user", "content": "mức án 2 năm tù về tội tàng trữ ma túy"},
        {"role": "assistant", "content": "..."},
    ]
    q = _retrieval_query("vậy mức đó hợp lý không?", history)
    assert "tàng trữ ma túy" in q and "hợp lý không" in q


def test_retrieval_query_luot_dau_giu_nguyen():
    assert _retrieval_query("câu hỏi đầu", []) == "câu hỏi đầu"


# ── Trích phần diễn biến hành vi ──────────────────────────────────────────────

def test_facts_section_giua_hai_moc():
    text = (
        "TÒA ÁN NHÂN DÂN...\nNỘI DUNG VỤ ÁN:\n"
        "Khoảng 20 giờ ngày 01/01, Nguyễn Văn A mua 0,5 gam heroin để sử dụng.\n"
        "NHẬN ĐỊNH CỦA TÒA ÁN:\nHành vi của bị cáo..."
    )
    facts = _facts_section(text)
    assert "heroin" in facts
    assert "NHẬN ĐỊNH" not in facts
    assert "TÒA ÁN NHÂN DÂN" not in facts


def test_facts_section_van_ban_rong():
    assert _facts_section("") == ""
