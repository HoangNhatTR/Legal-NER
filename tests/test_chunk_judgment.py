"""Test chunk_judgment + retrieve của RAG-trên-bản-án (api/judgment_rag.py).

Trọng tâm: chunk theo câu (không xé giữa câu), CÓ overlap giữa MỌI chunk, bền
khi PDF bị làm phẳng (mất xuống dòng), và retrieve trả đoạn theo thứ tự bản án.
"""
import numpy as np
import pytest

from api import judgment_rag
from api.judgment_rag import _tail_overlap, chunk_judgment, retrieve


# ── chunk_judgment ──────────────────────────────────────────────────────────

def test_empty_text_returns_empty():
    assert chunk_judgment("") == []
    assert chunk_judgment("   \n  ") == []


def test_short_text_single_chunk():
    chunks = chunk_judgment("Bị cáo Nguyễn Văn A phạm tội trộm cắp tài sản.")
    assert len(chunks) == 1
    assert "Nguyễn Văn A" in chunks[0]


def test_no_chunk_exceeds_target_plus_overlap():
    target, overlap = 200, 40
    text = " ".join(f"Đây là câu số {i} trong bản án thử nghiệm." for i in range(60))
    chunks = chunk_judgment(text, target=target, overlap=overlap)
    assert len(chunks) >= 2
    # gói theo câu + carry overlap -> mỗi chunk không vượt quá target + overlap
    assert all(len(c) <= target + overlap for c in chunks)


def test_overlap_between_consecutive_chunks():
    """Đầu mỗi chunk kế tiếp phải xuất hiện ở cuối chunk trước (overlap > 0)."""
    target, overlap = 200, 40
    text = " ".join(f"Câu thứ {i} mô tả một tình tiết của vụ án." for i in range(80))
    chunks = chunk_judgment(text, target=target, overlap=overlap)
    assert len(chunks) >= 3
    for prev, nxt in zip(chunks, chunks[1:]):
        head = nxt[:10]  # vài ký tự đầu của chunk sau
        assert head and head in prev, f"không thấy overlap: {head!r} ⊄ {prev!r}"


def test_flattened_text_still_splits_by_sentence():
    """PDF làm phẳng (không có '\\n') vẫn tách được nhờ dấu kết câu."""
    flat = "".join(f"Tình tiết {i} của vụ án được trình bày rõ ràng. " for i in range(80))
    chunks = chunk_judgment(flat, target=200, overlap=40)
    assert len(chunks) >= 3
    # không có chunk nào là toàn bộ văn bản (đã thực sự chia nhỏ)
    assert all(len(c) < len(flat) for c in chunks)


def test_very_long_sentence_hard_cut_with_overlap():
    """Câu liền không có ranh giới, dài hơn target -> cắt cứng có overlap."""
    target, overlap = 100, 30
    long_sent = "X" * 500  # không dấu câu, không khoảng trắng
    chunks = chunk_judgment(long_sent, target=target, overlap=overlap)
    assert len(chunks) > 1
    assert all(len(c) <= target for c in chunks)
    # cửa sổ kế tiếp overlap đúng (target - step) ký tự
    step = target - overlap
    assert chunks[1].startswith(chunks[0][step:])


def test_sentence_not_split_midway_in_packing():
    """Trong nhánh gói (câu ngắn), không có câu nào bị xé giữa chừng."""
    sentences = [f"Câu số {i} kết thúc đầy đủ." for i in range(40)]
    text = " ".join(sentences)
    chunks = chunk_judgment(text, target=150, overlap=30)
    joined = " ".join(chunks)
    # mọi câu gốc vẫn còn nguyên vẹn ở đâu đó trong kết quả
    for s in sentences:
        assert s in joined


def test_deterministic():
    text = " ".join(f"Nội dung {i} của bản án." for i in range(50))
    assert chunk_judgment(text) == chunk_judgment(text)


# ── _tail_overlap ───────────────────────────────────────────────────────────

def test_tail_overlap_zero_or_empty():
    assert _tail_overlap("abc", 0) == ""
    assert _tail_overlap("", 50) == ""


def test_tail_overlap_short_text_returned_whole():
    assert _tail_overlap("ngắn", 50) == "ngắn"


def test_tail_overlap_trims_to_word_boundary():
    text = "phần đầu không quan trọng lắm cuối cùng còn lại đoạn này"
    tail = _tail_overlap(text, 20)
    assert tail and not text.endswith("x")  # sanity
    assert text.endswith(tail)          # là hậu tố thật
    assert not tail.startswith(" ")      # không bắt đầu bằng khoảng trắng
    # không bắt đầu giữa chừng một từ: ký tự ngay trước tail là khoảng trắng
    idx = len(text) - len(tail)
    assert idx == 0 or text[idx - 1] == " "


# ── retrieve (monkeypatch _embed để khỏi cần mạng/Module 1) ─────────────────

@pytest.fixture(autouse=True)
def _clear_cache():
    judgment_rag._CACHE.clear()
    yield
    judgment_rag._CACHE.clear()


def _eye_embed_factory(query, n, hot):
    """Trả fake _embed: query -> vector one-hot theo dict `hot`; chunk batch -> eye."""
    def fake_embed(texts, timeout=120):
        if texts == [query]:
            q = np.zeros(n, dtype=np.float32)
            for idx, val in hot.items():
                q[idx] = val
            return q.reshape(1, n)
        return np.eye(n, dtype=np.float32)[: len(texts)]
    return fake_embed


def test_retrieve_returns_doc_order(monkeypatch):
    """retrieve hợp nhất vector+BM25 nhưng trả các đoạn theo THỨ TỰ trong bản án."""
    # 8 khối dài, mỗi khối có token DUY NHẤT để vector & BM25 cùng trỏ chunk 2,5.
    query = "TOPICFIVE TOPICTWO"
    blocks = []
    for i in range(8):
        tag = {2: "TOPICTWO", 5: "TOPICFIVE"}.get(i, f"TOPIC{i}")
        blocks.append(f"Khối {i} {tag}: " + ("dữ kiện vụ án " * 60))
    text = "\n\n".join(blocks)
    chunks = chunk_judgment(text)
    n = len(chunks)
    assert n >= 6

    monkeypatch.setattr(judgment_rag, "_embed", _eye_embed_factory(query, n, {5: 1.0, 2: 0.9}))
    hits = retrieve(text, query, top_k=2)
    assert len(hits) == 2
    # chunk 2 đứng trước chunk 5 dù chunk 5 điểm cao hơn -> đã sắp theo doc order
    returned = [c for c, _s in hits]
    assert returned == [chunks[2], chunks[5]]


def test_retrieve_bm25_rescues_exact_match(monkeypatch):
    """BM25 đưa chunk khớp CHÍNH XÁC chuỗi lên top dù vector cho điểm rất thấp."""
    query = "Điều 260"
    blocks = []
    for i in range(8):
        extra = "vi phạm quy định Điều 260 giao thông" if i == 6 else "nội dung khác"
        blocks.append(f"Khối {i}: {extra} " + ("văn bản dài " * 55))
    text = "\n\n".join(blocks)
    chunks = chunk_judgment(text)
    n = len(chunks)

    # Vector CỐ TÌNH bỏ qua chunk 6 (cho mọi chunk điểm 0) -> chỉ BM25 cứu được.
    monkeypatch.setattr(judgment_rag, "_embed", _eye_embed_factory(query, n, {}))
    hits = retrieve(text, query, top_k=3)
    returned = [c for c, _s in hits]
    assert chunks[6] in returned  # chunk chứa 'Điều 260' phải lọt top nhờ BM25


def test_retrieve_empty_text():
    assert retrieve("", "hỏi gì đó") == []
