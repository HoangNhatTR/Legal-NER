"""RAG trên CHÍNH bản án (hybrid với RAG kho luật của Module 1).

Vì sao có module này:
  Khung "Hỏi tiếp về bản án" trước đây nhồi NGUYÊN toàn văn vào prompt → bản án
  dài (>14k ký tự) bị cắt mất phần sau. Module này chunk bản án + embed rồi
  retrieve đúng đoạn liên quan theo từng câu hỏi → bền cho MỌI độ dài.

Không nạp BGE-M3 lần 2 (máy CPU, tránh OOM): mượn embedder của Module 1 qua
endpoint POST /v1/embed. Vector BGE-M3 đã L2-normalize → dot product = cosine.

Cache theo nội dung (sha1 toàn văn) → mỗi bản án chỉ embed MỘT lần dù hỏi nhiều câu.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import OrderedDict

import numpy as np
import requests

RAG_URL = os.environ.get("LEGAL_NER_RAG_URL", "http://localhost:8000").rstrip("/")
RAG_KEY = os.environ.get("LEGAL_NER_RAG_KEY", "legal-ai-local")

# Cache: hash toàn văn -> {"chunks": list[str], "vecs": np.ndarray (n, dim)}
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_MAX_CACHE = 8  # giữ tối đa 8 bản án gần nhất (LRU)


class JudgmentRagError(Exception):
    """Lỗi khi embed/retrieve bản án (vd Module 1 /v1/embed không phản hồi)."""


def _embed(texts: list[str], timeout: int = 120) -> np.ndarray:
    """Embed qua Module 1. Trả mảng (n, dim) float32 (đã normalize)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        r = requests.post(
            f"{RAG_URL}/v1/embed",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {RAG_KEY}",
            },
            json={"texts": texts},
            timeout=timeout,
        )
        r.raise_for_status()
        vecs = r.json().get("vectors", [])
    except requests.RequestException as exc:
        raise JudgmentRagError(f"không gọi được /v1/embed của Module 1 ({RAG_URL}): {exc}") from exc
    except (KeyError, ValueError) as exc:
        raise JudgmentRagError(f"phản hồi /v1/embed không hợp lệ: {exc}") from exc
    return np.asarray(vecs, dtype=np.float32)


# Ranh giới câu tiếng Việt: sau dấu kết câu (. ! ? ; … hoặc xuống dòng) + khoảng
# trắng. Dùng lookbehind để GIỮ dấu kết câu lại trong câu (cắt SAU dấu).
_SENT_SPLIT = re.compile(r"(?<=[.!?;…])\s+")


def _split_sentences(para: str) -> list[str]:
    """Tách một đoạn thành các câu (xấp xỉ). Bền cả khi PDF bị làm phẳng mất xuống
    dòng — vẫn tách được nhờ dấu kết câu."""
    sents = [s.strip() for s in _SENT_SPLIT.split(para) if s.strip()]
    if sents:
        return sents
    p = para.strip()
    return [p] if p else []


def _tail_overlap(text: str, overlap: int) -> str:
    """Phần đuôi ~overlap ký tự của ``text`` để mang sang đầu chunk kế (giữ liền
    mạch ngữ cảnh ở chỗ nối). Cắt gọn tại ranh giới câu — nếu không có thì tại
    khoảng trắng — để không bắt đầu giữa chừng một từ."""
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    m = re.search(r"[.!?;…]\s+", tail)
    if m:
        return tail[m.end():]
    sp = tail.find(" ")
    return tail[sp + 1:] if sp != -1 else tail


def chunk_judgment(text: str, target: int = 900, overlap: int = 150) -> list[str]:
    """Chunk bản án thành các đoạn ~target ký tự, CÓ overlap giữa MỌI chunk.

    Cải tiến so với bản gói-đoạn-thuần cũ:
      - Gói theo CÂU (không xé giữa câu), tách câu kể cả khi PDF mất xuống dòng.
      - Overlap áp dụng cho MỌI ranh giới chunk: mang ~overlap ký tự cuối của
        chunk trước sang đầu chunk sau → câu hỏi rơi đúng chỗ nối vẫn truy được.
      - Câu/đoạn liền quá dài (> target) vẫn được cắt cứng có overlap (lưới an toàn).

    Vector BGE-M3 nhận tới max_seq 1024 token (~2-3k ký tự tiếng Việt) nên target
    900 (+ overlap) không bao giờ tràn ngữ cảnh mã hoá.
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1) Tách toàn văn thành câu (gộp mọi đoạn; '\n' trở thành ranh giới câu).
    sentences: list[str] = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if para:
            sentences.extend(_split_sentences(para))

    # 2) Gói câu tới ~target, mang overlap từ đuôi chunk trước sang chunk sau.
    chunks: list[str] = []
    cur = ""
    for sent in sentences:
        if len(sent) > target:  # câu liền dài bất thường -> cắt cứng có overlap
            if cur:
                chunks.append(cur)
                cur = ""
            step = max(target - overlap, 1)
            for i in range(0, len(sent), step):
                chunks.append(sent[i : i + target])
            continue
        if cur and len(cur) + 1 + len(sent) > target:
            chunks.append(cur)
            carry = _tail_overlap(cur, overlap)
            cur = f"{carry} {sent}".strip() if carry else sent
        else:
            cur = f"{cur} {sent}".strip() if cur else sent
    if cur:
        chunks.append(cur)
    return chunks


def _hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


# ── Truy hồi LAI: vector (BGE-M3) + từ khoá (BM25) hợp nhất bằng RRF ──────────
# Vector bắt khớp NGỮ NGHĨA; BM25 bắt khớp CHÍNH XÁC chuỗi (tên người, số bản án,
# "Điều 260", số tiền…) mà embedding hay bỏ sót. Bản án nhỏ nên BM25 in-memory rẻ.

def _tokenize(s: str) -> list[str]:
    r"""Token hoá thô tiếng Việt: \w+ (Unicode) đã gồm chữ có dấu + chữ số."""
    return re.findall(r"\w+", (s or "").lower())


def _bm25_scores(
    docs_tokens: list[list[str]], query_tokens: list[str], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """Điểm BM25 của từng chunk so với câu hỏi (0 nếu không có từ nào trùng)."""
    n = len(docs_tokens)
    if n == 0 or not query_tokens:
        return [0.0] * n
    df: dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    avgdl = sum(len(t) for t in docs_tokens) / n or 1.0
    q_unique = set(query_tokens)
    scores = [0.0] * n
    for i, toks in enumerate(docs_tokens):
        if not toks:
            continue
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = len(toks)
        s = 0.0
        for t in q_unique:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores[i] = s
    return scores


def _rrf(ranked_indices: list[int], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: 1/(k + thứ hạng). ranked_indices xếp tốt-nhất-trước."""
    out: dict[int, float] = {}
    for rank, idx in enumerate(ranked_indices):
        out[idx] = out.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return out


def index_judgment(text: str) -> str:
    """Chunk + embed bản án (idempotent theo nội dung). Trả về hash key."""
    h = _hash(text)
    if h in _CACHE:
        _CACHE.move_to_end(h)  # LRU touch
        return h
    chunks = chunk_judgment(text)
    vecs = _embed(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)
    tokens = [_tokenize(c) for c in chunks]  # cho nhánh BM25 (token hoá 1 lần)
    _CACHE[h] = {"chunks": chunks, "vecs": vecs, "tokens": tokens}
    while len(_CACHE) > _MAX_CACHE:
        _CACHE.popitem(last=False)  # evict LRU cũ nhất
    return h


def retrieve(text: str, query: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Trả top-k (đoạn bản án, cosine) liên quan nhất tới câu hỏi — truy hồi LAI.

    Hợp nhất hai bảng xếp hạng bằng RRF: (1) cosine BGE-M3 (ngữ nghĩa) và (2) BM25
    (khớp chính xác tên/số hiệu/điều luật). CHỌN top-k theo điểm hợp nhất, nhưng
    XẾP các đoạn được chọn theo THỨ TỰ xuất hiện trong bản án → LLM đọc liền mạch.
    Điểm cosine vẫn trả kèm để caller hiển thị độ liên quan.
    """
    h = index_judgment(text)
    entry = _CACHE[h]
    chunks: list[str] = entry["chunks"]
    vecs: np.ndarray = entry["vecs"]
    tokens: list[list[str]] = entry.get("tokens") or [_tokenize(c) for c in chunks]
    if not chunks or vecs.size == 0:
        return []
    q = _embed([query])
    if q.size == 0:
        return []
    sims = vecs @ q[0]  # cả hai đã normalize -> dot = cosine

    vec_ranked = [int(i) for i in np.argsort(-sims)]
    kw_scores = _bm25_scores(tokens, _tokenize(query))
    # Chỉ xếp hạng BM25 cho chunk thực sự có từ trùng (score > 0) — tránh nhiễu.
    kw_ranked = [int(i) for i in np.argsort(-np.asarray(kw_scores)) if kw_scores[i] > 0]

    fused = _rrf(vec_ranked)
    for idx, val in _rrf(kw_ranked).items():
        fused[idx] = fused.get(idx, 0.0) + val

    k = min(top_k, len(chunks))
    chosen = sorted(fused, key=lambda i: -fused[i])[:k]
    order = sorted(int(i) for i in chosen)  # theo vị trí trong bản án
    return [(chunks[i], float(sims[i])) for i in order]
