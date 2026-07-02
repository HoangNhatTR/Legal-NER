"""Kho TIỀN LỆ cục bộ + engine so khớp bản án tương tự.

Mục đích:
  So khớp một bản án đang xem với (A) ÁN LỆ chính thức và (B) các bản án người
  dùng đã LƯU/XÁC NHẬN trước đó — để tham khảo "vụ tương tự đã xử thế nào",
  mức án có nhất quán không.

Vì sao KHÔNG so khớp bằng embedding TOÀN VĂN:
  Mọi bản án chia sẻ rất nhiều khuôn mẫu hành chính ("TÒA ÁN NHÂN DÂN... NHẬN
  ĐỊNH... QUYẾT ĐỊNH...") → embedding toàn văn khớp nhau ở BỐ CỤC, không phải
  NỘI DUNG. Nên so khớp ở đây là LAI:
    1) lọc cứng theo TỘI DANH + ĐIỀU LUẬT (từ NER) để chỉ giữ vụ "cùng loại";
    2) xếp hạng cosine trên một "khoá so khớp" = tội danh + điều luật + tóm tắt
       DIỄN BIẾN HÀNH VI (bỏ phần khuôn mẫu).

Lưu trữ: SQLite (data/precedents.db). Vector BGE-M3 (mượn embedder Module 1 qua
judgment_rag._embed) đã L2-normalize → dot = cosine; lưu blob float32.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from api.judgment_rag import _embed
from config import DATA_DIR

DB_PATH = Path(__import__("os").environ.get("LEGAL_NER_PRECEDENT_DB", str(DATA_DIR / "precedents.db")))

_ARTICLE_RE = re.compile(r"Điều\s+(\d+)", re.IGNORECASE)
# Phần DIỄN BIẾN HÀNH VI nằm giữa "NỘI DUNG VỤ ÁN" và "NHẬN ĐỊNH".
_SEC_FACTS = re.compile(r"NỘI\s*DUNG\s*VỤ\s*ÁN", re.IGNORECASE)
_SEC_REASON = re.compile(r"NHẬN\s*ĐỊNH", re.IGNORECASE)


# ── Trích đặc trưng để so khớp ───────────────────────────────────────────────

def _uniq(grouped: dict, label: str, n: int = 12) -> list[str]:
    seen: list[str] = []
    for e in grouped.get(label, []) or []:
        t = (e.get("text") if isinstance(e, dict) else str(e)).strip()
        if t and t not in seen:
            seen.append(t)
        if len(seen) >= n:
            break
    return seen


def normalize_articles(*texts: str) -> list[str]:
    """Rút các tham chiếu 'Điều N' (chuẩn hoá) từ một loạt chuỗi → ['Điều 250', ...]."""
    out: list[str] = []
    for t in texts:
        for m in _ARTICLE_RE.findall(t or ""):
            a = f"Điều {m}"
            if a not in out:
                out.append(a)
    return out


def crime_tokens(crimes: Iterable[str]) -> set[str]:
    """Tập token (lowercase) của các tội danh — để đo trùng lặp loại tội."""
    toks: set[str] = set()
    for c in crimes:
        toks |= set(re.findall(r"\w+", (c or "").lower()))
    # bỏ vài hư từ phổ biến để trùng lặp có nghĩa hơn
    return toks - {"tội", "về", "và", "các", "quy", "định"}


def facts_summary(text: str, limit: int = 1200) -> str:
    """Trích phần diễn biến hành vi (giữa 'NỘI DUNG VỤ ÁN' và 'NHẬN ĐỊNH').
    Fallback: đoạn giữa văn bản (bỏ phần đầu hành chính). Cắt cho gọn khoá so khớp."""
    if not text:
        return ""
    s, e = _SEC_FACTS.search(text), _SEC_REASON.search(text)
    if s and e and e.start() > s.end():
        seg = text[s.end():e.start()]
    elif s:
        seg = text[s.end():s.end() + limit]
    else:
        seg = text[300:300 + limit]
    return " ".join(seg.split())[:limit]


def build_case(
    case_meta: dict | None, grouped: dict | None, text: str = ""
) -> dict:
    """Dựng bản ghi tiền lệ (chưa có vector) từ dữ liệu NER của một bản án."""
    grouped = grouped or {}
    cm = case_meta or {}
    crimes = _uniq(grouped, "CRIME", 8)
    articles = normalize_articles(
        " ".join(_uniq(grouped, "ARTICLE", 20)),
        " ".join(_uniq(grouped, "LEGAL_BASIS", 8)),
    )
    return {
        "case_number": cm.get("case_number") or "",
        "court": (_uniq(grouped, "COURT", 1) or [""])[0],
        "case_type": cm.get("case_type") or "",
        "crimes": crimes,
        "articles": articles,
        "penalties": _uniq(grouped, "PENALTY", 6),
        "facts_summary": facts_summary(text),
        "full_text": text or "",
    }


def matching_key(case: dict) -> str:
    """Khoá văn bản để embed (substance, KHÔNG khuôn mẫu)."""
    return (
        f"Tội danh: {'; '.join(case.get('crimes') or []) or '—'}. "
        f"Điều luật: {'; '.join(case.get('articles') or []) or '—'}. "
        f"Tình tiết: {case.get('facts_summary') or ''}"
    ).strip()


def embed_case(case: dict) -> np.ndarray:
    """Vector BGE-M3 (1, dim) cho khoá so khớp của một vụ. [] nếu embed lỗi."""
    vecs = _embed([matching_key(case)])
    return vecs if vecs.size else np.zeros((0, 0), dtype=np.float32)


# ── Lưu trữ SQLite ───────────────────────────────────────────────────────────

class PrecedentStore:
    """Kho tiền lệ bền (SQLite). An toàn để tạo nhiều instance (mỗi cái 1 file)."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS precedents (
                id           TEXT PRIMARY KEY,
                source       TEXT NOT NULL,           -- 'an_le' | 'confirmed'
                title        TEXT,
                case_number  TEXT,
                court        TEXT,
                case_type    TEXT,
                crimes       TEXT,                    -- json list
                articles     TEXT,                    -- json list (Điều N)
                penalties    TEXT,                    -- json list
                facts_summary TEXT,
                full_text    TEXT,
                url          TEXT,
                dim          INTEGER,
                vector       BLOB,                    -- float32 normalized
                created_at   REAL
            )
            """
        )
        self._conn.commit()

    # -- ghi --
    def add(
        self, case: dict, vector: np.ndarray, *, source: str = "confirmed",
        title: str = "", url: str = "", id: Optional[str] = None,
    ) -> str:
        """Thêm/ghi đè một tiền lệ. Trả về id. case_number trùng (cùng source) -> cập nhật."""
        vec = np.asarray(vector, dtype=np.float32).ravel()
        rid = id or self._stable_id(source, case.get("case_number", ""))
        self._conn.execute(
            """
            INSERT INTO precedents
              (id, source, title, case_number, court, case_type, crimes, articles,
               penalties, facts_summary, full_text, url, dim, vector, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              source=excluded.source, title=excluded.title, court=excluded.court,
              case_type=excluded.case_type, crimes=excluded.crimes,
              articles=excluded.articles, penalties=excluded.penalties,
              facts_summary=excluded.facts_summary, full_text=excluded.full_text,
              url=excluded.url, dim=excluded.dim, vector=excluded.vector
            """,
            (
                rid, source, title, case.get("case_number", ""), case.get("court", ""),
                case.get("case_type", ""), json.dumps(case.get("crimes") or [], ensure_ascii=False),
                json.dumps(case.get("articles") or [], ensure_ascii=False),
                json.dumps(case.get("penalties") or [], ensure_ascii=False),
                case.get("facts_summary", ""), case.get("full_text", ""), url,
                int(vec.size), vec.tobytes(), time.time(),
            ),
        )
        self._conn.commit()
        return rid

    @staticmethod
    def _stable_id(source: str, case_number: str) -> str:
        key = f"{source}:{case_number}".strip(":")
        if case_number:
            return key
        return f"{source}:{uuid.uuid4().hex[:12]}"

    # -- đọc --
    def count(self, source: Optional[str] = None) -> int:
        if source:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM precedents WHERE source=?", (source,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM precedents").fetchone()
        return int(row[0])

    def _rows(self, sources: Optional[list[str]] = None) -> list[sqlite3.Row]:
        if sources:
            q = "SELECT * FROM precedents WHERE source IN (%s)" % ",".join("?" * len(sources))
            return self._conn.execute(q, sources).fetchall()
        return self._conn.execute("SELECT * FROM precedents").fetchall()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM precedents")
        self._conn.commit()

    # -- so khớp --
    def match(
        self, case: dict, query_vec: np.ndarray, *, top_k: int = 5,
        sources: Optional[list[str]] = None, min_score: float = 0.0,
    ) -> list[dict]:
        """Trả top-k tiền lệ tương tự nhất.

        1) ỨNG VIÊN: vụ chia ÍT NHẤT 1 điều luật HOẶC có token tội danh trùng.
           Nếu quá ít ứng viên (< top_k) thì nới ra TOÀN BỘ kho (vẫn xếp cosine).
        2) XẾP HẠNG: cosine(query_vec, vector) trên ứng viên.
        """
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        rows = self._rows(sources)
        if q.size == 0 or not rows:
            return []
        q_articles = set(case.get("articles") or [])
        q_crimes = crime_tokens(case.get("crimes") or [])

        scored: list[tuple[float, bool, sqlite3.Row]] = []
        for r in rows:
            vec = np.frombuffer(r["vector"], dtype=np.float32)
            if vec.size != q.size:
                continue
            sim = float(np.dot(q, vec))
            arts = set(json.loads(r["articles"] or "[]"))
            crimes = crime_tokens(json.loads(r["crimes"] or "[]"))
            related = bool(q_articles & arts) or bool(q_crimes & crimes)
            scored.append((sim, related, r))

        related = [s for s in scored if s[1]]
        pool = related if len(related) >= top_k else scored  # nới nếu thiếu
        pool.sort(key=lambda t: -t[0])
        out: list[dict] = []
        for sim, is_related, r in pool[:top_k]:
            if sim < min_score:
                continue
            out.append(self._row_to_dict(r, sim, is_related))
        return out

    def search(
        self, query_text: str, query_vec: np.ndarray, *, top_k: int = 8,
        sources: Optional[list[str]] = None,
    ) -> list[dict]:
        """Tìm kiếm TRỰC TIẾP bằng câu/từ khoá tự do.

        Xếp hạng = cosine (ngữ nghĩa) + boost nhỏ theo tỉ lệ từ khoá trùng trên
        (tiêu đề + tội danh + điều luật + diễn biến) → câu hỏi nêu đúng 'Điều 250'
        hay tên tội vẫn nổi lên dù embedding chưa sát.
        """
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        rows = self._rows(sources)
        if q.size == 0 or not rows:
            return []
        qtok = set(re.findall(r"\w+", (query_text or "").lower()))
        scored: list[tuple[float, float, sqlite3.Row]] = []
        for r in rows:
            vec = np.frombuffer(r["vector"], dtype=np.float32)
            if vec.size != q.size:
                continue
            sim = float(np.dot(q, vec))
            hay = " ".join([
                r["title"] or "", r["crimes"] or "", r["articles"] or "", r["facts_summary"] or "",
            ]).lower()
            kw = len(qtok & set(re.findall(r"\w+", hay))) / (len(qtok) or 1)
            scored.append((sim + 0.15 * kw, sim, r))
        scored.sort(key=lambda t: -t[0])
        return [self._row_to_dict(r, sim, False) for _f, sim, r in scored[:top_k]]

    @staticmethod
    def _row_to_dict(r: sqlite3.Row, score: float, related: bool) -> dict:
        return {
            "id": r["id"],
            "source": r["source"],
            "title": r["title"] or "",
            "case_number": r["case_number"] or "",
            "court": r["court"] or "",
            "case_type": r["case_type"] or "",
            "crimes": json.loads(r["crimes"] or "[]"),
            "articles": json.loads(r["articles"] or "[]"),
            "penalties": json.loads(r["penalties"] or "[]"),
            "facts_summary": r["facts_summary"] or "",
            "url": r["url"] or "",
            "score": round(float(score), 3),
            "related_by_law": bool(related),
        }


# ── Singleton dùng cho API ───────────────────────────────────────────────────
_STORE: Optional[PrecedentStore] = None


def get_store() -> PrecedentStore:
    global _STORE
    if _STORE is None:
        _STORE = PrecedentStore()
    return _STORE
