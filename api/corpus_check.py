"""Kiểm chứng viện dẫn trên KHO LUẬT LỚN (Module 1, ~4,9M chunks).

Bổ sung cho verify/citation.py — vốn CHỈ đối chiếu offline với BLHS 2015
(data/lawdb/blhs2015.db): bản án dân sự / lao động / hành chính viện dẫn
Bộ luật Dân sự, Bộ luật Lao động, các nghị định... đều "ngoài phạm vi".
Ở đây gọi ``POST /v1/retrieve`` của Module 1 (retrieve thuần, không LLM)
và kiểm tra: có chunk nào đúng "Điều N" thuộc đúng văn bản không.

Best-effort: Module 1 không chạy → trả status "skipped", không làm hỏng /verify.
"""
from __future__ import annotations

import re

import requests

from api.analyze import RAG_KEY, RAG_URL

# "điểm c khoản 1 Điều 250 của Bộ luật Hình sự năm 2015" → (250, "Bộ luật Hình sự")
_PAIR_RE = re.compile(
    r"Điều\s+(\d+)[^;.\n]{0,80}?"
    r"((?:Bộ\s*luật|Luật|Nghị\s*định|Thông\s*tư|Nghị\s*quyết|Pháp\s*lệnh|Hiến\s*pháp)"
    r"[^;,.\n(]{0,70})",
    re.IGNORECASE,
)
_YEAR_TAIL_RE = re.compile(r"\s*(?:năm\s+)?(?:19|20)\d{2}.*$")
_STOP_TOKENS = {"của", "và", "về", "số", "the"}


def _norm_law(name: str) -> str:
    """Chuẩn hóa tên luật để so khớp title: bỏ năm/số hiệu đuôi, thường hóa."""
    s = _YEAR_TAIL_RE.sub("", (name or "").strip())
    s = re.sub(r"\s+", " ", s).strip(" ,;.").lower()
    return s


def extract_citation_pairs(grouped: dict, limit: int = 8) -> list[dict]:
    """Bóc các cặp (Điều N, tên văn bản) từ thực thể LEGAL_BASIS/LAW_NAME.

    Giữ thứ tự xuất hiện, khử trùng. Điều không kèm tên luật → ghép với
    LAW_NAME đầu tiên (nếu có) làm dự đoán tốt nhất.
    """
    texts: list[str] = []
    for label in ("LEGAL_BASIS", "ARTICLE"):
        for e in grouped.get(label, []) or []:
            t = (e.get("text") or "").strip()
            if t:
                texts.append(t)
    law_names = [
        (e.get("text") or "").strip()
        for e in grouped.get("LAW_NAME", []) or []
        if (e.get("text") or "").strip()
    ]
    default_law = law_names[0] if law_names else ""

    pairs: list[dict] = []
    seen: set[tuple[int, str]] = set()

    def _add(article: int, law: str) -> None:
        key = (article, _norm_law(law))
        if key in seen:
            return
        seen.add(key)
        pairs.append({"article": article, "law": law.strip(" ,;.")})

    for t in texts:
        for m in _PAIR_RE.finditer(t):
            _add(int(m.group(1)), m.group(2))
    # Điều "trần" không bắt được luật đi kèm → gán luật mặc định (nếu có)
    for t in texts:
        for m in re.finditer(r"Điều\s+(\d+)", t):
            art = int(m.group(1))
            if default_law and not any(p["article"] == art for p in pairs):
                _add(art, default_law)

    return pairs[:limit]


# Cụm phân biệt cặp luật dễ nhầm: token 'bộ luật hình sự' là TẬP CON của
# 'bộ luật tố tụng hình sự' → phải cùng CÓ hoặc cùng KHÔNG có cụm này.
_DISCRIMINATORS = ("tố tụng",)


def _law_matches_title(law: str, title: str) -> bool:
    """Tên luật khớp title chunk: mọi token có nghĩa của tên luật nằm trong title.

    'Bộ luật Hình sự' khớp cả bản gốc lẫn 'Văn bản hợp nhất ... Bộ luật Hình sự',
    nhưng KHÔNG khớp 'Bộ luật Tố tụng hình sự' (discriminator 'tố tụng').
    """
    t = (title or "").lower()
    norm = _norm_law(law)
    tokens = [w for w in norm.split() if w not in _STOP_TOKENS]
    if not tokens or not all(w in t for w in tokens):
        return False
    for d in _DISCRIMINATORS:
        if (d in t) != (d in norm):
            return False
    return True


def _check_one(pair: dict, timeout: float) -> dict:
    """Kiểm 1 cặp qua /v1/retrieve. Trả {article, law, status, matched_title?}.

    rerank=False: kiểm TỒN TẠI không cần thứ hạng đẹp — RRF thô nhanh gấp
    5-10 lần CrossEncoder và cho phép top_k lớn (nhiều ứng viên = recall cao).
    Query dùng tên luật ĐÃ CHUẨN HÓA (bỏ đuôi 'năm... (sửa đổi, bổ sung...)')
    để khỏi nhiễu. Timeout ở 1 lượt → retry đúng 1 lần rồi mới bỏ.
    """
    art, law = pair["article"], pair["law"]
    query = f"{_norm_law(law)} Điều {art}"
    body = {"query": query, "top_k": 12, "model": "legal-ai-full", "rerank": False}
    last_exc: Exception | None = None
    for _attempt in range(2):
        try:
            r = requests.post(
                f"{RAG_URL}/v1/retrieve",
                headers={"Authorization": f"Bearer {RAG_KEY}"},
                json=body,
                timeout=timeout,
            )
            r.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
    else:
        raise last_exc  # type: ignore[misc]

    for c in r.json().get("chunks", []):
        if (c.get("article") or "") == f"Điều {art}" and _law_matches_title(
            law, c.get("title") or ""
        ):
            return {
                "article": art, "law": law, "status": "found",
                "matched_title": (c.get("title") or "")[:120],
                "doc_number": c.get("doc_number"),
            }
    return {"article": art, "law": law, "status": "not_found"}


def corpus_check(grouped: dict, timeout: float = 120.0) -> dict:
    """Kiểm chứng mọi viện dẫn của bản án trên kho luật Module 1.

    Trả về:
      {status: ok|skipped, checked, found, not_found, results: [...], note}
    'not_found' nghĩa là KHÔNG tìm thấy đúng Điều trong đúng văn bản — có thể
    do viện dẫn sai, NER bóc thiếu, hoặc corpus chưa có văn bản đó (xem note).
    Lỗi mạng/timeout ở MỘT viện dẫn → item đó status='error', các viện dẫn
    khác vẫn kiểm tiếp; tất cả lỗi → status='skipped'.
    """
    pairs = extract_citation_pairs(grouped)
    if not pairs:
        return {
            "status": "ok", "checked": 0, "found": 0, "not_found": 0,
            "results": [], "note": "Không bóc được viện dẫn nào để kiểm chứng.",
        }

    results: list[dict] = []
    errors = 0
    for p in pairs:
        try:
            results.append(_check_one(p, timeout))
        except requests.RequestException as exc:
            errors += 1
            results.append({
                "article": p["article"], "law": p["law"], "status": "error",
                "note": str(exc)[:120],
            })

    if errors == len(pairs):
        return {
            "status": "skipped", "checked": 0, "found": 0, "not_found": 0,
            "results": results,
            "note": f"Không gọi được kho luật Module 1 ({RAG_URL}).",
        }

    found = sum(1 for x in results if x["status"] == "found")
    not_found = sum(1 for x in results if x["status"] == "not_found")
    return {
        "status": "ok",
        "checked": len(results) - errors,
        "found": found,
        "not_found": not_found,
        "results": results,
        "note": (
            "Đối chiếu trên kho luật lớn (mọi ngành luật) — 'not_found' cần "
            "kiểm tra tay: viện dẫn sai, NER bóc thiếu, hoặc corpus thiếu văn bản."
        ),
    }
