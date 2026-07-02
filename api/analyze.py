"""Luồng phân tích so sánh riêng — KHÔNG đi qua validate_document của Module 1.

Vì sao có module này:
  Router của Module 1 đẩy mọi yêu cầu "có văn bản trong ngữ cảnh" sang tool
  ``validate_document`` — một bộ CHẤM THỂ THỨC/chất lượng (template cố định, bỏ
  qua câu hỏi người dùng, hay báo nhầm "thiếu yếu tố"). Để có một báo cáo ĐỐI
  CHIẾU thật (tội danh ↔ điều luật, mức án ↔ khung hình phạt) bám trên kho dữ
  liệu lớn, ta tự điều phối 2 lượt gọi RAG:

    1. TRA CỨU: hỏi Module 1 ở chế độ thường (intent "legal") NỘI DUNG từng điều
       luật được bản án viện dẫn. Đây là câu hỏi pháp lý thuần — KHÔNG có document
       trong ngữ cảnh — nên router đưa vào retrieve (RAG), không phải validate.
    2. TỔNG HỢP: đưa nội dung luật lấy được + dữ kiện bản án (từ NER) trở lại
       Module 1 như một câu TƯ VẤN ("đánh giá ...") để LLM so sánh và kết luận.

Cả hai lượt đều gọi API OpenAI-compatible /v1/chat/completions của Module 1.
"""

from __future__ import annotations

import json
import os
import re

import requests

# URL/keys tới Module 1 (RAG). Cấu hình qua env nếu chạy khác máy/cổng.
RAG_URL = os.environ.get("LEGAL_NER_RAG_URL", "http://localhost:8000").rstrip("/")
RAG_KEY = os.environ.get("LEGAL_NER_RAG_KEY", "legal-ai-local")

_DIEU_RE = re.compile(r"Điều\s+(\d+)")

# Dấu hiệu router Module 1 trả nhầm (chào hỏi / template validate) thay vì phân tích.
_GREETING_MARKERS = ("Xin chào! Tôi là", "trợ lý tư vấn pháp luật Việt Nam")
_VALIDATE_MARKER = "Báo cáo kiểm tra"


def _looks_useless(ans: str) -> bool:
    """True nếu RAG trả lời chào mẫu hoặc template 'Báo cáo kiểm tra' (validate)."""
    head = ans[:120]
    return any(m in head for m in _GREETING_MARKERS) or _VALIDATE_MARKER in head


class AnalyzeError(Exception):
    """RAG không phản hồi / lỗi khi điều phối phân tích."""


def _uniq(grouped: dict, label: str, n: int = 12) -> list[str]:
    seen: list[str] = []
    for e in grouped.get(label, []) or []:
        t = (e.get("text") or "").strip()
        if t and t not in seen:
            seen.append(t)
        if len(seen) >= n:
            break
    return seen


# Bắt cụm trích dẫn gọn: "(điểm x )(khoản n )Điều m"
_CITE_RE = re.compile(
    r"(?:điểm\s+\w+\s+)?(?:khoản\s+\d+\s+)?Điều\s+\d+", re.IGNORECASE
)


def _offense_basis(grouped: dict) -> str:
    """LEGAL_BASIS của TỘI DANH (ưu tiên nhắc 'hình sự' nhưng không phải 'tố tụng')."""
    lbs = _uniq(grouped, "LEGAL_BASIS", 4)
    for lb in lbs:
        low = lb.lower()
        if "hình sự" in low and "tố tụng" not in low:
            return lb
    return lbs[0] if lbs else ""


def _applied_citations(grouped: dict) -> tuple[str, str]:
    """Trích dẫn áp dụng (gọn) + tên luật, từ basis tội danh.

    Trả về ("điểm c khoản 1 Điều 250; điểm s khoản 1 Điều 51", "Bộ luật Hình sự ...").
    Bỏ chữ "Căn cứ/Áp dụng" và phần luật tố tụng/nghị quyết để câu hỏi NGẮN, tránh
    bị router hiểu nhầm thành "văn bản cần kiểm tra" (validate).
    """
    basis = _offense_basis(grouped)
    cites: list[str] = []
    for m in _CITE_RE.findall(basis):
        c = " ".join(m.split())  # gọn khoảng trắng
        if c not in cites:
            cites.append(c)
    if not cites:  # fallback: ghép từ ARTICLE rời
        cites = [f"Điều {n}" for n in _cited_articles(grouped)[:3]]
    law = "Bộ luật Hình sự năm 2015"
    lm = re.search(r"(Bộ luật [Hh]ình sự[^;.\n]*)", basis)
    if lm and "tố tụng" not in lm.group(1).lower():
        law = lm.group(1).strip()
    return "; ".join(cites[:4]), law


def _pronounced_penalty(grouped: dict) -> str:
    """Mức hình phạt TÒA TUYÊN (không phải mức VKS đề nghị).

    Heuristic: phần QUYẾT ĐỊNH viết dạng formal có số bằng chữ trong ngoặc
    ("02 (hai) năm tù") -> ưu tiên entity chứa '('; nếu không có thì lấy cái đầu.
    Tránh truyền NHIỀU mức (VD '03 năm tù; 02 năm tù') khiến LLM cộng dồn sai.
    """
    pens = _uniq(grouped, "PENALTY", 5)
    for p in pens:
        if "(" in p:
            return p
    return pens[0] if pens else "(không rõ mức hình phạt)"


def _cited_articles(grouped: dict) -> list[str]:
    """Số 'Điều N' duy nhất, gom từ ARTICLE + LEGAL_BASIS, giữ thứ tự xuất hiện."""
    nums: list[str] = []
    for t in _uniq(grouped, "ARTICLE", 20) + _uniq(grouped, "LEGAL_BASIS", 10):
        for m in _DIEU_RE.findall(t):
            if m not in nums:
                nums.append(m)
    return nums


def _chat_body(content: str, model: str, llm_model: str, stream: bool) -> dict:
    """Body cho /v1/chat/completions. ``llm_model`` = model LLM người dùng chọn ở
    chế độ trò chuyện (vd cc/claude-sonnet-4-6); để trống -> Module 1 dùng mặc định."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
        "temperature": 0.2,
    }
    if llm_model:
        body["llm_model"] = llm_model
    return body


def _rag_chat(content: str, model: str, llm_model: str = "", timeout: int = 240) -> str:
    """Một lượt hỏi Module 1 (stream=false). Trả về text trả lời."""
    try:
        r = requests.post(
            f"{RAG_URL}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {RAG_KEY}",
            },
            json=_chat_body(content, model, llm_model, stream=False),
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.RequestException as exc:
        raise AnalyzeError(f"không gọi được RAG Module 1 ({RAG_URL}): {exc}") from exc
    except (KeyError, ValueError) as exc:
        raise AnalyzeError(f"phản hồi RAG không hợp lệ: {exc}") from exc


def _build_scenario(case_meta: dict, grouped: dict) -> tuple[str, list[str]]:
    """Dựng câu hỏi "tình huống tư vấn" NGẮN GỌN + danh sách Điều đã viện dẫn.

    Câu ngắn dạng tư vấn -> router Module 1 đưa vào intent legal (retrieve), KHÔNG
    nhầm sang validate_document. (Câu dày 'Căn cứ/Áp dụng' + nhiều luật bị đẩy sang
    template validate.) Dùng tội danh + trích dẫn áp dụng (điểm/khoản/Điều của tội
    danh) + mức TÒA TUYÊN. Hỏi ĐÍCH DANH khoản chính để RAG không lấy nhầm khung.
    """
    arts = _cited_articles(grouped)[:6]
    crime = (_uniq(grouped, "CRIME", 1) or ["(không rõ tội danh)"])[0]
    penalty = _pronounced_penalty(grouped)
    applied, law = _applied_citations(grouped)
    main_cite = applied.split(";")[0].strip() if applied else "điều khoản áp dụng"

    scenario = (
        f'Một người bị Tòa án xử phạt {penalty} về tội "{crime}", '
        f"áp dụng {applied} {law}.\n\n"
        "Dựa trên quy định pháp luật, trả lời ngắn gọn bằng tiếng Việt, có đánh số:\n"
        f"1) Riêng {main_cite}: khung hình phạt tù là bao nhiêu năm (nêu CHÍNH XÁC "
        "khung của ĐÚNG khoản này, phân biệt rõ với các khoản khác của cùng điều)? "
        "Các điểm/khoản còn lại nêu trên quy định gì?\n"
        f"2) Mức hình phạt Tòa đã tuyên ({penalty}) có nằm trong khung của "
        f"{main_cite} không? So sánh cụ thể (chỉ xét đúng mức Tòa tuyên, không cộng dồn).\n"
        "3) Tội danh nêu trên có đúng với điều luật áp dụng không?\n"
        "4) Có điểm gì cần lưu ý không (tình tiết giảm nhẹ, mâu thuẫn về mức án)?\n\n"
        "Nhắc rằng đây là phân tích HỖ TRỢ, không phải kết luận pháp lý."
    )
    return scenario, arts


def run_analyze(
    case_meta: dict, grouped: dict, *, model: str = "legal-ai-graph", llm_model: str = ""
) -> dict:
    """Một lượt RAG kiểu "tình huống tư vấn" -> báo cáo đối chiếu (đồng bộ).

    Đã kiểm chứng cho câu trả lời đúng (khung Điều 250 = 2-7 năm, mức 2 năm nằm
    trong khung, tội danh khớp) kèm trích dẫn nguồn. Raises AnalyzeError khi RAG lỗi.
    """
    scenario, arts = _build_scenario(case_meta, grouped)
    analysis = _rag_chat(scenario, model, llm_model)

    # Guard: nếu router lỡ trả chào mẫu / template validate, thử LẠI một lần với
    # tiền tố ép rõ đây là câu hỏi tư vấn (không phải yêu cầu kiểm tra văn bản).
    if _looks_useless(analysis):
        retry = (
            "Đây là một CÂU HỎI TƯ VẤN PHÁP LUẬT về khung hình phạt, KHÔNG phải "
            "yêu cầu kiểm tra/thẩm định văn bản. Hãy trả lời trực tiếp:\n\n" + scenario
        )
        analysis = _rag_chat(retry, model, llm_model)

    return {
        "cited_articles": arts,
        "law_lookup": "",  # nội dung luật đã nằm trong phần phân tích (1 lượt)
        "analysis": analysis,
    }


def _sse(content: str) -> str:
    """Đóng gói 1 đoạn text thành SSE chunk kiểu OpenAI (client tái dùng parser chat)."""
    payload = json.dumps(
        {"choices": [{"delta": {"content": content}}]}, ensure_ascii=False
    )
    return f"data: {payload}\n\n"


def _sse_status(msg: str) -> str:
    """SSE chunk TRẠNG THÁI (không phải nội dung) — client hiện tạm rồi xoá."""
    return f"data: {json.dumps({'status': msg}, ensure_ascii=False)}\n\n"


def _sse_passages(passages: list[dict]) -> str:
    """SSE chunk ĐOẠN BẢN ÁN đã dùng (nguồn) — client hiện trong khối thu gọn."""
    return f"data: {json.dumps({'passages': passages}, ensure_ascii=False)}\n\n"


def _stream_rag_lines(scenario: str, model: str, llm_model: str = "", timeout: int = 300):
    """Generator: gọi Module 1 (stream=True) và CHUYỂN TIẾP nguyên các dòng SSE
    'data: {...}' kiểu OpenAI. Lỗi -> đẩy 1 chunk báo lỗi + '[DONE]'."""
    try:
        with requests.post(
            f"{RAG_URL}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {RAG_KEY}",
            },
            json=_chat_body(scenario, model, llm_model, stream=True),
            stream=True,
            timeout=timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line:  # chuyển tiếp dòng 'data: ...' (và 'data: [DONE]')
                    yield line + "\n\n"
    except requests.RequestException as exc:
        yield _sse(f"\n\n❌ Lỗi gọi RAG Module 1 ({RAG_URL}): {exc}")
        yield "data: [DONE]\n\n"


def _stream_rag_messages(messages: list[dict], model: str, llm_model: str = "", timeout: int = 300):
    """Như _stream_rag_lines nhưng gửi MẢNG messages (có system + lịch sử) thay vì
    một câu. Chuyển tiếp nguyên các dòng SSE 'data: {...}' từ Module 1."""
    body = {"model": model, "messages": messages, "stream": True, "temperature": 0.2}
    if llm_model:
        body["llm_model"] = llm_model
    try:
        with requests.post(
            f"{RAG_URL}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {RAG_KEY}",
            },
            json=body,
            stream=True,
            timeout=timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    yield line + "\n\n"
    except requests.RequestException as exc:
        yield _sse(f"\n\n❌ Lỗi gọi RAG Module 1 ({RAG_URL}): {exc}")
        yield "data: [DONE]\n\n"


# Nhãn tiếng Việt cho các trường HỒ SƠ VỤ ÁN bơm vào ngữ cảnh (gọn, ưu tiên dữ
# kiện hay được hỏi). Thứ tự = thứ tự hiển thị.
_FACT_FIELDS: list[tuple[str, str]] = [
    ("COURT", "Tòa án"),
    ("DEFENDANT", "Bị cáo"),
    ("PLAINTIFF", "Nguyên đơn"),
    ("VICTIM", "Bị hại"),
    ("CRIME", "Tội danh"),
    ("LEGAL_BASIS", "Căn cứ pháp lý"),
    ("ARTICLE", "Điều luật viện dẫn"),
    ("PENALTY", "Hình phạt"),
    ("COMPENSATION", "Bồi thường"),
    ("MONEY_AMOUNT", "Số tiền"),
    ("DECISION", "Quyết định"),
]


def _case_facts_block(case_meta: dict | None, grouped: dict | None) -> str:
    """Dựng khối HỒ SƠ VỤ ÁN cô đọng từ case_meta + thực thể NER (đã bóc ở /verify).

    Đặt cố định đầu ngữ cảnh → câu hỏi dữ kiện trả lời chuẩn ngay cả khi retrieve
    trượt. Rỗng -> trả "" (bỏ qua, giữ tương thích khi caller không truyền thực thể).
    """
    grouped = grouped or {}
    lines: list[str] = []
    cm = case_meta or {}
    head = "; ".join(
        v for v in (cm.get("case_number"), cm.get("case_type"), cm.get("procedure_stage")) if v
    )
    if head:
        lines.append(f"- Bản án: {head}")
    for key, label in _FACT_FIELDS:
        vals = _uniq(grouped, key, 6)
        if vals:
            lines.append(f"- {label}: {'; '.join(vals)}")
    if not lines:
        return ""
    return "HỒ SƠ VỤ ÁN (trích xuất tự động, dùng làm dữ kiện gốc):\n" + "\n".join(lines)


def _retrieval_query(question: str, history: list[dict]) -> str:
    """Câu hỏi nối tiếp ('vậy mức đó hợp lý không?') thiếu chủ ngữ → retrieve trượt.
    Gộp lượt NGƯỜI DÙNG gần nhất vào truy vấn (chỉ để tìm kiếm, không đưa cho LLM)
    nhằm khôi phục ngữ cảnh đã lược bỏ. Lượt đầu (không có history) giữ nguyên."""
    prev_user = ""
    for turn in reversed(history):
        if (turn.get("role") or "user") == "user" and (turn.get("content") or "").strip():
            prev_user = turn["content"].strip()
            break
    return f"{prev_user} {question}".strip() if prev_user else question


def stream_ask(
    text: str,
    question: str,
    history: list[dict] | None = None,
    *,
    case_meta: dict | None = None,
    grouped: dict | None = None,
    model: str = "legal-ai-graph",
    llm_model: str = "",
):
    """HYBRID hỏi-đáp: retrieve đoạn liên quan trong CHÍNH bản án + RAG kho luật.

    1. Chunk+embed bản án (mượn embedder Module 1) -> lấy top đoạn khớp câu hỏi
       (truy hồi LAI vector + BM25; câu nối tiếp được gộp ngữ cảnh lượt trước).
    2. Đưa HỒ SƠ VỤ ÁN (thực thể NER) + các đoạn đó làm ngữ cảnh (system) + lịch sử
       + câu hỏi xuống Module 1; Module 1 tự RAG kho luật rồi trả lời -> 3 nguồn.
    Cố ý KHÔNG dùng chữ 'file'/'văn bản này' để router không nhầm sang validate.
    """
    history = history or []
    yield _sse_status("Đang tìm đoạn liên quan trong bản án…")
    hits: list[tuple[str, float]] = []
    try:
        from api.judgment_rag import chunk_judgment, retrieve
        # top_k thích ứng: bản án dài (nhiều chunk) lấy nhiều ngữ cảnh hơn, nhưng
        # chặn trần để prompt không phình. Ngắn -> ít chunk, lấy hết phần liên quan.
        n_chunks = len(chunk_judgment(text)) if text else 0
        top_k = max(4, min(8, (n_chunks // 6) + 4)) if n_chunks else 5
        hits = retrieve(text, _retrieval_query(question, history), top_k=top_k)
    except Exception as exc:  # noqa: BLE001 — không truy được bản án thì vẫn trả lời theo kho luật
        yield _sse_status(f"(không truy xuất được bản án: {exc}) — trả lời theo kho luật")

    # Phát các đoạn đã dùng để client hiện nguồn (thu gọn). Kèm điểm cosine.
    if hits:
        yield _sse_passages(
            [{"text": c, "score": round(s, 3)} for c, s in hits]
        )

    facts = _case_facts_block(case_meta, grouped)
    ctx = "\n\n".join(f"[Đoạn {i + 1}] {c}" for i, (c, _s) in enumerate(hits))
    grounding = (
        "Bạn là trợ lý tư vấn pháp luật. Dưới đây là HỒ SƠ VỤ ÁN và các ĐOẠN trích "
        "từ MỘT bản án, đã chọn lọc theo câu hỏi và sắp theo thứ tự trong bản án. "
        "Quy tắc:\n"
        "- Trả lời dữ kiện vụ án (đương sự, hành vi, tội danh, mức án, quyết định…) "
        "DỰA TRƯỚC HẾT trên hồ sơ + các đoạn này; có thể dẫn 'Đoạn n' khi cần.\n"
        "- Nếu thông tin câu hỏi yêu cầu KHÔNG có trong hồ sơ/các đoạn, hãy nói rõ là "
        "bản án không nêu (hoặc đoạn trích chưa đủ) — TUYỆT ĐỐI không bịa dữ kiện.\n"
        "- Khi cần giải thích hoặc đối chiếu pháp lý, hãy tra cứu và TRÍCH DẪN điều "
        "luật từ kho luật.\n"
        "- Trả lời ngắn gọn, đúng trọng tâm, bằng tiếng Việt.\n\n"
        + (facts + "\n\n" if facts else "")
        + "ĐOẠN TRÍCH TỪ BẢN ÁN:\n" + (ctx or "(không tìm thấy đoạn phù hợp)")
    )
    messages: list[dict] = [{"role": "system", "content": grounding}]
    for turn in history[-6:]:
        role = turn.get("role") or "user"
        content = turn.get("content") or ""
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    yield _sse_status("Đang tra cứu kho luật và trả lời…")
    yield from _stream_rag_messages(messages, model, llm_model)


def stream_analyze(
    case_meta: dict, grouped: dict, *, model: str = "legal-ai-graph", llm_model: str = ""
):
    """STREAM phân tích ĐỐI CHIẾU PHÁP LÝ (citation ↔ tội danh ↔ khung hình phạt).

    Phát 1 status heartbeat ngay đầu rồi chuyển tiếp SSE từ Module 1. (Bỏ guard
    retry vì không thử lại được giữa luồng — scenario gọn đã route ổn định; có
    /analyze đồng bộ làm fallback.)
    """
    scenario, _arts = _build_scenario(case_meta, grouped)
    yield _sse_status("Đang tra cứu kho luật và đối chiếu… (có thể ~1–2 phút)")
    yield from _stream_rag_lines(scenario, model, llm_model)


# ── Phân tích CHUYÊN SÂU bản chất hành vi (cấu thành / định tội / hợp lý / logic) ──

_SECTION_FACTS = re.compile(r"NỘI\s*DUNG\s*VỤ\s*ÁN", re.IGNORECASE)
_SECTION_REASON = re.compile(r"NHẬN\s*ĐỊNH", re.IGNORECASE)


def _facts_section(text: str) -> str:
    """Trích phần DIỄN BIẾN HÀNH VI (giữa 'NỘI DUNG VỤ ÁN' và 'NHẬN ĐỊNH').

    Fallback: nếu không thấy mốc -> lấy một đoạn giữa văn bản (bỏ phần đầu hành
    chính). Cắt tối đa ~4000 ký tự cho gọn prompt.
    """
    if not text:
        return ""
    s = _SECTION_FACTS.search(text)
    e = _SECTION_REASON.search(text)
    if s and e and e.start() > s.end():
        seg = text[s.end() : e.start()]
    elif s:
        seg = text[s.end() : s.end() + 4000]
    else:
        seg = text[400:4000]
    return seg.strip(" :\n\t")[:4000]


def _strip_preamble(s: str) -> str:
    """Bỏ câu dẫn đầu mà router hay thêm ('Bạn…', 'Dưới đây là tóm tắt…')."""
    parts = re.split(r"(?<=[.:])\s+", s.strip())
    out: list[str] = []
    skipping = True
    for p in parts:
        low = p.lower()
        if skipping and any(
            k in low
            for k in ("bạn ", "dưới đây", "tóm tắt", "yêu cầu", "tôi hiểu",
                      "tôi sẽ", "như sau", "sự việc")
        ):
            continue
        skipping = False
        out.append(p)
    return (" ".join(out).strip() or s.strip())


def _conduct_summary(facts: str, model: str, llm_model: str = "") -> str:
    """Tóm tắt diễn biến hành vi bằng LLM thành ~2 câu NGẮN-SẠCH (để bước 2 route
    tư vấn, không bị validate). Dùng mode nhẹ (graph) cho nhanh; bỏ lời dẫn; cắt 350."""
    q = (
        "Tóm tắt sự việc sau trong ĐÚNG 2 câu, nêu: ai mua/vận chuyển/tàng trữ "
        "CHẤT gì và SỐ LƯỢNG, MỤC ĐÍCH, cách cất giấu/mang theo. Bắt đầu NGAY bằng "
        "nội dung, KHÔNG viết lời dẫn, KHÔNG trích dẫn luật:\n\n" + facts
    )
    return _strip_preamble(_rag_chat(q, model, llm_model, timeout=120).strip())[:350]


def _conduct_fallback(facts: str) -> str:
    """Dự phòng (không LLM): lấy ~2 câu đầu của diễn biến, cắt sạch ở ranh giới câu."""
    s = facts
    m = re.search(r"như sau\s*:?", s, re.IGNORECASE)
    if m:
        s = s[m.end():]
    s = re.sub(r"\s{2,}", " ", s.strip())[:320]
    cut = s.rfind(". ", 140)
    return (s[: cut + 1] if cut > 0 else s).strip()


# Map ĐẦY ĐỦ 31 nhãn thực thể (v3r) -> tên tiếng Việt, theo thứ tự config.ENTITY_TYPES.
# Phân tích sâu phải nhận HẾT thực thể NER bóc tách được, KHÔNG bỏ sót nhãn nào.
_ENTITY_VI = (
    ("CASE_NUMBER", "Số bản án/quyết định"),
    ("COURT", "Tòa án"),
    ("JUDGMENT_DATE", "Ngày xét xử/tuyên án"),
    ("CASE_TYPE", "Loại vụ án"),
    ("DEFENDANT", "Bị cáo"),
    ("PLAINTIFF", "Nguyên đơn"),
    ("VICTIM", "Bị hại"),
    ("RELATED_PARTY", "Người có quyền lợi, nghĩa vụ liên quan"),
    ("LAW_NAME", "Tên văn bản luật"),
    ("ARTICLE", "Điều"),
    ("CLAUSE", "Khoản"),
    ("POINT", "Điểm"),
    ("LEGAL_BASIS", "Căn cứ pháp lý"),
    ("CRIME", "Tội danh"),
    ("PENALTY", "Hình phạt"),
    ("MONEY_AMOUNT", "Số tiền"),
    ("COMPENSATION", "Bồi thường"),
    ("COURT_FEE", "Án phí/lệ phí"),
    ("DECISION", "Quyết định của tòa"),
    ("JUDGE", "Thẩm phán"),
    ("ASSESSOR", "Hội thẩm nhân dân"),
    ("PROSECUTOR", "Kiểm sát viên"),
    ("CLERK", "Thư ký phiên tòa"),
    ("LAWYER", "Luật sư/người bào chữa"),
    ("WITNESS", "Người làm chứng"),
    ("COURT_BEHAVIOR", "Thái độ/diễn biến tố tụng"),
    ("MITIGATING_FACTOR", "Tình tiết giảm nhẹ"),
    ("AGGRAVATING_FACTOR", "Tình tiết tăng nặng"),
    ("CRIMINAL_ACT", "Hành vi phạm tội"),
    ("QUANTITY", "Số lượng/khối lượng"),
    ("EVIDENCE_ITEM", "Tang vật, vật chứng"),
)


def _case_facts(grouped: dict) -> str:
    """Liệt kê TOÀN BỘ thực thể NER đã bóc tách (đủ 31 nhãn) thành khối dữ kiện
    cho prompt phân tích — KHÔNG bỏ sót nhãn nào có giá trị.

    Thứ tự theo config.ENTITY_TYPES; bất kỳ nhãn lạ nào (nếu mô hình đổi) vẫn được
    thêm ở cuối để không bao giờ rơi mất dữ kiện. Bỏ qua nhãn rỗng; trả "" nếu
    không có thực thể nào.
    """
    lines: list[str] = []
    used: set[str] = set()
    for label, show in _ENTITY_VI:
        used.add(label)
        vals = _uniq(grouped, label, 8)
        if vals:
            lines.append(f"- {show}: {'; '.join(vals)}")
    for label in grouped:  # an toàn: nhãn chưa map cũng đưa vào, không bỏ sót
        if label in used:
            continue
        vals = _uniq(grouped, label, 8)
        if vals:
            lines.append(f"- {label}: {'; '.join(vals)}")
    return "\n".join(lines)


def _build_deep_scenario(case_meta: dict, grouped: dict, conduct: str) -> str:
    """Bước 2: câu hỏi TƯ VẤN ĐỜI THƯỜNG về bản chất hành vi (4 chiều).

    QUAN TRỌNG: tránh thuật ngữ "cấu thành tội phạm / định tội / phân tích" — router
    Module 1 sẽ đẩy sang validate_document (template sai). Đã kiểm chứng: hỏi kiểu
    "gọi là tội X đúng chưa hay tội Y? mức án tương xứng không?" -> intent tư vấn,
    LLM vẫn trả lời đầy đủ bản chất (định tội, mức án, điểm bất hợp lý) + dẫn luật.

    Các dữ kiện NER (số lượng, tang vật, tình tiết) được CHÈN tường minh vào prompt
    để LLM dùng đúng, không "hỏi lại" những thứ đã bóc tách được.
    """
    crime = (_uniq(grouped, "CRIME", 1) or ["(không rõ tội danh)"])[0]
    penalty = _pronounced_penalty(grouped)
    applied, law = _applied_citations(grouped)
    main_cite = applied.split(";")[0].strip() if applied else "điều khoản áp dụng"
    facts_block = _case_facts(grouped)
    facts_part = (
        "\nTOÀN BỘ thực thể đã bóc tách từ bản án bằng mô hình NER (HÃY DÙNG ĐÚNG và "
        "ĐẦY ĐỦ các dữ kiện này, KHÔNG bỏ sót, KHÔNG hỏi lại hay đề nghị cung cấp "
        "thêm; dữ kiện nào không có thì nêu rõ 'bản án không thể hiện'):\n"
        + facts_block + "\n"
    ) if facts_block else ""
    return (
        "Cho tôi hỏi về pháp luật hình sự (chỉ để tìm hiểu). Trường hợp một người "
        f"có hành vi: {conduct} "
        f'Người này bị xử phạt {penalty} về tội "{crime}" (theo {main_cite} {law}). '
        + facts_part +
        "Tôi muốn hiểu rõ (trả lời bằng tiếng Việt, có đánh số):\n"
        f'1) Gọi là tội "{crime}" đã đúng chưa, hay đúng hơn có thể là một tội danh '
        "khác? Xét về mục đích, cách thức và ý thức, các tội này khác nhau ở dấu hiệu nào?\n"
        f"2) Mức {penalty} có tương xứng với tính chất, mức độ, số lượng và mục đích "
        "không (xét cả tình tiết tăng nặng và giảm nhẹ ĐÃ NÊU Ở TRÊN, không yêu cầu "
        "thêm thông tin)?\n"
        "3) Có điểm gì chưa hợp lý hoặc cần lưu ý thêm không?"
    )


def _sse_chunks(text: str, size: int = 280):
    """Cắt 1 câu trả lời thành nhiều SSE chunk để client hiện dần (mượt hơn 1 cục)."""
    for i in range(0, len(text), size):
        yield _sse(text[i : i + size])


def stream_deep_analyze(
    case_meta: dict, grouped: dict, text: str, *, model: str = "legal-ai-graph",
    llm_model: str = "",
):
    """STREAM phân tích CHUYÊN SÂU bản chất hành vi (định tội / mức án / điểm bất hợp lý).

    Bước 1: tóm tắt diễn biến hành vi (LLM mode nhẹ; lỗi -> fallback heuristic).
    Bước 2: hỏi RAG/LLM bằng NGÔN NGỮ TƯ VẤN. Router Module 1 là LLM (KHÔNG tất
    định) -> đôi khi vẫn đẩy sang validate; nên ở đây GỌI NON-STREAM + GUARD: nếu
    câu trả lời là template validate/chào thì THỬ LẠI 1 lần với tiền tố ép tư vấn,
    rồi mới phát kết quả (đánh đổi token-streaming lấy độ tin cậy). Heartbeat giữa
    các bước vẫn báo tiến trình.
    """
    yield _sse_status("Đang đọc & tóm tắt diễn biến hành vi…")
    facts = _facts_section(text)
    if len(facts) < 60:
        yield _sse(
            "\n\n⚠ Không trích được phần *diễn biến hành vi* từ văn bản (PDF có thể "
            "thiếu mục “NỘI DUNG VỤ ÁN”). Hãy thử bản án khác có đầy đủ phần nội dung."
        )
        yield "data: [DONE]\n\n"
        return
    try:
        conduct = _conduct_summary(facts, "legal-ai-graph", llm_model)
        if _looks_useless(conduct) or len(conduct) < 30:
            conduct = _conduct_fallback(facts)
    except AnalyzeError:
        conduct = _conduct_fallback(facts)

    yield _sse_status("Đang đối chiếu định tội, mức án với quy định pháp luật…")
    scenario = _build_deep_scenario(case_meta, grouped, conduct)
    # Router Module 1 (LLM) KHÔNG tất định -> thử tối đa 3 lần với tiền tố ép tư
    # vấn tăng dần; lấy câu trả lời hợp lệ đầu tiên.
    prefixes = (
        "",
        "Đây là câu hỏi TƯ VẤN pháp luật để tìm hiểu, KHÔNG phải yêu cầu kiểm tra "
        "văn bản. Hãy trả lời trực tiếp:\n\n",
        "Bạn là chuyên gia tư vấn luật hình sự. Trả lời TRỰC TIẾP câu hỏi tư vấn "
        "dưới đây, TUYỆT ĐỐI KHÔNG dùng mẫu “Báo cáo kiểm tra văn bản”:\n\n",
    )
    answer = ""
    try:
        for i, pre in enumerate(prefixes):
            if i > 0:
                yield _sse_status(f"Đang thử lại theo hướng tư vấn… (lần {i + 1}/3)")
            answer = _rag_chat(pre + scenario, model, llm_model)
            if not _looks_useless(answer):
                break
    except AnalyzeError as exc:
        yield _sse(f"\n\n❌ Lỗi gọi RAG Module 1: {exc}")
        yield "data: [DONE]\n\n"
        return

    if _looks_useless(answer):  # vẫn lỗi sau 3 lần -> báo rõ thay vì đổ template
        answer = (
            "⚠ Hệ thống RAG tạm thời định tuyến sai câu hỏi (trả về mẫu kiểm tra "
            "văn bản). Vui lòng bấm **Phân tích lại** — thường lần sau sẽ thành công."
        )
    yield from _sse_chunks(answer)
    yield "data: [DONE]\n\n"
