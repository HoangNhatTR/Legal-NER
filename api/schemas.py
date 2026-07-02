"""Pydantic v2 response models for the legal-NER API."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

# Reuse the Layer-2 / Layer-4 pydantic models as-is (both are pydantic v2
# BaseModels). Make the package root importable so ``verify.*`` resolves when
# uvicorn is launched from the legal_ner/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify.citation import CitationReport  # noqa: E402,F401
from verify.existence import ExistenceReport  # noqa: E402,F401
from verify.forgery import ForgeryReport  # noqa: E402,F401


class OdlHealth(BaseModel):
    """Best-effort status of the opendataloader (default) extraction backend."""

    java_ok: bool = Field(
        description="portable JDK 11+ (JAVA_HOME) present and runnable"
    )
    java_home: str = Field(description="JAVA_HOME the API points ODL at")
    hybrid_reachable: bool = Field(
        description="hybrid OCR server answers GET /health (needed for scans)"
    )
    hybrid_url: str = Field(description="host:port the hybrid OCR server is expected on")
    default_extractor: str = Field(description="extractor used when no ?extractor= given")


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' once the model is loaded")
    model_loaded: bool
    device: str = Field(description="'cuda' or 'cpu' the model runs on")
    model_path: str
    odl: OdlHealth | None = Field(
        default=None,
        description="opendataloader backend status (java + hybrid OCR server)",
    )

    # the field name starts with 'model_' which collides with pydantic's
    # protected namespace; disable it so no warning is emitted.
    model_config = {"protected_namespaces": ()}


class CaseMeta(BaseModel):
    case_number: str | None = None
    case_type: str | None = Field(
        default=None,
        description="e.g. 'hình sự', 'dân sự' — parsed from the case-number suffix",
    )
    procedure_stage: str | None = Field(
        default=None, description="'sơ thẩm' or 'phúc thẩm'"
    )


class Entity(BaseModel):
    type: str = Field(description="entity type, e.g. DEFENDANT, ARTICLE, PENALTY")
    text: str
    start: int = Field(description="char offset into the flattened text stream")
    end: int
    score: float | None = None


class OcrPage(BaseModel):
    page_index: int | None = None
    source_type: str | None = None
    engine: str | None = None
    n_blocks: int | None = None
    mean_confidence: float | None = None
    skipped: bool = False
    quality: str | None = None


class OcrInfo(BaseModel):
    """Present only when the PDF was scanned and routed through OCR."""

    engine: str = Field(description="OCR engine used: paddle | vl | layout")
    device: str = Field(description="'cuda' or 'cpu' the OCR subprocess ran on")
    page_count: int
    mean_confidence: float | None = Field(
        default=None, description="mean OCR block confidence across scan pages"
    )
    quality_warnings: int = 0
    skipped_pages: int = 0
    per_page: list[OcrPage] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    filename: str
    case_meta: CaseMeta
    entities: list[Entity]
    entities_grouped: dict[str, list[Entity]] = Field(
        description="entities bucketed by type, e.g. {'DEFENDANT': [...], 'ARTICLE': [...]}"
    )
    num_pages: int
    char_count: int = Field(description="length of the flattened text stream")
    warnings: list[str] = Field(default_factory=list)
    ocr: OcrInfo | None = Field(
        default=None,
        description="OCR metadata; null for native text-layer PDFs",
    )
    extractor: str = Field(
        default="pymupdf",
        description="extraction backend used: 'pymupdf' or 'opendataloader'",
    )
    text: str = Field(
        default="",
        description="full normalized text stream (for downstream RAG/validate)",
    )


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# /verify — full happy-path verification chain (L4 forgery + extract + L2)
# ---------------------------------------------------------------------------


class ExtractBlock(BaseModel):
    """The /extract result embedded inside the /verify response.

    Mirrors ExtractResponse minus ``filename`` (carried at the top level).
    """

    case_meta: CaseMeta
    entities: list[Entity]
    entities_grouped: dict[str, list[Entity]] = Field(
        description="entities bucketed by type, e.g. {'DEFENDANT': [...]}"
    )
    num_pages: int
    char_count: int = Field(description="length of the flattened text stream")
    warnings: list[str] = Field(default_factory=list)
    ocr: OcrInfo | None = None
    extractor: str = Field(
        default="pymupdf",
        description="extraction backend used: 'pymupdf' or 'opendataloader'",
    )
    text: str = Field(
        default="",
        description="full normalized text stream (for downstream RAG/validate)",
    )


class ExistenceSkipped(BaseModel):
    """Returned in place of an ExistenceReport when L2 was not run."""

    status: str = Field(description="'skipped'")
    reason: str = Field(description="why the existence lookup was not performed")


class CitationSkipped(BaseModel):
    """Returned in place of a CitationReport when L1/L3 were not run."""

    status: str = Field(description="'skipped'")
    reason: str = Field(description="why citation verification was not performed")


class OverallVerdict(BaseModel):
    verdict_vi: str = Field(
        description="honest, non-alarmist Vietnamese synthesis of all layers"
    )
    flags: list[str] = Field(
        default_factory=list,
        description="short Vietnamese flags surfacing each signal worth checking",
    )
    confidence_note_vi: str = Field(
        description="reminder that this is decision support, not a legal verdict"
    )


class VerifyResponse(BaseModel):
    filename: str
    extract: ExtractBlock
    forgery: ForgeryReport
    existence: ExistenceReport | ExistenceSkipped = Field(
        description="L2 result, or {status:'skipped', reason} when not run"
    )
    citation: CitationReport | CitationSkipped = Field(
        description=(
            "L1 (cited-law existence) + L3 (sentencing-frame) result over BLHS "
            "2015 citations, or {status:'skipped', reason} when not run"
        )
    )
    overall: OverallVerdict


# ---------------------------------------------------------------------------
# /analyze — phân tích so sánh qua RAG Module 1 (né validate_document)
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Body cho POST /analyze: dữ liệu đã bóc tách (từ /extract hoặc /verify)."""

    case_meta: CaseMeta | None = None
    entities_grouped: dict[str, list[Entity]] = Field(
        default_factory=dict,
        description="entities bucketed by type (lấy từ /extract hoặc /verify)",
    )
    rag_model: str = Field(
        default="legal-ai-graph",
        description="RAG mode của Module 1: legal-ai-graph | legal-ai-top15 | legal-ai-full",
    )
    llm_model: str = Field(
        default="",
        description="Model LLM người dùng chọn ở chế độ trò chuyện (vd cc/claude-sonnet-4-6); "
        "để trống -> Module 1 dùng LLM mặc định",
    )
    text: str = Field(
        default="",
        description="toàn văn bản án (cho /analyze/deep — trích diễn biến hành vi)",
    )


class AskTurn(BaseModel):
    role: str = "user"
    content: str = ""


class AskRequest(BaseModel):
    """Body cho POST /analyze/ask/stream: hỏi-đáp hybrid trên CHÍNH bản án + kho luật."""

    text: str = Field(default="", description="toàn văn bản án (để chunk + embed)")
    question: str = Field(description="câu hỏi của người dùng về bản án")
    history: list[AskTurn] = Field(
        default_factory=list, description="các lượt hỏi-đáp trước (để nhớ ngữ cảnh)"
    )
    # Thực thể đã bóc ở bước /verify — đưa vào ngữ cảnh như một HỒ SƠ VỤ ÁN cố định
    # để câu hỏi dữ kiện (ai là bị cáo, tội gì, mức án…) trả lời chuẩn dù retrieve trượt.
    case_meta: CaseMeta | None = None
    entities_grouped: dict[str, list[Entity]] = Field(
        default_factory=dict,
        description="thực thể bucket theo loại (từ /extract hoặc /verify); tuỳ chọn",
    )
    rag_model: str = Field(default="legal-ai-graph", description="RAG mode của Module 1")
    llm_model: str = Field(default="", description="model LLM chọn ở chế độ trò chuyện")


class AnalyzeResponse(BaseModel):
    cited_articles: list[str] = Field(description="số Điều luật đã tra cứu")
    law_lookup: str = Field(description="nội dung điều luật RAG tra được (markdown)")
    analysis: str = Field(description="báo cáo đối chiếu/so sánh (markdown)")


# ---------------------------------------------------------------------------
# /precedents — kho tiền lệ + so khớp bản án tương tự
# ---------------------------------------------------------------------------


class PrecedentAddRequest(BaseModel):
    """Lưu một bản án (đã trích xuất) vào kho tiền lệ cục bộ."""

    case_meta: CaseMeta | None = None
    entities_grouped: dict[str, list[Entity]] = Field(default_factory=dict)
    text: str = Field(default="", description="toàn văn bản án (để tóm tắt diễn biến)")
    source: str = Field(default="confirmed", description="'confirmed' (người dùng) | 'an_le'")
    title: str = Field(default="", description="nhãn hiển thị (vd 'Án lệ số 30/2020/AL ...')")
    url: str = Field(default="", description="nguồn (nếu có)")


class PrecedentMatchRequest(BaseModel):
    """Tìm các tiền lệ/bản án tương tự với bản án đang xem."""

    case_meta: CaseMeta | None = None
    entities_grouped: dict[str, list[Entity]] = Field(default_factory=dict)
    text: str = Field(default="")
    top_k: int = Field(default=5, ge=1, le=20)
    sources: list[str] | None = Field(
        default=None, description="lọc nguồn: ['an_le'] | ['confirmed'] | None = cả hai"
    )


class PrecedentSearchRequest(BaseModel):
    """Tìm kiếm tiền lệ TRỰC TIẾP bằng câu/từ khoá tự do (không cần upload bản án)."""

    query: str = Field(description="câu hoặc từ khoá tìm kiếm (vd 'cướp tài sản dùng vũ khí')")
    top_k: int = Field(default=8, ge=1, le=30)
    sources: list[str] | None = Field(default=None, description="['an_le'] | ['confirmed'] | None")


class PrecedentItem(BaseModel):
    id: str
    source: str
    title: str = ""
    case_number: str = ""
    court: str = ""
    case_type: str = ""
    crimes: list[str] = Field(default_factory=list)
    articles: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    facts_summary: str = ""
    url: str = ""
    score: float = Field(description="độ tương đồng cosine (0–1)")
    related_by_law: bool = Field(description="có chung điều luật/tội danh hay không")


class PrecedentMatchResponse(BaseModel):
    matches: list[PrecedentItem] = Field(default_factory=list)
    query_crimes: list[str] = Field(default_factory=list)
    query_articles: list[str] = Field(default_factory=list)


class PrecedentAddResponse(BaseModel):
    id: str
    total: int = Field(description="tổng số tiền lệ trong kho sau khi thêm")


class PrecedentStatsResponse(BaseModel):
    total: int
    an_le: int
    confirmed: int


# ---------------------------------------------------------------------------
# Async job queue (POST /jobs/extract, /jobs/verify -> poll GET /jobs/{id})
# ---------------------------------------------------------------------------


class JobSubmitResponse(BaseModel):
    """202 Accepted body returned when a job is enqueued."""

    job_id: str
    status: str = Field(description="always 'queued' at submit time")
    poll_url: str = Field(description="GET this URL to poll the job result")


class JobStatusResponse(BaseModel):
    """Full job record returned by GET /jobs/{job_id}."""

    job_id: str
    status: str = Field(description="queued | running | done | error")
    kind: str = Field(description="extract | verify")
    filename: str | None = None
    params: dict | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: dict | None = Field(
        default=None,
        description="full ExtractResponse/VerifyResponse when status=='done'",
    )
    error: str | None = Field(
        default=None, description="error message when status=='error'"
    )


class JobListItem(BaseModel):
    """Lightweight job row for GET /jobs (no result payload)."""

    job_id: str
    status: str
    kind: str
    filename: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
