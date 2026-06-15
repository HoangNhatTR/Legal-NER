"""FastAPI app: judgment PDF -> structured legal entities (v1).

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8100

The Model 1 NER checkpoint is loaded ONCE at startup (lifespan) and reused
across requests. Synchronous /extract; no auth, no job queue (later phases).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from api import jobs
from api.odl_adapter import (
    DEFAULT_JAVA_HOME,
    _ensure_java,
    _hybrid_base_url,
    hybrid_healthy,
    shutdown_hybrid_server,
)
from api.schemas import (
    ExtractResponse,
    HealthResponse,
    JobListItem,
    JobStatusResponse,
    JobSubmitResponse,
    OdlHealth,
)
from api.service import (
    DEFAULT_EXTRACTOR,
    EXTRACTORS,
    ModelHolder,
    NotAPdfError,
    ScannedPdfError,
    looks_like_pdf,
    run_extract,
)

_EXTRACTOR_DESC = (
    "Extraction backend: 'opendataloader' (DEFAULT — opendataloader-pdf: "
    "structured digital extraction for text-layer PDFs + hybrid docling/EasyOCR "
    "OCR for scans) or 'pymupdf' (fallback — PyMuPDF text layer + Phase-1 "
    "VietOCR for scans). The opendataloader backend requires JDK 11+ "
    "(LEGAL_NER_ODL_JAVA_HOME) and, for scans, a hybrid OCR server "
    "(LEGAL_NER_ODL_HYBRID_PORT; auto-started on first scan if not running)."
)


def _java_available() -> bool:
    """Best-effort check that the portable JDK java binary is runnable."""
    import subprocess

    java_home = _ensure_java()
    java_bin = Path(java_home) / "bin" / "java"
    try:
        out = subprocess.run(
            [str(java_bin), "-version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return out.returncode == 0
    except Exception:
        return False


holder = ModelHolder()


def _job_runner(job: dict, data: bytes) -> dict:
    """Execute one queued job in the worker thread; return a JSON-able result.

    Runs the SAME run_extract / run_verify as the sync endpoints against the
    shared GPU ``holder`` (one worker => OCR is serialized). The result is
    validated through the public response model so the stored JSON is byte-for-
    byte the same shape the sync endpoints return. NotAPdfError / ScannedPdfError
    propagate as ordinary exceptions -> the queue records them as job errors.
    """
    kind = job["kind"]
    filename = job.get("filename") or "upload.pdf"
    params = job.get("params") or {}

    if not holder.loaded:
        raise RuntimeError("model not loaded")
    if not looks_like_pdf(data, None):
        raise NotAPdfError(
            "uploaded file is not a PDF (expected %PDF magic bytes)"
        )

    if kind == "extract":
        result = run_extract(
            filename,
            data,
            holder,
            allow_ocr=params.get("allow_ocr", True),
            extractor=params.get("extractor", DEFAULT_EXTRACTOR),
        )
        return ExtractResponse(**result).model_dump()

    raise ValueError(f"unknown job kind {kind!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ODL is the default extractor and needs a JDK 11+ — point this process at
    # the portable JDK BEFORE serving any request (idempotent; adapter also does
    # this per-call, but we set it early so digital ODL works from cold start).
    _ensure_java()
    # load the model exactly once, before serving any request
    holder.load()
    print(
        f"[api] model loaded: path={holder.model_path} device={holder.device}",
        flush=True,
    )
    print(f"[api] JAVA_HOME={DEFAULT_JAVA_HOME} (ODL default extractor)", flush=True)
    # start the single background worker + reconcile any jobs left by a previous
    # run (running -> error; queued -> re-enqueue). See api/jobs.py for rationale.
    counts = jobs.init_jobs(_job_runner)
    print(
        f"[api] job queue started (in-process + SQLite); "
        f"orphaned_running={counts['orphaned_running']} "
        f"requeued_queued={counts['requeued_queued']}",
        flush=True,
    )
    yield
    # stop accepting work and let the in-flight job drain (bounded), then tear
    # down the managed hybrid OCR subprocess (if we auto-started one).
    jobs.shutdown_jobs()
    print("[api] job queue stopped", flush=True)
    shutdown_hybrid_server()


app = FastAPI(
    title="Legal NER API",
    version="1.0.0",
    description="Vietnamese judgment PDF -> structured entities (Model 1, v1).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health():
    # ODL status is best-effort and MUST NOT make /health fail.
    try:
        odl = OdlHealth(
            java_ok=_java_available(),
            java_home=DEFAULT_JAVA_HOME,
            hybrid_reachable=hybrid_healthy(timeout=1.5),
            hybrid_url=_hybrid_base_url(),
            default_extractor=DEFAULT_EXTRACTOR,
        )
    except Exception:
        odl = None
    return HealthResponse(
        status="ok" if holder.loaded else "loading",
        model_loaded=holder.loaded,
        device=holder.device,
        model_path=holder.model_path,
        odl=odl,
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(...),
    allow_ocr: bool = Query(
        True,
        description=(
            "Route scanned PDFs (no text layer) through the OCR pipeline. "
            "Set false for fast-only behavior: scanned PDFs then return 422."
        ),
    ),
    extractor: str = Query(DEFAULT_EXTRACTOR, description=_EXTRACTOR_DESC),
):
    """Synchronous extraction (blocks until done).

    Best for fast text-layer PDFs. For SCANNED PDFs the OCR step can take
    minutes and will block the HTTP request the whole time — use the async
    variant ``POST /jobs/extract`` (submit -> job_id -> poll GET /jobs/{id})
    instead.
    """
    if not holder.loaded:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    if extractor not in EXTRACTORS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown extractor {extractor!r}; expected one of {list(EXTRACTORS)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    if not looks_like_pdf(data, file.content_type):
        raise HTTPException(
            status_code=400,
            detail="uploaded file is not a PDF (expected %PDF magic bytes or a PDF content-type)",
        )

    try:
        result = run_extract(
            file.filename or "upload.pdf",
            data,
            holder,
            allow_ocr=allow_ocr,
            extractor=extractor,
        )
    except NotAPdfError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ScannedPdfError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ExtractResponse(**result)


# ---------------------------------------------------------------------------
# Async job queue — submit slow (scanned-PDF OCR) work without blocking HTTP.
# Submit -> {job_id, status:"queued"} (202) -> poll GET /jobs/{id} for result.
# Jobs are SERIALIZED (one OCR at a time; single GPU + single hybrid OCR server)
# and persisted to data/jobs.db. See api/jobs.py for the design rationale.
# ---------------------------------------------------------------------------


async def _read_and_validate(file: UploadFile, extractor: str) -> bytes:
    """Shared upload validation for the async submit endpoints."""
    if not holder.loaded:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if extractor not in EXTRACTORS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown extractor {extractor!r}; expected one of {list(EXTRACTORS)}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    # NOTE: we DO NOT reject non-PDF uploads here. A bad upload is accepted as a
    # job and fails inside the worker (status -> error with a clear message),
    # which is exactly what test #4 exercises. Magic-byte mismatch surfaces as a
    # NotAPdfError in the runner.
    return data


def _submit_response(job_id: str, request: Request) -> JSONResponse:
    poll_url = str(request.url_for("get_job_status", job_id=job_id))
    body = JobSubmitResponse(
        job_id=job_id, status="queued", poll_url=poll_url
    ).model_dump()
    return JSONResponse(status_code=202, content=body)


@app.post("/jobs/extract", response_model=JobSubmitResponse, status_code=202)
async def submit_extract_job(
    request: Request,
    file: UploadFile = File(...),
    allow_ocr: bool = Query(True, description="route scans through OCR (else job errors)"),
    extractor: str = Query(DEFAULT_EXTRACTOR, description=_EXTRACTOR_DESC),
):
    """Enqueue an extraction job (async). Returns 202 + job_id immediately.

    Use this for SCANNED PDFs whose OCR takes minutes. Poll GET /jobs/{job_id}
    until status=='done'; the result field then holds the full ExtractResponse.
    """
    data = await _read_and_validate(file, extractor)
    job_id = jobs.submit_job(
        "extract",
        file.filename or "upload.pdf",
        {"allow_ocr": allow_ocr, "extractor": extractor},
        data,
    )
    return _submit_response(job_id, request)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, name="get_job_status")
async def get_job_status(job_id: str):
    """Poll a job. status in {queued, running, done, error}.

    When status=='done', ``result`` carries the full ExtractResponse /
    VerifyResponse. When status=='error', ``error`` carries the message.
    """
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job with id {job_id!r}")
    return JobStatusResponse(**job)


@app.get("/jobs", response_model=list[JobListItem])
async def list_jobs(limit: int = Query(50, ge=1, le=500)):
    """Recent jobs (newest first), metadata only (no result payloads)."""
    return [JobListItem(**j) for j in jobs.recent_jobs(limit)]
