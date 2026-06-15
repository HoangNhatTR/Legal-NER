"""OCR adapter: run the Phase-1 OCR pipeline on a scanned PDF as a subprocess.

Why a subprocess (not an import)?
  * The OCR pipeline's heavy deps (vietocr, pix2tex, ultralytics, easyocr,
    paddleocr, pytesseract) live in a SEPARATE venv
    ``OCR/phase1/.venv`` — they are NOT installed in the API's HoangEnv.
  * The OCR code is cwd-sensitive (uses relative paths like ``./corrector_vi``,
    ``phase2/...``), so it must run with ``cwd=OCR/phase1``.

So we shell out to ``OCR/phase1/.venv/bin/python pipeline.py <pdf> -o <out> ...``
in that working directory, then read back the produced ``{stem}.json`` and
concatenate every block's text IN READING ORDER into one plain-text string —
the same kind of stream the native text-layer path feeds into normalize/infer.

GPU note: the OCR subprocess loads its OWN models (PP-DocLayoutV3, VietOCR,
PaddleOCR, ...). The GPU is shared and near-full (~3-4 GB free) and the NER
checkpoint is already resident in the API process, so the GPU run can OOM. We
try ``--gpu`` first; on OOM / CUDA failure we transparently retry on CPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# ── Locations (resolved once) ──────────────────────────────────────────────
_AIHOANG_ROOT = Path(__file__).resolve().parents[3]          # /home/tts/AI/AIHoang
OCR_DIR = _AIHOANG_ROOT / "OCR" / "phase1"
OCR_PYTHON = OCR_DIR / ".venv" / "bin" / "python"
OCR_PIPELINE = OCR_DIR / "pipeline.py"

# Defaults — overridable via env so ops can tune without code changes.
#   LEGAL_NER_OCR_ENGINE    : paddle | vl | layout      (default: layout)
#   LEGAL_NER_OCR_TIMEOUT   : seconds for the whole subprocess (default: 1800)
#   LEGAL_NER_OCR_USE_GPU   : "1"/"0" — try GPU first    (default: 1)
#   LEGAL_NER_OCR_CPU_FALLBACK : "1"/"0" — retry on CPU after a GPU OOM (default: 1)
#
# Engine note (measured on this box): the ``paddle`` engine SEGFAULTS at
# PaddleOCR predictor init in this venv (PIR parameter load), on both GPU and
# CPU — it is unusable here. The ``layout`` engine (PP-DocLayoutV3 + VietOCR)
# works on GPU and is the default. It is slow (~1-2 min/page incl. a one-time
# ~20-40s model load), so the timeout is generous and OCR requests can take
# several minutes for multi-page scans.
DEFAULT_ENGINE = os.environ.get("LEGAL_NER_OCR_ENGINE", "layout")
DEFAULT_TIMEOUT = int(os.environ.get("LEGAL_NER_OCR_TIMEOUT", "1800"))
DEFAULT_USE_GPU = os.environ.get("LEGAL_NER_OCR_USE_GPU", "1") == "1"
DEFAULT_CPU_FALLBACK = os.environ.get("LEGAL_NER_OCR_CPU_FALLBACK", "1") == "1"

# Substrings that signal a CUDA out-of-memory / GPU failure in the subprocess
# output, used to decide whether a CPU retry is worth attempting. A bare
# segfault (SIGSEGV) at predictor init under VRAM pressure is included.
_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda error",
    "cublas",
    "cudnn",
    "no kernel image",
    "cuda runtime error",
    "segmentation fault",
    "sigsegv",
)


class OcrError(Exception):
    """The OCR subprocess failed in a way we cannot recover from."""


@dataclass
class OcrResult:
    text: str
    engine: str
    device: str                      # "cuda" | "cpu"
    page_count: int
    mean_confidence: float | None    # None if no confidences available
    quality_warnings: int = 0
    skipped_pages: int = 0
    per_page: list[dict] = field(default_factory=list)


def _venv_ok() -> tuple[bool, str]:
    if not OCR_PYTHON.exists():
        return False, f"OCR venv python not found: {OCR_PYTHON}"
    if not OCR_PIPELINE.exists():
        return False, f"OCR pipeline not found: {OCR_PIPELINE}"
    return True, ""


def _build_cmd(pdf_path: Path, out_dir: Path, engine: str, use_gpu: bool) -> list[str]:
    cmd = [
        str(OCR_PYTHON),
        str(OCR_PIPELINE),
        str(pdf_path),
        "-o",
        str(out_dir),
        "--engine",
        engine,
        "--quiet",            # keep stdout clean; we read the JSON file, not stdout
    ]
    if use_gpu:
        cmd.append("--gpu")
    return cmd


def _run_subprocess(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    # Force CUDA allocator to be a little more forgiving about fragmentation;
    # harmless on CPU runs.
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return subprocess.run(
        cmd,
        cwd=str(OCR_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _looks_like_oom(proc: subprocess.CompletedProcess) -> bool:
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    blob = blob.lower()
    return any(m in blob for m in _OOM_MARKERS)


def _block_text(block: dict) -> str:
    """Plain text of one block: prefer OCR/digital text, fall back to table HTML.

    Tables are kept as their HTML so downstream normalize() still sees the cell
    text (tags are stripped later by flatten/normalize); figures/formulas with
    no text contribute nothing.
    """
    txt = block.get("text") or ""
    if txt:
        return txt
    # table_html carries cell text; keep it so e.g. tabular party lists survive.
    if block.get("table_html"):
        return block["table_html"]
    if block.get("embedded_text"):
        return block["embedded_text"]
    return ""


def _concat_reading_order(doc_json: dict) -> tuple[str, list[dict], list[float]]:
    """Concatenate every page's blocks IN READING ORDER -> one text stream.

    Pages are already sorted by (page_index, sub_page) by the pipeline. Within a
    page we re-sort by the block ``order`` field to be safe, then join.
    Returns (full_text, per_page_summaries, all_confidences).
    """
    parts: list[str] = []
    per_page: list[dict] = []
    confidences: list[float] = []

    for page in doc_json.get("pages", []):
        blocks = page.get("blocks", []) or []
        # sort by reading order when present; stable for blocks without 'order'
        blocks = sorted(blocks, key=lambda b: b.get("order", 0))

        page_text_parts: list[str] = []
        page_confs: list[float] = []
        for b in blocks:
            t = _block_text(b)
            if t:
                page_text_parts.append(t)
            conf = b.get("confidence")
            # confidence 1.0 on digital pages is not an OCR signal; only count
            # genuine OCR confidences (< 1.0 or explicitly present on scans)
            if isinstance(conf, (int, float)) and page.get("source_type") == "scan":
                page_confs.append(float(conf))

        page_text = "\n".join(page_text_parts)
        if page_text:
            parts.append(page_text)

        confidences.extend(page_confs)
        per_page.append(
            {
                "page_index": page.get("page_index"),
                "source_type": page.get("source_type"),
                "engine": page.get("engine"),
                "n_blocks": len(blocks),
                "mean_confidence": (
                    round(sum(page_confs) / len(page_confs), 4) if page_confs else None
                ),
                "skipped": bool(page.get("skipped", False)),
                "quality": (page.get("quality") or {}).get("quality"),
            }
        )

    full_text = "\n\n".join(parts)
    return full_text, per_page, confidences


def _parse_output(out_dir: Path, stem: str, engine: str, device: str) -> OcrResult:
    json_path = out_dir / f"{stem}.json"
    doc_json = json.loads(json_path.read_text(encoding="utf-8"))
    full_text, per_page, confs = _concat_reading_order(doc_json)
    mean_conf = round(sum(confs) / len(confs), 4) if confs else None
    return OcrResult(
        text=full_text,
        engine=engine,
        device=device,
        page_count=doc_json.get("total_pages", len(per_page)),
        mean_confidence=mean_conf,
        quality_warnings=doc_json.get("quality_warnings", 0),
        skipped_pages=doc_json.get("skipped_pages", 0),
        per_page=per_page,
    )


def run_ocr_on_pdf(
    data: bytes,
    *,
    engine: str = DEFAULT_ENGINE,
    timeout: int = DEFAULT_TIMEOUT,
    use_gpu: bool = DEFAULT_USE_GPU,
    cpu_fallback: bool = DEFAULT_CPU_FALLBACK,
) -> OcrResult:
    """Run the OCR pipeline on PDF bytes and return concatenated text + summary.

    Strategy:
      * If ``use_gpu``: try GPU. On success, return.
      * On a GPU OOM / segfault-at-init (VRAM pressure) AND ``cpu_fallback``,
        retry the whole pipeline on CPU.
      * A GPU *timeout* is NOT retried on CPU (CPU is slower — it would only
        time out again); it raises immediately so the caller fails fast.
      * If ``use_gpu`` is False, run straight on CPU.

    Raises ``OcrError`` if no attempt yields usable output.
    """
    ok, why = _venv_ok()
    if not ok:
        raise OcrError(why)

    with tempfile.TemporaryDirectory(prefix="legal_ner_ocr_") as tmp:
        tmp_dir = Path(tmp)
        pdf_path = tmp_dir / "input.pdf"
        out_dir = tmp_dir / "out"
        pdf_path.write_bytes(data)
        stem = pdf_path.stem

        # ── GPU attempt ──────────────────────────────────────────────────
        if use_gpu:
            cmd = _build_cmd(pdf_path, out_dir, engine, use_gpu=True)
            try:
                proc = _run_subprocess(cmd, timeout)
                if proc.returncode == 0 and (out_dir / f"{stem}.json").exists():
                    return _parse_output(out_dir, stem, engine, "cuda")
                gpu_err = (
                    f"GPU OCR failed (rc={proc.returncode}). "
                    f"stderr tail: {(proc.stderr or '')[-800:]}"
                )
                gpu_oom = _looks_like_oom(proc)
            except subprocess.TimeoutExpired as exc:
                # GPU timed out — CPU would be slower; do not retry.
                raise OcrError(f"GPU OCR timed out after {timeout}s") from exc

            # GPU produced a hard failure. Retry on CPU only for VRAM-type
            # failures (OOM / segfault at predictor init) when allowed.
            if not (cpu_fallback and gpu_oom):
                raise OcrError(gpu_err)
            # else fall through to CPU
        # ── CPU attempt ──────────────────────────────────────────────────
        cmd = _build_cmd(pdf_path, out_dir, engine, use_gpu=False)
        try:
            proc = _run_subprocess(cmd, timeout)
        except subprocess.TimeoutExpired as exc:
            raise OcrError(f"CPU OCR timed out after {timeout}s") from exc

        if proc.returncode == 0 and (out_dir / f"{stem}.json").exists():
            return _parse_output(out_dir, stem, engine, "cpu")

        raise OcrError(
            f"CPU OCR failed (rc={proc.returncode}). "
            f"stderr tail: {(proc.stderr or '')[-800:]}"
        )
