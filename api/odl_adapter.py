"""opendataloader-pdf (ODL) extraction adapter — alternative to ocr_adapter.

Two modes, both via the SAME ``opendataloader_pdf.convert`` Python API:

  * digital (ocr=False) — fast, structured native extraction. Parses headings /
    paragraphs / lists in reading order out of ODL's JSON tree. Does NOT OCR
    scans (a scanned page yields little/no text).
  * hybrid OCR (ocr=True) — for scans. Requires a running ODL hybrid backend
    server (``opendataloader-pdf-hybrid``) reachable on LEGAL_NER_ODL_HYBRID_PORT;
    docling (EasyOCR/RapidOCR) does the actual OCR. We pass hybrid="docling-fast".

Java: ODL is a Java pipeline wrapped in Python; it needs JDK 11+. System java is
8, so we point JAVA_HOME at the portable JDK 21 (LEGAL_NER_ODL_JAVA_HOME) and
prepend its bin to PATH *for the current process* before calling convert().

Output contract (mirrors the spirit of OcrResult but lighter):
    extract_with_odl(data, ocr=...) -> {
        "text":   concatenated reading-order plain text (NOT yet normalized),
        "blocks": list of {type, page, bbox, text}  (flattened reading order),
        "source": "opendataloader-digital" | "opendataloader-hybrid",
        "num_pages": int,
        "meta": {...}            # title/author/raw counts, best-effort
    }

The returned ``text`` still contains source-font artifacts (e.g. legacy 'ƣ');
callers MUST run corpus.normalize on it exactly as for every other extractor.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Config (env-overridable so ops can tune without code changes) ───────────
#   LEGAL_NER_ODL_JAVA_HOME   : portable JDK 11+ home (REQUIRED — system java is 8)
#   LEGAL_NER_ODL_HYBRID_PORT : port the hybrid backend server listens on
#   LEGAL_NER_ODL_HYBRID_HOST : host for the hybrid backend (default 127.0.0.1)
#   LEGAL_NER_ODL_HYBRID_TIMEOUT_MS : per-request hybrid timeout ms (0 = none)
DEFAULT_JAVA_HOME = os.environ.get(
    "LEGAL_NER_ODL_JAVA_HOME", "/home/tts/jdk/jdk-21.0.11+10"
)
DEFAULT_HYBRID_PORT = os.environ.get("LEGAL_NER_ODL_HYBRID_PORT", "5002")
DEFAULT_HYBRID_HOST = os.environ.get("LEGAL_NER_ODL_HYBRID_HOST", "127.0.0.1")
DEFAULT_HYBRID_TIMEOUT_MS = os.environ.get("LEGAL_NER_ODL_HYBRID_TIMEOUT_MS", "0")
DEFAULT_HYBRID_BACKEND = os.environ.get(
    "LEGAL_NER_ODL_HYBRID_BACKEND", "docling-fast"
)
#   LEGAL_NER_ODL_HYBRID_AUTOSTART : "1" (default) auto-spawn the hybrid server
#                                    when it is not reachable; "0" -> raise a
#                                    clear error instead (supervised/prod case).
#   LEGAL_NER_ODL_HYBRID_OCR_LANG  : OCR languages for the hybrid server.
#   LEGAL_NER_ODL_HYBRID_START_TIMEOUT_S : how long to wait for the first-run
#                                    model load before giving up (default 180s).
DEFAULT_HYBRID_AUTOSTART = os.environ.get("LEGAL_NER_ODL_HYBRID_AUTOSTART", "1")
DEFAULT_HYBRID_OCR_LANG = os.environ.get("LEGAL_NER_ODL_HYBRID_OCR_LANG", "vi,en")
DEFAULT_HYBRID_START_TIMEOUT_S = float(
    os.environ.get("LEGAL_NER_ODL_HYBRID_START_TIMEOUT_S", "180")
)


class OdlError(Exception):
    """ODL extraction failed in a way we cannot recover from."""


# ── Hybrid OCR server lifecycle (health-check + managed auto-start) ──────────
# The hybrid server (``opendataloader-pdf-hybrid``) is a separate process that
# loads EasyOCR models once and serves them over HTTP. We:
#   1. health-check it on host:port before any OCR call;
#   2. if down and autostart is enabled, spawn it ONCE as a module-level
#      singleton subprocess (CPU OCR via CUDA_VISIBLE_DEVICES="") and poll
#      /health until ready (first run loads models, can take ~1-2 min);
#   3. reuse that subprocess across requests; clean it up at interpreter exit.
# A lock serialises concurrent first-request races so we never spawn twice.

_hybrid_proc: subprocess.Popen | None = None
_hybrid_lock = threading.Lock()


def _hybrid_base_url() -> str:
    host = os.environ.get("LEGAL_NER_ODL_HYBRID_HOST", DEFAULT_HYBRID_HOST)
    port = os.environ.get("LEGAL_NER_ODL_HYBRID_PORT", DEFAULT_HYBRID_PORT)
    return f"http://{host}:{port}"


def hybrid_healthy(timeout: float = 2.0) -> bool:
    """True if the hybrid OCR server answers GET /health on host:port."""
    url = f"{_hybrid_base_url()}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _hybrid_env() -> dict:
    """Environment for the hybrid subprocess: CPU OCR + portable JDK on PATH."""
    env = dict(os.environ)
    java_home = env.get("LEGAL_NER_ODL_JAVA_HOME", DEFAULT_JAVA_HOME)
    if java_home:
        env["JAVA_HOME"] = java_home
        env["PATH"] = str(Path(java_home) / "bin") + os.pathsep + env.get("PATH", "")
    # EasyOCR OOMs on the shared GPU — force CPU OCR.
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _hybrid_launcher_cmd() -> list[str]:
    """Build the ``opendataloader-pdf-hybrid`` launch command from env config."""
    port = os.environ.get("LEGAL_NER_ODL_HYBRID_PORT", DEFAULT_HYBRID_PORT)
    host = os.environ.get("LEGAL_NER_ODL_HYBRID_HOST", DEFAULT_HYBRID_HOST)
    ocr_lang = os.environ.get("LEGAL_NER_ODL_HYBRID_OCR_LANG", DEFAULT_HYBRID_OCR_LANG)
    # Prefer the console script next to the running interpreter; fall back to
    # ``python -m`` invocation of the server's main if it isn't on PATH.
    exe = Path(sys.executable).with_name("opendataloader-pdf-hybrid")
    if exe.exists():
        cmd = [str(exe)]
    else:
        cmd = [sys.executable, "-m", "opendataloader_pdf.hybrid_server"]
    cmd += [
        "--host", host,
        "--port", str(port),
        "--force-ocr",
        "--ocr-lang", ocr_lang,
        "--device", "cpu",
    ]
    return cmd


def ensure_hybrid_server() -> None:
    """Ensure the hybrid OCR server is reachable before an OCR conversion.

    Already healthy -> return immediately. Otherwise, if autostart is enabled,
    spawn it once (module-level singleton) and poll /health until ready or the
    start timeout elapses. If autostart is disabled OR the spawn never becomes
    healthy, raise OdlError with an actionable message (how to start it).
    """
    if hybrid_healthy():
        return

    autostart = os.environ.get(
        "LEGAL_NER_ODL_HYBRID_AUTOSTART", DEFAULT_HYBRID_AUTOSTART
    ) not in ("0", "false", "False", "")

    base = _hybrid_base_url()
    launcher_hint = (
        "start it with scripts/start_ocr_server.sh (or run "
        f"`opendataloader-pdf-hybrid --port "
        f"{os.environ.get('LEGAL_NER_ODL_HYBRID_PORT', DEFAULT_HYBRID_PORT)} "
        f"--force-ocr --ocr-lang "
        f"{os.environ.get('LEGAL_NER_ODL_HYBRID_OCR_LANG', DEFAULT_HYBRID_OCR_LANG)}` "
        "with JAVA_HOME set and CUDA_VISIBLE_DEVICES='')"
    )

    if not autostart:
        raise OdlError(
            f"hybrid OCR server not reachable at {base} and autostart is "
            f"disabled (LEGAL_NER_ODL_HYBRID_AUTOSTART=0); {launcher_hint}"
        )

    global _hybrid_proc
    with _hybrid_lock:
        # re-check inside the lock: another thread may have started it.
        if hybrid_healthy():
            return
        # if our managed proc died, reap it before respawning.
        if _hybrid_proc is not None and _hybrid_proc.poll() is not None:
            _hybrid_proc = None
        if _hybrid_proc is None:
            cmd = _hybrid_launcher_cmd()
            try:
                _hybrid_proc = subprocess.Popen(
                    cmd,
                    env=_hybrid_env(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,  # detach from API's signal group
                )
            except Exception as exc:
                raise OdlError(
                    f"failed to auto-start hybrid OCR server ({exc}); {launcher_hint}"
                ) from exc

        timeout_s = DEFAULT_HYBRID_START_TIMEOUT_S
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            # surface an early crash with a clear message
            if _hybrid_proc is not None and _hybrid_proc.poll() is not None:
                rc = _hybrid_proc.returncode
                _hybrid_proc = None
                raise OdlError(
                    f"hybrid OCR server exited during startup (rc={rc}); "
                    f"{launcher_hint}"
                )
            if hybrid_healthy(timeout=2.0):
                return
            time.sleep(2.0)

        raise OdlError(
            f"hybrid OCR server did not become healthy within {timeout_s:.0f}s "
            f"at {base} (first-run model load can be slow); {launcher_hint}"
        )


def shutdown_hybrid_server() -> None:
    """Terminate the managed hybrid subprocess if we started one (idempotent)."""
    global _hybrid_proc
    with _hybrid_lock:
        proc = _hybrid_proc
        _hybrid_proc = None
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        pass


atexit.register(shutdown_hybrid_server)


def _ensure_java() -> str:
    """Point this process at a JDK 11+ (idempotent). Returns the JAVA_HOME used."""
    java_home = os.environ.get("LEGAL_NER_ODL_JAVA_HOME", DEFAULT_JAVA_HOME)
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        bin_dir = str(Path(java_home) / "bin")
        path = os.environ.get("PATH", "")
        if bin_dir not in path.split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + path
    return java_home


# ── Reading-order text extraction from the ODL JSON tree ────────────────────
# The JSON is a tree: root has "kids"; each node has "type" (heading/paragraph/
# list), "page number", "bounding box", and either "content" (str) or, for
# lists, "list items" (each a node with its own "content" + nested "kids").
# Tree order == reading order (ODL already applies xycut reading-order sort).

def _node_text(node: dict) -> str:
    """Plain text contributed by one node's own content (not its kids)."""
    c = node.get("content")
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):  # defensive: some nodes may carry list-of-str
        return " ".join(str(x).strip() for x in c if x)
    return ""


def _walk(nodes: list, blocks: list[dict]) -> None:
    """Depth-first, in array order — append a block per text-bearing node."""
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        txt = _node_text(node)
        if txt:
            blocks.append(
                {
                    "type": node.get("type"),
                    "page": node.get("page number"),
                    "bbox": node.get("bounding box"),
                    "text": txt,
                }
            )
        # list nodes carry their text inside "list items"
        items = node.get("list items")
        if isinstance(items, list):
            _walk(items, blocks)
        # generic children
        kids = node.get("kids")
        if isinstance(kids, list):
            _walk(kids, blocks)


def _concat_reading_order(doc_json: dict) -> tuple[str, list[dict]]:
    """Flatten the ODL JSON tree to (text, blocks) in reading order."""
    blocks: list[dict] = []
    _walk(doc_json.get("kids", []), blocks)
    text = "\n".join(b["text"] for b in blocks)
    return text, blocks


def _read_json(out_dir: Path, stem: str) -> dict:
    """Load ``{stem}.json`` from out_dir; tolerate ODL's exact-name output."""
    candidate = out_dir / f"{stem}.json"
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    # fallback: any single .json in the dir (ODL preserves the input stem,
    # including spaces/parentheses, which we control via the temp filename).
    jsons = sorted(out_dir.glob("*.json"))
    if not jsons:
        raise OdlError(f"ODL produced no JSON in {out_dir}")
    return json.loads(jsons[0].read_text(encoding="utf-8"))


def extract_with_odl(data: bytes, *, ocr: bool = False) -> dict:
    """Extract text from PDF bytes via opendataloader-pdf.

    ocr=False -> digital (native structured extraction; fast).
    ocr=True  -> hybrid OCR via a running docling backend (for scans).

    Returns the dict described in the module docstring. Raises OdlError on
    unrecoverable failure. Temp files are always cleaned up.
    """
    import opendataloader_pdf  # local import: heavy Java bridge

    _ensure_java()

    with tempfile.TemporaryDirectory(prefix="legal_ner_odl_") as tmp:
        tmp_dir = Path(tmp)
        # stem with no spaces/parentheses so the JSON name is predictable
        pdf_path = tmp_dir / "input.pdf"
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(data)

        kwargs: dict = dict(
            input_path=[str(pdf_path)],
            output_dir=str(out_dir),
            format="json",
            quiet=True,
            image_output="off",  # we only want text; skip image dumping
        )
        if ocr:
            # make sure the hybrid OCR backend is up (auto-start if needed)
            # BEFORE we hand the PDF to convert(); raises OdlError otherwise.
            ensure_hybrid_server()
            host = os.environ.get("LEGAL_NER_ODL_HYBRID_HOST", DEFAULT_HYBRID_HOST)
            port = os.environ.get("LEGAL_NER_ODL_HYBRID_PORT", DEFAULT_HYBRID_PORT)
            timeout_ms = os.environ.get(
                "LEGAL_NER_ODL_HYBRID_TIMEOUT_MS", DEFAULT_HYBRID_TIMEOUT_MS
            )
            kwargs.update(
                hybrid=DEFAULT_HYBRID_BACKEND,
                hybrid_mode="full",  # send all pages to the OCR backend (it's a scan)
                hybrid_url=f"http://{host}:{port}",
                hybrid_timeout=str(timeout_ms),
            )

        try:
            opendataloader_pdf.convert(**kwargs)
        except Exception as exc:  # Java bridge / hybrid server unreachable / etc.
            raise OdlError(
                f"opendataloader convert failed (ocr={ocr}): {type(exc).__name__}: {exc}"
            ) from exc

        doc_json = _read_json(out_dir, pdf_path.stem)
        text, blocks = _concat_reading_order(doc_json)

        return {
            "text": text,
            "blocks": blocks,
            "source": "opendataloader-hybrid" if ocr else "opendataloader-digital",
            "num_pages": doc_json.get("number of pages", 0),
            "meta": {
                "title": doc_json.get("title"),
                "author": doc_json.get("author"),
                "n_blocks": len(blocks),
            },
        }
