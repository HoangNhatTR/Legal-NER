"""In-process async job queue for slow OCR extraction/verification.

WHY THIS BACKEND (in-process + SQLite, NOT arq+Redis)
-----------------------------------------------------
The deployment is a SINGLE GPU server. At build time:
  * the docker daemon socket is permission-denied (cannot `docker run redis`),
  * Redis is not installed and not running, and
  * arq is not installed.
Adding a hard Redis/arq dependency would make the API fail to start in this
environment. The OCR step is also GPU/CPU-bound and MUST be serialized (one
hybrid-OCR run at a time — the GPU and the single hybrid OCR server are a
single shared resource), so a distributed broker buys us nothing here.

Therefore we use a lightweight, dependency-free design:
  * a single background WORKER thread (ThreadPoolExecutor max_workers=1) that
    pulls job_ids off an in-memory queue.Queue and runs the blocking
    run_extract / run_verify against the shared GPU `holder`. One worker ==
    natural serialization of OCR; the FastAPI event loop is never blocked
    (the blocking work happens off the event loop, in the worker thread).
  * a SQLite job table (data/jobs.db, WAL mode) for status/result persistence
    so a submitted job survives across HTTP requests and process restarts.
    Each DB operation opens a short-lived connection (no cross-thread handle
    sharing — sqlite3 connections are not safe to share across threads).

If Redis/arq ever become available, swap `JobQueue` for an arq enqueue without
touching the endpoints (they only call submit_job / get_job).

JOB STATES
----------
    queued  -> running -> done
                       -> error
job_id: a fresh uuid4 hex per submission (NOT content-hash). We deliberately do
NOT dedupe by content: the same PDF may legitimately be re-verified (the live
portal lookup / law DB can change between runs), and idempotency-by-hash would
hide that. Callers wanting dedupe can do it upstream.

RESTART SEMANTICS
-----------------
On startup `recover_orphans()` finds any job left in `running` (the process
died mid-job — the in-flight OCR did NOT survive) and marks it `error` with a
clear message. We do NOT auto re-enqueue: an interrupted multi-minute OCR is
better resubmitted explicitly by the client than silently retried. `queued`
jobs from before the restart ARE re-enqueued into the in-memory queue so they
still run.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# data/jobs.db lives next to the rest of the project data.
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DB_PATH = _DATA_DIR / "jobs.db"
# uploaded PDF bytes are spooled here (keyed by job_id) instead of in the DB:
# multi-hundred-KB blobs do not belong in a status table, and the worker thread
# reads the file lazily right before running.
_UPLOAD_DIR = _DATA_DIR / "job_uploads"

VALID_STATUSES = ("queued", "running", "done", "error")
VALID_KINDS = ("extract", "verify")


def _now() -> str:
    """UTC ISO-8601 timestamp (second precision is plenty for jobs)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a short-lived connection in WAL mode.

    WAL lets the worker thread write while a request thread reads without
    blocking. timeout handles the brief lock window. We open/close per op
    rather than sharing a handle across threads (sqlite3 forbids that).
    """
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    filename    TEXT,
    params      TEXT,            -- JSON
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    result      TEXT,            -- JSON (ExtractResponse / VerifyResponse dict)
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
"""


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for k in ("params", "result"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except (TypeError, ValueError):
                pass
    return d


class JobStore:
    """SQLite-backed job persistence. All methods open short-lived conns."""

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    # --- writes ---------------------------------------------------------
    def create(self, kind: str, filename: str, params: dict) -> str:
        job_id = uuid.uuid4().hex
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, status, kind, filename, params, created_at) "
                "VALUES (?, 'queued', ?, ?, ?, ?)",
                (job_id, kind, filename, json.dumps(params), _now()),
            )
            conn.commit()
        return job_id

    def mark_running(self, job_id: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status='running', started_at=? WHERE job_id=?",
                (_now(), job_id),
            )
            conn.commit()

    def mark_done(self, job_id: str, result: dict) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status='done', finished_at=?, result=? WHERE job_id=?",
                (_now(), json.dumps(result, ensure_ascii=False), job_id),
            )
            conn.commit()

    def mark_error(self, job_id: str, error: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status='error', finished_at=?, error=? WHERE job_id=?",
                (_now(), error, job_id),
            )
            conn.commit()

    # --- reads ----------------------------------------------------------
    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT job_id, status, kind, filename, created_at, started_at, "
                "finished_at, error FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_by_status(self, status: str) -> list[str]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE status=? ORDER BY created_at", (status,)
            ).fetchall()
        return [r["job_id"] for r in rows]


class JobQueue:
    """Single-worker async job queue.

    `runner(job, data)` is the callable that actually executes one job dict
    (with the uploaded PDF bytes) and returns a result dict; it runs in the
    worker thread and may block for minutes. Exceptions are caught and recorded
    as job errors — the worker never dies and never crashes the app.
    """

    def __init__(
        self, store: JobStore, runner: Callable[[dict, bytes], dict]
    ) -> None:
        self.store = store
        self.runner = runner
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jobworker"
        )
        self._stop = threading.Event()
        self._started = False
        self._worker_future = None
        # set to the job_id currently executing (for diagnostics/shutdown)
        self.current_job_id: Optional[str] = None

    # --- lifecycle ------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._worker_future = self._executor.submit(self._worker_loop)

    def recover_orphans(self) -> dict[str, int]:
        """Reconcile DB state left by a previous (possibly crashed) process.

        - running -> error  (the in-flight OCR did not survive the restart)
        - queued  -> re-enqueue so it still runs this session
        Returns counts for logging.
        """
        orphaned = self.store.list_by_status("running")
        for jid in orphaned:
            self.store.mark_error(
                jid,
                "job was running when the server stopped; the in-flight OCR did "
                "not survive the restart. Please resubmit.",
            )
            # the worker's per-job cleanup never ran (process was killed), so
            # delete the spooled upload here to avoid leaking large .bin files.
            try:
                (_UPLOAD_DIR / f"{jid}.bin").unlink(missing_ok=True)
            except OSError:
                pass
        requeued = self.store.list_by_status("queued")
        for jid in requeued:
            self._q.put(jid)
        return {"orphaned_running": len(orphaned), "requeued_queued": len(requeued)}

    def stop(self, drain_timeout: float = 5.0) -> None:
        """Stop accepting work; let the worker exit BETWEEN jobs (bounded wait).

        We do NOT kill an in-flight job: there is no safe way to interrupt the
        OCR subprocess mid-run, and ThreadPoolExecutor cannot cancel a task that
        has already started. So:
          * signal stop + push the sentinel so an IDLE worker exits immediately;
          * wait up to ``drain_timeout`` for the worker thread to finish.

        If a job is mid-OCR when stop() is called, the worker is busy and will
        NOT observe the stop flag until that job ends — which can be minutes.
        We deliberately bound the wait so SIGTERM shutdown stays prompt: after
        ``drain_timeout`` we return without blocking. Any job still ``running``
        at process exit is reconciled to ``error`` on the next startup
        (recover_orphans). We do NOT join the executor with wait=True, precisely
        to avoid an unbounded block on an in-flight multi-minute OCR.
        """
        self._stop.set()
        self._q.put(None)  # sentinel to wake a blocked get()
        fut = self._worker_future
        if fut is not None:
            try:
                # bounded: returns when the worker loop exits, or raises
                # TimeoutError if a job is still mid-flight past the deadline.
                fut.result(timeout=drain_timeout)
            except TimeoutError:
                print(
                    f"[jobs] worker still busy after {drain_timeout}s "
                    f"(job {self.current_job_id} mid-flight); shutting down "
                    "without waiting — it will be reconciled to error on restart",
                    flush=True,
                )
            except Exception:  # noqa: BLE001 — never block shutdown on this
                pass
        # do not block on in-flight work; let the process exit.
        self._executor.shutdown(wait=False, cancel_futures=True)

    # --- enqueue --------------------------------------------------------
    def submit(self, kind: str, filename: str, params: dict, data: bytes) -> str:
        job_id = self.store.create(kind, filename, params)
        # spool the upload to disk keyed by job_id; the worker reads it later.
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        (_UPLOAD_DIR / f"{job_id}.bin").write_bytes(data)
        self._q.put(job_id)
        return job_id

    # --- worker ---------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            if job_id is None:  # shutdown sentinel
                break
            self._run_one(job_id)

    def _run_one(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        # A job already terminal (e.g. errored during recover) should be skipped.
        if job["status"] not in ("queued",):
            return
        self.current_job_id = job_id
        self.store.mark_running(job_id)
        upload_path = _UPLOAD_DIR / f"{job_id}.bin"
        try:
            data = upload_path.read_bytes()
            result = self.runner(job, data)
            self.store.mark_done(job_id, result)
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            tb = traceback.format_exc()
            print(f"[jobs] job {job_id} failed: {exc}\n{tb}", flush=True)
            self.store.mark_error(job_id, f"{type(exc).__name__}: {exc}")
        finally:
            self.current_job_id = None
            # the upload bytes are no longer needed once the job is terminal.
            try:
                upload_path.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Module-level singletons wired by the FastAPI lifespan.
# ---------------------------------------------------------------------------

_store: Optional[JobStore] = None
_queue: Optional[JobQueue] = None


def init_jobs(
    runner: Callable[[dict, bytes], dict], db_path: Path = _DB_PATH
) -> dict[str, int]:
    """Create the store + queue, start the worker, reconcile orphans.

    `runner` maps a job dict -> result dict; it is supplied by main.py so this
    module stays decoupled from service.run_extract/run_verify and the holder.
    Returns the recover_orphans() counts for startup logging.
    """
    global _store, _queue
    _store = JobStore(db_path)
    _queue = JobQueue(_store, runner)
    counts = _queue.recover_orphans()
    _queue.start()
    return counts


def shutdown_jobs() -> None:
    if _queue is not None:
        _queue.stop()


def submit_job(kind: str, filename: str, params: dict, data: bytes) -> str:
    if _queue is None:
        raise RuntimeError("job queue not initialized")
    if kind not in VALID_KINDS:
        raise ValueError(f"unknown job kind {kind!r}; expected one of {VALID_KINDS}")
    return _queue.submit(kind, filename, params, data)


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    if _store is None:
        raise RuntimeError("job queue not initialized")
    return _store.get(job_id)


def recent_jobs(limit: int = 50) -> list[dict[str, Any]]:
    if _store is None:
        raise RuntimeError("job queue not initialized")
    return _store.recent(limit)


def current_job_id() -> Optional[str]:
    return _queue.current_job_id if _queue is not None else None
