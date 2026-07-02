"""Query API over the multi-document Vietnamese law SQLite database.

Powers verification Layer 1 (cited-law existence + validity-at-date) and
Layer 3 (sentencing frames). All functions open the DB read-only and return
plain dicts so callers need no SQLite knowledge.

Every lookup is scoped by ``law_key`` (default ``BLHS2015`` for backward
compatibility). Known keys: BLHS2015, BLTTHS2015, BLDS2015, NQ326_2016.

Public API:
    get_article(so_dieu, law_key=...)              -> article + its khoản tree
    get_clause(so_dieu, so_khoan, law_key=...)     -> one khoản + its penalty frame
    penalty_frame(so_dieu, so_khoan, law_key=...)  -> compact sentencing frame (L3)
    check_validity(law_key, so_dieu, on_date)      -> in-force verdict at a date
    verify_citation(so_dieu, khoan?, diem?, law_key=..., on_date=...) -> verdict

Backward compatibility:
    * ``law_key`` defaults to BLHS2015, so existing positional calls keep working.
    * ``on_date=None`` (default) skips the validity check entirely.
    * If ``lawvn.db`` is absent it falls back to the legacy ``blhs2015.db``.

CLI:
    python -m lawdb.lookup --dieu 250 --khoan 2
    python -m lawdb.lookup --law BLTTHS2015 --dieu 106
    python -m lawdb.lookup --law BLHS2015 --dieu 250 --on-date 2017-06-01
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

from lawdb import LAW_REGISTRY, normalize_law_key

MODULE_DIR = Path(__file__).resolve().parent
_DATA_DIR = MODULE_DIR.parent / "data" / "lawdb"
DEFAULT_DB = _DATA_DIR / "lawvn.db"
_LEGACY_DB = _DATA_DIR / "blhs2015.db"

PENALTY_LABEL_VI = {
    "tu_co_thoi_han": "tù có thời hạn",
    "tu_chung_than": "tù chung thân",
    "tu_hinh": "tử hình",
    "phat_tien": "phạt tiền",
    "cai_tao_khong_giam_giu": "cải tạo không giam giữ",
    "canh_cao": "cảnh cáo",
    "truc_xuat": "trục xuất",
}


def _resolve_db(db_path: Path | str | None) -> Path:
    if db_path is not None:
        return Path(db_path)
    if DEFAULT_DB.exists():
        return DEFAULT_DB
    return _LEGACY_DB  # legacy single-doc fallback


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = _resolve_db(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Law DB not found at {path}. Build it: python -m lawdb.build"
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _doc_id(conn: sqlite3.Connection, law_key: str) -> int | None:
    """Resolve a law_key to its documents.id, or None if absent.

    Legacy ``blhs2015.db`` has no ``law_key`` column; in that case the single
    BLHS document (id=1) is returned for the BLHS key.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)")}
    if "law_key" in cols:
        row = conn.execute(
            "SELECT id FROM documents WHERE law_key=?", (law_key,)
        ).fetchone()
        return row["id"] if row else None
    return 1 if law_key == "BLHS2015" else None


# ---------------------------------------------------------------------------
# Internal fetch helpers
# ---------------------------------------------------------------------------
def _frames_for_clause(conn: sqlite3.Connection, clause_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT penalty_type, min_value, max_value, unit, raw_text "
        "FROM penalty_frames WHERE clause_id=? ORDER BY id",
        (clause_id,),
    ).fetchall()
    return [
        {
            "type": r["penalty_type"],
            "label_vi": PENALTY_LABEL_VI.get(r["penalty_type"], r["penalty_type"]),
            "min": r["min_value"],
            "max": r["max_value"],
            "unit": r["unit"],
            "raw_text": r["raw_text"],
        }
        for r in rows
    ]


def _points_for_clause(conn: sqlite3.Connection, clause_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT ky_hieu, noi_dung FROM points WHERE clause_id=? ORDER BY id",
        (clause_id,),
    ).fetchall()
    return [{"ky_hieu": r["ky_hieu"], "noi_dung": r["noi_dung"]} for r in rows]


def _law_label(law_key: str) -> str:
    """Readable Vietnamese label, e.g. 'Bộ luật Hình sự 2015 (100/2015/QH13)'."""
    reg = LAW_REGISTRY.get(law_key)
    if not reg:
        return law_key
    year = (reg.get("ngay_ban_hanh") or "")[:4]
    suffix = f" {year}" if year else ""
    return f"{reg['ten']}{suffix} ({reg['so_hieu']})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_article(
    so_dieu: int,
    db_path: Path | str | None = None,
    law_key: str = "BLHS2015",
) -> dict:
    """Return the article with its full khoản/điểm tree, or ton_tai=False.

    Scoped to ``law_key`` (default BLHS2015 for backward compatibility).
    """
    law_key = normalize_law_key(law_key)
    conn = _connect(db_path)
    try:
        doc_id = _doc_id(conn, law_key)
        if doc_id is None:
            return {"ton_tai": False, "so_dieu": so_dieu, "law_key": law_key}
        art = conn.execute(
            "SELECT id, so_dieu, tieu_de, phan, chuong, muc, noi_dung, is_offense "
            "FROM articles WHERE doc_id=? AND so_dieu=?",
            (doc_id, so_dieu),
        ).fetchone()
        if art is None:
            return {"ton_tai": False, "so_dieu": so_dieu, "law_key": law_key}

        khoan_rows = conn.execute(
            "SELECT id, so_khoan, noi_dung FROM clauses WHERE article_id=? "
            "ORDER BY so_khoan",
            (art["id"],),
        ).fetchall()
        khoan = [
            {
                "so_khoan": k["so_khoan"],
                "noi_dung": k["noi_dung"],
                "diem": _points_for_clause(conn, k["id"]),
                "penalty_frames": _frames_for_clause(conn, k["id"]),
            }
            for k in khoan_rows
        ]
        return {
            "ton_tai": True,
            "law_key": law_key,
            "so_dieu": art["so_dieu"],
            "tieu_de": art["tieu_de"],
            "phan": art["phan"],
            "chuong": art["chuong"],
            "muc": art["muc"],
            "is_offense": bool(art["is_offense"]),
            "noi_dung": art["noi_dung"],
            "khoan": khoan,
        }
    finally:
        conn.close()


def get_clause(
    so_dieu: int,
    so_khoan: int,
    db_path: Path | str | None = None,
    law_key: str = "BLHS2015",
) -> dict:
    """Return one khoản with its penalty frame, or ton_tai=False."""
    law_key = normalize_law_key(law_key)
    conn = _connect(db_path)
    try:
        doc_id = _doc_id(conn, law_key)
        if doc_id is None:
            return {"ton_tai": False, "so_dieu": so_dieu, "so_khoan": so_khoan,
                    "law_key": law_key, "ly_do": "không có văn bản"}
        row = conn.execute(
            "SELECT c.id AS cid, c.noi_dung AS noi_dung "
            "FROM clauses c JOIN articles a ON a.id=c.article_id "
            "WHERE a.doc_id=? AND a.so_dieu=? AND c.so_khoan=?",
            (doc_id, so_dieu, so_khoan),
        ).fetchone()
        if row is None:
            art = conn.execute(
                "SELECT 1 FROM articles WHERE doc_id=? AND so_dieu=?",
                (doc_id, so_dieu),
            ).fetchone()
            return {
                "ton_tai": False,
                "law_key": law_key,
                "so_dieu": so_dieu,
                "so_khoan": so_khoan,
                "ly_do": "không có điều" if art is None else "không có khoản",
            }
        frames = _frames_for_clause(conn, row["cid"])
        return {
            "ton_tai": True,
            "law_key": law_key,
            "so_dieu": so_dieu,
            "so_khoan": so_khoan,
            "noi_dung": row["noi_dung"],
            "diem": _points_for_clause(conn, row["cid"]),
            "penalty_frame": frames,
        }
    finally:
        conn.close()


def penalty_frame(
    so_dieu: int,
    so_khoan: int,
    db_path: Path | str | None = None,
    law_key: str = "BLHS2015",
) -> dict:
    """Compact sentencing frame for Layer 3 (charge<->article<->sentence)."""
    cl = get_clause(so_dieu, so_khoan, db_path, law_key=law_key)
    if not cl["ton_tai"]:
        return {"ton_tai": False, "so_dieu": so_dieu, "so_khoan": so_khoan}
    frames = cl["penalty_frame"]
    custodial_order = ["tu_co_thoi_han", "tu_chung_than", "tu_hinh"]
    principal = None
    for t in custodial_order:
        principal = next((f for f in frames if f["type"] == t), None)
        if principal:
            break
    if principal is None and frames:
        principal = frames[0]
    return {
        "ton_tai": True,
        "so_dieu": so_dieu,
        "so_khoan": so_khoan,
        "type": principal["type"] if principal else None,
        "min": principal["min"] if principal else None,
        "max": principal["max"] if principal else None,
        "unit": principal["unit"] if principal else None,
        "all_frames": frames,
    }


# ---------------------------------------------------------------------------
# Validity at judgment date
# ---------------------------------------------------------------------------
def _parse_date(d: str | date | None) -> date | None:
    if d is None:
        return None
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d).strip()[:10])


def check_validity(
    law_key: str,
    so_dieu: int | None = None,
    on_date: str | date | None = None,
    db_path: Path | str | None = None,
) -> dict:
    """Was the document (and, where possible, the article) IN FORCE on ``on_date``?

    Validity is checked at DOCUMENT level: each document carries
    ``effective_from`` / ``effective_to``. Article-level (per-Điều) amendment
    granularity is NOT captured — this is stated honestly in ``message_vi`` and
    the ``granularity`` field.

    Returns:
        in_force       : bool | None  (None = date missing or doc unknown)
        granularity    : "document"
        effective_from : ISO date | None
        effective_to   : ISO date | None
        on_date        : echoed ISO date | None
        status         : registry status (in_force | repealed | ...)
        amendments     : list of {amending_ten, so_hieu, effective_from, mo_ta}
        message_vi     : human-readable Vietnamese verdict
    """
    law_key = normalize_law_key(law_key)
    label = _law_label(law_key)

    conn = _connect(db_path)
    try:
        doc_id = _doc_id(conn, law_key)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)")}
        eff_from = eff_to = status = None
        amendments: list[dict] = []
        if doc_id is not None and "effective_from" in cols:
            row = conn.execute(
                "SELECT effective_from, effective_to, status FROM documents WHERE id=?",
                (doc_id,),
            ).fetchone()
            if row:
                eff_from, eff_to, status = (
                    row["effective_from"],
                    row["effective_to"],
                    row["status"],
                )
            if "amendments" in {
                r["name"] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }:
                amendments = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT amending_ten, so_hieu, effective_from, mo_ta "
                        "FROM amendments WHERE target_doc_id=?",
                        (doc_id,),
                    ).fetchall()
                ]
        else:
            # fall back to registry metadata (legacy DB / missing columns)
            reg = LAW_REGISTRY.get(law_key)
            if reg:
                eff_from, eff_to, status = (
                    reg["effective_from"],
                    reg["effective_to"],
                    reg["status"],
                )
                amendments = list(reg.get("amendments", []))
    finally:
        conn.close()

    base = {
        "law_key": law_key,
        "so_dieu": so_dieu,
        "granularity": "document",
        "effective_from": eff_from,
        "effective_to": eff_to,
        "status": status,
        "on_date": None,
        "amendments": amendments,
    }

    on = _parse_date(on_date)
    ef = _parse_date(eff_from)
    et = _parse_date(eff_to)

    if on is None:
        base["in_force"] = None
        base["message_vi"] = (
            f"{label}: hiệu lực từ {eff_from or '?'}"
            + (f" đến {eff_to}" if eff_to else " (chưa bị thay thế)")
            + ". Không truyền ngày để kiểm tra hiệu lực."
        )
        return base

    base["on_date"] = on.isoformat()

    if ef is None:
        base["in_force"] = None
        base["message_vi"] = (
            f"{label}: không có dữ liệu ngày hiệu lực để kiểm tra tại {on.isoformat()}."
        )
        return base

    if on < ef:
        base["in_force"] = False
        base["message_vi"] = (
            f"BẤT THƯỜNG: tại ngày {on.isoformat()}, {label} CHƯA có hiệu lực "
            f"(hiệu lực từ {ef.isoformat()}). Việc viện dẫn văn bản này tại thời "
            f"điểm đó là không hợp lệ. (kiểm tra ở cấp văn bản, chưa theo từng điều)"
        )
        return base

    if et is not None and on > et:
        base["in_force"] = False
        base["message_vi"] = (
            f"Tại ngày {on.isoformat()}, {label} ĐÃ HẾT hiệu lực "
            f"(hết hiệu lực {et.isoformat()}). (kiểm tra ở cấp văn bản)"
        )
        return base

    base["in_force"] = True
    base["message_vi"] = (
        f"Tại ngày {on.isoformat()}, {label} ĐANG có hiệu lực "
        f"(hiệu lực từ {ef.isoformat()}"
        + (f" đến {et.isoformat()}" if et else ", chưa bị thay thế")
        + "). (kiểm tra ở cấp văn bản, chưa theo từng điều)"
    )
    return base


# ---------------------------------------------------------------------------
# Citation verification (L1) + optional validity-at-date
# ---------------------------------------------------------------------------
def verify_citation(
    so_dieu: int,
    so_khoan: int | None = None,
    diem: str | None = None,
    db_path: Path | str | None = None,
    law_key: str = "BLHS2015",
    on_date: str | date | None = None,
) -> dict:
    """Verify a citation 'Điều X [khoản Y] [điểm Z]' against a law document.

    Layer 1 primitive. ``law_key`` selects the document (default BLHS2015 for
    backward compatibility — old positional calls keep working). When
    ``on_date`` is given, a ``validity`` block (from ``check_validity``) is added;
    ``on_date=None`` (default) omits it entirely, preserving prior behaviour.

    Returns:
        exists        : bool — whether the cited unit exists
        law_key       : the resolved document key
        content       : the matched text (article/clause/point)
        penalty_frame : frames of the cited clause (if a clause was cited)
        validity      : (only if on_date given) the check_validity dict
        message_vi    : human-readable Vietnamese verdict
    """
    law_key = normalize_law_key(law_key)
    label = _law_label(law_key)
    reg = LAW_REGISTRY.get(law_key, {})
    total = reg.get("total_dieu")

    def _attach_validity(result: dict) -> dict:
        if on_date is not None:
            result["validity"] = check_validity(law_key, so_dieu, on_date, db_path)
        return result

    art = get_article(so_dieu, db_path, law_key=law_key)
    if not art["ton_tai"]:
        cnt = f" ({label} có {total} điều)" if total else ""
        return _attach_validity({
            "exists": False,
            "law_key": law_key,
            "so_dieu": so_dieu,
            "so_khoan": so_khoan,
            "diem": diem,
            "content": None,
            "penalty_frame": [],
            "message_vi": f"Không tồn tại Điều {so_dieu} trong {label}{cnt}.",
        })

    cite = f"Điều {so_dieu}"
    if so_khoan is None:
        return _attach_validity({
            "exists": True,
            "law_key": law_key,
            "so_dieu": so_dieu,
            "so_khoan": None,
            "diem": None,
            "tieu_de": art["tieu_de"],
            "is_offense": art["is_offense"],
            "content": art["noi_dung"],
            "penalty_frame": [],
            "message_vi": f"{cite} tồn tại trong {label}: \"{art['tieu_de']}\".",
        })

    khoan = next((k for k in art["khoan"] if k["so_khoan"] == so_khoan), None)
    if khoan is None:
        return _attach_validity({
            "exists": False,
            "law_key": law_key,
            "so_dieu": so_dieu,
            "so_khoan": so_khoan,
            "diem": diem,
            "tieu_de": art["tieu_de"],
            "content": None,
            "penalty_frame": [],
            "message_vi": (
                f"{cite} (\"{art['tieu_de']}\") tồn tại trong {label} nhưng "
                f"KHÔNG có khoản {so_khoan}."
            ),
        })

    if diem is None:
        return _attach_validity({
            "exists": True,
            "law_key": law_key,
            "so_dieu": so_dieu,
            "so_khoan": so_khoan,
            "diem": None,
            "tieu_de": art["tieu_de"],
            "is_offense": art["is_offense"],
            "content": khoan["noi_dung"],
            "penalty_frame": khoan["penalty_frames"],
            "message_vi": f"{cite} khoản {so_khoan} tồn tại trong {label}.",
        })

    diem_norm = diem.strip().rstrip(")").lower()
    pt = next((p for p in khoan["diem"] if p["ky_hieu"] == diem_norm), None)
    if pt is None:
        return _attach_validity({
            "exists": False,
            "law_key": law_key,
            "so_dieu": so_dieu,
            "so_khoan": so_khoan,
            "diem": diem,
            "tieu_de": art["tieu_de"],
            "content": None,
            "penalty_frame": khoan["penalty_frames"],
            "message_vi": (
                f"{cite} khoản {so_khoan} tồn tại trong {label} nhưng KHÔNG có "
                f"điểm {diem_norm})."
            ),
        })
    return _attach_validity({
        "exists": True,
        "law_key": law_key,
        "so_dieu": so_dieu,
        "so_khoan": so_khoan,
        "diem": diem_norm,
        "tieu_de": art["tieu_de"],
        "is_offense": art["is_offense"],
        "content": pt["noi_dung"],
        "penalty_frame": khoan["penalty_frames"],
        "message_vi": f"{cite} khoản {so_khoan} điểm {diem_norm}) tồn tại trong {label}.",
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Query the Vietnamese law DB.")
    ap.add_argument("--dieu", type=int, required=True, help="article number")
    ap.add_argument("--khoan", type=int, default=None, help="clause number")
    ap.add_argument("--diem", type=str, default=None, help="point letter (a,b,...)")
    ap.add_argument("--law", type=str, default="BLHS2015", help="law_key")
    ap.add_argument("--on-date", type=str, default=None, help="validity check date (ISO)")
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--json", action="store_true", help="raw JSON output")
    args = ap.parse_args()

    result = verify_citation(
        args.dieu, args.khoan, args.diem,
        db_path=args.db, law_key=args.law, on_date=args.on_date,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"law         : {result['law_key']}")
    print(f"exists      : {result['exists']}")
    print(f"message_vi  : {result['message_vi']}")
    if result.get("tieu_de"):
        print(f"tiêu đề     : {result['tieu_de']}")
    if result.get("validity"):
        v = result["validity"]
        print(f"in_force    : {v['in_force']}  ({v['message_vi']})")
    if result.get("penalty_frame"):
        print("penalty_frame:")
        for f in result["penalty_frame"]:
            rng = ""
            if f["min"] is not None or f["max"] is not None:
                rng = f" [{f['min']}–{f['max']} {f['unit']}]"
            print(f"   - {f['label_vi']}{rng}  «{f['raw_text']}»")


if __name__ == "__main__":
    sys.exit(main())
