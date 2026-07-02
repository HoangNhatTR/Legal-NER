"""Build the multi-document Vietnamese law SQLite database: fetch -> parse -> load.

Loads every document in ``lawdb.LAW_REGISTRY`` (BLHS / BLTTHS / BLDS / NQ326)
into one DB at ``data/lawdb/lawvn.db``, distinguishable by ``law_key``, with
each document's effective-date metadata and amendment rows.

CLI:
    python -m lawdb.build                  # fetch (if needed) + parse + load ALL
    python -m lawdb.build --refetch        # force re-download of raw text
    python -m lawdb.build --law BLTTHS2015 # build just one document into the DB
    python -m lawdb.build --db PATH        # custom output path
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from lawdb import LAW_REGISTRY
from lawdb.fetch import (
    SOURCES,
    fetch_document,
    raw_txt_path,
    read_article_sections,
    read_raw_articles,
)
from lawdb.parser import parse_all

MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR.parent / "data" / "lawdb"
DEFAULT_DB = DATA_DIR / "lawvn.db"
SCHEMA_PATH = MODULE_DIR / "schema.sql"


def _load_document(conn: sqlite3.Connection, law_key: str, refetch: bool) -> dict:
    """Fetch (if needed), parse and insert one document. Returns per-doc stats."""
    reg = LAW_REGISTRY[law_key]
    if refetch or not raw_txt_path(law_key).exists():
        fetch_document(law_key, force=refetch)

    articles_raw = read_raw_articles(law_key)
    sections = read_article_sections(law_key)
    if not articles_raw:
        raise RuntimeError(f"No raw articles available for {law_key}.")

    is_penal_code = law_key == "BLHS2015"
    parsed = parse_all(articles_raw, sections, is_penal_code=is_penal_code)

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (law_key, ten, so_hieu, loai, total_dieu, "
        "ngay_ban_hanh, effective_from, effective_to, status, nguon_url, version_note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            reg["law_key"],
            reg["ten"],
            reg["so_hieu"],
            reg["loai"],
            reg["total_dieu"],
            reg["ngay_ban_hanh"],
            reg["effective_from"],
            reg["effective_to"],
            reg["status"],
            SOURCES[law_key]["url"],
            reg["version_note"],
        ),
    )
    doc_id = cur.lastrowid

    for am in reg.get("amendments", []):
        cur.execute(
            "INSERT INTO amendments (target_doc_id, amending_key, amending_ten, "
            "so_hieu, effective_from, mo_ta) VALUES (?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                am.get("amending_key"),
                am["amending_ten"],
                am.get("so_hieu"),
                am.get("effective_from"),
                am.get("mo_ta"),
            ),
        )

    n_art = n_clause = n_point = n_frame = 0
    for art in parsed:
        cur.execute(
            "INSERT INTO articles (doc_id, so_dieu, tieu_de, phan, chuong, muc, "
            "noi_dung, is_offense) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                art.so_dieu,
                art.tieu_de,
                art.phan,
                art.chuong,
                art.muc,
                art.noi_dung,
                int(art.is_offense),
            ),
        )
        article_id = cur.lastrowid
        n_art += 1
        # Defensive de-dup: the luatvietnam "nội dung hợp nhất" view occasionally
        # duplicates a clause block (a comparison/consolidation artifact). A real
        # article never has two khoản with the same number — keep the first.
        _seen_khoan: set[int] = set()
        for cl in art.clauses:
            if cl.so_khoan in _seen_khoan:
                continue
            _seen_khoan.add(cl.so_khoan)
            cur.execute(
                "INSERT INTO clauses (article_id, so_khoan, noi_dung) VALUES (?, ?, ?)",
                (article_id, cl.so_khoan, cl.noi_dung),
            )
            clause_id = cur.lastrowid
            n_clause += 1
            for pt in cl.points:
                cur.execute(
                    "INSERT INTO points (clause_id, ky_hieu, noi_dung) VALUES (?, ?, ?)",
                    (clause_id, pt.ky_hieu, pt.noi_dung),
                )
                n_point += 1
            for fr in cl.penalty_frames:
                cur.execute(
                    "INSERT INTO penalty_frames (clause_id, penalty_type, min_value, "
                    "max_value, unit, raw_text) VALUES (?, ?, ?, ?, ?, ?)",
                    (clause_id, fr.penalty_type, fr.min_value, fr.max_value, fr.unit, fr.raw_text),
                )
                n_frame += 1

    return {
        "law_key": law_key,
        "articles": n_art,
        "expected": reg["total_dieu"],
        "clauses": n_clause,
        "points": n_point,
        "penalty_frames": n_frame,
        "offense_articles": sum(a.is_offense for a in parsed),
    }


def build(db_path: Path = DEFAULT_DB, refetch: bool = False, only: str | None = None) -> dict:
    keys = [only] if only else list(LAW_REGISTRY)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    per_doc: list[dict] = []
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        for k in keys:
            per_doc.append(_load_document(conn, k, refetch))
        conn.commit()
    finally:
        conn.close()
    return {"db_path": str(db_path), "documents": per_doc}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build multi-document Vietnamese law DB.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="output DB path")
    ap.add_argument("--law", type=str, default=None, help="single law_key (default: all)")
    ap.add_argument("--refetch", action="store_true", help="force re-download")
    args = ap.parse_args()

    stats = build(db_path=args.db, refetch=args.refetch, only=args.law)
    print("\n=== lawvn.db build complete ===")
    print(f"  db : {stats['db_path']}\n")
    print(f"  {'law_key':12s} {'articles':>14s} {'clauses':>8s} {'points':>7s} "
          f"{'frames':>7s} {'offense':>8s}")
    for d in stats["documents"]:
        cov = f"{d['articles']}/{d['expected']}"
        print(f"  {d['law_key']:12s} {cov:>14s} {d['clauses']:>8d} {d['points']:>7d} "
              f"{d['penalty_frames']:>7d} {d['offense_articles']:>8d}")
        pct = 100.0 * d["articles"] / d["expected"] if d["expected"] else 0
        print(f"  {'':12s} coverage = {pct:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
