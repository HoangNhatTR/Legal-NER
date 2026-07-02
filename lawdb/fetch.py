"""Acquire the full text of Vietnamese law documents (real data only).

Generalised multi-document fetcher. Each document is keyed by a ``law_key``
(see ``lawdb.LAW_REGISTRY``) and acquired from a luatvietnam.vn "nội dung hợp
nhất" full-text page, which renders clean continuous article bodies (penalty
phrases such as "phạt tù từ 02 năm đến 07 năm" survive intact).

  BLHS2015    luatvietnam + HuggingFace backfill (10 articles LV omits)
  BLTTHS2015  luatvietnam only
  BLDS2015    luatvietnam only
  NQ326_2016  luatvietnam only

For every document the merged result is written to
``data/lawdb/<law_key>_raw.txt`` (one article per ``=== Điều N ===`` block) plus
a sidecar ``<law_key>_raw.meta.json`` recording the source of every article, so
the parse step is fully reproducible offline.

CLI:  python -m lawdb.fetch                       # fetch ALL registry docs
      python -m lawdb.fetch --law BLTTHS2015      # fetch one document
      python -m lawdb.fetch --offline             # report cached coverage only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import requests
import urllib3

from lawdb import LAW_REGISTRY

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR.parent / "data" / "lawdb"

# Per-document acquisition config. ``url`` is the luatvietnam full-text page.
# ``backfill_hf`` (BLHS only) lists the HuggingFace fallback for LV-missing arts.
SOURCES: dict[str, dict] = {
    "BLHS2015": {
        "url": "https://luatvietnam.vn/hinh-su/bo-luat-hinh-su-2015-101324-d1.html",
        "lv_missing": (33, 69, 93, 94, 95, 96, 97, 101, 105, 106),
        "hf_repo": "xuanhungttm/bo-luat-hinh-su-2015",
        "hf_file": "data/train-00000-of-00001.parquet",
    },
    "BLTTHS2015": {
        "url": "https://luatvietnam.vn/hinh-su/bo-luat-to-tung-hinh-su-2015-101322-d1.html",
    },
    "BLDS2015": {
        "url": "https://luatvietnam.vn/dan-su/bo-luat-dan-su-2015-moi-nhat-so-91-2015-qh13-101333-d1.html",
    },
    "NQ326_2016": {
        "url": "https://luatvietnam.vn/thue/nghi-quyet-326-2016-ubtvqh14-uy-ban-thuong-vu-quoc-hoi-111767-d1.html",
    },
}

# Backward-compat constant some callers may import.
LUATVIETNAM_URL = SOURCES["BLHS2015"]["url"]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REQUEST_TIMEOUT = 60

ARTICLE_BLOCK_RE = re.compile(r"^=== Điều (\d+) ===$")
CHUONG_LINE_RE = re.compile(r"^Chương ([IVXLC]+)$")
DIEU_LINE_RE = re.compile(r"^Điều\s+(\d+)\.")


def raw_txt_path(law_key: str) -> Path:
    return DATA_DIR / f"{law_key}_raw.txt"


def raw_meta_path(law_key: str) -> Path:
    return DATA_DIR / f"{law_key}_raw.meta.json"


# ---------------------------------------------------------------------------
# luatvietnam.vn — generic full-text parser
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "")
    return text


def fetch_luatvietnam(url: str) -> tuple[dict[int, str], dict[int, dict]]:
    """Return ``(articles, sections)`` parsed from a luatvietnam full-text page.

    * ``articles`` : {so_dieu: full_article_text} — body runs from the
      "Điều N. <title>" heading to just before the next "Điều M." heading.
    * ``sections`` : {so_dieu: {"phan", "chuong", "muc"}} — Phần/Chương/Mục
      context in force when each article begins, by walking the body and
      tracking the most recent standalone section header.

    A standalone ``Chương <roman>`` header is confirmed only when its following
    line is an ALL-CAPS title (filters body cross-references like "Chương XIII").
    Generic across documents — no per-document hard-coding except the URL.
    """
    from bs4 import BeautifulSoup

    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, verify=False)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    art = soup.select_one("article")
    if art is None:
        raise RuntimeError(f"luatvietnam: <article> container not found at {url}")
    for tag in art(["script", "style"]):
        tag.decompose()

    raw = _clean_text(art.get_text("\n", strip=True))
    lines = [
        ln.strip()
        for ln in raw.split("\n")
        if ln.strip() and ln.strip() != "Đang theo dõi"
    ]
    start = next((i for i, ln in enumerate(lines) if DIEU_LINE_RE.match(ln)), -1)
    if start < 0:
        raise RuntimeError(f"luatvietnam: could not locate first 'Điều' at {url}")
    lines = lines[start:]

    def is_title_line(s: str) -> bool:
        letters = [c for c in s if c.isalpha()]
        return bool(letters) and all(c == c.upper() for c in letters) and len(s) > 4

    cur_phan: str | None = None
    cur_chuong: str | None = None
    cur_muc: str | None = None
    articles: dict[int, str] = {}
    sections: dict[int, dict] = {}
    cur_num: int | None = None
    buf: list[str] = []

    def flush():
        if cur_num is not None and cur_num not in articles:
            articles[cur_num] = "\n".join(buf).strip()

    n = len(lines)
    PHAN_RE = re.compile(r"^Phần thứ (nhất|hai|ba|tư|năm|sáu|bảy|tám)\b", re.IGNORECASE)
    for i, ln in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < n else ""

        m_phan = PHAN_RE.match(ln)
        m_chuong = CHUONG_LINE_RE.match(ln)
        m_muc = re.match(r"^(Mục \d+)\.?\s*(.*)$", ln) if ln.startswith("Mục ") else None
        m_dieu = DIEU_LINE_RE.match(ln)

        if m_phan and is_title_line(nxt):
            cur_phan = f"{ln} - {nxt}"
            cur_chuong = None
            cur_muc = None
            continue
        if m_chuong and is_title_line(nxt):
            cur_chuong = f"{ln} - {nxt}"
            cur_muc = None
            continue
        if m_muc and m_muc.group(2):
            cur_muc = ln
            continue
        if m_dieu:
            flush()
            cur_num = int(m_dieu.group(1))
            buf = [ln]
            sections[cur_num] = {
                "phan": cur_phan,
                "chuong": cur_chuong,
                "muc": cur_muc,
            }
            continue
        buf.append(ln)
    flush()
    return articles, sections


# ---------------------------------------------------------------------------
# HuggingFace backfill (BLHS only — the 10 articles luatvietnam omits)
# ---------------------------------------------------------------------------
def fetch_hf_articles(repo: str, hf_file: str, wanted: set[int]) -> dict[int, str]:
    """Reconstruct best-effort article text for ``wanted`` from an HF parquet.

    The parquet splits each article into (Khoản, Điểm, text) fragments; we
    concatenate them in row order. Lower fidelity than luatvietnam, hence used
    only for the few articles luatvietnam lacks.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, hf_file, repo_type="dataset")
    df = pd.read_parquet(path)

    def cell(v) -> str:
        if v is None:
            return ""
        s = str(v)
        return "" if s == "nan" else s.strip()

    out: dict[int, str] = {}
    for num in sorted(wanted):
        rows = df[df["Điều"].astype(str).str.startswith(f"Điều {num}.")]
        if not len(rows):
            continue
        parts: list[str] = [cell(rows.iloc[0]["Điều"])]
        seen_khoan: set[str] = set()
        for _, r in rows.iterrows():
            khoan, diem, tail = cell(r["Khoản"]), cell(r["Điểm"]), cell(r["text"])
            if khoan and khoan not in seen_khoan:
                parts.append(khoan)
                seen_khoan.add(khoan)
            if diem:
                parts.append(diem)
            if tail:
                parts.append(tail)
        out[num] = "\n".join(p for p in parts if p)
    return out


# ---------------------------------------------------------------------------
# Merge + persist (per law_key)
# ---------------------------------------------------------------------------
def _write_raw(
    law_key: str,
    articles: dict[int, str],
    sources: dict[int, str],
    sections: dict[int, dict],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reg = LAW_REGISTRY[law_key]
    blocks = [f"=== Điều {num} ===\n{articles[num].strip()}" for num in sorted(articles)]
    raw_txt_path(law_key).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    total = reg["total_dieu"]
    meta = {
        "law_key": law_key,
        "document": f"{reg['ten']} ({reg['so_hieu']})",
        "total_articles_obtained": len(articles),
        "total_articles_expected": total,
        "sources": {"primary": {"name": "luatvietnam.vn", "url": SOURCES[law_key]["url"]}},
        "article_source": {str(k): sources[k] for k in sorted(sources)},
        "article_section": {str(k): sections.get(k, {}) for k in sorted(articles)},
        "missing_articles": sorted(set(range(1, total + 1)) - set(articles)),
    }
    raw_meta_path(law_key).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def fetch_document(law_key: str, force: bool = False) -> dict:
    """Fetch one document by law_key and write its raw cache. Returns summary."""
    if law_key not in SOURCES:
        raise KeyError(f"Unknown law_key {law_key!r}; known: {sorted(SOURCES)}")
    if raw_txt_path(law_key).exists() and not force:
        return load_cached(law_key)

    cfg = SOURCES[law_key]
    articles: dict[int, str] = {}
    sources: dict[int, str] = {}
    sections: dict[int, dict] = {}

    print(f"[fetch:{law_key}] PRIMARY luatvietnam: {cfg['url']}")
    try:
        lv, lv_sections = fetch_luatvietnam(cfg["url"])
        for n, t in lv.items():
            articles[n] = t
            sources[n] = "luatvietnam"
        sections.update(lv_sections)
        print(f"[fetch:{law_key}] luatvietnam -> {len(lv)} articles")
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch:{law_key}] luatvietnam FAILED: {exc!r}")

    total = LAW_REGISTRY[law_key]["total_dieu"]
    missing = sorted(set(range(1, total + 1)) - set(articles))
    if missing and "hf_repo" in cfg:
        wanted = set(missing) | set(cfg.get("lv_missing", ()))
        print(f"[fetch:{law_key}] BACKFILL huggingface for {len(wanted)} articles")
        try:
            hf = fetch_hf_articles(cfg["hf_repo"], cfg["hf_file"], wanted)
            for n, t in hf.items():
                if n not in articles:
                    articles[n] = t
                    sources[n] = "huggingface"
            print(f"[fetch:{law_key}] huggingface -> filled {len(hf)} articles")
        except Exception as exc:  # noqa: BLE001
            print(f"[fetch:{law_key}] huggingface FAILED: {exc!r}")

    if not articles:
        raise RuntimeError(f"No articles acquired for {law_key} from any source.")

    _write_raw(law_key, articles, sources, sections)
    return _summary(law_key, articles, sources)


def fetch_all(force: bool = False, only: str | None = None) -> dict[str, dict]:
    """Fetch every registry document (or just ``only``). Returns {law_key: summary}."""
    keys = [only] if only else list(SOURCES)
    return {k: fetch_document(k, force=force) for k in keys}


def _summary(law_key: str, articles: dict[int, str], sources: dict[int, str]) -> dict:
    total = LAW_REGISTRY[law_key]["total_dieu"]
    by_src: dict[str, int] = {}
    for s in sources.values():
        by_src[s] = by_src.get(s, 0) + 1
    return {
        "law_key": law_key,
        "obtained": len(articles),
        "expected": total,
        "by_source": by_src,
        "missing": sorted(set(range(1, total + 1)) - set(articles)),
        "raw_txt": str(raw_txt_path(law_key)),
        "raw_meta": str(raw_meta_path(law_key)),
    }


def load_cached(law_key: str) -> dict:
    """Parse the cached raw file for ``law_key`` into {so_dieu: text} + summary."""
    if not raw_txt_path(law_key).exists():
        raise FileNotFoundError(
            f"No cached raw text for {law_key} at {raw_txt_path(law_key)}; run fetch."
        )
    meta = (
        json.loads(raw_meta_path(law_key).read_text(encoding="utf-8"))
        if raw_meta_path(law_key).exists()
        else {}
    )
    sources = {int(k): v for k, v in meta.get("article_source", {}).items()}
    articles = read_raw_articles(law_key)
    return _summary(law_key, articles, sources)


def read_raw_articles(law_key: str) -> dict[int, str]:
    """Read the cached raw file back into {so_dieu: full_article_text}."""
    text = raw_txt_path(law_key).read_text(encoding="utf-8")
    articles: dict[int, str] = {}
    cur_num: int | None = None
    buf: list[str] = []
    for line in text.split("\n"):
        m = ARTICLE_BLOCK_RE.match(line)
        if m:
            if cur_num is not None:
                articles[cur_num] = "\n".join(buf).strip()
            cur_num = int(m.group(1))
            buf = []
        else:
            buf.append(line)
    if cur_num is not None:
        articles[cur_num] = "\n".join(buf).strip()
    return articles


def read_article_sections(law_key: str) -> dict[int, dict]:
    """Read {so_dieu: {phan, chuong, muc}} from the meta sidecar."""
    if raw_meta_path(law_key).exists():
        meta = json.loads(raw_meta_path(law_key).read_text(encoding="utf-8"))
        return {int(k): v for k, v in meta.get("article_section", {}).items()}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Vietnamese law full text.")
    ap.add_argument("--law", type=str, default=None, help="single law_key (default: all)")
    ap.add_argument("--force", action="store_true", help="re-fetch even if cached")
    ap.add_argument("--offline", action="store_true", help="report cached coverage only")
    args = ap.parse_args()

    keys = [args.law] if args.law else list(SOURCES)
    summaries = {}
    for k in keys:
        summaries[k] = load_cached(k) if args.offline else fetch_document(k, force=args.force)

    print("\n=== Acquisition summary ===")
    for k, s in summaries.items():
        print(
            f"  {k:12s} {s['obtained']}/{s['expected']} articles "
            f"by_source={s['by_source']} missing={len(s['missing'])}"
        )


if __name__ == "__main__":
    sys.exit(main())
