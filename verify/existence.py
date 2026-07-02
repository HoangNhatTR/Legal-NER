"""Layer 2 verification: judgment EXISTENCE lookup on the public portal.

This is Layer 2 of "Model 2" verification. Given a judgment's extracted
fields (case_number / court / judgment_date from the Model 1 NER), it searches
the official portal ``congbobanan.toaan.gov.vn`` and reports whether a matching
judgment was found.

GUIDING PRINCIPLE — ABSENCE IS NOT PROOF OF FORGERY
---------------------------------------------------
A "not found" result is **INCONCLUSIVE**, never "fake". Reasons:
  * Publication lag — a real judgment may not be published yet.
  * Excluded categories — many judgments are never published at all
    (state-secret, juvenile, morality, family-privacy, etc. — see
    Nghị quyết 03/2017/NQ-HĐTP, Điều 4).
  * Imperfect search — typos, suffix variants, OCR noise in the source
    fields, and portal index gaps cause false negatives.
So statuses are:
    found          — strong positive match (case no. + court + date align)
    found_partial  — case number matches, but court/date differ or missing
    not_found      — no match (INCONCLUSIVE, carries the caveat)
    error          — portal unreachable / search failed (also inconclusive)

Search mechanism (verified live 2026-06-11)
-------------------------------------------
The portal homepage hosts a single ASP.NET WebForms ``aspnetForm``. To search:
  1. GET ``/`` and harvest the hidden fields
     (__VIEWSTATE, __VIEWSTATEGENERATOR, __EVENTTARGET, __EVENTARGUMENT,
      and the feedback hdnProcess flag). There is NO __EVENTVALIDATION and
      NO captcha on this page.
  2. POST ``/`` with those hidden fields plus
        ctl00$Content_home_Public$ctl00$txtKeyword = <case number>
        ctl00$Content_home_Public$ctl00$cmd_search_home = "Tìm kiếm"
     The server 200-redirects to ``/0t15at1cvn1/Tra-cu-ban-an`` and returns a
     results page (20 hits/page). Each hit is an
        <a class="echo_id_pub" href="/2ta{id}t1cvn/chi-tiet-ban-an">
            <span>số {case_no} ngày {dd/mm/yyyy} của {court} ({publish_date})
     element, from which we extract case number, court, date and detail URL.

The keyword search is substring/relevance based (querying "17/2018/HS-ST" also
returns "217/2018/HS-ST"), so the match logic re-checks the normalized case
number strictly rather than trusting result ordering.

Public entry point:
    from verify.existence import check_existence, ExistenceReport
    report = check_existence("17/2018/HS-ST", court=..., judgment_date=...)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Literal

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PORTAL_BASE, REQUEST_TIMEOUT  # noqa: E402
from crawler.portal_client import (  # noqa: E402
    SEARCH_KEYWORD_FIELD,
    SEARCH_SUBMIT_FIELD,
    SEARCH_SUBMIT_VALUE,
    make_session,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Status = Literal["found", "found_partial", "not_found", "error"]

CAVEAT_VI = (
    "Không tìm thấy KHÔNG đồng nghĩa với giả mạo — cổng có độ trễ công bố "
    "và nhiều bản án không được công bố (các trường hợp loại trừ theo "
    "Nghị quyết 03/2017/NQ-HĐTP), và việc tìm kiếm không hoàn hảo."
)

# Span on each result row, e.g.
#   "số 17/2018/HS-ST  ngày 20/03/2019 của  TAND Quận 11,  TP. Hồ Chí Minh"
_SPAN_RE = re.compile(
    r"số\s+(?P<case>.+?)\s+ngày\s+(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+của\s+(?P<court>.+?)\s*$"
)
_DATE_TAIL_RE = re.compile(r"\s*\([^)]*\)\s*$")  # trailing "(01.06.2019)" publish stamp


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QueriedFields(BaseModel):
    case_number: str | None = None
    court: str | None = None
    judgment_date: str | None = None


class MatchEntry(BaseModel):
    case_number: str | None = None
    court: str | None = None
    date: str | None = None
    detail_url: str | None = None
    score: float = Field(ge=0.0, le=1.0)


class ExistenceReport(BaseModel):
    status: Status
    confidence: float = Field(ge=0.0, le=1.0)
    queried: QueriedFields
    matches: list[MatchEntry] = Field(default_factory=list)
    caveat_vi: str = CAVEAT_VI
    summary_vi: str = ""


# ---------------------------------------------------------------------------
# Normalization & matching
# ---------------------------------------------------------------------------


def _strip_accents(text: str) -> str:
    # Đ/đ (U+0110/U+0111) are distinct letters, not base+combining marks, so
    # NFD leaves them intact — map them to D/d before decomposing the rest.
    text = text.replace("Đ", "D").replace("đ", "d")
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def normalize_case_number(raw: str | None) -> str:
    """Canonical key for a judgment number, tolerant of real-world variants.

    Handles the corpus suffix variants: ``DS-ST`` vs ``DSST`` vs ``DS_ST``,
    stray spacing, accent on Đ/đ, ``HSPT-QĐ`` style reorderings and OCR
    lowercase. The canonical form keeps only ``[A-Z0-9]`` after upper-casing
    and accent-stripping, so ``17/2018/HS-ST`` and ``17 / 2018 / HSST`` and
    ``17/2018/hs_st`` all collapse to ``172018HSST``.
    """
    if not raw:
        return ""
    text = _strip_accents(raw).upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def _core_number(canon: str) -> str:
    """Leading ``<num><year>`` portion (e.g. ``172018`` from ``172018HSST``).

    Used to guard against substring relevance matches: the portal returns
    "2172018HSST" when you search "172018HSST"; the cores ``172018`` vs
    ``2172018`` differ, so we reject it.
    """
    m = re.match(r"^(\d+)", canon)
    return m.group(1) if m else canon


def _norm_court(raw: str | None) -> str:
    if not raw:
        return ""
    text = _strip_accents(raw).upper()
    text = re.sub(r"[^A-Z0-9]", " ", text)
    # collapse common abbreviation noise: "TP" / "Q" / "TAND" kept as tokens
    return " ".join(text.split())


def _norm_date(raw: str | None) -> str:
    """Return ``YYYY-MM-DD`` from common Vietnamese date renderings, else ''."""
    if not raw:
        return ""
    raw = raw.strip()
    # ISO already
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", raw)
    if m:
        y, mo, d = m.groups()
    else:
        m = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$", raw)
        if not m:
            return ""
        d, mo, y = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def _court_similarity(a: str, b: str) -> float:
    """Token-overlap (Jaccard) on normalized court strings, 0..1."""
    sa, sb = set(_norm_court(a).split()), set(_norm_court(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def score_match(
    q_case: str,
    q_court: str | None,
    q_date: str | None,
    cand: MatchEntry,
) -> tuple[float, bool]:
    """Score a candidate against the query. Returns (score, case_number_match).

    case_number_match is True only when the canonical numbers AND their core
    ``num/year`` agree — this is the gate for any positive status.
    """
    qc = normalize_case_number(q_case)
    cc = normalize_case_number(cand.case_number)
    if not qc or not cc:
        return 0.0, False

    if qc == cc:
        case_score = 1.0
    elif _core_number(qc) == _core_number(cc):
        # same case+year but suffix mismatch (e.g. HS-ST vs HS-PT) — weak
        case_score = 0.6
    else:
        return 0.0, False  # substring-relevance noise: reject

    case_match = qc == cc

    # court component
    if q_court:
        court_sim = _court_similarity(q_court, cand.court or "")
    else:
        court_sim = None

    # date component
    if q_date:
        date_ok = _norm_date(q_date) and _norm_date(q_date) == _norm_date(cand.date)
    else:
        date_ok = None

    score = case_score * 0.6
    score += 0.25 * (court_sim if court_sim is not None else 0.5)
    score += 0.15 * (1.0 if date_ok else (0.5 if date_ok is None else 0.0))
    return round(min(score, 1.0), 3), case_match


# ---------------------------------------------------------------------------
# Portal search
# ---------------------------------------------------------------------------


def _harvest_hidden(soup: BeautifulSoup) -> dict[str, str]:
    fields: dict[str, str] = {}
    for inp in soup.find_all("input", {"type": "hidden"}):
        name = inp.get("name")
        if name:
            fields[name] = inp.get("value", "")
    return fields


def _parse_results(html: str, max_results: int = 20) -> list[MatchEntry]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[MatchEntry] = []
    for a in soup.find_all("a", class_="echo_id_pub"):
        href = a.get("href") or ""
        span = a.find("span")
        if not span:
            continue
        text = span.get_text(" ", strip=True)
        text = _DATE_TAIL_RE.sub("", text)  # drop trailing publish stamp
        m = _SPAN_RE.search(text)
        if not m:
            continue
        case = m.group("case").strip()
        date_raw = m.group("date").strip()
        court = re.sub(r"\s+", " ", m.group("court")).strip()
        detail = PORTAL_BASE + href if href.startswith("/") else href
        out.append(
            MatchEntry(
                case_number=case,
                court=court or None,
                date=_norm_date(date_raw) or date_raw,
                detail_url=detail,
                score=0.0,
            )
        )
        if len(out) >= max_results:
            break
    return out


def search_portal(
    session: requests.Session,
    case_number: str,
    timeout: int = REQUEST_TIMEOUT,
) -> list[MatchEntry]:
    """Run one search-by-case-number against the portal. Raises on transport
    error (caller handles retry / error status)."""
    resp = session.get(PORTAL_BASE + "/", timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    data = _harvest_hidden(soup)
    data[SEARCH_KEYWORD_FIELD] = case_number
    data[SEARCH_SUBMIT_FIELD] = SEARCH_SUBMIT_VALUE
    # carry the feedback flag if present (harmless, mirrors the browser POST)
    feedback = soup.find("input", {"name": "ctl00$Feedback_Home$hdnProcess"})
    if feedback is not None:
        data["ctl00$Feedback_Home$hdnProcess"] = feedback.get("value", "TRUE")
    post = session.post(PORTAL_BASE + "/", data=data, timeout=timeout)
    post.raise_for_status()
    return _parse_results(post.text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_existence(
    case_number: str,
    court: str | None = None,
    judgment_date: str | None = None,
    *,
    session: requests.Session | None = None,
    delay: float = 1.0,
    timeout: int = REQUEST_TIMEOUT,
    retries: int = 1,
) -> ExistenceReport:
    """Look up a judgment on the portal and return a structured report.

    ``retries`` extra attempts are made on transient transport errors, with a
    polite ``delay`` between any portal requests.
    """
    queried = QueriedFields(
        case_number=case_number, court=court, judgment_date=judgment_date
    )
    own_session = session is None
    sess = session or make_session()

    raw: list[MatchEntry] | None = None
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = search_portal(sess, case_number, timeout=timeout)
            break
        except requests.RequestException as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(delay)  # back off before the next attempt
    if own_session:
        sess.close()

    if raw is None:
        return ExistenceReport(
            status="error",
            confidence=0.0,
            queried=queried,
            matches=[],
            summary_vi=(
                "Lỗi truy vấn cổng công bố bản án "
                f"({type(last_err).__name__ if last_err else 'unknown'}). "
                "Kết quả KHÔNG kết luận — không thể xác nhận hay phủ nhận sự tồn tại."
            ),
        )

    # Score every candidate; keep the ones that pass the case-number gate.
    scored: list[tuple[MatchEntry, bool]] = []
    for cand in raw:
        s, case_match = score_match(case_number, court, judgment_date, cand)
        if s > 0.0:
            cand.score = s
            scored.append((cand, case_match))
    scored.sort(key=lambda t: t[0].score, reverse=True)
    matches = [c for c, _ in scored]

    if not scored:
        return ExistenceReport(
            status="not_found",
            confidence=0.0,
            queried=queried,
            matches=[],
            summary_vi=(
                f"Không tìm thấy bản án số {case_number} trên cổng công bố. "
                "Đây là kết quả KHÔNG kết luận. " + CAVEAT_VI
            ),
        )

    best, best_case_match = scored[0]
    # "found" requires an exact case-number match AND good corroboration
    # (court overlap and/or date agreement pushing the score high).
    if best_case_match and best.score >= 0.85:
        status: Status = "found"
        confidence = best.score
        summary = (
            f"Tìm thấy bản án số {best.case_number} của {best.court} "
            f"(ngày {best.date}) trên cổng công bố — khớp số/tòa/ngày."
        )
    elif best_case_match:
        status = "found_partial"
        confidence = best.score
        summary = (
            f"Tìm thấy bản án có số trùng khớp ({best.case_number}) nhưng "
            "tòa án hoặc ngày không khớp hoàn toàn hoặc thiếu thông tin để đối chiếu. "
            "Cần kiểm tra thủ công."
        )
    else:
        status = "found_partial"
        confidence = best.score
        summary = (
            f"Chỉ tìm thấy bản án gần giống ({best.case_number}) — cùng số/năm "
            "nhưng khác hậu tố loại án. Cần kiểm tra thủ công."
        )

    return ExistenceReport(
        status=status,
        confidence=round(confidence, 3),
        queried=queried,
        matches=matches[:10],
        summary_vi=summary,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 2: judgment existence lookup on congbobanan.toaan.gov.vn"
    )
    parser.add_argument("--case-number", required=True, help='e.g. "17/2018/HS-ST"')
    parser.add_argument("--court", default=None, help="court name (optional)")
    parser.add_argument("--date", default=None, help="judgment date, dd/mm/yyyy or ISO")
    parser.add_argument("--delay", type=float, default=1.0, help="polite delay (s)")
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    report = check_existence(
        args.case_number,
        court=args.court,
        judgment_date=args.date,
        delay=args.delay,
        timeout=args.timeout,
        retries=args.retries,
    )

    if args.json:
        print(report.model_dump_json(indent=2))
        return

    print(f"Case number : {args.case_number}")
    print(f"Status      : {report.status}")
    print(f"Confidence  : {report.confidence}")
    print(f"Summary     : {report.summary_vi}")
    if report.matches:
        print(f"Matches ({len(report.matches)}):")
        for mt in report.matches:
            print(
                f"  [{mt.score:>5.3f}] {mt.case_number} | {mt.court} | "
                f"{mt.date} | {mt.detail_url}"
            )
    if report.status in ("not_found", "error"):
        print(f"\nLƯU Ý: {report.caveat_vi}")


if __name__ == "__main__":
    main()
