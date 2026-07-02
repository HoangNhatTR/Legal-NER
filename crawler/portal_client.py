"""HTTP client for congbobanan.toaan.gov.vn.

Verified portal behavior (2026-06-10):
  * Detail page  : GET /2ta{id}t1cvn/chi-tiet-ban-an  -> HTML with metadata
                   and a direct PDF link of the form /5ta{id}.../<name>.pdf
  * Fulltext URL : GET /3ta{id}t1cvn                  -> raw PDF bytes
  * TLS chain is incomplete -> verify=False (warning suppressed).
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CIVIL_MARKERS,
    CRIMINAL_MARKERS,
    DETAIL_URL,
    FULLTEXT_URL,
    PORTAL_BASE,
    REQUEST_TIMEOUT,
    USER_AGENT,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PDF_LINK_RE = re.compile(r'href="(/5ta\d+[^"]*\.pdf)"', re.IGNORECASE)
TITLE_RE = re.compile(r"<title>\s*(.*?)\s*</title>", re.DOTALL | re.IGNORECASE)


@dataclass
class DetailInfo:
    judgment_id: int
    title: str
    pdf_path: str | None  # relative /5ta... link, if present
    is_criminal: bool
    is_civil: bool  # civil / family / administrative (NON-criminal)


def make_session() -> requests.Session:
    session = requests.Session()
    session.verify = False  # incomplete TLS chain on the portal
    session.headers["User-Agent"] = USER_AGENT
    return session


# Search form on the homepage is an ASP.NET WebForms postback. These are the
# field names verified live (2026-06-11): the keyword box accepts a judgment
# number, the submit button name triggers the search. There is NO captcha and
# NO __EVENTVALIDATION field on this page (only __VIEWSTATE / generator).
SEARCH_KEYWORD_FIELD = "ctl00$Content_home_Public$ctl00$txtKeyword"
SEARCH_SUBMIT_FIELD = "ctl00$Content_home_Public$ctl00$cmd_search_home"
SEARCH_SUBMIT_VALUE = "Tìm kiếm"


def fetch_detail(session: requests.Session, judgment_id: int) -> DetailInfo | None:
    """Fetch + parse a judgment detail page. None when id has no judgment."""
    url = DETAIL_URL.format(id=judgment_id)
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        return None
    html = resp.text
    title_match = TITLE_RE.search(html)
    title = " ".join(title_match.group(1).split()) if title_match else ""
    pdf_match = PDF_LINK_RE.search(html)
    if not pdf_match and not title:
        return None  # empty placeholder page -> treat as missing id
    lowered = title.lower()
    is_criminal = any(marker in lowered for marker in CRIMINAL_MARKERS)
    # Civil = matches a civil/family/admin marker AND is not criminal. The
    # criminal check takes precedence so titles carrying both a topic word
    # and an explicit HS suffix are never mislabelled as civil.
    is_civil = (not is_criminal) and any(
        marker in lowered for marker in CIVIL_MARKERS
    )
    return DetailInfo(
        judgment_id=judgment_id,
        title=title,
        pdf_path=pdf_match.group(1) if pdf_match else None,
        is_criminal=is_criminal,
        is_civil=is_civil,
    )


def download_pdf(session: requests.Session, info: DetailInfo) -> bytes | None:
    """Download judgment PDF: prefer the 5ta link, fall back to 3ta."""
    candidates = []
    if info.pdf_path:
        candidates.append(PORTAL_BASE + info.pdf_path)
    candidates.append(FULLTEXT_URL.format(id=info.judgment_id))
    for url in candidates:
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            continue
        if resp.status_code == 200 and resp.content[:5] == b"%PDF-":
            return resp.content
    return None
