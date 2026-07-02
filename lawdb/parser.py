"""Parse BLHS 2015 raw text into a điều/khoản/điểm tree with penalty frames.

Input  : the cached raw articles produced by ``lawdb.fetch`` — a dict
         {so_dieu: full_article_text}, plus the per-article section context
         (Phần/Chương/Mục) from the fetch meta sidecar.

Output : a list of ``Article`` objects, each carrying its clauses (khoản),
         each clause its points (điểm) and any extracted ``PenaltyFrame``s.

Hierarchy detected from the real luatvietnam text:
  * "Điều {N}. {tiêu đề}"          -> article heading (line 1 of each block)
  * line-initial "1." "2." ...     -> khoản
  * line-initial "a)" "b)" "đ)" .. -> điểm (belongs to the current khoản)

Penalty frames (offense articles, Phần "Các tội phạm") are extracted per khoản
from the canonical sentencing phrases, e.g.:
  * "bị phạt tù từ 02 năm đến 07 năm"        -> prison 2..7 years
  * "phạt tù 20 năm, tù chung thân hoặc tử hình" -> prison fixed 20y + life + death
  * "phạt tiền từ 5.000.000 đồng đến 50.000.000 đồng" -> fine
  * "cải tạo không giam giữ đến 03 năm"      -> non-custodial reform 0..3y
  * standalone "tù chung thân" / "tử hình"   -> life / death
Robust to month/year units ("06 tháng", "03 năm") and dotted-thousand money.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
PenaltyType = str  # tu_co_thoi_han | tu_chung_than | tu_hinh | phat_tien |
# cai_tao_khong_giam_giu | canh_cao | truc_xuat


@dataclass
class PenaltyFrame:
    penalty_type: PenaltyType
    min_value: float | None
    max_value: float | None
    unit: str | None  # "nam" | "thang" | "dong" | None
    raw_text: str


@dataclass
class Point:
    ky_hieu: str  # "a", "b", "đ", ...
    noi_dung: str


@dataclass
class Clause:
    so_khoan: int
    noi_dung: str
    points: list[Point] = field(default_factory=list)
    penalty_frames: list[PenaltyFrame] = field(default_factory=list)


@dataclass
class Article:
    so_dieu: int
    tieu_de: str
    phan: str | None
    chuong: str | None
    muc: str | None
    noi_dung: str
    is_offense: bool
    clauses: list[Clause] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Line classifiers
# ---------------------------------------------------------------------------
ARTICLE_HEAD_RE = re.compile(r"^Điều\s+(\d+)\.\s*(.*)$")
KHOAN_RE = re.compile(r"^(\d{1,2})\.\s+(.*)$")
# Vietnamese point markers: a) b) ... including đ. Single lowercase letter + ")".
DIEM_RE = re.compile(r"^([a-zđ])\)\s+(.*)$")

# ---------------------------------------------------------------------------
# Penalty parsing
# ---------------------------------------------------------------------------
# A numeric value: dotted thousands (2.000.000), decimals with comma (0,1) or
# plain integers, optionally zero-padded ("02", "06").
_NUM = r"\d[\d.,]*"


def _to_number(token: str, unit: str | None) -> float | None:
    """Convert a Vietnamese-formatted numeric token to float.

    Money uses '.' as thousands separator -> strip dots.
    Quantities may use ',' as decimal -> convert to '.'.
    """
    t = token.strip()
    if not t:
        return None
    if unit == "dong":
        t = t.replace(".", "").replace(",", "")
    else:
        # years/months: drop any thousands dot, treat comma as decimal
        t = t.replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


# Range prison: "phạt tù từ 02 năm đến 07 năm" / "từ 06 tháng đến 03 năm"
# Capture min value+unit and max value+unit (units may differ).
_PRISON_RANGE_RE = re.compile(
    r"phạt tù\s+từ\s+(" + _NUM + r")\s*(năm|tháng)\s+đến\s+(" + _NUM + r")\s*(năm|tháng)",
    re.IGNORECASE,
)
# Fixed-term prison stated as a single point, e.g. "phạt tù 20 năm, tù chung thân"
_PRISON_FIXED_RE = re.compile(
    r"phạt tù\s+(" + _NUM + r")\s*(năm|tháng)\b(?!\s*đến)", re.IGNORECASE
)
_LIFE_RE = re.compile(r"tù chung thân", re.IGNORECASE)
_DEATH_RE = re.compile(r"tử hình", re.IGNORECASE)
_FINE_RANGE_RE = re.compile(
    r"phạt tiền\s+từ\s+(" + _NUM + r")\s*đồng\s+đến\s+(" + _NUM + r")\s*đồng",
    re.IGNORECASE,
)
# Single fine: "phạt tiền 50.000.000 đồng" (rare in frames)
_FINE_FIXED_RE = re.compile(
    r"phạt tiền\s+(" + _NUM + r")\s*đồng\b(?!\s*đến)", re.IGNORECASE
)
_REFORM_RE = re.compile(
    r"cải tạo không giam giữ\s+đến\s+(" + _NUM + r")\s*(năm|tháng)", re.IGNORECASE
)
_WARNING_RE = re.compile(r"phạt cảnh cáo", re.IGNORECASE)
_EXPEL_RE = re.compile(r"\btrục xuất\b", re.IGNORECASE)


def parse_penalty_frames(text: str) -> list[PenaltyFrame]:
    """Extract structured penalty frames from a clause's full text.

    A single khoản can carry several frames (e.g. prison range + life + death,
    or reform + prison). Each distinct sentencing option becomes one frame.
    """
    frames: list[PenaltyFrame] = []

    for m in _PRISON_RANGE_RE.finditer(text):
        lo = _to_number(m.group(1), "nam")
        hi = _to_number(m.group(3), "nam")
        lo_unit, hi_unit = m.group(2).lower(), m.group(4).lower()
        # Normalize to years; keep months as fractional years for comparability,
        # but record the dominant unit (max side) for display.
        min_y = _months_to_years(lo, lo_unit)
        max_y = _months_to_years(hi, hi_unit)
        frames.append(
            PenaltyFrame(
                "tu_co_thoi_han", min_y, max_y, "nam", m.group(0).strip()
            )
        )

    # Fixed-term prison only when no range matched at that position.
    if not _PRISON_RANGE_RE.search(text):
        m = _PRISON_FIXED_RE.search(text)
        if m:
            val = _months_to_years(_to_number(m.group(1), "nam"), m.group(2).lower())
            frames.append(
                PenaltyFrame("tu_co_thoi_han", val, val, "nam", m.group(0).strip())
            )

    if _LIFE_RE.search(text):
        frames.append(PenaltyFrame("tu_chung_than", None, None, None, "tù chung thân"))
    if _DEATH_RE.search(text):
        frames.append(PenaltyFrame("tu_hinh", None, None, None, "tử hình"))

    m = _FINE_RANGE_RE.search(text)
    if m:
        frames.append(
            PenaltyFrame(
                "phat_tien",
                _to_number(m.group(1), "dong"),
                _to_number(m.group(2), "dong"),
                "dong",
                m.group(0).strip(),
            )
        )
    elif _FINE_FIXED_RE.search(text):
        m = _FINE_FIXED_RE.search(text)
        v = _to_number(m.group(1), "dong")
        frames.append(PenaltyFrame("phat_tien", v, v, "dong", m.group(0).strip()))

    m = _REFORM_RE.search(text)
    if m:
        max_y = _months_to_years(_to_number(m.group(1), "nam"), m.group(2).lower())
        frames.append(
            PenaltyFrame("cai_tao_khong_giam_giu", 0.0, max_y, "nam", m.group(0).strip())
        )

    if _WARNING_RE.search(text):
        frames.append(PenaltyFrame("canh_cao", None, None, None, "phạt cảnh cáo"))

    return frames


def _months_to_years(value: float | None, unit: str) -> float | None:
    if value is None:
        return None
    if unit == "tháng":
        return round(value / 12.0, 4)
    return value


# ---------------------------------------------------------------------------
# Article body -> tree
# ---------------------------------------------------------------------------
def _is_offense(so_dieu: int, phan: str | None, is_penal_code: bool) -> bool:
    """Offense-defining articles live in Phần thứ hai (Các tội phạm).

    Only the substantive criminal code (BLHS) has offense articles. Identified
    by Phần label when available, else by BLHS article range 108..425. For
    procedural/civil codes and resolutions this is always False.
    """
    if not is_penal_code:
        return False
    if phan and "CÁC TỘI PHẠM" in phan.upper():
        return True
    return 108 <= so_dieu <= 425


def parse_article(
    so_dieu: int,
    text: str,
    section: dict | None = None,
    is_penal_code: bool = True,
) -> Article:
    section = section or {}
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]

    tieu_de = ""
    body_lines: list[str] = []
    if lines:
        m = ARTICLE_HEAD_RE.match(lines[0])
        if m:
            tieu_de = m.group(2).strip()
            body_lines = lines[1:]
        else:
            body_lines = lines

    phan = section.get("phan")
    chuong = section.get("chuong")
    muc = section.get("muc")
    is_offense = _is_offense(so_dieu, phan, is_penal_code)

    art = Article(
        so_dieu=so_dieu,
        tieu_de=tieu_de,
        phan=phan,
        chuong=chuong,
        muc=muc,
        noi_dung=text.strip(),
        is_offense=is_offense,
    )

    cur_clause: Clause | None = None
    # Buffer for an implicit "khoản 0" — body before the first numbered khoản
    # (articles that have no numbered clauses, e.g. single-paragraph articles).
    preamble: list[str] = []

    for ln in body_lines:
        mk = KHOAN_RE.match(ln)
        md = DIEM_RE.match(ln)
        if mk:
            cur_clause = Clause(so_khoan=int(mk.group(1)), noi_dung=mk.group(2).strip())
            art.clauses.append(cur_clause)
        elif md and cur_clause is not None:
            cur_clause.points.append(Point(ky_hieu=md.group(1), noi_dung=md.group(2).strip()))
            # Append point text to clause body so penalty parsing sees it too.
            cur_clause.noi_dung += "\n" + ln.strip()
        else:
            if cur_clause is not None:
                # continuation line of the current khoản (or a point)
                if cur_clause.points:
                    cur_clause.points[-1].noi_dung += " " + ln.strip()
                cur_clause.noi_dung += "\n" + ln.strip()
            else:
                preamble.append(ln.strip())

    # Article with no numbered khoản -> wrap its body as a single khoản 1.
    if not art.clauses and preamble:
        art.clauses.append(Clause(so_khoan=1, noi_dung="\n".join(preamble)))

    # Extract penalty frames per clause for offense articles.
    if is_offense:
        for cl in art.clauses:
            cl.penalty_frames = parse_penalty_frames(cl.noi_dung)

    return art


def parse_all(
    articles: dict[int, str],
    sections: dict[int, dict] | None = None,
    is_penal_code: bool = True,
) -> list[Article]:
    sections = sections or {}
    out: list[Article] = []
    for so_dieu in sorted(articles):
        out.append(
            parse_article(
                so_dieu, articles[so_dieu], sections.get(so_dieu), is_penal_code
            )
        )
    return out
