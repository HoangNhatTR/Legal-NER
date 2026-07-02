"""Layer 1 + Layer 3 verification: cited-law verification & sentencing logic.

Standalone, dependency-light (only the lawdb query API + stdlib). The optional
sentence-transformers tiebreaker for charge<->article matching is lazily
imported and degrades silently if unavailable.

Scope (MVP): ONLY citations to Bộ luật Hình sự 2015 (BLHS 2015) are verified.
Citations to other laws (BLTTHS, Nghị quyết, Luật THADS, ...) are recognised but
marked ``out_of_scope`` — they are outside the current DB.

Public API:
    from verify.citation import check_citations, CitationReport
    report = check_citations(entities)            # entities: list[dict]
    report = check_citations(entities, law_filter="BLHS")

Each entity dict is the shape produced by either the NER inference layer
(``{"label","start","end","text"}``) or the API extract flow
(``{"type","text","start","end"}``). Both ``type`` and ``label`` keys are
accepted.

Layers implemented here:
    L1  — citation existence + content (Điều / khoản / điểm in DB?)
    L1+ — charge<->article semantic match (CRIME text vs cited offence title)
    L3  — sentencing-frame check (pronounced PENALTY within statutory min/max?)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from pydantic import BaseModel, Field

from lawdb import LAW_REGISTRY
from lawdb.lookup import get_article, penalty_frame, verify_citation

# Coarse law tags (classify_law) -> registry law_key for documents we hold in
# the DB. Tags absent here (e.g. 'OTHER' = Luật THADS / Nghị quyết) remain out
# of scope. 'NQ326' would be addable once NQ coverage is complete.
LAW_TAG_TO_KEY: dict[str, str] = {
    "BLHS": "BLHS2015",
    "BLTTHS": "BLTTHS2015",
    "BLDS": "BLDS2015",
}
# law_keys actually present in the DB are the verifiable scope.
IN_SCOPE_KEYS = set(LAW_REGISTRY)

# ---------------------------------------------------------------------------
# Law-name classification
# ---------------------------------------------------------------------------
# We only verify BLHS 2015. To do that reliably we must NOT confuse it with the
# Bộ luật TỐ TỤNG Hình sự (BLTTHS) — both contain the substring "hình sự".
# Strategy: detect BLTTHS / other laws first; only what is left and looks like a
# criminal code is treated as BLHS.

_BLTTHS_RE = re.compile(r"tố\s*tụng\s*hình\s*sự", re.IGNORECASE)
_BLHS_RE = re.compile(r"\bbộ\s*luật\s*hình\s*sự\b", re.IGNORECASE)
# Bộ luật Dân sự (substantive civil code) — but NOT "tố tụng dân sự" (procedure,
# which we do not hold). Tested before _OTHER_LAW_RE so it isn't swept into OTHER.
_BLDS_RE = re.compile(r"\bbộ\s*luật\s*dân\s*sự\b", re.IGNORECASE)
_BLTTDS_RE = re.compile(r"tố\s*tụng\s*dân\s*sự", re.IGNORECASE)
_OTHER_LAW_RE = re.compile(
    r"nghị\s*quyết|thông\s*tư|nghị\s*định|"
    r"luật\s*thi\s*hành\s*án|luật\s*\w|hiến\s*pháp|"
    r"tố\s*tụng\s*dân\s*sự",
    re.IGNORECASE,
)

# The DB holds ONLY the 2015-vintage codes (BLHS/BLTTHS/BLDS 2015). A judgment may
# also cite the OLD codes ("Bộ luật Hình sự năm 1999", "... năm 1985"). Those are
# different documents we do NOT hold, so they must be out-of-scope — NOT verified
# (and certainly not flagged) against the 2015 article tree. We read the FIRST
# year that follows the code name as its vintage; trailing "(sửa đổi, bổ sung năm
# 2017/2025)" are AMENDMENTS to the 2015 code, not a different vintage, so only
# the first year counts.
_LAW_YEAR_RE = re.compile(r"năm\s*((?:19|20)\d{2})", re.IGNORECASE)
_DB_VINTAGE_YEAR = "2015"


def _is_db_vintage(law_text: str | None) -> bool:
    """True if a substantive/procedure code carries the 2015 vintage (or no year).

    No explicit year -> assume the in-DB 2015 vintage (the common case). An
    explicit non-2015 year (e.g. 1999) -> a different document we do not hold.
    """
    if not law_text:
        return True
    m = _LAW_YEAR_RE.search(law_text)
    if not m:
        return True
    return m.group(1) == _DB_VINTAGE_YEAR


def classify_law(law_text: str | None) -> str:
    """Return a coarse law tag: 'BLHS' | 'BLTTHS' | 'BLDS' | 'OTHER' | 'UNKNOWN'.

    Order matters: BLTTHS (tố tụng hình sự) and BLTTDS (tố tụng dân sự) are tested
    before BLHS / BLDS because their names contain the substantive-code names as
    substrings once "tố tụng" is ignored.

    A criminal/civil code cited with an explicit NON-2015 vintage (e.g. "Bộ luật
    Hình sự năm 1999") is tagged ``OTHER`` — the DB only holds the 2015 codes, so
    the old code is out of scope rather than a (false) not-found against 2015.
    """
    if not law_text:
        return "UNKNOWN"
    if _BLTTHS_RE.search(law_text):
        return "BLTTHS" if _is_db_vintage(law_text) else "OTHER"
    if _BLTTDS_RE.search(law_text):
        return "OTHER"  # Bộ luật Tố tụng dân sự — not held in DB
    if _BLHS_RE.search(law_text):
        return "BLHS" if _is_db_vintage(law_text) else "OTHER"
    if _BLDS_RE.search(law_text):
        return "BLDS" if _is_db_vintage(law_text) else "OTHER"
    if _OTHER_LAW_RE.search(law_text):
        return "OTHER"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------
# A judgment cites like:
#   "điểm a khoản 2 Điều 173 của Bộ luật Hình sự"
#   "Căn cứ điểm c khoản 1 Điều 250; điểm s khoản 1 Điều 51 Bộ luật Hình sự 2015"
# i.e. a chain of (điểm? khoản? Điều) segments, semicolon/comma separated, with a
# trailing law name that applies to the whole chain (until a new law name
# appears mid-chain — handled per-segment from the nearest following law token).

_DIEU_RE = re.compile(r"đi[eề]u\s*0*(\d{1,3})", re.IGNORECASE)
_KHOAN_RE = re.compile(r"kho[aả]n\s*0*(\d{1,3})", re.IGNORECASE)
# điểm letters: single Vietnamese/latin letter token(s) e.g. "a", "c", "s".
# We capture the letter(s) that directly follow "điểm" (possibly "a, c").
_DIEM_RE = re.compile(r"đi[eể]m\s+([a-zđ](?:\s*,\s*[a-zđ])*)", re.IGNORECASE)


def _strip_accents_lower(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower()


class ParsedCitation(BaseModel):
    """One structured citation extracted from NER output."""

    raw: str = Field(description="the source text the citation was parsed from")
    so_dieu: int | None = None
    so_khoan: int | None = None
    diem: str | None = None
    law: str = Field(description="coarse law tag: BLHS | BLTTHS | OTHER | UNKNOWN")
    law_text: str | None = Field(
        default=None, description="the matched law-name substring, if any"
    )
    chain_id: int = Field(
        default=-1,
        description=(
            "id grouping citations that came from the SAME chain/sentence; used "
            "by the DB-validated re-attachment safety net to recover a điểm/khoản "
            "mis-grouped onto an adjacent article in the chain"
        ),
    )


def _law_in_segment(segment: str) -> tuple[str, str | None]:
    """Find the nearest law name within a segment; return (tag, matched_text)."""
    # Prefer an explicit law mention inside this very segment. Order mirrors
    # classify_law (procedure codes before substantive codes).
    for rx in (_BLTTHS_RE, _BLTTDS_RE, _BLHS_RE, _BLDS_RE):
        m = rx.search(segment)
        if m:
            return classify_law(segment), segment[m.start() : m.start() + 40]
    m = _OTHER_LAW_RE.search(segment)
    if m:
        return "OTHER", segment[m.start() : m.start() + 40]
    return "UNKNOWN", None


# Token kinds emitted by the token-stream scanner.
#   DIEM  -> letter(s) e.g. "a" or "a, c"
#   KHOAN -> int
#   DIEU  -> int
#   LAW   -> (tag, matched_text)
_TOKEN_RE = re.compile(
    r"(?P<diem>đi[eể]m\s+[a-zđ](?:\s*,\s*[a-zđ])*)"
    r"|(?P<khoan>kho[aả]n\s*0*\d{1,3})"
    r"|(?P<dieu>đi[eề]u\s*0*\d{1,3})",
    re.IGNORECASE,
)
# Law-name tokens, with their coarse tag. Procedure codes BEFORE substantive
# codes (mirrors classify_law) so "tố tụng hình sự" is not read as BLHS.
_LAW_TOKEN_RES: list[tuple[re.Pattern[str], str]] = [
    (_BLTTHS_RE, "BLTTHS"),
    (_BLTTDS_RE, "OTHER"),
    (_BLHS_RE, "BLHS"),
    (_BLDS_RE, "BLDS"),
    (_OTHER_LAW_RE, "OTHER"),
]


def _scan_law_tokens(text: str) -> list[tuple[int, str, str]]:
    """Return (pos, tag, matched_text) for every law-name mention, in order.

    Each character position is claimed by at most one law tag (the
    highest-priority one), so "tố tụng hình sự" yields a single BLTTHS token and
    is not double-counted as BLHS.
    """
    claimed: list[tuple[int, int]] = []
    out: list[tuple[int, str, str]] = []
    for rx, tag in _LAW_TOKEN_RES:
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            if any(s < ce and e > cs for cs, ce in claimed):
                continue  # overlaps a higher-priority law token already taken
            claimed.append((s, e))
            window = text[s : s + 40]
            # Demote a NON-2015-vintage substantive/procedure code to OTHER
            # (out of scope): the DB only holds the 2015 codes, so verifying an
            # old code (e.g. BLHS 1999) against 2015 would be a false anomaly.
            if tag in {"BLHS", "BLTTHS", "BLDS"} and not _is_db_vintage(window):
                tag = "OTHER"
            out.append((s, tag, window))
    out.sort(key=lambda t: t[0])
    return out


def _diem_letters(raw: str) -> list[str]:
    """Split a 'điểm a, c' token into its individual point letters."""
    m = _DIEM_RE.search(raw)
    if not m:
        return []
    return [p.strip().lower() for p in re.split(r"\s*,\s*", m.group(1)) if p.strip()]


def parse_citation_text(text: str, *, chain_id: int = -1) -> list[ParsedCitation]:
    """Parse a free-text citation / chained legal-basis string into citations.

    Vietnamese citation grammar reads RIGHT-TO-LEFT within a segment:
    ``điểm X khoản Y Điều Z`` -> (Z, Y, X). A ``điểm`` / ``khoản`` qualifier
    therefore binds FORWARD to the FIRST ``Điều`` that follows it, NOT to the
    article that happens to precede it in the string. This is the core fix for
    chained citations such as::

        điểm c khoản 1 Điều 250; điểm s khoản 1 Điều 51

    which must yield (250,1,c) and (51,1,s) — never (250,1,s).

    Implementation: scan the whole string into an ordered token stream
    (điểm / khoản / Điều / law-name). Walk it left-to-right, buffering the
    điểm/khoản qualifiers seen since the last Điều; when a Điều token appears,
    flush the buffered qualifiers onto THAT article. The law name for an article
    is the first law token between this Điều and the next Điều, else the chain's
    trailing (last) law mention.
    """
    if not text or not _DIEU_RE.search(text):
        return []

    # Chain-level fallback law = last law mention anywhere in the string.
    law_tokens = _scan_law_tokens(text)
    inherited_tag, inherited_text = ("UNKNOWN", None)
    if law_tokens:
        inherited_tag, inherited_text = law_tokens[-1][1], law_tokens[-1][2]

    # Ordered structural tokens.
    toks: list[tuple[int, str, object]] = []
    for m in _TOKEN_RE.finditer(text):
        if m.group("diem") is not None:
            toks.append((m.start(), "DIEM", m.group("diem")))
        elif m.group("khoan") is not None:
            km = _KHOAN_RE.search(m.group("khoan"))
            toks.append((m.start(), "KHOAN", int(km.group(1))))
        elif m.group("dieu") is not None:
            dm = _DIEU_RE.search(m.group("dieu"))
            toks.append((m.start(), "DIEU", int(dm.group(1))))

    parsed: list[ParsedCitation] = []
    # Buffered qualifiers waiting for their (forward) Điều. We keep the MOST
    # RECENT điểm and khoản before the article: "khoản 4 khoản 5 Điều 250" uses
    # the nearest (khoản 5); a stray earlier qualifier from OCR garbage does not
    # leak across an article boundary because the buffer is cleared on flush.
    pending_diem_raw: str | None = None
    pending_khoan: int | None = None

    dieu_positions = [t[0] for t in toks if t[1] == "DIEU"]

    def _law_for(article_pos: int) -> tuple[str, str | None]:
        # next Điều position after this one (segment end for law search)
        nexts = [p for p in dieu_positions if p > article_pos]
        seg_end = min(nexts) if nexts else len(text)
        for pos, tag, txt in law_tokens:
            if article_pos < pos < seg_end:
                return tag, txt
        return inherited_tag, inherited_text

    for pos, kind, val in toks:
        if kind == "DIEM":
            pending_diem_raw = val  # nearest điểm wins
        elif kind == "KHOAN":
            pending_khoan = val      # nearest khoản wins
        elif kind == "DIEU":
            so_dieu = int(val)  # type: ignore[arg-type]
            so_khoan = pending_khoan
            diem_letters = _diem_letters(pending_diem_raw) if pending_diem_raw else []
            law, law_text = _law_for(pos)
            raw_parts = [p for p in (
                pending_diem_raw,
                f"khoản {so_khoan}" if so_khoan is not None else None,
                f"Điều {so_dieu}",
                law_text,
            ) if p]
            raw = " ".join(raw_parts)
            if diem_letters:
                parsed.extend(
                    ParsedCitation(
                        raw=raw, so_dieu=so_dieu, so_khoan=so_khoan,
                        diem=d, law=law, law_text=law_text, chain_id=chain_id,
                    )
                    for d in diem_letters
                )
            else:
                parsed.append(
                    ParsedCitation(
                        raw=raw, so_dieu=so_dieu, so_khoan=so_khoan,
                        diem=None, law=law, law_text=law_text, chain_id=chain_id,
                    )
                )
            # clear buffered qualifiers — they belonged to THIS article only
            pending_diem_raw = None
            pending_khoan = None

    return parsed


# ---------------------------------------------------------------------------
# Entity normalisation + reconstruction of citations from separate spans
# ---------------------------------------------------------------------------
def _etype(ent: dict) -> str:
    return (ent.get("type") or ent.get("label") or "").upper()


def _etext(ent: dict) -> str:
    return (ent.get("text") or "").strip()


def _estart(ent: dict) -> int:
    return int(ent.get("start", 0) or 0)


def reconstruct_from_spans(entities: list[dict], *, chain_base: int = 0) -> list[ParsedCitation]:
    """Reconstruct citations from separate ARTICLE/CLAUSE/POINT/LAW_NAME spans.

    Real NER often emits ``điểm b`` (POINT), ``khoản 4`` (CLAUSE), ``Điều 250``
    (ARTICLE), ``Bộ luật Hình sự ...`` (LAW_NAME) as adjacent but separate
    entities. We sort by start offset and, for each ARTICLE, attach the most
    recent preceding CLAUSE/POINT and the nearest LAW_NAME (preceding or
    immediately following) within a small offset window.

    ``chain_base`` offsets the chain ids assigned here so they do not collide
    with chain ids handed out to LEGAL_BASIS chains. Spans whose ARTICLE tokens
    sit close together (gap <= ``CHAIN_GAP``) share a chain id, so the
    DB-validated re-attachment safety net can recover a điểm/khoản that NER drift
    parked on an adjacent article in the same run.
    """
    spans = sorted(
        [e for e in entities if _etype(e) in {"ARTICLE", "CLAUSE", "POINT", "LAW_NAME"}],
        key=_estart,
    )
    citations: list[ParsedCitation] = []
    WINDOW = 60  # chars: how far a CLAUSE/POINT/LAW may sit from its Điều
    CHAIN_GAP = 80  # chars between adjacent ARTICLEs that still count as one chain

    prev_article_start: int | None = None
    chain_id = chain_base - 1

    for i, ent in enumerate(spans):
        if _etype(ent) != "ARTICLE":
            continue
        dieu_m = _DIEU_RE.search(_etext(ent))
        if not dieu_m:
            continue
        so_dieu = int(dieu_m.group(1))
        a_start = _estart(ent)

        # open a new chain when this ARTICLE is far from the previous one
        if prev_article_start is None or a_start - prev_article_start > CHAIN_GAP:
            chain_id += 1
        prev_article_start = a_start

        so_khoan: int | None = None
        diem: str | None = None
        law_text: str | None = None

        # look back over a few preceding spans for the nearest CLAUSE/POINT/LAW
        for prev in reversed(spans[max(0, i - 6) : i]):
            if a_start - _estart(prev) > WINDOW:
                break
            pt = _etype(prev)
            if pt == "CLAUSE" and so_khoan is None:
                km = _KHOAN_RE.search(_etext(prev))
                if km:
                    so_khoan = int(km.group(1))
            elif pt == "POINT" and diem is None:
                dm = _DIEM_RE.search(_etext(prev))
                if dm:
                    diem = dm.group(1).split(",")[0].strip().lower()
            elif pt == "LAW_NAME" and law_text is None:
                law_text = _etext(prev)

        # look forward for a LAW_NAME directly following the Điều
        if law_text is None:
            for nxt in spans[i + 1 : i + 4]:
                if _estart(nxt) - a_start > WINDOW:
                    break
                if _etype(nxt) == "LAW_NAME":
                    law_text = _etext(nxt)
                    break

        law = classify_law(law_text)
        raw_parts = [p for p in (
            f"điểm {diem}" if diem else None,
            f"khoản {so_khoan}" if so_khoan else None,
            f"Điều {so_dieu}",
            law_text,
        ) if p]
        citations.append(
            ParsedCitation(
                raw=" ".join(raw_parts),
                so_dieu=so_dieu,
                so_khoan=so_khoan,
                diem=diem,
                law=law,
                law_text=law_text,
                chain_id=chain_id,
            )
        )
    return citations


def _dedupe(cits: list[ParsedCitation]) -> list[ParsedCitation]:
    seen: set[tuple] = set()
    out: list[ParsedCitation] = []
    for c in cits:
        key = (c.so_dieu, c.so_khoan, c.diem, c.law)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def collect_citations(entities: list[dict]) -> list[ParsedCitation]:
    """Collect citations from BOTH LEGAL_BASIS chained text AND span groups.

    LEGAL_BASIS spans are the richest source (they hold full chains). We also
    reconstruct from separate ARTICLE/CLAUSE/POINT spans to catch citations the
    model didn't bundle into a LEGAL_BASIS. Results are de-duplicated.
    """
    cits: list[ParsedCitation] = []
    chain_id = 0
    for ent in entities:
        if _etype(ent) == "LEGAL_BASIS":
            cits.extend(parse_citation_text(_etext(ent), chain_id=chain_id))
            chain_id += 1
    # span-reconstructed chains continue numbering after the LEGAL_BASIS chains
    cits.extend(reconstruct_from_spans(entities, chain_base=chain_id))
    return _dedupe(cits)


# ---------------------------------------------------------------------------
# Charge <-> article matching (L1+)
# ---------------------------------------------------------------------------
def _charge_match(crime_text: str, article_title: str) -> str:
    """Return 'true' | 'false' | 'uncertain' for CRIME vs article title.

    String-first: normalise (drop accents/case), check substring either way.
    The article title is like "Tội vận chuyển trái phép chất ma túy"; the CRIME
    entity is usually the bare offence "Vận chuyển trái phép chất ma túy".
    """
    if not crime_text or not article_title:
        return "uncertain"
    c = _strip_accents_lower(crime_text).strip()
    t = _strip_accents_lower(article_title).strip()
    # title often prefixed with "toi " — strip for comparison
    t_core = re.sub(r"^toi\s+", "", t)
    if len(c) < 4:
        return "uncertain"
    if c in t_core or t_core in c:
        return "true"
    # token-overlap heuristic before declaring a mismatch
    c_tok = set(c.split())
    t_tok = set(t_core.split())
    if c_tok and t_tok:
        overlap = len(c_tok & t_tok) / len(c_tok)
        if overlap >= 0.7:
            return "true"
        if overlap >= 0.4:
            return _charge_match_semantic(crime_text, article_title)
    return _charge_match_semantic(crime_text, article_title)


_ST_MODEL = None
_ST_TRIED = False


def _charge_match_semantic(crime_text: str, article_title: str) -> str:
    """Optional sentence-transformers cosine tiebreaker. Degrades to 'false'.

    Loaded lazily; kept fast/CPU. Any failure (not installed, GPU full, ...)
    returns the conservative 'uncertain' so we never crash the report.
    """
    global _ST_MODEL, _ST_TRIED
    if _ST_TRIED and _ST_MODEL is None:
        return "uncertain"
    try:
        if _ST_MODEL is None:
            _ST_TRIED = True
            from sentence_transformers import SentenceTransformer, util  # noqa

            _ST_MODEL = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                device="cpu",
            )
        from sentence_transformers import util

        emb = _ST_MODEL.encode([crime_text, article_title], convert_to_tensor=True)
        sim = float(util.cos_sim(emb[0], emb[1])[0][0])
        if sim >= 0.65:
            return "true"
        if sim <= 0.35:
            return "false"
        return "uncertain"
    except Exception:
        return "uncertain"


# ---------------------------------------------------------------------------
# Sentencing-frame check (L3)
# ---------------------------------------------------------------------------
_PENALTY_DEATH_RE = re.compile(r"tử\s*hình", re.IGNORECASE)
_PENALTY_LIFE_RE = re.compile(r"chung\s*thân", re.IGNORECASE)
# "08 năm 09 tháng tù", "20 năm tù", "02 (hai) năm tù", "7 năm tù"
_PENALTY_YM_RE = re.compile(
    r"0*(\d{1,3})\s*năm(?:\s*(?:\(.*?\)\s*)?0*(\d{1,2})\s*tháng)?", re.IGNORECASE
)
_PENALTY_M_ONLY_RE = re.compile(r"0*(\d{1,2})\s*tháng", re.IGNORECASE)


def parse_penalty(text: str) -> dict | None:
    """Parse a pronounced PENALTY entity into {type, years, raw}.

    Returns years as a float (months folded in). Only custodial / life / death
    are parsed; fines and probation-only return None (out of L3 scope).
    """
    if not text:
        return None
    if _PENALTY_DEATH_RE.search(text):
        return {"type": "tu_hinh", "years": None, "raw": text}
    if _PENALTY_LIFE_RE.search(text):
        return {"type": "tu_chung_than", "years": None, "raw": text}
    # require the word "tù" somewhere to treat a number as a prison term
    if "tù" not in text.lower():
        return None
    m = _PENALTY_YM_RE.search(text)
    if m:
        years = float(m.group(1))
        if m.group(2):
            years += int(m.group(2)) / 12.0
        return {"type": "tu_co_thoi_han", "years": round(years, 3), "raw": text}
    m = _PENALTY_M_ONLY_RE.search(text)
    if m:
        return {"type": "tu_co_thoi_han", "years": round(int(m.group(1)) / 12.0, 3), "raw": text}
    return None


# severity rank for comparing custodial penalty types
_PEN_RANK = {"tu_co_thoi_han": 1, "tu_chung_than": 2, "tu_hinh": 3}


def check_sentencing(pen: dict, frame: dict) -> dict:
    """Compare a parsed penalty against a clause penalty_frame.

    Status:
      within_frame — value within [min, max] (or matching life/death frame)
      out_of_frame — OVER the statutory max  (the real red flag)
      note         — UNDER the statutory min (legally possible via Điều 54)
      uncertain    — frame has no numeric bounds / types incomparable

    Conservative by design: under-frame is a 'note', not a flag, because courts
    may reduce below the frame under Điều 54 (mitigating circumstances).
    """
    if not frame or not frame.get("ton_tai"):
        return {"status": "uncertain", "reason_vi": "không có khung hình phạt trong CSDL"}

    ptype = pen["type"]
    # Death / life: compare against the presence of such a frame type.
    if ptype in ("tu_hinh", "tu_chung_than"):
        types = {f["type"] for f in frame.get("all_frames", [])}
        if ptype in types:
            return {"status": "within_frame", "reason_vi": "loại hình phạt nằm trong khung"}
        # is the pronounced penalty MORE severe than the frame allows?
        max_rank = max((_PEN_RANK.get(t, 0) for t in types), default=0)
        if _PEN_RANK[ptype] > max_rank:
            return {
                "status": "out_of_frame",
                "reason_vi": (
                    f"hình phạt tuyên ({pen['raw']}) NẶNG hơn khung hình phạt tối đa"
                ),
            }
        return {"status": "note", "reason_vi": "hình phạt tuyên nhẹ hơn khung (có thể do giảm nhẹ)"}

    # Fixed-term prison: compare ONLY against a custodial (tù có thời hạn) frame
    # measured in years. The clause's principal frame may be a fine (phạt tiền,
    # unit=đồng) — comparing a prison term against đồng bounds is meaningless, so
    # we re-select the fixed-term-prison frame from all_frames.
    all_frames = frame.get("all_frames", [])
    prison = next(
        (f for f in all_frames if f["type"] == "tu_co_thoi_han" and f.get("unit") == "nam"),
        None,
    )
    if prison is None:
        types = {f["type"] for f in all_frames}
        if types & {"tu_chung_than", "tu_hinh"}:
            # clause allows life/death but no fixed-term frame: any term is below
            return {
                "status": "note",
                "reason_vi": "khung khoản này là chung thân/tử hình; mức tù có thời hạn tuyên nhẹ hơn",
            }
        return {
            "status": "uncertain",
            "reason_vi": "khoản viện dẫn không có khung tù có thời hạn để so sánh (vd: phạt tiền)",
        }

    fmin = prison.get("min")
    fmax = prison.get("max")
    yrs = pen["years"]
    if yrs is None or (fmin is None and fmax is None):
        return {"status": "uncertain", "reason_vi": "thiếu dữ liệu số năm để so sánh"}

    # If the frame also allows life/death, an over-max fixed term is still fine.
    types = {f["type"] for f in all_frames}
    higher_allowed = bool(types & {"tu_chung_than", "tu_hinh"})

    if fmax is not None and yrs > fmax + 1e-6 and not higher_allowed:
        return {
            "status": "out_of_frame",
            "reason_vi": (
                f"mức tù tuyên {yrs:g} năm VƯỢT trần khung {fmax:g} năm — cần kiểm tra"
            ),
        }
    if fmin is not None and yrs < fmin - 1e-6:
        return {
            "status": "note",
            "reason_vi": (
                f"mức tù tuyên {yrs:g} năm THẤP hơn sàn khung {fmin:g} năm "
                "(hợp lệ nếu áp dụng Điều 54 giảm nhẹ)"
            ),
        }
    return {
        "status": "within_frame",
        "reason_vi": f"mức tù tuyên {yrs:g} năm nằm trong khung [{fmin:g}–{fmax:g}] năm",
    }


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------
class CitationResult(BaseModel):
    raw: str
    so_dieu: int | None = None
    so_khoan: int | None = None
    diem: str | None = None
    law: str
    status: str = Field(
        description="valid | not_found | out_of_scope | unparseable | needs_review"
    )
    article_title: str | None = None
    message_vi: str | None = None
    reattached: dict | None = Field(
        default=None,
        description=(
            "set when a parsed điểm/khoản did not exist on the parsed article but "
            "WAS found elsewhere in the same citation chain (mis-grouping artifact); "
            "holds the corrected {so_dieu, so_khoan, diem} and a Vietnamese note"
        ),
    )
    charge_match: str | None = Field(
        default=None, description="true | false | uncertain (offence-article match)"
    )
    sentencing: dict | None = Field(
        default=None, description="L3 result: {status, reason_vi, ...} or null"
    )
    law_key: str | None = Field(
        default=None, description="resolved DB document key (BLHS2015/BLTTHS2015/...)"
    )
    validity: dict | None = Field(
        default=None,
        description="validity-at-date block from lookup.check_validity (if on_date given)",
    )


class CitationSummary(BaseModel):
    total: int = 0
    valid: int = 0
    not_found: int = 0
    out_of_scope: int = 0
    unparseable: int = 0
    charge_mismatches: int = 0
    sentencing_flags: int = 0
    validity_flags: int = 0
    needs_review: int = 0
    reattached: int = 0


class CitationReport(BaseModel):
    law_scope: str = "BLHS 2015 + BLTTHS 2015 + BLDS 2015"
    citations: list[CitationResult] = Field(default_factory=list)
    summary: CitationSummary = Field(default_factory=CitationSummary)
    flags_vi: list[str] = Field(default_factory=list)
    note_vi: str = (
        "Kiểm chứng các viện dẫn thuộc BLHS 2015, Bộ luật Tố tụng hình sự 2015 "
        "và Bộ luật Dân sự 2015. Các văn bản khác (Nghị quyết, Luật THADS, Bộ "
        "luật Tố tụng dân sự, ...) hiện nằm ngoài phạm vi CSDL. Kiểm tra hiệu "
        "lực ở cấp văn bản (chưa theo từng điều)."
    )


# ---------------------------------------------------------------------------
# DB-validated re-attachment (safety net for chained-citation mis-grouping)
# ---------------------------------------------------------------------------
def _try_reattach(
    cit: ParsedCitation,
    chain_cits: list[ParsedCitation],
    target_key: str,
    *,
    on_date: str | None,
    doc_cits: list[ParsedCitation] | None = None,
) -> dict | None:
    """Attempt to recover a not-found điểm/khoản by re-attaching within the chain.

    Called only when ``cit`` failed L1 because its (Điều, khoản, điểm) triple does
    not exist in the DB, BUT the Điều itself exists (an article that doesn't exist
    at all is a genuine error and is never re-attached here).

    Search order, all DB-validated (we only re-attach to combinations that truly
    EXIST), most-local first:
      1. same Điều, a DIFFERENT khoản cited elsewhere in the chain that DOES
         carry this điểm  (fixes "khoản 4 khoản 5" OCR picking the wrong khoản);
      2. same Điều, ANY khoản in the DB that carries this điểm;
      3. a DIFFERENT Điều cited in the same chain whose (khoản, điểm) exists
         (fixes a điểm parked on the adjacent article in the chain);
      4. DOCUMENT-WIDE fallback (only when ``doc_cits`` is given): the SAME
         (khoản, điểm) was ALSO parsed as a DB-valid citation under a DIFFERENT
         Điều somewhere else in the judgment. This recovers span-reconstruction
         drift where NER parked ``điểm s`` on ``Điều 250`` even though the
         judgment's own LEGAL_BASIS chain cites ``điểm s khoản 1 Điều 51``.
         Strict: we only re-attach to a triple that ALREADY EXISTS as an
         independent valid citation in the same document — we never invent one.

    Returns a ``reattached`` dict (corrected target + Vietnamese note) or None
    when the điểm/khoản exists under NONE of the chain's (or document's)
    articles — i.e. a genuine anomaly that must still be flagged.
    """
    if cit.diem is None and cit.so_khoan is None:
        return None  # nothing to re-attach (bare article handled upstream)

    diem = cit.diem
    # khoản numbers cited elsewhere in the chain for the SAME article
    same_dieu_khoans = sorted({
        c.so_khoan for c in chain_cits
        if c.so_dieu == cit.so_dieu and c.so_khoan is not None
        and c.so_khoan != cit.so_khoan
    })

    def _exists(d: int, k: int | None, p: str | None) -> bool:
        return bool(verify_citation(d, k, p, law_key=target_key, on_date=on_date)["exists"])

    # --- 1. same Điều, a different khoản from the chain that carries this điểm
    if diem is not None:
        for k in same_dieu_khoans:
            if _exists(cit.so_dieu, k, diem):
                return {
                    "so_dieu": cit.so_dieu, "so_khoan": k, "diem": diem,
                    "reason_vi": (
                        f"điểm {diem}) không nằm ở khoản {cit.so_khoan} mà ở khoản {k} "
                        f"Điều {cit.so_dieu} (gộp nhầm khoản trong chuỗi viện dẫn — đã hiệu chỉnh)"
                    ),
                }

    # --- 2. same Điều, ANY khoản in the DB that carries this điểm
    if diem is not None:
        art = get_article(cit.so_dieu, law_key=target_key)
        if art.get("ton_tai"):
            for kh in art.get("khoan", []):
                if kh["so_khoan"] == cit.so_khoan:
                    continue
                if any(p["ky_hieu"] == diem for p in kh.get("diem", [])):
                    return {
                        "so_dieu": cit.so_dieu, "so_khoan": kh["so_khoan"], "diem": diem,
                        "reason_vi": (
                            f"điểm {diem}) thuộc khoản {kh['so_khoan']} Điều {cit.so_dieu} "
                            f"(không phải khoản {cit.so_khoan} — gộp nhầm khoản, đã hiệu chỉnh)"
                        ),
                    }

    # --- 3. a DIFFERENT Điều cited in the chain whose (khoản, điểm) exists
    other_dieus = sorted({
        c.so_dieu for c in chain_cits
        if c.so_dieu is not None and c.so_dieu != cit.so_dieu
    })
    for d in other_dieus:
        # try the citation's own khoản first, then any khoản that chain peers used
        khoan_candidates: list[int | None] = []
        for c in chain_cits:
            if c.so_dieu == d and c.so_khoan is not None and c.so_khoan not in khoan_candidates:
                khoan_candidates.append(c.so_khoan)
        if cit.so_khoan is not None and cit.so_khoan not in khoan_candidates:
            khoan_candidates.append(cit.so_khoan)
        if not khoan_candidates:
            khoan_candidates = [cit.so_khoan]
        for k in khoan_candidates:
            if _exists(d, k, diem):
                return {
                    "so_dieu": d, "so_khoan": k, "diem": diem,
                    "reason_vi": (
                        f"điểm {diem}) {('khoản ' + str(k) + ' ') if k else ''}"
                        f"thực chất thuộc Điều {d} (cùng chuỗi viện dẫn), không phải "
                        f"Điều {cit.so_dieu} — gộp nhầm điều trong chuỗi, đã hiệu chỉnh"
                    ),
                }

    # --- 4. document-wide: same (khoản, điểm) cited as a valid triple under a
    #        DIFFERENT Điều elsewhere in the judgment. High confidence because we
    #        re-attach only to an independently-parsed, DB-valid citation.
    if diem is not None and doc_cits is not None:
        for c in doc_cits:
            if c.so_dieu is None or c.so_dieu == cit.so_dieu:
                continue
            if c.diem != diem:
                continue
            # require the SAME khoản number (a điểm letter is meaningless across
            # different khoản numbers, so don't blur khoản here)
            if cit.so_khoan is not None and c.so_khoan != cit.so_khoan:
                continue
            if _exists(c.so_dieu, c.so_khoan, diem):
                return {
                    "so_dieu": c.so_dieu, "so_khoan": c.so_khoan, "diem": diem,
                    "reason_vi": (
                        f"điểm {diem}) {('khoản ' + str(c.so_khoan) + ' ') if c.so_khoan else ''}"
                        f"thực chất thuộc Điều {c.so_dieu} (đã được viện dẫn đúng ở chỗ "
                        f"khác trong bản án), không phải Điều {cit.so_dieu} — NER gộp nhầm "
                        f"điều, đã hiệu chỉnh"
                    ),
                }
    return None


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------
def check_citations(
    entities: Iterable[dict],
    *,
    law_filter: str | None = None,
    on_date: str | None = None,
) -> CitationReport:
    """Verify all citations found in NER ``entities`` against the law DB.

    Pipeline: collect citations (chained LEGAL_BASIS + span reconstruction) ->
    route each citation to its law document (BLHS / BLTTHS / BLDS) -> L1
    existence -> L1+ charge match (CRIME vs offence title, BLHS only) -> L3
    sentencing check (pronounced PENALTY vs cited clause frame, BLHS only).

    Scope: any citation whose law maps to a document held in the DB
    (BLHS2015 / BLTTHS2015 / BLDS2015) is verified against that document.
    UNKNOWN-law citations default to BLHS. Laws not in the DB (Luật THADS,
    Nghị quyết, Bộ luật Tố tụng dân sự, ...) remain ``out_of_scope``.

    ``law_filter`` (legacy, optional): if set to a coarse tag (e.g. 'BLHS'),
    only that law is verified and everything else is out_of_scope — preserves
    the original single-law behaviour for callers that pass it.

    ``on_date`` (optional ISO date, e.g. the judgment date): when given, each
    verified citation carries a ``validity`` block flagging anomalies such as a
    code cited before it took effect.
    """
    entities = list(entities)
    parsed = collect_citations(entities)

    # index citations by chain_id so the re-attachment safety net can see the
    # other articles/khoản cited in the SAME chain/sentence.
    chains_by_id: dict[int, list[ParsedCitation]] = {}
    for c in parsed:
        chains_by_id.setdefault(c.chain_id, []).append(c)

    # gather CRIME + PENALTY entities once for L1+/L3
    crimes = [
        _etext(e)
        for e in entities
        if _etype(e) == "CRIME" and len(_etext(e)) >= 4
    ]
    penalties_raw = [
        _etext(e) for e in entities if _etype(e) == "PENALTY"
    ]
    penalties = [p for p in (parse_penalty(t) for t in penalties_raw) if p]
    # the principal custodial penalty (most severe) drives the L3 check
    principal_penalty = None
    if penalties:
        principal_penalty = max(
            penalties,
            key=lambda p: (_PEN_RANK.get(p["type"], 0), p.get("years") or 0.0),
        )

    results: list[CitationResult] = []
    summary = CitationSummary()
    flags: list[str] = []

    # de-dup the offence article used for charge match (the heaviest/first
    # offence Điều). We compute charge match against the offence article(s).
    for cit in parsed:
        summary.total += 1

        if cit.so_dieu is None:
            summary.unparseable += 1
            results.append(
                CitationResult(
                    raw=cit.raw, law=cit.law, status="unparseable",
                    message_vi="Không nhận diện được số Điều từ viện dẫn.",
                )
            )
            continue

        # Route the citation to a DB document. UNKNOWN defaults to BLHS.
        # ``law_filter`` (legacy) pins verification to one coarse tag.
        if law_filter is not None:
            if cit.law != law_filter and cit.law != "UNKNOWN":
                target_key = None
            else:
                target_key = LAW_TAG_TO_KEY.get(law_filter, "BLHS2015")
        else:
            target_key = LAW_TAG_TO_KEY.get(
                cit.law, "BLHS2015" if cit.law == "UNKNOWN" else None
            )

        if target_key is None or target_key not in IN_SCOPE_KEYS:
            # law not held in the DB -> out of scope
            summary.out_of_scope += 1
            results.append(
                CitationResult(
                    raw=cit.raw, so_dieu=cit.so_dieu, so_khoan=cit.so_khoan,
                    diem=cit.diem, law=cit.law, status="out_of_scope",
                    message_vi=(
                        f"Điều {cit.so_dieu} thuộc {cit.law} — ngoài phạm vi CSDL "
                        "(văn bản chưa có trong cơ sở dữ liệu)."
                    ),
                )
            )
            continue

        # verify against the resolved document, with optional validity-at-date
        verdict = verify_citation(
            cit.so_dieu, cit.so_khoan, cit.diem,
            law_key=target_key, on_date=on_date,
        )
        title = verdict.get("tieu_de")
        validity = verdict.get("validity")

        # surface a validity anomaly (e.g. code cited before it took effect)
        if validity is not None and validity.get("in_force") is False:
            summary.validity_flags += 1
            flags.append(validity["message_vi"])

        if not verdict["exists"]:
            # The article itself may exist while the cited khoản/điểm does not.
            # Before flagging a genuine "không tồn tại", try the DB-validated
            # re-attachment safety net: the điểm/khoản may simply belong to an
            # adjacent article/khoản in the SAME chain (mis-grouping / NER drift).
            article_exists = get_article(cit.so_dieu, law_key=target_key).get("ton_tai", False)
            reattached = None
            if article_exists:
                reattached = _try_reattach(
                    cit, chains_by_id.get(cit.chain_id, []), target_key,
                    on_date=on_date, doc_cits=parsed,
                )

            if reattached is not None:
                # Mis-grouping artifact — silently corrected, NOT an anomaly.
                # Re-verify the corrected target to surface its real title/frame.
                fixed = verify_citation(
                    reattached["so_dieu"], reattached["so_khoan"], reattached["diem"],
                    law_key=target_key, on_date=on_date,
                )
                summary.valid += 1
                summary.reattached += 1
                results.append(
                    CitationResult(
                        raw=cit.raw,
                        so_dieu=reattached["so_dieu"],
                        so_khoan=reattached["so_khoan"],
                        diem=reattached["diem"],
                        law=cit.law, status="valid",
                        article_title=fixed.get("tieu_de"),
                        message_vi=fixed.get("message_vi"),
                        law_key=target_key, validity=validity,
                        reattached=reattached,
                    )
                )
                # NOT added to flags_vi — honest, non-alarmist: this is a parsing
                # artifact that was corrected, not a forged/wrong citation.
                continue

            summary.not_found += 1
            results.append(
                CitationResult(
                    raw=cit.raw, so_dieu=cit.so_dieu, so_khoan=cit.so_khoan,
                    diem=cit.diem, law=cit.law, status="not_found",
                    article_title=title, message_vi=verdict["message_vi"],
                    law_key=target_key, validity=validity,
                )
            )
            flags.append(verdict["message_vi"])
            continue

        # valid citation
        summary.valid += 1
        art = get_article(cit.so_dieu, law_key=target_key)
        is_offense = art.get("is_offense", False)

        charge_match: str | None = None
        if is_offense and crimes and title:
            best = "false"
            for crime in crimes:
                m = _charge_match(crime, title)
                if m == "true":
                    best = "true"
                    break
                if m == "uncertain" and best != "true":
                    best = "uncertain"
            charge_match = best
            if charge_match == "false":
                summary.charge_mismatches += 1
                flags.append(
                    f"Tội danh nêu trong bản án có thể KHÔNG khớp Điều {cit.so_dieu} "
                    f"(\"{title}\") — cần kiểm tra."
                )

        # L3 sentencing: only for offence clauses with a frame + a penalty
        sentencing: dict | None = None
        if is_offense and cit.so_khoan is not None and principal_penalty is not None:
            frame = penalty_frame(cit.so_dieu, cit.so_khoan, law_key=target_key)
            if frame.get("ton_tai"):
                sentencing = check_sentencing(principal_penalty, frame)
                sentencing["penalty_raw"] = principal_penalty["raw"]
                # describe the clause frame using its custodial (prison) bounds
                # when present, else fall back to the principal frame summary.
                prison = next(
                    (f for f in frame.get("all_frames", [])
                     if f["type"] == "tu_co_thoi_han" and f.get("unit") == "nam"),
                    None,
                )
                if prison is not None:
                    sentencing["frame_vi"] = (
                        f"khung tù Điều {cit.so_dieu} khoản {cit.so_khoan}: "
                        f"{prison.get('min')}–{prison.get('max')} năm"
                    )
                else:
                    sentencing["frame_vi"] = (
                        f"khung Điều {cit.so_dieu} khoản {cit.so_khoan}: "
                        f"{frame.get('type')} {frame.get('min')}–{frame.get('max')} "
                        f"{frame.get('unit')}"
                    )
                if sentencing["status"] == "out_of_frame":
                    summary.sentencing_flags += 1
                    flags.append(
                        f"Hình phạt tuyên vượt khung Điều {cit.so_dieu} khoản "
                        f"{cit.so_khoan}: {sentencing['reason_vi']}."
                    )

        results.append(
            CitationResult(
                raw=cit.raw, so_dieu=cit.so_dieu, so_khoan=cit.so_khoan,
                diem=cit.diem, law=cit.law, status="valid",
                article_title=title, message_vi=verdict["message_vi"],
                charge_match=charge_match, sentencing=sentencing,
                law_key=target_key, validity=validity,
            )
        )

    return CitationReport(
        citations=results,
        summary=summary,
        flags_vi=flags,
    )
