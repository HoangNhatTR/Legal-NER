"""Major-section segmentation of a Vietnamese court judgment.

A judgment (criminal Mẫu / civil Mẫu 52-DS) is written with a fixed set of
ALL-CAPS heading anchors. Splitting on those anchors lets downstream
inference treat each major section as an INDEPENDENT sequence, so an
inference window never bleeds context across unrelated sections (e.g. the
operative QUYẾT ĐỊNH rulings must not be predicted with NỘI DUNG narrative
context, and vice-versa).

Verified template anchors (criminal + civil), in document order:

    PREAMBLE            text before "NHÂN DANH" (cover: court, case no., panel)
    NHAN_DANH           "NHÂN DANH NƯỚC CỘNG HÒA ..." block
    NOI_DUNG            "NỘI DUNG VỤ ÁN"        — narrative of the case
    NHAN_DINH           "NHẬN ĐỊNH [CỦA TÒA ÁN]" — court reasoning
    QUYET_DINH          "QUYẾT ĐỊNH"            — operative rulings

Civil judgments (Mẫu 52-DS) and short decisions differ: an anchor may be
missing (e.g. no "NHẬN ĐỊNH", or only a "QUYẾT ĐỊNH" header on a bare
decision). The segmenter degrades gracefully: it keeps only the anchors that
actually occur and stitches the gaps so the returned sections ALWAYS cover the
full ``[0, len(text))`` range with no gaps and no overlaps.

The QUYẾT ĐỊNH anchor is resolved with the SAME rule the labeler uses
(``labeling.patterns._operative_zone_start``: the LAST "QUYẾT ĐỊNH" marker is
the operative header — earlier hits live in the cover page / index). This
keeps the sectioner and the weak-labeler in agreement.
"""

from __future__ import annotations

import re

from labeling.patterns import QUYET_DINH_MARKER_RE, _operative_zone_start

# Section names (stable identifiers used by the windowing layer / debugging).
SECTION_PREAMBLE = "PREAMBLE"
SECTION_NHAN_DANH = "NHAN_DANH"
SECTION_NOI_DUNG = "NOI_DUNG"
SECTION_NHAN_DINH = "NHAN_DINH"
SECTION_QUYET_DINH = "QUYET_DINH"

# Heading anchors. ALL-CAPS, matched as whole phrases. Order matters: this is
# the canonical document order used to build contiguous sections.
#   * NHÂN DANH        — opening of the "NHÂN DANH NƯỚC CỘNG HÒA ..." block.
#   * NỘI DUNG VỤ ÁN   — narrative (criminal + civil share the phrase).
#   * NHẬN ĐỊNH        — reasoning; "[CỦA TÒA ÁN]" suffix is optional.
#   * QUYẾT ĐỊNH       — operative block; resolved via _operative_zone_start
#                        (LAST marker) rather than a plain first-match.
_NHAN_DANH_RE = re.compile(r"NHÂN\s+DANH")
_NOI_DUNG_RE = re.compile(r"NỘI\s+DUNG\s+VỤ\s+ÁN")
_NHAN_DINH_RE = re.compile(r"NHẬN\s+ĐỊNH(?:\s+CỦA\s+TÒA\s+ÁN)?")


def _first(rx: re.Pattern, text: str, after: int = 0) -> int | None:
    """Start offset of the first ``rx`` match at/after ``after`` (or None)."""
    m = rx.search(text, after)
    return m.start() if m else None


def segment_sections(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into major judgment sections.

    Returns ``[(section_name, start_char, end_char), ...]`` in document order,
    contiguous and gap-free: ``sections[0][1] == 0``,
    ``sections[-1][2] == len(text)``, and ``s[i][2] == s[i+1][1]``.

    ``text`` should be the normalized + flattened stream (the same one
    inference tokenizes), so the returned offsets index into it directly.

    Robustness:
      * Missing anchors are simply skipped — only anchors present in the
        document produce boundaries.
      * Anchors are accepted only in increasing order (an anchor appearing
        before an already-consumed boundary is ignored), so a stray ALL-CAPS
        phrase quoted inside the narrative cannot reorder the structure.
      * If NO anchor is found at all (e.g. an OCR fragment, or an atypical
        template), the whole document is returned as a single PREAMBLE
        section — the windowing layer then behaves like the old whole-doc
        path, just without false section cuts.
    """
    n = len(text)
    if n == 0:
        return [(SECTION_PREAMBLE, 0, 0)]

    # Resolve each anchor start (None when absent). Each anchor is searched
    # only AFTER the previous one so the order is enforced.
    boundaries: list[tuple[str, int]] = []
    cursor = 0

    nhan_danh = _first(_NHAN_DANH_RE, text, cursor)
    if nhan_danh is not None:
        boundaries.append((SECTION_NHAN_DANH, nhan_danh))
        cursor = nhan_danh

    noi_dung = _first(_NOI_DUNG_RE, text, cursor)
    if noi_dung is not None:
        boundaries.append((SECTION_NOI_DUNG, noi_dung))
        cursor = noi_dung

    nhan_dinh = _first(_NHAN_DINH_RE, text, cursor)
    if nhan_dinh is not None:
        boundaries.append((SECTION_NHAN_DINH, nhan_dinh))
        cursor = nhan_dinh

    # QUYẾT ĐỊNH: align with the labeler — the LAST marker is the operative
    # header. _operative_zone_start returns the marker END; we need its START
    # for the section boundary, so re-derive it as the last marker whose start
    # is >= cursor (falling back to the last marker overall).
    qd_start = _last_quyet_dinh_start(text, cursor)
    if qd_start is not None:
        boundaries.append((SECTION_QUYET_DINH, qd_start))

    # No anchor at all -> single section spanning the whole document.
    if not boundaries:
        return [(SECTION_PREAMBLE, 0, n)]

    sections: list[tuple[str, int, int]] = []

    # Everything before the first anchor is the PREAMBLE (cover page / panel).
    first_start = boundaries[0][1]
    if first_start > 0:
        sections.append((SECTION_PREAMBLE, 0, first_start))

    # Each anchor section runs until the next anchor (or end of text).
    for i, (name, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] if i + 1 < len(boundaries) else n
        sections.append((name, start, end))

    return sections


def _last_quyet_dinh_start(text: str, after: int) -> int | None:
    """Start offset of the operative QUYẾT ĐỊNH marker.

    Mirrors ``_operative_zone_start`` (which returns the marker END): the LAST
    "QUYẾT ĐỊNH" marker is the operative one. We prefer the last marker at or
    after ``after`` (so it sits past NHẬN ĐỊNH); if every marker precedes
    ``after`` (atypical), fall back to the last marker overall so a bare
    decision document (only a title "QUYẾT ĐỊNH") still yields a boundary.
    """
    # Cheap consistency guard: keep using the shared helper so any future
    # change to operative-zone logic stays in one place. We only need its
    # existence signal here; the precise start is recomputed below.
    if _operative_zone_start(text) is None:
        return None
    last_after = None
    last_any = None
    for m in QUYET_DINH_MARKER_RE.finditer(text):
        last_any = m.start()
        if m.start() >= after:
            last_after = m.start()
    return last_after if last_after is not None else last_any
