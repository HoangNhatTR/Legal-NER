"""Text normalization for Vietnamese judgment PDFs with legacy-font artifacts.

Observed artifacts in real portal PDFs (verified on demo judgment
17/2018/HS-ST):
  * U+01A3 'ƣ' / U+01A2 'Ƣ' (LATIN LETTER OI) used instead of 'ư' / 'Ư'
    (TCVN3/ABC legacy font leftovers, e.g. "Dƣơng" -> "Dương").
  * Combining diacritics detached from their base letter by a stray space,
    e.g. "Thi ̣ Kim" = "Thi" + SPACE + U+0323 -> "Thị Kim".
  * Decomposed (NFD-like) sequences -> fixed by NFC.
"""

import re
import unicodedata

# Legacy-font codepoint fixes (extend as new artifacts are observed).
LEGACY_CHAR_MAP = {
    "ƣ": "ư",  # ƣ -> ư
    "Ƣ": "Ư",  # Ƣ -> Ư
}

# Vietnamese combining marks seen detached: grave, acute, hook above,
# tilde, dot below (U+0300, U+0301, U+0309, U+0303, U+0323).
_COMBINING = "̣̀́̃̉"
# One or more spaces (incl. NBSP) sitting between a letter and a combining
# mark — glue the mark back onto the preceding letter.
_DETACHED_MARK_RE = re.compile(rf"(\w)[  ]+([{_COMBINING}])")

_MULTI_SPACE_RE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def fix_legacy_chars(text: str) -> str:
    for src, dst in LEGACY_CHAR_MAP.items():
        text = text.replace(src, dst)
    return text


def reattach_combining_marks(text: str) -> str:
    # Repeat until stable: a syllable can carry the mark several letters back
    # only in pathological cases; one pass suffices for observed data but the
    # loop is cheap.
    prev = None
    while prev != text:
        prev = text
        text = _DETACHED_MARK_RE.sub(r"\1\2", text)
    return text


def normalize_text(text: str) -> str:
    """Full normalization: legacy chars -> reattach marks -> NFC -> spaces."""
    text = fix_legacy_chars(text)
    text = reattach_combining_marks(text)
    text = unicodedata.normalize("NFC", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    # strip trailing spaces per line, collapse blank-line runs
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def flatten_for_matching(text: str) -> str:
    """Collapse hard line-wraps into a single space-separated stream.

    PDF extraction wraps sentences mid-span; entity regexes need an
    unwrapped stream. Page-number-only lines are dropped.
    """
    lines = [ln for ln in text.split("\n") if ln.strip() and not ln.strip().isdigit()]
    flat = " ".join(lines)
    return _MULTI_SPACE_RE.sub(" ", flat).strip()
