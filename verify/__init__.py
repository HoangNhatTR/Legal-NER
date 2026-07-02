"""Layer 4 verification: forgery / tampering detection for judgment PDFs.

Standalone, dependency-light (pikepdf + PyMuPDF). No ML, no network.

Public API:
    from verify.forgery import analyze_pdf, ForgeryReport
    report = analyze_pdf("/path/to/judgment.pdf")   # path str/Path or raw bytes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid eager import so `python -m verify.forgery` is clean
    from .forgery import ForgeryReport, Finding, PdfMetadata, analyze_pdf
    from .existence import ExistenceReport, MatchEntry, check_existence
    from .citation import (
        CitationReport,
        CitationResult,
        ParsedCitation,
        check_citations,
        parse_citation_text,
    )

__all__ = [
    # Layer 4 — forgery / tampering signals
    "ForgeryReport", "Finding", "PdfMetadata", "analyze_pdf",
    # Layer 2 — judgment existence lookup
    "ExistenceReport", "MatchEntry", "check_existence",
    # Layer 1 + Layer 3 — cited-law verification & sentencing logic
    "CitationReport", "CitationResult", "ParsedCitation",
    "check_citations", "parse_citation_text",
]

_FORGERY = {"ForgeryReport", "Finding", "PdfMetadata", "analyze_pdf"}
_EXISTENCE = {"ExistenceReport", "MatchEntry", "check_existence"}
_CITATION = {
    "CitationReport", "CitationResult", "ParsedCitation",
    "check_citations", "parse_citation_text",
}


def __getattr__(name: str):  # PEP 562 lazy re-export
    if name in _FORGERY:
        from . import forgery

        return getattr(forgery, name)
    if name in _EXISTENCE:
        from . import existence

        return getattr(existence, name)
    if name in _CITATION:
        from . import citation

        return getattr(citation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
