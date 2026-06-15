"""PDF -> normalized text extraction (PyMuPDF).

Happy path only: text-layer PDFs. Scanned PDFs (no text layer) are skipped.

CLI:
    python -m corpus.extract --in data/raw --out data/text
    python -m corpus.extract --pdf "/path/to/one.pdf" --out data/text
"""

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, TEXT_DIR  # noqa: E402
from corpus.normalize import normalize_text  # noqa: E402

# A real text layer on a judgment yields far more than this per document.
MIN_TEXT_CHARS = 200


def extract_pdf_text(pdf_path: Path) -> str | None:
    """Return raw extracted text, or None when there is no usable text layer."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # corrupt download
        print(f"  [skip] cannot open {pdf_path.name}: {exc}")
        return None
    text = "".join(page.get_text() for page in doc)
    doc.close()
    if len(text.strip()) < MIN_TEXT_CHARS:
        return None  # scanned / empty -> not happy path
    return text


def process_one(pdf_path: Path, out_dir: Path, force: bool = False) -> bool:
    out_path = out_dir / (pdf_path.stem + ".txt")
    if out_path.exists() and not force:
        print(f"  [done] {out_path.name} already exists")
        return True
    raw = extract_pdf_text(pdf_path)
    if raw is None:
        print(f"  [scan] {pdf_path.name}: no text layer, skipped")
        return False
    out_path.write_text(normalize_text(raw), encoding="utf-8")
    print(f"  [ok]   {pdf_path.name} -> {out_path.name} ({len(raw)} chars)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract+normalize text from judgment PDFs")
    parser.add_argument("--in", dest="in_dir", default=str(RAW_DIR),
                        help="directory containing PDFs (default: data/raw)")
    parser.add_argument("--pdf", default=None, help="process a single PDF file instead of a directory")
    parser.add_argument("--out", default=str(TEXT_DIR), help="output directory (default: data/text)")
    parser.add_argument("--force", action="store_true", help="re-extract even if output exists")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = [Path(args.pdf)] if args.pdf else sorted(Path(args.in_dir).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.in_dir}")
        return

    ok = sum(process_one(p, out_dir, args.force) for p in pdfs)
    print(f"\nExtracted {ok}/{len(pdfs)} PDFs with text layers -> {out_dir}")


if __name__ == "__main__":
    main()
