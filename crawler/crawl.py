"""Resumable crawler for congbobanan.toaan.gov.vn (sequential id probing).

Idempotent: visited ids live in a JSON state file; re-running continues
where the previous run stopped and never re-downloads existing PDFs.

CLI:
    python -m crawler.crawl --target-count 20
    python -m crawler.crawl --target-count 1000 --start-id 100000 --criminal-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CRAWL_DELAY_SECONDS,
    CRAWL_STATE_FILE,
    DEFAULT_START_ID,
    RAW_DIR,
)
from crawler.portal_client import download_pdf, fetch_detail, make_session  # noqa: E402


def load_state(state_path: Path, start_id: int) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"next_id": start_id, "downloaded": [], "misses": 0, "probed": 0}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(state_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl judgments from the public portal")
    parser.add_argument("--target-count", type=int, default=20,
                        help="stop after this many NEW judgments downloaded (default: 20)")
    parser.add_argument("--start-id", type=int, default=DEFAULT_START_ID,
                        help=f"first judgment id to probe (default: {DEFAULT_START_ID}); "
                             "ignored when a state file already exists")
    parser.add_argument("--out", default=str(RAW_DIR), help="output dir (default: data/raw)")
    parser.add_argument("--delay", type=float, default=CRAWL_DELAY_SECONDS,
                        help="seconds between requests (default: 1.0)")
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument("--criminal-only", action="store_true",
                            help="keep only criminal cases (title contains 'hình sự'/HS markers)")
    type_group.add_argument("--civil-only", action="store_true",
                            help="keep only civil/family/administrative cases "
                                 "(DS/HNGĐ/KDTM/LĐ/HC markers; rejects HS)")
    parser.add_argument("--max-probes", type=int, default=None,
                        help="hard cap on ids probed this run (default: 20x target)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(CRAWL_STATE_FILE) if args.out == str(RAW_DIR) \
        else out_dir / "crawl_state.json"
    state = load_state(state_path, args.start_id)
    meta_path = out_dir / "metadata.jsonl"
    max_probes = args.max_probes or args.target_count * 20

    session = make_session()
    new_downloads = 0
    probes = 0
    print(f"Resuming at id {state['next_id']} "
          f"({len(state['downloaded'])} already downloaded)")

    while new_downloads < args.target_count and probes < max_probes:
        judgment_id = state["next_id"]
        state["next_id"] = judgment_id + 1
        probes += 1
        state["probed"] += 1

        pdf_path = out_dir / f"{judgment_id}.pdf"
        if pdf_path.exists():
            continue  # idempotency: already on disk from a previous run

        try:
            info = fetch_detail(session, judgment_id)
        except requests.RequestException as exc:
            print(f"  [err ] id {judgment_id}: {type(exc).__name__}: {exc}")
            save_state(state_path, state)
            time.sleep(args.delay)
            continue

        if info is None:
            state["misses"] += 1
        elif args.criminal_only and not info.is_criminal:
            print(f"  [skip] id {judgment_id}: not criminal ({info.title[:60]})")
        elif args.civil_only and not info.is_civil:
            print(f"  [skip] id {judgment_id}: not civil ({info.title[:60]})")
        else:
            pdf_bytes = download_pdf(session, info)
            if pdf_bytes is None:
                print(f"  [skip] id {judgment_id}: no PDF available")
            else:
                pdf_path.write_bytes(pdf_bytes)
                with meta_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "id": judgment_id,
                        "title": info.title,
                        "criminal": info.is_criminal,
                        "civil": info.is_civil,
                        "pdf_link": info.pdf_path,
                        "bytes": len(pdf_bytes),
                    }, ensure_ascii=False) + "\n")
                state["downloaded"].append(judgment_id)
                new_downloads += 1
                print(f"  [ok  ] id {judgment_id}: {len(pdf_bytes)} bytes "
                      f"| {info.title[:70]}")

        save_state(state_path, state)
        time.sleep(args.delay)

    print(f"\nRun finished: {new_downloads} new judgments "
          f"({probes} ids probed, {state['misses']} total misses). "
          f"State: {state_path}")


if __name__ == "__main__":
    main()
