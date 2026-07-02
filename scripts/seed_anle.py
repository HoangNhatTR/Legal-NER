"""Nạp ÁN LỆ chính thức (anle.toaan.gov.vn) vào kho tiền lệ (source='an_le').

Portal là app Oracle ADF (render JS) + nội dung án lệ là PDF SCAN trên content
server → không lấy text bằng requests thuần được. Cách tiếp cận:

  1. crawl (Playwright): mở trang danh sách án lệ, lật hết các trang, thu
     (số án lệ, tiêu đề, dDocName, URL chi tiết, URL PDF) → ghi JSONL.
     Tiêu đề án lệ chính là NGUYÊN TẮC PHÁP LÝ cô đọng → đủ để so khớp theo chủ đề
     mà KHÔNG cần OCR. (--pdf-dir: tải kèm PDF scan để OCR làm giàu sau qua Module 2.)
  2. seed (embed): đọc JSONL -> dựng bản ghi -> embed (Module 1 /v1/embed) -> ghi kho.
  3. parse_anle(text): tách 1 án lệ từ TEXT (vd sau khi OCR) thành bản ghi đầy đủ
     (số AL, tội danh/từ khoá, điều luật, diễn biến). TEST OFFLINE được.

Cách dùng (chạy TỪ TRONG legal_ner/):
  python -m scripts.seed_anle --crawl --out data/anle.jsonl            # cần mạng
  python -m scripts.seed_anle --seed-jsonl data/anle.jsonl --dry-run   # xem trước
  python -m scripts.seed_anle --seed-jsonl data/anle.jsonl             # cần Module 1
  python -m scripts.seed_anle --from-dir data/anle_ocr_txt             # seed text/OCR
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Cho phép 'import api.*' khi chạy -m scripts.seed_anle từ legal_ner/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api.precedent_store as _ps  # noqa: E402
from api.precedent_store import (  # noqa: E402
    embed_case,
    facts_summary,
    get_store,
    normalize_articles,
)


def use_offline_embedder() -> None:
    """Nạp BGE-M3 TRỰC TIẾP (không qua Module 1 /v1/embed) để seed offline.

    Ghi đè api.precedent_store._embed bằng embedder cục bộ (model đã cache HF).
    Cần venv có sentence-transformers/torch (venv Chatbot). Model BAAI/bge-m3.
    """
    import numpy as np

    proj_root = Path(__file__).resolve().parents[2]  # ProjectGenAI_2
    sys.path.insert(0, str(proj_root))
    from src.embedding import Embedder

    emb = Embedder("BAAI/bge-m3")
    print("  (offline-embed) đang nạp BGE-M3 từ cache HF…")

    def _embed(texts, timeout: int = 120):
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(emb.encode(list(texts)), dtype=np.float32)

    _ps._embed = _embed  # embed_case dùng biến module-level này

PORTAL = "https://anle.toaan.gov.vn"
LIST_URL = PORTAL + "/webcenter/portal/anle/anle"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_AL_NUM = re.compile(r"(\d+/\d{4}/AL)", re.IGNORECASE)
# Mục nêu CÁC ĐIỀU LUẬT CHÍNH của án lệ (chính xác hơn là quét mọi 'Điều N').
_SEC_LAW = re.compile(
    r"Quy\s*định\s*c[ủu]a\s*ph[áa]p\s*lu[ậa]t\s*li[êe]n\s*quan[^:]*:?", re.IGNORECASE
)
_KEYWORDS = re.compile(r"T[ừu]\s*kh[óo]a\s*c[ủu]a\s*[áa]n\s*l[ệe]\s*:?(.+)", re.IGNORECASE)
_QUOTED = re.compile(r"[\"“”']([^\"“”']{2,80})[\"“”']")
_CASE_TYPE_MARKERS = [
    ("hình sự", "hình sự"), ("dân sự", "dân sự"), ("hành chính", "hành chính"),
    ("kinh doanh", "kinh doanh thương mại"), ("thương mại", "kinh doanh thương mại"),
    ("lao động", "lao động"), ("hôn nhân", "hôn nhân & gia đình"),
    ("nuôi con nuôi", "hôn nhân & gia đình"),
]


def _pick_title(cands: list[str]) -> str:
    """Chọn tiêu đề NGUYÊN TẮC án lệ tốt nhất trong các text ứng viên."""
    if not cands:
        return ""
    def score(t: str) -> float:
        low = t.lower()
        s = 0.0
        if re.search(r"al\s+về|^về\s", low):
            s += 3
        if low.startswith(("quyết định", "bản án", "số:", "quyết đinh")):
            s -= 3
        return s + min(len(t), 120) / 120
    best = max(cands, key=score)
    # bỏ tiền tố 'Án lệ số N/AL về' -> giữ phần nguyên tắc cho gọn (vẫn ok nếu không khớp)
    return re.sub(r"^Án lệ số\s*\d+/\d{4}/AL\s*", "", best).strip() or best


def pdf_url(d_doc_name: str) -> str:
    return f"{PORTAL}/webcenter/ShowProperty?nodeId=/UCMServer/{d_doc_name}//idcPrimaryFile&revision=latestreleased"


def _case_type(text: str) -> str:
    low = text.lower()
    for marker, label in _CASE_TYPE_MARKERS:
        if marker in low:
            return label
    return ""


# ── parse 1 án lệ từ TEXT (dùng cho bản OCR / file văn bản) ──────────────────

def parse_anle(text: str, *, url: str = "") -> dict | None:
    """Tách một án lệ thành bản ghi tiền lệ. None nếu không nhận ra là án lệ."""
    text = (text or "").strip()
    if not text:
        return None
    mnum = _AL_NUM.search(text)
    if not mnum or "AL" not in mnum.group(1).upper():
        return None
    case_no = mnum.group(1)

    crimes: list[str] = []
    km = _KEYWORDS.search(text)
    if km:
        line = km.group(1).splitlines()[0]
        crimes = [q.strip() for q in _QUOTED.findall(line)]
        if not crimes:
            crimes = [p.strip(" .\t") for p in line.split(";") if 2 < len(p.strip()) < 80][:6]

    return {
        "case_number": case_no,
        "court": "Tòa án nhân dân tối cao",
        "case_type": _case_type(text),
        "crimes": crimes,
        "articles": normalize_articles(text),
        "penalties": [],
        "facts_summary": facts_summary(text),
        "full_text": text,
        "_title": f"Án lệ số {case_no}",
        "_url": url,
    }


# ── dựng bản ghi từ metadata crawl (số + tiêu đề) ────────────────────────────

def _related_articles(full_text: str, title: str) -> list[str]:
    """Điều luật CHÍNH của án lệ: ưu tiên mục 'Quy định của pháp luật liên quan…'
    (vài dòng sau tiêu mục) để bộ lọc luật chính xác, tránh nhiễu từ điều phụ rải
    rác toàn văn. Fallback: điều luật nêu trong tiêu đề."""
    m = _SEC_LAW.search(full_text or "")
    if m:
        seg = full_text[m.end(): m.end() + 600]
        # dừng trước mục kế tiếp để không lẫn điều luật rải rác phần nội dung
        seg = re.split(
            r"T[ừu]\s*kh[óo]a|N[Ộộ]I\s*DUNG|NH[Ậậ]N\s*[Đđ]|\n\s*\n", seg
        )[0]
        arts = normalize_articles(seg)
        if arts:
            return arts[:8]
    return normalize_articles(title)


def record_from_meta(case_no: str, title: str, *, full_text: str = "") -> dict:
    """Bản ghi tiền lệ từ (số án lệ, tiêu đề). full_text (nếu có OCR) làm giàu."""
    blob = full_text or title
    crimes: list[str] = []
    km = _KEYWORDS.search(full_text or "")
    if km:
        crimes = [q.strip() for q in _QUOTED.findall(km.group(1).splitlines()[0])]
    # Khoá so khớp = NGUYÊN TẮC (tiêu đề) + diễn biến (nếu có full_text). Tiêu đề
    # là tín hiệu mạnh nhất nên đặt trước; diễn biến bổ sung chiều sâu ngữ nghĩa.
    facts = facts_summary(full_text) if full_text else ""
    combined = ". ".join(p for p in (title, facts) if p).strip(". ") or title
    return {
        "case_number": case_no,
        "court": "Tòa án nhân dân tối cao",
        "case_type": _case_type(blob),
        "crimes": crimes,
        "articles": _related_articles(full_text, title),
        "penalties": [],
        "facts_summary": combined,
        "full_text": full_text or title,
    }


# ── CRAWL danh sách án lệ (Playwright) ───────────────────────────────────────

def crawl(out_jsonl: Path, *, limit: int = 200, delay: float = 0.6,
          pdf_dir: Path | None = None, headless: bool = True) -> int:
    """Lật hết các trang danh sách án lệ, ghi metadata ra JSONL. Trả số bản ghi."""
    from playwright.sync_api import sync_playwright

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if pdf_dir:
        pdf_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}  # dDocName -> rec (dedup)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=headless, args=["--disable-blink-features=AutomationControlled"])
        pg = b.new_context(user_agent=UA, locale="vi-VN").new_page()
        pg.goto(LIST_URL, wait_until="networkidle", timeout=60000)
        pg.wait_for_timeout(2500)

        _SKIP_TXT = {"Thuộc tính", "Văn bản liên quan", "Tải về"}
        for page_i in range(1, 30):
            before = len(records)
            # Mỗi án lệ có 2 link cùng dDocName: một mang SỐ ('90/2026/AL'), một
            # mang TIÊU ĐỀ ('Về ...'). Gom cả hai vào cùng bản ghi.
            for txt, href in pg.eval_on_selector_all(
                "a", "els=>els.map(e=>[e.innerText.trim(), e.getAttribute('href')||''])"
            ):
                m = re.search(r"chitietanle\?dDocName=(TAND\d+)", href)
                if not m or "duthao" in href or "Tab=" in href:
                    continue
                dd = m.group(1)
                rec = records.setdefault(dd, {
                    "dDocName": dd, "case_number": "", "_cands": [],
                    "detail_url": f"{PORTAL}/webcenter/portal/anle/chitietanle?dDocName={dd}",
                    "pdf_url": pdf_url(dd),
                })
                txt = (txt or "").strip()
                numm = _AL_NUM.search(txt)
                if numm and txt.upper().endswith("/AL"):
                    rec["case_number"] = numm.group(1)
                elif len(txt) > 10 and txt not in _SKIP_TXT and txt not in rec["_cands"]:
                    rec["_cands"].append(txt[:300])
            print(f"  trang {page_i}: tổng {len(records)} án lệ")
            if len(records) >= limit:
                break
            nxt = pg.query_selector("xpath=//a[normalize-space(text())='Sau']")
            if not nxt:
                break
            nxt.click()
            pg.wait_for_timeout(1700)
            if len(records) == before:  # trang không thêm bản ghi -> hết
                pg.wait_for_timeout(1200)
                if len(records) == before:
                    break

        # Chọn tiêu đề tốt nhất: ưu tiên anchor nêu NGUYÊN TẮC ('... AL về ...' /
        # bắt đầu 'Về'), tránh số bản án nguồn ('Quyết định …'/'Bản án …').
        for rec in records.values():
            rec["title"] = _pick_title(rec.pop("_cands", []))
        recs = [r for r in records.values() if r["case_number"]][:limit]
        # tuỳ chọn tải PDF scan để OCR làm giàu sau
        if pdf_dir:
            import requests
            B = str(Path(__file__).parent / "toaan_ca_bundle.pem")
            for r in recs:
                dest = pdf_dir / f"{r['dDocName']}.pdf"
                if dest.exists():
                    continue
                try:
                    resp = requests.get(r["pdf_url"], headers={"User-Agent": UA}, timeout=60, verify=B)
                    if resp.headers.get("content-type", "").startswith("application/pdf"):
                        dest.write_bytes(resp.content)
                        print(f"     PDF {r['case_number']} ({len(resp.content)//1024} KB)")
                except Exception as exc:  # noqa: BLE001
                    print(f"     ! PDF lỗi {r['case_number']}: {exc}")
                time.sleep(delay)
        b.close()

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  => ghi {len(recs)} án lệ vào {out_jsonl}")
    return len(recs)


# ── ENRICH: tải file gốc + trích text (PDF-text / DOCX / OCR scan) ───────────

def _docx_text(data: bytes) -> str:
    """Trích text từ .docx (dependency-free: unzip word/document.xml, bỏ tag)."""
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = xml.replace("</w:p>", "\n").replace("</w:tr>", "\n")
    return re.sub(r"<[^>]+>", "", xml)


def _extract_text(data: bytes, content_type: str, *, do_ocr: bool) -> tuple[str, str]:
    """Trả (text, kind). kind: 'pdf-text' | 'docx' | 'ocr' | 'scan-skip' | 'unknown'."""
    if data[:4] == b"%PDF" or "pdf" in content_type:
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        txt = "".join(p.get_text() for p in doc)
        if len(txt.strip()) > 400:
            return txt, "pdf-text"
        if not do_ocr:
            return "", "scan-skip"
        from api.ocr_cloud import ocrspace_available, run_ocrspace_on_pdf

        if ocrspace_available():
            return run_ocrspace_on_pdf(data, max_pages=20).text, "ocr"
        try:  # fallback EasyOCR offline (chậm)
            from api.ocr_easyocr import run_easyocr_on_pdf

            return run_easyocr_on_pdf(data, max_pages=20).text, "ocr"
        except Exception:  # noqa: BLE001
            return "", "scan-skip"
    if data[:2] == b"PK":  # OOXML .docx (ZIP). .doc cũ (OLE2 \xd0\xcf) -> unknown.
        try:
            return _docx_text(data), "docx"
        except Exception:  # noqa: BLE001 — zip lạ / thiếu word/document.xml
            return "", "docx-fail"
    return "", "unknown"


def enrich_jsonl(jsonl: Path, *, pdf_dir: Path, do_ocr: bool = False) -> None:
    """Tải file gốc từng án lệ, trích text, ghi 'full_text' vào JSONL (in-place)."""
    import requests

    pdf_dir.mkdir(parents=True, exist_ok=True)
    B = str(Path(__file__).parent / "toaan_ca_bundle.pem")
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    stats: dict[str, int] = {}
    for r in rows:
        if r.get("full_text"):  # đã enrich -> bỏ qua (resume)
            stats["cached"] = stats.get("cached", 0) + 1
            continue
        try:
            resp = requests.get(r["pdf_url"], headers={"User-Agent": UA}, timeout=60, verify=B)
            text, kind = _extract_text(resp.content, resp.headers.get("content-type", ""), do_ocr=do_ocr)
        except Exception as exc:  # noqa: BLE001
            text, kind = "", f"err:{repr(exc)[:30]}"
        stats[kind] = stats.get(kind, 0) + 1
        if text.strip():
            r["full_text"] = re.sub(r"[ \t]+", " ", text).strip()[:20000]
        print(f"  {r['case_number']:>12} | {kind:10} | {len(r.get('full_text',''))} ký tự")
    jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print("  thống kê:", stats)


# ── SEED từ JSONL (embed + ghi kho) ──────────────────────────────────────────

def seed_jsonl(jsonl: Path, *, dry_run: bool = False) -> int:
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    store = None if dry_run else get_store()
    added = 0
    for r in rows:
        case = record_from_meta(r["case_number"], r.get("title", ""), full_text=r.get("full_text", ""))
        title = f"Án lệ số {r['case_number']}" + (f" — {r['title']}" if r.get("title") else "")
        print(f"  [{'DRY' if dry_run else 'SEED'}] {case['case_number']} | "
              f"{case['case_type'] or '—'} | điều: {case['articles'] or '—'} | {r.get('title','')[:50]}")
        if dry_run:
            added += 1
            continue
        vec = embed_case(case)
        if vec.size == 0:
            print(f"     ! embed lỗi (Module 1 /v1/embed?) — bỏ {case['case_number']}")
            continue
        store.add(case, vec, source="an_le", title=title, url=r.get("detail_url", ""))
        added += 1
    return added


# ── SEED từ thư mục file văn bản/OCR (.txt) ──────────────────────────────────

def seed_from_dir(dir_path: Path, *, dry_run: bool = False) -> int:
    files = sorted(dir_path.glob("*.txt"))
    if not files:
        print(f"  (không có file .txt trong {dir_path})")
        return 0
    store = None if dry_run else get_store()
    added = 0
    for f in files:
        case = parse_anle(f.read_text(encoding="utf-8", errors="ignore"))
        if not case:
            print(f"  [bỏ qua] {f.name}: không nhận ra án lệ (thiếu 'N/YYYY/AL')")
            continue
        title = case.pop("_title"); url = case.pop("_url")
        print(f"  [{'DRY' if dry_run else 'SEED'}] {case['case_number']} | "
              f"tội: {case['crimes'] or '—'} | điều: {case['articles'] or '—'}")
        if dry_run:
            added += 1
            continue
        vec = embed_case(case)
        if vec.size == 0:
            print(f"     ! embed lỗi — bỏ {case['case_number']}")
            continue
        store.add(case, vec, source="an_le", title=title, url=url)
        added += 1
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description="Nạp án lệ chính thức vào kho tiền lệ")
    ap.add_argument("--crawl", action="store_true", help="crawl danh sách án lệ -> JSONL (Playwright)")
    ap.add_argument("--enrich", type=Path, metavar="JSONL",
                    help="tải file gốc + trích text (PDF-text/DOCX/OCR) -> ghi full_text vào JSONL")
    ap.add_argument("--ocr", action="store_true", help="bật OCR cho bản scan khi --enrich (cần OCR.space key / EasyOCR)")
    ap.add_argument("--out", type=Path, default=Path("data/anle.jsonl"), help="file JSONL đầu ra của --crawl")
    ap.add_argument("--pdf-dir", type=Path, default=Path("data/anle_pdf"), help="thư mục cache file gốc án lệ")
    ap.add_argument("--seed-jsonl", type=Path, help="seed từ JSONL đã crawl (cần Module 1 /v1/embed)")
    ap.add_argument("--from-dir", type=Path, help="seed từ thư mục .txt (vd bản OCR)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--delay", type=float, default=0.6)
    ap.add_argument("--no-headless", action="store_true", help="hiện trình duyệt khi crawl")
    ap.add_argument("--offline-embed", action="store_true",
                    help="nạp BGE-M3 trực tiếp để seed (không cần Module 1; cần venv có torch)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ xem, không embed/ghi")
    args = ap.parse_args()

    if args.crawl:
        crawl(args.out, limit=args.limit, delay=args.delay, pdf_dir=None,
              headless=not args.no_headless)
    if args.enrich:
        enrich_jsonl(args.enrich, pdf_dir=args.pdf_dir, do_ocr=args.ocr)
    if args.offline_embed and not args.dry_run:
        use_offline_embedder()
    if args.seed_jsonl:
        n = seed_jsonl(args.seed_jsonl, dry_run=args.dry_run)
        _report(n, args.dry_run)
    elif args.from_dir:
        n = seed_from_dir(args.from_dir, dry_run=args.dry_run)
        _report(n, args.dry_run)
    elif not (args.crawl or args.enrich):
        ap.print_help()


def _report(n: int, dry: bool) -> None:
    if dry:
        print(f"\n(DRY) nhận ra {n} án lệ hợp lệ.")
    else:
        print(f"\nĐã nạp {n} án lệ. Kho hiện có {get_store().count('an_le')} án lệ.")


if __name__ == "__main__":
    main()
