"""Central configuration for the legal_ner pipeline.

Paths, crawl settings and the NER label inventory live here so every CLI
shares the same defaults.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # downloaded PDFs / HTML + crawl metadata
TEXT_DIR = DATA_DIR / "text"        # normalized plain-text judgments
LABELED_DIR = DATA_DIR / "labeled"  # weak-labeled JSONL (BIO)
MODELS_DIR = DATA_DIR / "models"    # fine-tuned checkpoints

# ---------------------------------------------------------------------------
# Crawler settings (congbobanan.toaan.gov.vn)
# ---------------------------------------------------------------------------
PORTAL_BASE = "https://congbobanan.toaan.gov.vn"
DETAIL_URL = PORTAL_BASE + "/2ta{id}t1cvn/chi-tiet-ban-an"
FULLTEXT_URL = PORTAL_BASE + "/3ta{id}t1cvn"  # serves the judgment PDF bytes
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
CRAWL_DELAY_SECONDS = 1.0      # polite delay between requests
REQUEST_TIMEOUT = 30           # seconds
DEFAULT_START_ID = 100000      # verified criminal-case region of the id space
CRAWL_STATE_FILE = RAW_DIR / "crawl_state.json"

# Substrings (lowercased) identifying criminal cases in the detail-page title.
CRIMINAL_MARKERS = ("hình sự", "hs-st", "hs-pt", "hsst", "hspt")

# Substrings (lowercased) identifying civil/family/administrative cases in the
# detail-page title. Two complementary signal families:
#   1. case-number type suffixes  -> "08/2018/QĐDS-ST", "31/2018/HCPT", ...
#        DS   = dan su (civil)            HNGD = hon nhan gia dinh (family)
#        KDTM = kinh doanh thuong mai     LD   = lao dong (labour)
#        HC   = hanh chinh (administrative)
#   2. topic phrases for the many older titles that carry no type suffix
#        (e.g. "số 124 ngày ... Vụ án ly hôn ...").
# A judgment counts as civil only when it matches one of these AND is NOT
# criminal -> this rejects criminal titles that lack an explicit HS suffix.
CIVIL_MARKERS = (
    # --- case-number type suffixes (DS / HNGĐ / KDTM / LĐ / HC) ---
    "ds-st", "ds-pt", "dsst", "dspt", "qđds", "qдds",
    "hngđ", "hngd", "hn-st", "hn-pt", "hnst", "hnpt",
    "kdtm", "kdtm-st", "kdtm-pt",
    "lđ-st", "lđ-pt", "lđst", "lđpt", "ld-st", "ld-pt",
    "hc-st", "hc-pt", "hcst", "hcpt", "qđhc",
    # --- explicit case-type words ---
    "dân sự", "hôn nhân", "gia đình", "hành chính",
    "kinh doanh", "thương mại", "lao động",
    # --- common civil/family/admin topic phrases (suffix-less titles) ---
    "ly hôn", "tranh chấp", "khiếu kiện", "yêu cầu",
    "chia tài sản", "thừa kế", "đòi nợ", "hợp đồng",
)

# ---------------------------------------------------------------------------
# NER label inventory (BIO scheme)
# ---------------------------------------------------------------------------
# v3r schema: 31 entity types -> 63 BIO tags (O + B-/I- per type).
# (v3 was 32 entity types -> 65 BIO tags; v2 was 20 -> 41.) v3r drops the
# legacy VIOLATION_ACT label, consolidating act narratives into CRIMINAL_ACT
# (see note below). Because VIOLATION_ACT sat in the MIDDLE of the list, the
# label ids of every entity AFTER it (PENALTY, MONEY_AMOUNT, ... and all 12
# dynamic labels) shift by one vs v3 — v3r is trained from scratch, so this is
# fine. The BIO tag list / LABEL2ID / ID2LABEL all derive from ENTITY_TYPES
# below and regenerate automatically.
#
# VIOLATION_ACT vs CRIMINAL_ACT (v3r CONSOLIDATION):
# v3 kept BOTH labels, but they fired on the IDENTICAL anchor
# ("(có|đã|thực hiện) hành vi ...") and so split the act-narrative signal:
# VIOLATION_ACT won most spans (1130) while CRIMINAL_ACT was starved (109,
# F1=0.0). Per NER_UPGRADE_PLAN, CRIMINAL_ACT is the *upgrade* of the sparse
# VIOLATION_ACT. v3r therefore CONSOLIDATES: VIOLATION_ACT is DROPPED and ALL
# act narrative routes to the single, richer CRIMINAL_ACT label (one act
# label, full signal). Anti-regression no longer applies to VIOLATION_ACT
# (it no longer exists) — this is expected and accepted by the recovery plan.
ENTITY_TYPES = [
    # A. document metadata
    "CASE_NUMBER",    # judgment/decision number: "17/2018/HS-ST"
    "COURT",          # court name
    "JUDGMENT_DATE",  # header / hearing / pronouncement date
    "CASE_TYPE",      # explicit case-type mention: "hình sự", "dân sự", ...
    # B. parties (anonymized names after role anchors)
    "DEFENDANT",      # name after "bị cáo / các bị cáo"
    "PLAINTIFF",      # name after "nguyên đơn"
    "VICTIM",         # name after "bị hại / người bị hại"
    "RELATED_PARTY",  # name after "người có quyền lợi, nghĩa vụ liên quan"
    # C. legal references
    "LAW_NAME",       # "Bộ luật Hình sự năm 2015 (sửa đổi, bổ sung năm 2017)"
    "ARTICLE",        # "Điều 250"
    "CLAUSE",         # "khoản 1"
    "POINT",          # "điểm c"
    "LEGAL_BASIS",    # full "Căn cứ ..."/"Áp dụng ..." citation sentences
    # D. offense
    "CRIME",          # offense name (quoted or unquoted)
    # (v3r) VIOLATION_ACT dropped — consolidated into CRIMINAL_ACT (see note above)
    # E. outcome
    "PENALTY",        # "02 (hai) năm tù", "tù chung thân", "án treo", ...
    "MONEY_AMOUNT",   # money NOT in compensation/fee context
    "COMPENSATION",   # money in "bồi thường" context
    "COURT_FEE",      # money in "án phí / lệ phí" context
    "DECISION",       # verdict-sentence remainder in QUYET DINH section
    # -----------------------------------------------------------------------
    # v3 NEW (12) -- "dynamic" labels (NER_UPGRADE_PLAN Pillar A)
    # -----------------------------------------------------------------------
    # F. people conducting the proceedings (strong anchors -> regex)
    "JUDGE",              # name after "Thẩm phán" / "Chủ tọa phiên tòa"
    "ASSESSOR",           # name after "Hội thẩm nhân dân"
    "PROSECUTOR",         # name after "Kiểm sát viên"
    "CLERK",              # name after "Thư ký phiên tòa" / "Thư ký"
    "LAWYER",             # name after "Luật sư" / "người bào chữa"
    "WITNESS",            # name after "người làm chứng"
    # G. courtroom behavior / procedural status (bounded short spans)
    "COURT_BEHAVIOR",     # "có mặt", "vắng mặt", "kháng cáo", "xin giảm nhẹ", ...
    # H. sentencing factors (gazetteer Điều 51/52 BLHS 2015)
    "MITIGATING_FACTOR",  # tình tiết giảm nhẹ (Điều 51)
    "AGGRAVATING_FACTOR", # tình tiết tăng nặng (Điều 52)
    # I. criminal act narrative (richer upgrade of VIOLATION_ACT; stays weak)
    "CRIMINAL_ACT",       # act narrative after "có hành vi"/"thực hiện hành vi"/"đã ..."
    # J. quantities & physical evidence
    "QUANTITY",           # "0,1488 gam", "01 viên", "X kg", "X gói"
    "EVIDENCE_ITEM",      # tang vật / vật chứng: "01 xe mô tô ...", "01 điện thoại ..."
]

LABELS = ["O"]
for _ent in ENTITY_TYPES:
    LABELS.append(f"B-{_ent}")
    LABELS.append(f"I-{_ent}")

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
MODEL_NAME = "xlm-roberta-base"
MAX_SEQ_LENGTH = 256
# Weak-label chunking: target chunk size in syllable tokens.
CHUNK_TOKENS = 150

# ---------------------------------------------------------------------------
# Inference windowing
# ---------------------------------------------------------------------------
# A forward pass sees at most INFER_WINDOW syllable tokens. INFER_STRIDE is the
# advance between consecutive windows; the (WINDOW - STRIDE) overlap keeps an
# entity that straddles a window edge intact for "first-decided-wins" merge.
INFER_WINDOW = 200          # syllable tokens per window
INFER_STRIDE = 150          # advance per window (50-token overlap)
# Subword budget for ONE forward pass. xlm-roberta supports 512 positions; a
# window is at most INFER_WINDOW + (INFER_WINDOW - INFER_STRIDE) = 250 syllables
# (the sentence-extension hard cap), which is ~320 subwords on dense judgment
# prose (money/dates/case-numbers/foreign names tokenize >1 subword each). The
# old 256 cap silently TRUNCATED such windows: the tail syllables produced no
# word_ids and were left "O" — a real recall hole at section/sentence cuts.
# 512 covers the 250-syllable hard cap for any realistic subword ratio (<2.0)
# so no in-window syllable is ever dropped. Sequences shorter than this are
# unaffected (padding/truncation only ever applied at the cap).
INFER_MAX_SUBWORDS = 512
# Section-aware inference (default): windows never cross a major-section
# boundary (PREAMBLE / NHÂN DANH / NỘI DUNG / NHẬN ĐỊNH / QUYẾT ĐỊNH), and a
# section longer than INFER_WINDOW is sub-chunked at SENTENCE boundaries near
# the target size rather than cut mid-token. Set False to fall back to the
# legacy blind sliding window (for before/after comparison).
INFER_SECTION_AWARE = True
