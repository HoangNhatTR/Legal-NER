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
# This MUST match the served checkpoint's config.json id2label
# (legal-ner-v3r-full, micro-F1 ~0.970). The legacy VIOLATION_ACT label was
# DROPPED and its act-narrative signal consolidated into CRIMINAL_ACT; that
# shifts the id of every label after it vs the old 20-label schema, so the
# v3r-full model was trained from scratch on this inventory.
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
    # (v3r) VIOLATION_ACT dropped — consolidated into CRIMINAL_ACT (below)
    # E. outcome
    "PENALTY",        # "02 (hai) năm tù", "tù chung thân", "án treo", ...
    "MONEY_AMOUNT",   # money NOT in compensation/fee context
    "COMPENSATION",   # money in "bồi thường" context
    "COURT_FEE",      # money in "án phí / lệ phí" context
    "DECISION",       # verdict-sentence remainder in QUYET DINH section
    # F. people conducting the proceedings
    "JUDGE",              # "Thẩm phán" / "Chủ tọa phiên tòa"
    "ASSESSOR",           # "Hội thẩm nhân dân"
    "PROSECUTOR",         # "Kiểm sát viên"
    "CLERK",              # "Thư ký phiên tòa"
    "LAWYER",             # "Luật sư" / "người bào chữa"
    "WITNESS",            # "người làm chứng"
    # G. courtroom behavior / procedural status
    "COURT_BEHAVIOR",     # "có mặt", "vắng mặt", "kháng cáo", "xin giảm nhẹ", ...
    # H. sentencing factors (Điều 51/52 BLHS 2015)
    "MITIGATING_FACTOR",  # tình tiết giảm nhẹ (Điều 51)
    "AGGRAVATING_FACTOR", # tình tiết tăng nặng (Điều 52)
    # I. criminal act narrative (upgrade of the dropped VIOLATION_ACT)
    "CRIMINAL_ACT",       # act narrative after "có hành vi"/"thực hiện hành vi"
    # J. quantities & physical evidence
    "QUANTITY",           # "0,1488 gam", "01 viên", "X kg"
    "EVIDENCE_ITEM",      # tang vật / vật chứng: "01 xe mô tô ...", "01 điện thoại"
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
# Inference windowing (v3r-full)
# ---------------------------------------------------------------------------
# A forward pass sees at most INFER_WINDOW syllable tokens. INFER_STRIDE is the
# advance between consecutive windows; the (WINDOW - STRIDE) overlap keeps an
# entity that straddles a window edge intact for "first-decided-wins" merge.
INFER_WINDOW = 200          # syllable tokens per window
INFER_STRIDE = 150          # advance per window (50-token overlap)
# Subword budget for ONE forward pass. xlm-roberta supports 512 positions; a
# window can reach INFER_WINDOW + (INFER_WINDOW - INFER_STRIDE) = 250 syllables
# (the sentence-extension hard cap) ≈ ~320 subwords on dense judgment prose. The
# old 256 cap silently TRUNCATED such windows (tail syllables produced no
# word_ids → left "O", a recall hole). 512 covers the hard cap for any realistic
# subword ratio (<2.0) so no in-window syllable is dropped.
INFER_MAX_SUBWORDS = 512
# Section-aware inference (default): windows never cross a major-section
# boundary (PREAMBLE / NHÂN DANH / NỘI DUNG / NHẬN ĐỊNH / QUYẾT ĐỊNH); a section
# longer than INFER_WINDOW is sub-chunked at SENTENCE boundaries. Set False for
# the legacy blind sliding window (before/after comparison).
INFER_SECTION_AWARE = True
