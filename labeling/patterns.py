"""Regex patterns for the 20-entity legal NER schema, grounded on real
judgment text (demo judgment 17/2018/HS-ST Di An / Binh Duong, plus crawled
portal judgments 100000-100014: HS-ST, HS-PT, QDST-HNGD, QDDS-ST layouts).

All patterns operate on NORMALIZED + FLATTENED text (see corpus.normalize).
`find_entities(text)` returns non-overlapping spans; `derive_doc_meta(spans)`
returns document-level metadata (case_type / procedure_stage parsed from the
CASE_NUMBER suffix).

OVERLAP PRIORITY POLICY (flat BIO -- one label per token):
  1. LEGAL_BASIS swallows LAW_NAME / ARTICLE / CLAUSE / POINT nested inside
     "Căn cứ ..." / "Áp dụng ..." citation sentences (BIO cannot nest).
  2. DECISION is the LOWEST priority label: fine-grained entities (PENALTY,
     CRIME, DEFENDANT, COURT_FEE, COMPENSATION, ARTICLE, ...) win inside
     decision sentences; DECISION only labels the remaining GAP tokens of the
     sentence ("Xử phạt bị cáo <DEFENDANT> <PENALTY>." -> "Xử phạt bị cáo"
     and the trailing punctuation-side tokens become DECISION). This
     preserves penalty/crime extraction quality inside verdicts.
  3. COMPENSATION / COURT_FEE beat MONEY_AMOUNT -- decided at detection time
     from a context window around the amount (60 chars before, 40 after; the
     corpus shows the cue AFTER the amount too: "nộp 200.000 ... đồng án phí
     hình sự sơ thẩm").
  4. Among the rest: fixed priority rank, then longer match, then
     left-to-right (greedy keep in `_resolve`).
"""

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Character classes
# ---------------------------------------------------------------------------
UPPER_VI = (
    "A-Z"
    "ĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
    "ÈÉẺẼẸÊỀẾỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰ"
    "ỲÝỶỸỴ"
)
LOWER_VI = (
    "a-z"
    "đàáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵ"
)
# A capitalized proper-noun word: "Dĩ", "An", "Bình" (also bare initial "T").
CAP_WORD = rf"[{UPPER_VI}][{LOWER_VI}]*"
# A fully uppercase word of >=2 chars (court headers). The lookahead stops
# before the national motto / section headers interleaved by two-column PDFs.
UP_WORD = (
    r"(?!CỘNG\b|NƯỚC\b|NHẬN\b|XÉT\b|QUYẾT\b|ĐỘC\b|THẨM\b|TM\b|HỘI\b|NHÂN\s+DANH)"
    rf"[{UPPER_VI}][{UPPER_VI}0-9.]+"
)

# ---------------------------------------------------------------------------
# A. CASE_NUMBER -- "17/2018/HS-ST", "111/2017/HSPT-QĐ", "118 /2017/QĐST-HNGĐ"
# Only the number token is labeled (anchors like "Bản án số:" stay O).
# "Nghị quyết số 326/2016/UBTVQH14" has the same shape but is swallowed by
# the higher-priority LAW_NAME match that covers it.
# ---------------------------------------------------------------------------
CASE_NUMBER_RE = re.compile(
    r"\b\d{1,4}\s*/\s*\d{4}/[A-ZĐ][A-ZĐ0-9]*(?:\s*-\s*[A-ZĐ][A-ZĐ0-9]*)*"
)

# Doc-level metadata derived from the CASE_NUMBER suffix (HS/DS/HC/HNGĐ/...).
# Order matters: longer codes first so "HNGĐ" is not read as "HC"/"DS".
_SUFFIX_CASE_TYPES = [
    ("HNGĐ", "hôn nhân và gia đình"),
    ("HNGD", "hôn nhân và gia đình"),
    ("KDTM", "kinh doanh thương mại"),
    ("LĐ", "lao động"),
    ("HS", "hình sự"),
    ("DS", "dân sự"),
    ("HC", "hành chính"),
]

# ---------------------------------------------------------------------------
# A. COURT -- "TÒA ÁN NHÂN DÂN THỊ XÃ DĨ AN, TỈNH BÌNH DƯƠNG" (also "TOÀ ÁN"
# spelling), mixed-case "Tòa án nhân dân thị xã Dĩ An, tỉnh Bình Dương".
# ---------------------------------------------------------------------------
COURT_UPPER_RE = re.compile(
    rf"T[OÒ][AÀ]\s+ÁN\s+(?:NHÂN\s+DÂN|QUÂN\s+SỰ)(?:(?:,\s*|\s+){UP_WORD})*"
)
COURT_MIXED_RE = re.compile(
    r"[Tt][oò][aà]\s+án\s+(?:nhân\s+dân|quân\s+sự)"
    r"(?:\s+(?:tối\s+cao|cấp\s+cao\s+tại|thành\s+phố|tỉnh|huyện|quận|thị\s+xã|khu\s+vực))?"
    rf"(?:\s+{CAP_WORD})*"
    rf"(?:\s*,\s*(?:tỉnh|thành\s+phố|TP\.?)(?:\s+{CAP_WORD})+)?"
)

# ---------------------------------------------------------------------------
# A. JUDGMENT_DATE (context-filtered in find_entities)
#   header : "Bản án số: 17/2018/HS-ST Ngày 26-01-2018", "Số: 111/2017/...
#             ... ngày 20 tháng 12 năm 2017"
#   hearing: "Ngày 26 tháng 01 năm 2018, tại trụ sở ... xét xử ..."
#   pronouncement: date near "tuyên án"
# ---------------------------------------------------------------------------
DATE_NUMERIC_RE = re.compile(r"[Nn]gày\s+\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{4}")
DATE_VERBOSE_RE = re.compile(r"[Nn]gày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}")
_HEADER_ANCHOR = re.compile(r"(?:Bản\s+án|Quyết\s+định)\s+(?:số|phúc\s+thẩm|sơ\s+thẩm)|Số\s*:")
_HEARING_ANCHOR = re.compile(r"xét\s+xử|tuyên\s+án")
_PRONOUNCE_BEFORE = re.compile(r"[Tt]uyên\s+án\s*:?\s*$")
# Dates glued to a case number ("thụ lý số 402/2017/HSST ngày 29 tháng 12
# năm 2017") are docket dates, not the hearing date.
_CASE_NUMBER_BEFORE = re.compile(r"\d+/\d{4}/[\w-]+\s*$")

# ---------------------------------------------------------------------------
# A. CASE_TYPE -- "vụ án hình sự sơ thẩm" -> label "hình sự" (group 1).
# Doc-level case_type is ALSO derived from the CASE_NUMBER suffix (metadata,
# not a token label) -- see derive_doc_meta().
# ---------------------------------------------------------------------------
CASE_TYPE_RE = re.compile(
    r"vụ\s+án\s+(hình\s+sự|dân\s+sự|hành\s+chính"
    r"|hôn\s+nhân(?:\s+(?:và\s+)?gia\s+đình)?"
    r"|kinh\s+doanh,?\s+thương\s+mại|lao\s+động)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# B. Parties -- anonymized names ("Nguyễn Văn T", "NGUYỄN THỊ T", "Lê Thế A")
# after role anchors. The corpus shows separators/bullets/list numbers and
# honorifics between anchor and name:
#   "bị cáo: Lê Thế A", "- Bị cáo có kháng cáo: NGUYỄN THỊ T – sinh năm",
#   "Nguyên đơn: Chị Nguyễn Thị H, SN 1988",
#   "Bị hại: 1. Chị Huỳnh Thị Kim T", "...nghĩa vụ liên quan: + Anh Lê Tấn V"
# Only the NAME is labeled (honorific stays O).
# ---------------------------------------------------------------------------
_HONORIFIC = r"(?:[Ôô]ng|[Bb]à|[Aa]nh|[Cc]hị|[Ee]m|[Cc]háu)"
_NAME_TOKEN = rf"[{UPPER_VI}][{LOWER_VI}{UPPER_VI}]*"
_NAME = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,5}}"
_PARTY_GLUE = rf"\s*:?\s*(?:\d+\.\s*)?(?:[+\-–]\s*)?(?:{_HONORIFIC}\s+)?"

PARTY_RES = {
    "DEFENDANT": re.compile(
        rf"(?:[Cc]ác\s+)?[Bb]ị\s+cáo(?:\s+có\s+kháng\s+cáo)?{_PARTY_GLUE}({_NAME})"
    ),
    "PLAINTIFF": re.compile(rf"[Nn]guyên\s+đơn{_PARTY_GLUE}({_NAME})"),
    "VICTIM": re.compile(rf"(?:[Nn]gười\s+)?[Bb]ị\s+hại{_PARTY_GLUE}({_NAME})"),
    "RELATED_PARTY": re.compile(
        r"[Nn]gười\s+có\s+(?:quyền\s+lợi,?\s+(?:và\s+)?nghĩa\s+vụ\s+liên\s+quan|QLNVLQ)"
        rf"{_PARTY_GLUE}({_NAME})"
    ),
}

# Capitalized words that start a sentence/section after a role mention but
# are never a person name ("Bị cáo Tại phiên tòa ..." must not yield "Tại").
_NAME_STOPWORDS = {
    "Tại", "Theo", "Trong", "Trước", "Sau", "Khi", "Nếu", "Việc", "Về", "Và",
    "Các", "Những", "Không", "Có", "Đã", "Đang", "Là", "Do", "Người", "Bị",
    "Nguyên", "Hội", "Tòa", "Toà", "Viện", "Cơ", "Nhà", "Bộ", "Trại", "Ngày",
    "Số", "Bản", "Quyết", "Căn", "Áp", "Xét", "Tuyên", "Xử", "Buộc", "Đối",
    "Công", "Phòng", "Chi", "Sở", "Ủy", "Uỷ", "Thôn", "Ấp", "Xã", "Phường",
    "Khu", "Đường", "Lời", "Một", "Hai", "Ba", "Bốn", "Năm", "Sáu", "Bảy",
    "Tám", "Chín", "Mười", "Vì", "Nay", "Đến", "Từ", "Như", "Nơi", "Được",
}

# ---------------------------------------------------------------------------
# C. LAW_NAME / ARTICLE / CLAUSE / POINT
#   "Bộ luật Hình sự năm 2015 (sửa đổi, bổ sung năm 2017)", "Bộ luật Hình sự
#   2015", "Bộ luật tố tụng dân sự", "Luật hôn nhân và gia đình năm 2014",
#   "Nghị quyết số 326/2016/UBTVQH14 ngày 30/12/2016", "Nghị định .../NĐ-CP"
# ---------------------------------------------------------------------------
_LAW_TAIL = (
    r"(?:\s+(?:năm\s+)?\d{4})?"
    r"(?:\s*,?\s*\(?\s*sửa\s+đổi,?\s+bổ\s+sung\s+năm\s+\d{4}\s*\)?)?"
)
_LAW_DOC_NO = (
    r"(?:số\s*:?\s*)?\d+/\d{4}/[A-ZĐ0-9.-]+"
    r"(?:\s+ngày\s+(?:\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}))?"
)
LAW_NAME_RE = re.compile(
    r"(?:"
    r"Bộ\s+[Ll]uật\s+(?:[Tt]ố\s+tụng\s+)?"
    r"(?:[Hh]ình\s+sự|[Dd]ân\s+sự|[Ll]ao\s+động|[Hh]àng\s+hải)"
    r"|Luật\s+(?:[Tt]ố\s+tụng\s+)?(?:"
    r"hôn\s+nhân\s+(?:và\s+)?gia\s+đình|đất\s+đai|nhà\s+ở|doanh\s+nghiệp"
    r"|thương\s+mại|đầu\s+tư|xây\s+dựng|giao\s+thông\s+đường\s+bộ"
    r"|bảo\s+hiểm\s+xã\s+hội|lao\s+động|hình\s+sự|dân\s+sự|thi\s+hành\s+án(?:\s+[a-zđ]\S*)*"
    rf"|{CAP_WORD}(?:\s+[{LOWER_VI}]+){{0,6}}"
    r")"
    rf"|Nghị\s+quyết\s+{_LAW_DOC_NO}"
    rf"|Nghị\s+định\s+{_LAW_DOC_NO}"
    rf"|Pháp\s+lệnh\s+[{LOWER_VI}]+(?:\s+[{LOWER_VI}]+){{0,6}}"
    r")"
    + _LAW_TAIL
)
ARTICLE_RE = re.compile(r"[Đđ]iều\s+\d+[a-zđ]?\b")
CLAUSE_RE = re.compile(r"[Kk]hoản\s+\d+\b")
# "điểm c khoản 1 Điều 250" -- require a khoản/Điều within 40 chars so bare
# "điểm" in prose ("địa điểm x") is not matched.
POINT_RE = re.compile(r"[Đđ]iểm\s+[a-zđ]\b(?=[^.]{0,40}?(?:[Kk]hoản|[Đđ]iều))")

# ---------------------------------------------------------------------------
# Ruling-verb opener vocabulary shared by LEGAL_BASIS (as a span terminator)
# and DECISION (as a span anchor). Covers every operative verb that opens a
# verdict sentence in the QUYẾT ĐỊNH section. Built once, reused below.
# ---------------------------------------------------------------------------
_RULING_VERB = (
    r"(?:Căn\s+cứ|Áp\s+dụng|Xét|XÉT|QUYẾT|[Xx]ử\s+phạt|[Pp]hạt\s+bổ\s+sung"
    r"|[Tt]uyên\s+(?:bố|xử|án)|[Bb]uộc\b|[Cc]hấp\s+nhận|[Kk]hông\s+chấp\s+nhận"
    r"|[Tt]ịch\s+thu|[Tt]rả\s+lại|[Đđ]ình\s+chỉ|[Mm]iễn\b|[Gg]iao\b|[Ss]ửa\b"
    r"|[Kk]iến\s+nghị)"
)

# ---------------------------------------------------------------------------
# C. LEGAL_BASIS -- "Căn cứ ..." / "Áp dụng ..." citation clauses. Applies
# anywhere (QUYẾT ĐỊNH section AND procedural headers).
# Guards:
#   * "(?!\s+biện\s+pháp)" keeps "Áp dụng biện pháp ..." for DECISION;
#   * a statute keyword must occur before the first [.;:] so non-citation
#     sentences ("Căn cứ vào biên bản ghi nhận ...") are skipped;
#   * the span TERMINATES at the first of:
#       - '.' followed by whitespace (number-internal '.' is never space-fed);
#       - ';'/',' immediately introducing the next citation sentence
#         ("Căn cứ ..." / "Áp dụng ...") or a DECISION ruling verb (the
#         operative verdict that follows the citation chain:
#         "...1999; Xử phạt ...", "...dân sự; Căn cứ ...");
#       - end of text.
#   * nested LAW_NAME/ARTICLE/CLAUSE/POINT inside this span are swallowed by
#     the priority engine.
# ---------------------------------------------------------------------------
_LEGAL_BASIS_STOP_VERB = (
    r"(?:Căn\s+cứ|Áp\s+dụng|Xét|QUYẾT|[Xx]ử\s+phạt|[Pp]hạt\s+bổ\s+sung"
    r"|[Tt]uyên\s+(?:bố|xử|án)|[Bb]uộc)"
)
LEGAL_BASIS_RE = re.compile(
    r"(?:Căn\s+cứ(?:\s+vào)?|Áp\s+dụng)(?!\s+biện\s+pháp)\s+"
    r"[^.;]{0,80}?(?:[Đđ]iều\s+\d|Bộ\s+[Ll]uật|Luật\s|Nghị\s+quyết|Nghị\s+định|Pháp\s+lệnh)"
    r"[^.]{0,400}?"
    r"(?=\.(?=\s|$)"
    r"|[;,]\s*" + _LEGAL_BASIS_STOP_VERB +
    r"|$)"
)

# ---------------------------------------------------------------------------
# D. CRIME -- offense names, quoted and unquoted; the offense NAME is labeled
# (the word "tội" stays O).
#   quoted  : về tội "Vận chuyển trái phép chất ma túy" / phạm tội: "Cướp
#             tài sản" / phạm vào tội "Tàng trữ trái phép chất ma tuý"
#   unquoted: về tội Vận chuyển trái phép chất ma túy quy định tại ... and
#             lowercase "phạm tội vận chuyển trái phép chất ma túy" (lowercase
#             variant restricted to a known offense-head vocabulary to stay
#             conservative).
# ---------------------------------------------------------------------------
CRIME_QUOTED_RE = re.compile(r"tội\s*:?\s*[“\"]([^”\"]{3,120})[”\"]")
_CRIME_HEAD = (
    r"(?:vận\s+chuyển|tàng\s+trữ|mua\s+bán|sản\s+xuất|trộm\s+cắp"
    r"|cướp(?:\s+giật)?|cưỡng\s+đoạt|lừa\s+đảo|lạm\s+dụng\s+tín\s+nhiệm"
    r"|giết\s+người|cố\s+ý\s+gây\s+thương\s+tích|đánh\s+bạc"
    r"|tổ\s+chức\s+đánh\s+bạc|hiếp\s+dâm|giao\s+cấu|dâm\s+ô"
    r"|chống\s+người\s+thi\s+hành\s+công\s+vụ|vi\s+phạm\s+quy\s+định"
    r"|gây\s+rối\s+trật\s+tự|chứa\s+chấp|tiêu\s+thụ|làm\s+giả|tham\s+ô"
    r"|nhận\s+hối\s+lộ|đưa\s+hối\s+lộ|bắt\s+cóc|hủy\s+hoại\s+tài\s+sản)"
)
CRIME_UNQUOTED_RE = re.compile(
    rf"(?:về|phạm)\s+(?:vào\s+)?tội\s*:?\s+"
    rf"((?:{CAP_WORD}|{_CRIME_HEAD})[^.;,:“”\"()]{{0,90}}?)"
    r"(?=\s+(?:quy\s+định|theo)\b|\s*[.;,:“”\"()])"
)

# ---------------------------------------------------------------------------
# D. VIOLATION_ACT -- the act phrase after a "hành vi" anchor
# ("có/đã có/(đã) thực hiện hành vi"). The span runs from the verb phrase up
# to the first clause break .;, (not number-internal) or end of text.
# ---------------------------------------------------------------------------
VIOLATION_ACT_RE = re.compile(
    r"(?:có|đã\s+(?:thực\s+hiện|có)|thực\s+hiện)\s+hành\s+vi\s+"
    r"(?!phạm\s+tội\s+quả\s+tang\b)"
    r"([^.;,]{10,150}?)"
    r"(?=(?<!\d)[.;,](?!\d)"
    r"|$)"
)

# ---------------------------------------------------------------------------
# E. PENALTY -- prison terms (incl. compound "01 (một) năm 06 (sáu) tháng
# tù"), non-custodial reform, suspended sentence, probation, life/death.
# Fines ("phạt tiền X đồng") are handled by the MONEY family instead.
# ---------------------------------------------------------------------------
_NUM_SPELLED = r"\d{1,3}\s*(?:\([^)]{1,60}\))?"
PENALTY_RES = [
    re.compile(
        rf"{_NUM_SPELLED}\s*năm(?:\s+{_NUM_SPELLED}\s*tháng)?\s+tù"
        r"(?:\s+nhưng\s+cho\s+hưởng\s+án\s+treo)?(?:\s+giam)?"
    ),
    re.compile(rf"{_NUM_SPELLED}\s*tháng\s+tù(?:\s+nhưng\s+cho\s+hưởng\s+án\s+treo)?(?:\s+giam)?"),
    re.compile(rf"{_NUM_SPELLED}\s*(?:năm|tháng)\s+cải\s+tạo\s+không\s+giam\s+giữ"),
    re.compile(rf"{_NUM_SPELLED}\s*(?:năm|tháng)\s+thử\s+thách"),
    re.compile(rf"thử\s+thách\s+(?:là\s+)?{_NUM_SPELLED}\s*(?:năm|tháng)"),
    re.compile(r"(?:cho\s+hưởng\s+)?án\s+treo"),
    re.compile(r"tù\s+chung\s+thân"),
    re.compile(r"tử\s+hình"),
]

# ---------------------------------------------------------------------------
# E. MONEY family -- "5.000.000 đồng", "200.000đ (Hai trăm nghìn đồng)",
# "200.000 (Hai trăm nghìn) đồng". Label decided by context window:
#   "án phí"/"lệ phí" nearby  -> COURT_FEE
#   "bồi thường" nearby       -> COMPENSATION
#   otherwise                 -> MONEY_AMOUNT
# ---------------------------------------------------------------------------
MONEY_RE = re.compile(
    r"(?<![\d./–-])(?:\d{1,3}(?:[.,]\d{3})+|\d{4,9})"
    r"(?:\s*\([^)]{1,80}\))?\s*(?:đồng|đ\b)"
)
_MONEY_CTX_BEFORE = 60
_MONEY_CTX_AFTER = 40

# ---------------------------------------------------------------------------
# E. DECISION -- operative ruling block in the QUYẾT ĐỊNH section, anchored on
# a ruling verb and running to the end of the sentence. The span ENDS at the
# FIRST of:
#   * '.'/';' followed by whitespace (number-internal '.' of "200.000đ" never
#     space-followed, so the sentence is not cut mid-amount);
#   * end of text.
# DECISION stays LOWEST priority: fine-grained entities (PENALTY, CRIME,
# DEFENDANT, MONEY, ...) win inside; only the GAP tokens become DECISION
# (see _decision_gaps).
# ---------------------------------------------------------------------------
_DECISION_VERB = (
    r"(?:Tuyên\s+bố|Tuyên\s+xử|Xử\s+phạt|Buộc"
    r"|Áp\s+dụng\s+biện\s+pháp)"
)
DECISION_RE = re.compile(
    rf"{_DECISION_VERB}\b"
    r".{5,400}?"
    r"(?=[.;](?=\s|$)"
    r"|$)"
)
QUYET_DINH_MARKER_RE = re.compile(r"QUYẾT\s+ĐỊNH\s*:?")


def _operative_zone_start(text: str) -> int | None:
    """Char offset where the operative QUYẾT ĐỊNH ruling block begins.

    The LAST "QUYẾT ĐỊNH" marker is the operative header (earlier occurrences
    appear in the cover page / table of contents). Returns None when the
    document has no operative section. Used by corpus.sectioner (section-aware
    inference) to align the QUYẾT ĐỊNH boundary with the weak-labeler.
    """
    end = None
    for m in QUYET_DINH_MARKER_RE.finditer(text):
        end = m.end()
    return end


# Priority for overlap resolution: higher first. DECISION is handled
# separately (gap filling) and is therefore strictly lowest.
PRIORITY = [
    "LEGAL_BASIS",
    "COURT_FEE", "COMPENSATION",
    "PENALTY", "CRIME",
    "LAW_NAME",
    "CASE_NUMBER", "COURT", "JUDGMENT_DATE", "CASE_TYPE",
    "DEFENDANT", "PLAINTIFF", "VICTIM", "RELATED_PARTY",
    "ARTICLE", "CLAUSE", "POINT",
    "MONEY_AMOUNT",
    "VIOLATION_ACT",
    "DECISION",
]


@dataclass
class Span:
    start: int
    end: int
    label: str
    text: str


def _spans(regex: re.Pattern, text: str, label: str, group: int = 0) -> list[Span]:
    return [
        Span(m.start(group), m.end(group), label, m.group(group))
        for m in regex.finditer(text)
        if m.group(group)
    ]


def _find_dates(text: str) -> list[Span]:
    """Judgment-date candidates filtered by document context."""
    out = []
    for m in DATE_NUMERIC_RE.finditer(text):
        if _HEADER_ANCHOR.search(text[max(0, m.start() - 120): m.start()]):
            out.append(Span(m.start(), m.end(), "JUDGMENT_DATE", m.group()))
    for m in DATE_VERBOSE_RE.finditer(text):
        if _CASE_NUMBER_BEFORE.search(text[max(0, m.start() - 30): m.start()]):
            continue  # docket date glued to a case number
        before = text[max(0, m.start() - 160): m.start()]
        after = text[m.end(): m.end() + 160]
        if (_HEARING_ANCHOR.search(after) or _HEADER_ANCHOR.search(before)
                or _PRONOUNCE_BEFORE.search(text[max(0, m.start() - 40): m.start()])):
            out.append(Span(m.start(), m.end(), "JUDGMENT_DATE", m.group()))
    return out


def _find_parties(text: str) -> list[Span]:
    out = []
    for label, regex in PARTY_RES.items():
        for m in regex.finditer(text):
            name = m.group(1)
            if name.split()[0] in _NAME_STOPWORDS:
                continue
            out.append(Span(m.start(1), m.end(1), label, name))
    return out


def _find_money(text: str) -> list[Span]:
    out = []
    for m in MONEY_RE.finditer(text):
        ctx = (text[max(0, m.start() - _MONEY_CTX_BEFORE): m.start()]
               + " | " + text[m.end(): m.end() + _MONEY_CTX_AFTER]).lower()
        if "án phí" in ctx or "lệ phí" in ctx:
            label = "COURT_FEE"
        elif "bồi thường" in ctx:
            label = "COMPENSATION"
        else:
            label = "MONEY_AMOUNT"
        out.append(Span(m.start(), m.end(), label, m.group()))
    return out


def _resolve(spans: list[Span]) -> list[Span]:
    """Priority rank, then longer span, then earlier start; greedy keep."""
    rank = {label: i for i, label in enumerate(PRIORITY)}
    spans = sorted(spans, key=lambda s: (rank[s.label], -(s.end - s.start), s.start))
    kept: list[Span] = []
    for s in spans:
        if all(s.end <= k.start or s.start >= k.end for k in kept):
            kept.append(s)
    return kept


_WORD_RE = re.compile(r"\w", re.UNICODE)


def _decision_gaps(text: str, kept: list[Span]) -> list[Span]:
    """DECISION = the uncovered remainder of decision sentences (lowest
    priority by construction: only gaps between already-kept entities)."""
    qd = None
    for m in QUYET_DINH_MARKER_RE.finditer(text):
        qd = m  # LAST occurrence = the ruling header
    if qd is None:
        return []
    out: list[Span] = []
    occupied = sorted([k for k in kept], key=lambda s: s.start)
    for m in DECISION_RE.finditer(text, qd.end()):
        cur = m.start()
        inside = [k for k in occupied + out
                  if k.start < m.end() and k.end > m.start()]
        inside.sort(key=lambda s: s.start)
        gaps = []
        for k in inside:
            if k.start > cur:
                gaps.append((cur, k.start))
            cur = max(cur, k.end)
        if cur < m.end():
            gaps.append((cur, m.end()))
        for gs, ge in gaps:
            seg = text[gs:ge]
            first = _WORD_RE.search(seg)
            if not first:
                continue
            # trim to word-char bounds
            last = len(seg)
            while last > 0 and not _WORD_RE.match(seg[last - 1]):
                last -= 1
            s2, e2 = gs + first.start(), gs + last
            if e2 - s2 >= 2:
                out.append(Span(s2, e2, "DECISION", text[s2:e2]))
    return out


def find_entities(text: str) -> list[Span]:
    """Find all entity spans; resolve overlaps per the module-level policy."""
    spans: list[Span] = []
    spans += _spans(LEGAL_BASIS_RE, text, "LEGAL_BASIS")
    spans += _spans(CASE_NUMBER_RE, text, "CASE_NUMBER")
    spans += _spans(COURT_UPPER_RE, text, "COURT")
    spans += _spans(COURT_MIXED_RE, text, "COURT")
    spans += _find_dates(text)
    spans += _spans(CASE_TYPE_RE, text, "CASE_TYPE", group=1)
    spans += _find_parties(text)
    spans += _spans(LAW_NAME_RE, text, "LAW_NAME")
    spans += _spans(ARTICLE_RE, text, "ARTICLE")
    spans += _spans(CLAUSE_RE, text, "CLAUSE")
    spans += _spans(POINT_RE, text, "POINT")
    spans += _spans(CRIME_QUOTED_RE, text, "CRIME", group=1)
    spans += _spans(CRIME_UNQUOTED_RE, text, "CRIME", group=1)
    spans += _spans(VIOLATION_ACT_RE, text, "VIOLATION_ACT", group=1)
    for rx in PENALTY_RES:
        spans += _spans(rx, text, "PENALTY")
    spans += _find_money(text)

    kept = _resolve(spans)
    kept += _decision_gaps(text, kept)
    kept.sort(key=lambda s: s.start)
    return kept


def derive_doc_meta(spans: list[Span]) -> dict:
    """Document-level metadata from the FIRST CASE_NUMBER span (the header
    number): case_type and procedure_stage parsed from its suffix
    (HS/DS/HC/HNGĐ/KDTM/LĐ + ST/PT). Metadata only -- not a token label."""
    meta = {"case_number": None, "case_type": None, "procedure_stage": None}
    for s in sorted(spans, key=lambda x: x.start):
        if s.label != "CASE_NUMBER":
            continue
        parts = re.sub(r"\s+", "", s.text).split("/", 2)
        if len(parts) < 3:
            continue
        suffix = parts[2]
        meta["case_number"] = "/".join(parts)
        for code, vi_type in _SUFFIX_CASE_TYPES:
            if code in suffix:
                meta["case_type"] = vi_type
                break
        if "PT" in suffix:
            meta["procedure_stage"] = "phúc thẩm"
        elif "ST" in suffix:
            meta["procedure_stage"] = "sơ thẩm"
        break
    return meta
