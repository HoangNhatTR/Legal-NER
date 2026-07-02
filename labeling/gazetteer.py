"""Sentencing-factor gazetteer — mitigating (Điều 51) / aggravating (Điều 52)
circumstances of the 2015 Penal Code (Bộ luật Hình sự 2015, sửa đổi 2017).

``find_entities`` (labeling.patterns._find_factors) uses this to weak-label the
MITIGATING_FACTOR / AGGRAVATING_FACTOR spans that train the NER model. Each
entry is the SHORT diagnostic phrase that recurs verbatim in the NHẬN ĐỊNH
reasoning of real judgments (not the full statutory clause), so the regex built
in ``patterns._to_regex`` matches the way courts actually write them.

Contract (consumed by labeling.patterns._load_gazetteer_regexes):
    load_gazetteer() -> {
        "mitigating":  [{"phrase": str}, ...],
        "aggravating": [{"phrase": str}, ...],
    }

Phrases are matched diacritic- and case-insensitively (patterns._fold_keep_len)
with punctuation/whitespace treated flexibly, so write them naturally with
diacritics here.

NOTE: this list is RECONSTRUCTED from the statute + observed judgment wording to
restore the weak-labeling pipeline (the original gazetteer file was not shipped
with this inference bundle). It is deliberately CONSERVATIVE — it omits phrases
that collide with COURT_BEHAVIOR cues ("tự thú", "đầu thú") or that are
ambiguous between the two factor classes ("phụ nữ có thai", "người đủ 70 tuổi
trở lên", "khuyết tật nặng" — these appear in BOTH Điều 51 and Điều 52 with
opposite meaning depending on whether the offender or the victim is described).
Extend it as new judgment wording is observed; re-run ``labeling.weak_label``
to regenerate training data after any change.
"""

from __future__ import annotations

# --- Điều 51: tình tiết giảm nhẹ trách nhiệm hình sự -------------------------
_MITIGATING: list[str] = [
    # b) tự nguyện sửa chữa, bồi thường thiệt hại hoặc khắc phục hậu quả
    "tự nguyện bồi thường",
    "tự nguyện sửa chữa",
    "bồi thường thiệt hại",
    "khắc phục hậu quả",
    # a) đã ngăn chặn hoặc làm giảm bớt tác hại của tội phạm
    "ngăn chặn hoặc làm giảm bớt tác hại",
    "làm giảm bớt tác hại",
    # c/d/đ) vượt quá giới hạn phòng vệ / tình thế cấp thiết / khi bắt giữ
    "vượt quá giới hạn phòng vệ chính đáng",
    "vượt quá yêu cầu của tình thế cấp thiết",
    # e) bị kích động về tinh thần do hành vi trái pháp luật của nạn nhân
    "bị kích động về tinh thần",
    # g) hoàn cảnh đặc biệt khó khăn mà không phải do mình tự gây ra
    "hoàn cảnh đặc biệt khó khăn",
    # h) chưa gây thiệt hại hoặc gây thiệt hại không lớn
    "chưa gây thiệt hại",
    "gây thiệt hại không lớn",
    # i) phạm tội lần đầu và thuộc trường hợp ít nghiêm trọng
    "phạm tội lần đầu",
    # k) bị người khác đe dọa hoặc cưỡng bức
    "bị đe dọa hoặc cưỡng bức",
    # m) phạm tội do lạc hậu
    "phạm tội do lạc hậu",
    # r) tự thú handled by COURT_BEHAVIOR; s) thành khẩn khai báo, ăn năn hối cải
    "thành khẩn khai báo",
    "ăn năn hối cải",
    # t) tích cực hợp tác với cơ quan có trách nhiệm
    "tích cực hợp tác",
    # u) đã lập công chuộc tội
    "lập công chuộc tội",
    # v) có thành tích xuất sắc trong sản xuất, chiến đấu, học tập, công tác
    "thành tích xuất sắc",
    # x) người có công với cách mạng
    "người có công với cách mạng",
]

# --- Điều 52: tình tiết tăng nặng trách nhiệm hình sự ------------------------
_AGGRAVATING: list[str] = [
    # a) phạm tội có tổ chức
    "phạm tội có tổ chức",
    # b) có tính chất chuyên nghiệp
    "có tính chất chuyên nghiệp",
    # c) lợi dụng chức vụ, quyền hạn để phạm tội
    "lợi dụng chức vụ, quyền hạn",
    # d) có tính chất côn đồ
    "có tính chất côn đồ",
    # đ) phạm tội vì động cơ đê hèn
    "động cơ đê hèn",
    # e) cố tình thực hiện tội phạm đến cùng
    "cố tình thực hiện tội phạm đến cùng",
    # g) phạm tội 02 lần trở lên
    "phạm tội 02 lần trở lên",
    "phạm tội 2 lần trở lên",
    "phạm tội nhiều lần",
    # h) tái phạm hoặc tái phạm nguy hiểm  (longer alt is tried first)
    "tái phạm nguy hiểm",
    "tái phạm",
    # l) lợi dụng hoàn cảnh chiến tranh / tình trạng khẩn cấp / thiên tai, dịch bệnh
    "lợi dụng hoàn cảnh chiến tranh",
    "lợi dụng tình trạng khẩn cấp",
    "lợi dụng thiên tai, dịch bệnh",
    # m) dùng thủ đoạn tinh vi, xảo quyệt, tàn ác
    "thủ đoạn tinh vi",
    "thủ đoạn xảo quyệt",
    "thủ đoạn tàn ác",
    # n) dùng thủ đoạn, phương tiện có khả năng gây nguy hại cho nhiều người
    "gây nguy hại cho nhiều người",
    # o) xúi giục người dưới 18 tuổi phạm tội
    "xúi giục người dưới 18 tuổi",
    # p) hành động xảo quyệt, hung hãn nhằm trốn tránh, che giấu tội phạm
    "che giấu tội phạm",
]


def load_gazetteer() -> dict[str, list[dict[str, str]]]:
    """Return the mitigating/aggravating phrase lists in the schema
    ``labeling.patterns`` expects: a dict of two lists of ``{"phrase": str}``."""
    return {
        "mitigating": [{"phrase": p} for p in _MITIGATING],
        "aggravating": [{"phrase": p} for p in _AGGRAVATING],
    }
