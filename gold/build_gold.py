"""Build gold/gold.jsonl from human-chosen substrings (annotations below).

Each annotation is (LABEL, exact-substring [, occurrence]). The substring is
copied from the FLATTENED gold text (gold/texts/<id>.txt) so char offsets line
up EXACTLY with the model/labeler stream. Annotation was done by READING each
judgment and deciding the correct spans per the gold entity definitions —
INDEPENDENT of the regex labeler.

Priority entities (annotated thoroughly): DECISION, LEGAL_BASIS, VIOLATION_ACT.
Anchors (CASE_NUMBER, CRIME) annotated where unambiguous for a subset.
"""

import json
from pathlib import Path

from annotate import build_record

GOLD_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# ANNOTATIONS: {doc_id: (source, [(label, substring[, occ]), ...])}
# DECISION  = one operative ruling clause (no time-credit / procedural tail)
# LEGAL_BASIS = one "Căn cứ ..."/"Áp dụng ..." statutory-citation clause
# VIOLATION_ACT = concrete violating-act verb-phrase (tight)
# ---------------------------------------------------------------------------
ANN: dict[str, tuple[str, list]] = {}

# ===== CRIMINAL — short "đình chỉ phúc thẩm" decisions =====
ANN["104295"] = ("criminal", [
    ("CASE_NUMBER", "08/2017/HSPT – QĐ"),
    ("LEGAL_BASIS", "Căn cứ vào Điều 39 và Điều 238 của Bộ luật tố tụng hình sự"),
    ("DECISION", "Đình chỉ xét xử phúc thẩm vụ án hình sự đối với các bị cáo Khổng Trường G và Phạm Văn K phạm tội “Cướp tài sản” theo điểm c khoản 2 Điều 133 Bộ luật Hình sự"),
    ("CRIME", "Cướp tài sản"),
])
ANN["104476"] = ("criminal", [
    ("CASE_NUMBER", "11/2018/HSPT-QĐ"),
    ("LEGAL_BASIS", "Căn cứ các điều 45, 342 và 348 của Bộ luật Tố tụng hình sự"),
    ("DECISION", "Đình chỉ xét xử phúc thẩm vụ án hình sự phúc thẩm thụ lý số: 33/2018/TLPT-HS ngày 04 tháng 5 năm 2018 đối với bị cáo Phạm Văn V phạm tội “Trộm cắp tài sản”"),
    ("CRIME", "Trộm cắp tài sản"),
])
ANN["100201"] = ("criminal", [
    ("CASE_NUMBER", "09/2018/HSPT-QĐ"),
    ("LEGAL_BASIS", "Căn cứ các điều 45, 342 và 348 của Bộ luật Tố tụng hình sự"),
    ("DECISION", "Đình chỉ xét xử phúc thẩm vụ án hình sự phúc thẩm thụ lý số 04/2018/TLPT-HS ngày 04 tháng 01 năm 2018 đối với bị cáo Nguyễn Tiến N phạm tội “Lừa đảo chiếm đoạt tài sản”"),
    ("CRIME", "Lừa đảo chiếm đoạt tài sản"),
])
ANN["100203"] = ("criminal", [
    ("CASE_NUMBER", "117/2017/HSPT- QĐ"),
    ("LEGAL_BASIS", "Căn cứ vào Điều 39 và Điều 238 của Bộ luật tố tụng hình sự"),
    ("DECISION", "Đình chỉ xét xử phúc thẩm vụ án hình sự phúc thẩm thụ lý số:172/2017/TLPT-HS đối với bị cáo Trương Hữu C, sinh năm 1995; STQ: Thôn A, xã G, huyện T, tỉnh Bắc Ninh phạm tội “Cố ý gây thương tích” theo khoản 3 Điều 104 của Bộ luật hình sự"),
    ("CRIME", "Cố ý gây thương tích"),
])
ANN["102641"] = ("criminal", [
    ("CASE_NUMBER", "27/2018/HSPT-QĐ"),
    ("LEGAL_BASIS", "Căn cứ các điều 45, 342 và 348 của Bộ luật Tố tụng hình sự"),
    ("DECISION", "Đình chỉ xét xử phúc thẩm vụ án hình sự phúc thẩm thụ lý số 38/2018/HSPT ngày 06 tháng 4 năm 2018 đối với bị cáo Vũ Văn Th, sinh năm 1975; nơi cư trú: Thôn K, xã D, huyện H, tỉnh Thái Bình phạm tội “Đánh bạc” theo khoản 1, 3 Điều 248 Bộ luật hình sự năm 1999"),
    ("CRIME", "Đánh bạc"),
])
ANN["106588"] = ("criminal", [
    ("CASE_NUMBER", "124/ 2018 HSPT-QĐ"),
    ("LEGAL_BASIS", "Căn cứ các điều 45, 342 và 348 của Bộ luật Tố tụng hình sự"),
    ("DECISION", "Đình chỉ xét xử phúc thẩm vụ án hình sự phúc thẩm thụ lý số: 268/2018/TLPT-HS ngày 29 tháng 3 năm 2018 đối với bị cáo Phạm Quang T phạm tội Vận chuyển trái phép chất ma túy"),
    ("CRIME", "Vận chuyển trái phép chất ma túy"),
])

# ===== CRIMINAL — full verdicts =====
ANN["102375"] = ("criminal", [
    ("VIOLATION_ACT", "Mua bán trái phép chất ma túy tại nhà của Giàng Xu T", 1),
    ("DECISION", "Tuyên bố bị cáo Lý A N phạm tội: Mua bán trái phép chất ma tuý"),
    ("LEGAL_BASIS", "Áp dụng khoản 1 Điều 194, điểm p khoản 1 Điều 46, của Bộ luật hình sự năm 1999"),
    ("DECISION", "Xử phạt bị cáo Lý A N 24 (hai mươi bốn) tháng tù"),
    ("LEGAL_BASIS", "Áp dụng Điều 41 Bộ luật hình sự năm 1999 và Điều 76 Bộ luật tố tụng hình sự năm 2003"),
    ("DECISION", "Tuyên tiêu hủy 01 phong bì còn nguyên niêm phong bên ngoài ghi"),
    ("DECISION", "Tuyên trả lại cho Vàng Thị X 01 chiếc xe máy nhãn hiệu HONDA, loại xe Wave S, sơn màu đỏ - xám - đen, Biển kiểm soát 25T1 - 0292, số khung 084335, số máy 0621157 xe cũ đã qua sử dụng"),
    ("DECISION", "Tuyên sung công quỹ Nhà nước số tiền 50.000đ, tiền thật do Ngân hàng Nhà nước Việt Nam phát hành"),
    ("DECISION", "Buộc bị cáo Lý A N phải chịu 200.000đ án phí hình sự sơ thẩm"),
    ("CRIME", "Mua bán trái phép chất ma tuý"),
])
ANN["104263"] = ("criminal", [
    ("VIOLATION_ACT", "tàng trữ trái phép chất bột nghi là ma túy", 1),
    ("LEGAL_BASIS", "Căn cứ vào: khoản 3 Điều 7, điểm g khoản 2 Điều 249, điểm s khoản 1 và khoản 2 Điều 51 của Bộ luật hình sự năm 2015 (được sửa đổi, bổ sung theo Luật số: 12/2017/QH14 ngày 20/6/2017 của Quốc hội); Nghị quyết số: 41/2017/QH14 ngày 20/6/2017 của Quốc hội; khoản 1, điểm a khoản 2 và điểm a khoản 3 Điều 106, khoản 2 Điều 136, khoản 1 Điều 331, khoản 1 Điều 333 của Bộ luật tố tụng hình sự năm 2015; Nghị quyết số: 326/2016/UBTVQH14 ngày 30/12/2016 của Uỷ ban thường vụ Quốc hội quy định về mức thu, miễn, giảm, thu, nộp, quản lý và sử dụng án phí và lệ phí Tòa án"),
    ("DECISION", "Tuyên bố: Bị cáo Phạm Tiến H phạm tội “Tàng trữ trái phép chất ma túy”"),
    ("DECISION", "Xử phạt bị cáo Phạm Tiến H 08 (tám) năm 06 (sáu) tháng tù"),
    ("DECISION", "Tịch thu và tiêu hủy 01 (một) phong bì thư hoàn lại sau giám định số: 763/GĐMT của Phòng kỹ thuật hình sự Công an tỉnh Quảng Ninh"),
    ("DECISION", "Trả lại cho bị cáo Phạm Tiến H số tiền 2.000.000đ (hai triệu đồng)"),
    ("DECISION", "Buộc bị cáo Phạm Tiến H phải nộp 200.000đ (hai trăm nghìn đồng) tiền án phí hình sự sơ thẩm"),
    ("CRIME", "Tàng trữ trái phép chất ma túy", 1),
])
ANN["104071"] = ("criminal", [
    ("VIOLATION_ACT", "sử dụng trái phép chất ma túy", 2),
    ("DECISION", "Tuyên bố bị cáo: Cao Xuân S phạm tội: \"Mua bán trái phép chất ma túy\", bị cáo Nguyễn Thái L phạm tội “Tàng trữ trái phép chất ma túy”"),
    ("LEGAL_BASIS", "¸p dông: Điểm b, c khoản 2 Điều 251, điểm s khoản 1 và khoản 2 Điều 51, Điều 38 của Bộ luật hình sự 2015, sửa đổi, bổ sung năm 2017 đối với bị cáo Cao Xuân S"),
    ("LEGAL_BASIS", "Áp dụng: Điểm c khoản 1 Điều 249, điểm s khoản 1 và khoản 2 Điều 51, Điều 38 của Bộ luật hình sự 2015, sửa đổi, bổ sung năm 2017 đối với bị cáo Nguyễn Thái L"),
    ("DECISION", "Xử phạt bị cáo Cao Xuân S 08 (tám) năm tù"),
    ("DECISION", "Xử phạt bị cáo Nguyễn Thái L 15 (mười lăm) tháng tù"),
    ("CRIME", "Mua bán trái phép chất ma túy", 1),
])
ANN["101464"] = ("criminal", [
    ("VIOLATION_ACT", "lợi dụng sơ hở của người bị hại vào nhà để lén lút chiếm đoạt 01 sợi dây chuyền bằng vàng của bị hại M"),
    ("DECISION", "Tuyên bố: bị cáo Thạch S (X), phạm tội “Trộm cắp tài sản”"),
    ("LEGAL_BASIS", "Áp dụng khoản 1, Điều 138; các điểm b, h, p, khoản 1 và khoản 2, Điều 46; Điều 33, Điều 45 Bộ luật Hình sự năm 1999; khoản 2, Điều 136; Điều 299, Điều 333, Điều 338 Bộ luật Tố tụng hình sự; khoản 1, Điều 23 Nghị quyết số 326/2016/UBTVQH14, ngày 30/12/2016 của Ủy ban Thường vụ Quốc Hội, quy định về án phi, lệ phí Tòa án"),
    ("DECISION", "Xử phạt: Bị cáo Thạch S (X) 06 (sáu) tháng tù"),
    ("DECISION", "Buộc bị cáo Thạch S phải nộp 200.000 đồng án phí hình sự sơ thẩm"),
    ("CRIME", "Trộm cắp tài sản", 1),
])
ANN["101476"] = ("criminal", [
    ("DECISION", "Tuyên bố bị cáo Đậu Ngọc G phạm Tội “Giết người do vượt quá giới hạn phòng vệ chính đáng” và Tội “Cố ý gây thương tích do vượt quá giới hạn phòng vệ chính đáng”"),
    ("LEGAL_BASIS", "Áp dụng khoản 1 Điều 96; điểm b, p khoản 1, khoản 2 Điều 46; điểm g khoản 1 Điều 48; Điều 33 của Bộ luật Hình sự 1999, được sửa đổi, bổ sung năm 2009", 1),
    ("DECISION", "Xử phạt bị cáo Đậu Ngọc G 02 (hai) năm tù về tội “Giết người do vượt quá giới hạn phòng vệ chính đáng”"),
    ("LEGAL_BASIS", "Áp dụng khoản 1 Điều 106; điểm b, p khoản 1, khoản 2 Điều 46; điểm g khoản 1 Điều 48; Điều 33 của Bộ luật Hình sự 1999, được sửa đổi, bổ sung năm 2009"),
    ("DECISION", "Xử phạt bị cáo Đậu Ngọc G 01 (một) năm tù về tội “Cố ý gây thương tích do vượt quá giới hạn phòng vệ chính đáng”"),
    ("LEGAL_BASIS", "Áp dụng Điều 50 của Bộ luật Hình sự 1999, được sửa đổi, bổ sung năm 2009"),
    ("DECISION", "Tổng hợp hình phạt của hai tội buộc bị cáo Đậu Ngọc G phải chấp hành là 03 (ba) năm tù"),
    ("DECISION", "Buộc bị cáo Đậu Ngọc G phải bồi thường cho người Đại diện hợp pháp của người bị hại Trần Văn B số tiền 192.751.000 đồng (Một trăm chín mươi hai triệu bảy trăm năm mươi mốt nghìn đồng)"),
])
ANN["101809"] = ("criminal", [
    ("VIOLATION_ACT", "chặn xe nhiều lần để chiếm đoạt tài sản của nhiều người bị hại"),
    ("LEGAL_BASIS", "Căn cứ điểm c, khoản 1 Điều 355; điểm b khoản 1, điểm b khoản 2 Điều 358 Bộ luật Tố tụng hình sự năm 2015"),
    ("DECISION", "Hủy bản án hình sự sơ thẩm số: 43/2017/HS-ST ngày 15/12/2017 của Tòa án nhân dân huyện Thoại Sơn, tỉnh An Giang đã xét xử các bị cáo Hồ Văn M, Hồ Phước C, Lê Văn T, Phạm Văn Q (Q Ruồi), Lê Văn H, Lưu Minh Tr, Nguyễn Văn T1, Trần Hoàng T2, Nguyễn Văn H phạm tội “Cưỡng đoạt tài sản”"),
    ("DECISION", "Giao toàn bộ hồ sơ vụ án cho cấp sơ thẩm điều tra, truy tố, xét xử lại theo đúng quy định của pháp luật"),
    ("DECISION", "Tiếp tục tạm giam bị cáo Hồ Văn M cho đến khi các cơ quan tiến hành tố tụng huyện Thoại Sơn thụ lý lại vụ án"),
    ("CRIME", "Cưỡng đoạt tài sản", 1),
])
ANN["102186"] = ("criminal", [
    ("DECISION", "Không chấp nhận kháng cáo của bị cáo Trương Văn T và giữ nguyên bản án sơ thẩm"),
    ("DECISION", "Tuyên bố bị cáo Trương Văn T phạm tội “Vi phạm quy định về điều khiển phương tiện giao thông đường bộ”"),
    ("LEGAL_BASIS", "Áp dụng khoản 1 Điều 202; điểm b, p khoản 1, khoản 2 Điều 46 Bộ luật Hình sự"),
    ("DECISION", "Xử phạt bị cáo Trương Văn T 01 (một) năm 03 (ba) tháng tù"),
    ("CRIME", "Vi phạm quy định về điều khiển phương tiện giao thông đường bộ"),
])
ANN["100640"] = ("criminal", [
    ("VIOLATION_ACT", "đánh bạc dưới hình thức ghi “Lô đề” với Nguyễn Thị Hoa S", 1),
    ("LEGAL_BASIS", "Căn cứ điểm a, b khoản 1 Điều 355, điểm c, e khoản 1, điểm a khoản 2 Điều 357 BLTTHS"),
    ("DECISION", "Chấp nhận kháng nghị của VKSND tỉnh Hà Tĩnh đối với nội dung tăng hình phạt đối với bị cáo Tôn Nữ Quỳnh T"),
    ("DECISION", "không chấp nhận kháng cáo của bị cáo Tôn Nữ Quỳnh T"),
    ("DECISION", "Chấp nhận một phần kháng nghị của VKSND tỉnh Hà Tĩnh đối với bị cáo Nguyễn Thị Hoa S"),
    ("DECISION", "Chấp nhận kháng cáo của các bị cáo: Nguyễn Thị Thu H, Đặng Thị Tố H1, Võ Viết C, Lại Thế T1"),
    ("DECISION", "Chấp nhận một phần kháng cáo của bị cáo Đặng Thị L"),
    ("DECISION", "Sửa bản án sơ thẩm"),
    ("DECISION", "Tuyên bố bị cáo Tôn Nữ Quỳnh T phạm tội “Tổ chức đánh bạc” và “Đánh bạc”; bị cáo Nguyễn Thị Hoa S phạm tội “Tổ chức đánh bạc”; bị cáo Nguyễn Thị Thu H, Đặng Thị L, Đặng Thị Tố H1, Võ Viết C, Lại Thế T1 phạm tội “Đánh bạc”"),
])
ANN["101260"] = ("criminal", [
    ("DECISION", "Tuyên bố: Các bị cáo Hoàng Văn L, Chu Đức T, Nguyễn Duy S phạm tội “Trộm cắp tài sản”"),
    ("LEGAL_BASIS", "Áp dụng điểm b khoản 2 Điều 138; điểm b, p khoản 1, khoản 2 Điều 46; Điều 69; Điều 74 của Bộ luật Hình sự 1999"),
    ("DECISION", "Xử phạt: Bị cáo Hoàng Văn L 02 ( Hai) năm tù"),
    ("DECISION", "Xử phạt: Bị cáo Chu Đức T 02 ( Hai) năm tù"),
    ("LEGAL_BASIS", "Áp dụng khoản 1 Điều 138; điểm b, p khoản 1, khoản 2 Điều 46; điểm g khoản 1 Điều 48; Điều 50; khoản 2 Điều 51; khoản 5 Điều 60 của Bộ luật Hình sự 1999"),
    ("CRIME", "Trộm cắp tài sản", 1),
])
ANN["102835"] = ("criminal", [
    ("DECISION", "Tuyên bố: bị cáo Trần Trung D, Nguyễn Thanh T, Nguyễn Xuân Th, Phạm Tiến L, Lý Hoàng H và Nguyễn Mạnh Ph phạm tội: “Trộm cắp tài sản”"),
    ("LEGAL_BASIS", "áp dụng: điểm e khoản 2 Điều 138; điểm g khoản 1 Điều 48 (phạm tội nhiều lần, tái phạm); điểm p; o (tự thú) khoản 1 Điều 46; điểm a khoản 1 Điều 50; khoản 1 Điều 51; Điều 20; Điều 53 Bộ luật hình sự", 1),
    ("DECISION", "Xử phạt bị cáo Trần Trung D 42 (bốn mươi hai) tháng tù"),
    ("DECISION", "Xử phạt bị cáo Nguyễn Thanh T 30 (ba mươi) tháng tù"),
    ("CRIME", "Trộm cắp tài sản", 1),
])
ANN["103640"] = ("criminal", [
    ("VIOLATION_ACT", "lừa đảo chiếm đoạt tài sản bằng cách sử dụng các facebook trên kết bạn với nhiều phụ nữ Việt Nam"),
    ("DECISION", "Tuyên bố bị cáo Lê Thị B phạm tội: Lừa đảo chiếm đoạt tài sản"),
    ("LEGAL_BASIS", "Căn cứ điểm a khoản 4 Điều 139 của Bộ luật Hình sự năm 1999 và điểm a khoản 4 Điều 174 của Bộ luật Hình sự năm 2015; điểm b, p khoản 1 Điều 51; điểm g khoản 1 Điều 52; Điều 38 của Bộ luật Hình sự năm 2015 (được sửa đổi, bổ sung năm 2017)"),
    ("DECISION", "Xử phạt: Bị cáo Lê Thị B 14 (Mười bốn) năm tù"),
    ("LEGAL_BASIS", "Áp dụng Điều 48 của Bộ luật hình sự năm 2015 (được sửa đổi, bổ sung năm 2017)"),
    ("LEGAL_BASIS", "Áp dụng Điều 292 của Bộ luật Tố tụng hình sự 2015 (được sửa đổi, bổ sung năm 2017)"),
])
ANN["104821"] = ("criminal", [
    ("VIOLATION_ACT", "lập khống tổng số tiền trong Bảng thanh toán tiền lương, phụ cấp lương hàng tháng của nhà trường"),
    ("LEGAL_BASIS", "Căn cứ Điều 355, điểm c khoản1 Điều 357, điểm a khoản 1 Điều 358 Bộ luật tố tụng hình sự năm 2015"),
    ("DECISION", "Hủy phần quyết định về tội danh, hình phạt và biện pháp tư pháp đối với các bị cáo Phạm Văn T và Phạm Thị S, tại Bản án hình sự sơ thẩm số 40/2017/HS-ST ngày 28/8/2017 của Tòa án nhân dân tỉnh Đắk Lắk"),
    ("DECISION", "Giao hồ sơ cho Viện kiểm sát nhân dân tỉnh Đắk Lắk để tiến hành điều tra lại theo đúng quy định của pháp luật"),
    ("DECISION", "Chấp nhận kháng cáo xin được hưởng án treo của bị cáo Y Lim N, sửa bản án sơ thẩm đối với phần quyết định về biện pháp chấp hành hình phạt đối với bị cáo Y Lim N"),
    ("LEGAL_BASIS", "Áp dụng: Điểm c Khoản 2 Điều 360 Bộ luật hình sự năm 2015, các điểm s, t, v khoản 1 khoản 2 Điều 51, Điều 65, khoản 3 Điều 7 Nghị quyết số 41/2017/QH14 của Quốc hội"),
    ("DECISION", "Xử phạt: bị cáo Y Lim N 02 năm tù nhưng cho bị cáo được hưởng án treo"),
])

# ===== CIVIL =====
ANN["civil_100585"] = ("civil", [
    ("CASE_NUMBER", "21/2018/QĐST-HNGĐ"),
    ("LEGAL_BASIS", "Căn cứ vào Điều 48, 217, 218, 219 và khoản 2 Điều 273 của Bộ luật tố tụng dân sự"),
    ("DECISION", "Đình chỉ giải quyết vụ án dân sự thụ lý số: 60/2018/TLST-HNGĐ ngày 28 tháng 02 năm 2018 về việc “Ly hôn” giữa: Nguyên đơn: Anh Thạch B, sinh năm 1992 Bị đơn: Chị Thạch Thị Hồng Đ, sinh năm 1992 Cùng địa chỉ cư trú: ấp R, xã C, huyện S, tỉnh Vĩnh Long"),
])
ANN["civil_100052"] = ("civil", [
    ("CASE_NUMBER", "06/2018/QĐST-HC"),
    ("LEGAL_BASIS", "Căn cứ vào khoản 5 Điều 38, Điều 143 và Điều 144 của Luật tố tụng hành chính"),
    ("DECISION", "Đình chỉ giải quyết vụ án hành chính thụ lý số 36/2016/TLST-HC ngày 04 tháng 11 năm 2016 về việc “Yêu cầu hủy Giấy chứng nhận quyền sử dụng đất”"),
    ("DECISION", "Trả lại cho bà Nguyễn Thị R 200.000đ (hai trăm nghìn đồng) tạm ứng án phí đã nộp theo biên lai thu tiền số AA/2016/0000105 ngày 03/11/2016 của Cục Thi hành án dân sự tỉnh Khánh Hòa"),
])
ANN["civil_100221"] = ("civil", [
    ("CASE_NUMBER", "27/2018/QĐST- HNGĐ"),
    ("LEGAL_BASIS", "Căn cứ vào các điều 48, 217, 218, 219 và khoản 2 Điều 273 của Bộ luật tố tụng dân sự"),
    ("DECISION", "Đình chỉ giải quyết vụ án dân sự thụ lý số: 179/2018/TLST - HNGĐ , ngày 23/4/2018 về việc tranh chấp “Hôn nhân và gia đình” giữa: Nguyên đơn: Anh Trần Văn D, sinh năm 1978. Bị đơn: Chị Đào Thị H, sinh năm 1982. Cùng có HKTT: thôn Phù Lang, xã Phù Lương, huyện Quế Võ, tỉnh Bắc Ninh"),
    ("DECISION", "Hoàn trả anh Trần Văn D 300.000® tiền tạm ứng án phí tại biên lai số AA/2017/0002618 ngày 23/4/2018 của Chi cục Thi hành án dân sự huyện Quế Võ, tỉnh Bắc Ninh"),
])
ANN["civil_100586"] = ("civil", [
    ("CASE_NUMBER", "1224/2017/QĐST-HC"),
    ("LEGAL_BASIS", "Căn cứ vào Khoản 5 Điều 38, Điều 143 và Điều 144 của Luật Tố tụng hành chính"),
    ("LEGAL_BASIS", "Căn cứ Pháp lệnh án phí, lệ phí Tòa án số 10/2009/UBTVQH12 ngày 27/02/2009 và Nghị quyết 326/2016/UBTVQH14 ngày 30-12-2016 của Ủy ban Thường vụ Quốc hội về mức thu, miễn, giảm, thu, nộp, quản lý và sử dụng án phí và lệ phí tòa án"),
    ("DECISION", "Đình chỉ giải quyết vụ án hành chính thụ lý số 26/2016/TLST-HC ngày 23 tháng 3 năm 2016 về việc “Khiếu kiện quyết định hành chính”"),
    ("DECISION", "Sung công quỹ 200.000 đồng tiền tạm ứng án phí bà Lê Thị Minh T đã nộp theo Biên lai số 08268 ngày 18/01/2012 của Chi cục Thi hành án dân sự Quận H"),
])
ANN["civil_100618"] = ("civil", [
    ("CASE_NUMBER", "21/2018/QĐST-HNGĐ"),
    ("LEGAL_BASIS", "Căn cứ vào Điều 212 và Điều 213 của Bộ luật tố tụng dân sự"),
    ("LEGAL_BASIS", "Căn cứ vào Điều 55 của Luật hôn nhân và gia đình"),
    ("LEGAL_BASIS", "Căn cứ vào Điểm a Khoản 1 Điều 24; Khoản 7 Điều 26; Điểm a Khoản 5 Điều 27 Nghị quyết số 326/2016/UBTVQH 14 quy định về án phí, lệ phí Tòa án"),
    ("DECISION", "Công nhận sự thuận tình ly hôn giữa: Anh Phạm Thanh H và Chị Đường Thị H"),
])
ANN["civil_100645"] = ("civil", [
    ("CASE_NUMBER", "62/QĐST/HNGĐ"),
    ("LEGAL_BASIS", "Căn cứ vào Điều 212 và Điều 213 của Bộ luật tố tụng dân sự"),
    ("LEGAL_BASIS", "Căn cứ vào các điều 55, 58, 81, 82, 83 của Luật hôn nhân và gia đình"),
    ("DECISION", "Công nhận sự thuận tình ly hôn giữa: chị Nguyễn Thị H và anh Lý Sinh T"),
])
ANN["civil_100883"] = ("civil", [
    ("CASE_NUMBER", "41/2018/QĐST-HNGĐ"),
    ("LEGAL_BASIS", "Căn cứ vào Điều 212 và Điều 213 Bộ luật Tố tụng dân sự"),
    ("LEGAL_BASIS", "Căn cứ vào các điều 55, 58, 81, 82 và 83 Luật Hôn nhân và gia đình"),
    ("DECISION", "Công nhận sự thuận tình ly hôn giữa: Bà Nguyễn Thị Tú Q và ông Đặng Chí B"),
])
ANN["civil_101005"] = ("civil", [
    ("CASE_NUMBER", "28/2018/QĐST- DS"),
    ("LEGAL_BASIS", "Căn cứ vào Điều 212 và Điều 213 của Bộ luật tố tụng dân sự"),
    ("DECISION", "Công nhận sự thỏa thuận của các đương sự"),
])
ANN["civil_100070"] = ("civil", [
    ("CASE_NUMBER", "95/2018/QĐ-PT"),
    ("LEGAL_BASIS", "Căn cứ vào điểm a khoản 5 Điều 243 của Luật Tố tụng hành chính 2015"),
    ("DECISION", "Giữ nguyên Quyết định đình chỉ giải quyết vụ án hành chính sơ thẩm số 26/2017/QĐST-HC ngày 13 tháng 9 năm 2017, Tòa án nhân dân tỉnh Tiền Giang"),
    ("DECISION", "Bà Phan Thị D phải chịu 300.000 đồng án phí hành chính phúc thẩm"),
])
ANN["civil_100556"] = ("civil", [
    ("CASE_NUMBER", "299/2018/HNGĐ-PT"),
    ("LEGAL_BASIS", "Căn cứ Điều 228, điểm b Khoản 1 Điều 289, Khoản 2 Điều 308 Bộ luật Tố tụng dân sự"),
    ("LEGAL_BASIS", "Căn cứ Điều 33, 55, 59 Luật Hôn nhân và gia đình"),
    ("LEGAL_BASIS", "Căn cứ Điều 213, Điều 468 Bộ luật dân sự"),
    ("DECISION", "Chấp nhận một phần yêu cầu kháng cáo của ông Trần Lý S"),
    ("DECISION", "Không chấp nhận yêu cầu kháng cáo của bà Vòng A M"),
    ("DECISION", "Đình chỉ xét xử phúc thẩm đối với kháng cáo của Công ty TNHH TM Đầu tư M"),
    ("DECISION", "Sửa một phần bản án sơ thẩm về việc chia tài sản chung, như sau"),
])
ANN["civil_100043"] = ("civil", [
    ("CASE_NUMBER", "54/2018/DS-PT"),
])
ANN["civil_100062"] = ("civil", [
    ("CASE_NUMBER", "34/2017/HC-ST", 1),
    ("LEGAL_BASIS", "Căn cứ vào khoản 1 Điều 241 của Luật Tố tụng hành chính"),
    ("LEGAL_BASIS", "Áp dụng Nghị quyết số 326/2016/UBTVQH14 ngày 30-12-2016 của Uỷ ban thường vụ Quốc hội quy định về mức thu, miễn, giảm, thu, nộp, quản lý và sử dụng án phí và lệ phí Tòa án"),
    ("DECISION", "Bác yêu cầu kháng cáo của Người khởi kiện Hợp tác xã chế biến gỗ X Quảng Ngãi"),
    ("DECISION", "Giữ nguyên quyết định Bản án hành chính sơ thẩm số 34/2017/HC-ST ngày 13/9/ 2017 của Tòa án nhân dân tỉnh Quảng Ngãi"),
    ("DECISION", "Bác yêu cầu khởi kiện của Hợp tác xã chế biến gỗ X Quảng Ngãi về việc buộc Văn phòng đăng ký đất đai tỉnh Quảng Ngãi nhận lại hồ sơ đăng ký, cấp giấy chứng nhận quyền sử dụng đất, quyền sở hữu tài sản gắn liền với đất do Hợp tác xã chế biến gỗ X Quảng Ngãi đã nộp ngày 06/4/2016 đối với diện tích đất 347m2 thuộc thửa 243, tờ bản đồ số 16, phường T, thành phố Q, tỉnh Quảng Ngãi"),
])


def main() -> None:
    records = []
    for doc_id, (source, anns) in ANN.items():
        records.append(build_record(doc_id, source, anns))
    out = GOLD_DIR / "gold.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # stats
    from collections import Counter
    c = Counter()
    for r in records:
        for s in r["spans"]:
            c[s["label"]] += 1
    print(f"Wrote {len(records)} gold docs -> {out}")
    print(f"Total spans: {sum(c.values())}")
    for lab in sorted(c):
        print(f"  {lab:14s} {c[lab]}")


if __name__ == "__main__":
    main()
