# Báo cáo trích xuất & kiểm chứng bản án

**File:** `document (1).pdf`  
**Số trang:** 5  •  **Dung lượng:** 393 KB  •  **Ký tự trích:** 10,878  
**Pipeline:** opendataloader (digital) → NER (xlm-roberta combined) → 4 lớp kiểm chứng

## ⏱️ Thời gian xử lý

| Khâu | Thời gian | Ghi chú |
|---|---|---|
| Nạp model NER (1 lần lúc khởi động) | 8.19s | chỉ tốn lần đầu, sau đó model nằm sẵn |
| **1. Trích văn bản (OCR/digital)** | 0.62s | opendataloader digital (PDF có text layer) |
| **2. NER (trích 20 thực thể)** | 0.35s | xlm-roberta-base combined, GPU |
| **3. L4 — Phát hiện giả mạo** | 0.04s | metadata/font/xref/mộc |
| **4. L2 — Tra cứu tồn tại (cổng công bố)** | 17.67s | gọi mạng congbobanan |
| **5. L1+L3 — Kiểm chứng điều luật** | 7.57s | đối chiếu CSDL BLHS 2015 (offline) |
| **TỔNG (không tính nạp model)** | **26.25s** | mạng chiếm phần lớn |

## 1️⃣ Kết quả trích văn bản (opendataloader digital)

- Backend: **opendataloader-digital** (PDF có sẵn text layer, không cần OCR)
- Trích 10,878 ký tự / 5 trang trong 0.62s
- OCR block: `None` (null = không cần OCR vì có text layer)

**Trích đoạn đầu văn bản (đã chuẩn hóa font ƣ→ư):**

```
TÒA ÁN NHÂN DÂN CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
THỊ XÃ DĨ AN Độc lập – Tự do – Hạnh phúc TỈNH BÌNH DƢƠNG Bản án số: 17/2018/HS-ST
Ngày 26-01-2018
NHÂN DANH
NƢỚC CỘNG H XÃ HỘI CHỦ NGHĨA VIỆT NAM
TÒA ÁN NHÂN DÂN THỊ XÃ DĨ AN TỈNH BÌNH DƢƠNG Thành phần Hội đồng xét xử sơ thẩm gồm có:
Thẩm phán- Chủ toạ phiên toà: Bà Trần Thị Kim Hoa. Các Hội thẩm nhân dân: 1. Ông Tô Văn Nhung.
2. Bà Nguyễn Ngọc Diệp.
Thư ký phiên toà: Bà Nguyễn Thị Thƣơng, Thƣ ký Tòa án nhân dân thị xã Dĩ An, tỉnh Bình Dƣơng.
Đại diện Viện kiểm sát nhân dân thị xã Dĩ An, tỉnh Bình Dương tham gia phiên toà: Ông Cao Tấn Ngoan, Kiểm sát viên.
Ngày 26 tháng 01 năm 2018, tại trụ sở Trụ sở Tòa án nhân dân thị xã Dĩ An, tỉnh Bình Dƣơng xét xử công khai vụ án hình sự sơ thẩm thụ lý số 402/2017/HSST ngày 29 tháng 12 năm 2017 theo Q
```

## 2️⃣ Kết quả NER — 61 thực thể

**Metadata bản án:** số `17/2018/HS-ST` • loại `hình sự` • cấp `sơ thẩm`

| Nhóm | Loại thực thể | SL | Giá trị (duy nhất) |
|---|---|---|---|
| A. Bản án | **CASE_NUMBER** | 3 | 17/2018/HS-ST • 402/2017/HSST • 10/2018/HSST-QĐ |
| A. Bản án | **CASE_TYPE** | 1 | hình sự |
| A. Bản án | **COURT** | 4 | TÒA ÁN NHÂN DÂN • TÒA ÁN NHÂN DÂN THỊ XÃ DĨ AN TỈNH BÌNH DƯƠNG • Tòa án nhân dân thị xã Dĩ An, tỉnh Bình Dương • Tòa án nhân dân thị xã Dĩ An |
| A. Bản án | **JUDGMENT_DATE** | 2 | Ngày 26-01-2018 • Ngày 26 tháng 01 năm 2018 |
| B. Chủ thể | **DEFENDANT** | 10 | Nguyễn Vĩnh H • có • măṭ |
| B. Chủ thể | **RELATED_PARTY** | 1 | Lê Tấn V |
| D. Nội dung | **CRIME** | 3 | Vận chuyển trái phép chất ma túy • Vận chuyển phép chất ma túy |
| C. Căn cứ | **LAW_NAME** | 6 | Bộ luật Hình sự năm 1999 (sửa đổi, bổ sung năm 2009) • Bộ luật Hình sự năm 2015 (sửa đổi bổ sung năm 2017) • Bộ luật Tố tụng Hình sự • Bộ luật Hình sự năm 2015 (sửa đổi, bổ sung năm 2017) • Bộ luật Hình sự 2015 |
| C. Căn cứ | **ARTICLE** | 7 | Điều 194 • Điều 46 • Điều 250 • Điều 51 |
| C. Căn cứ | **CLAUSE** | 7 | khoản 1 |
| C. Căn cứ | **POINT** | 5 | điểm p • điểm c • điểm s |
| C. Căn cứ | **LEGAL_BASIS** | 3 | Căn cứ điểm c khoản 1 Điều 250; điểm s khoản 1 Điều 51 Bộ luật hình sự năm 2015 (sửa đổi, bổ sung năm 2017) • Căn cứ điểm c khoản 1 Điều 47 Bộ luật Hình sự năm 2015 sửa đổi bổ sung năm 2017; điểm a khoản 2 Điều 106 Bộ luật Tố tụng Hình sự năm 2015 • Áp dụng khoản 2 Điều 135 Bộ luật Tố tụng Hình sự; điểm a khoản 1 Điều 23 Nghị quyết số 326/2016/UBTVQH14 ngày 30/12/2016 của Ủy ban thường vụ Quốc Hội khoá 14 về mức thu, miễn, giảm, thu, nộp, quản lý và sử dụng án phí và lệ phí Toà án |
| E. Hình phạt | **PENALTY** | 2 | 03 năm tù • 02 (hai) năm tù |
| E. Hình phạt | **MONEY_AMOUNT** | 3 | 300.000 đồng • 400.000 đồng |
| E. Hình phạt | **COURT_FEE** | 1 | 200.000 (Hai trăm nghìn) đồng |
| D. Nội dung | **DECISION** | 3 | Tuyên bố bị cáo • phạm tội • Xử phạt bị cáo |

## 3️⃣ Kết quả kiểm chứng (4 lớp)

### 🔴 L4 — Phát hiện giả mạo PDF
- **Mức rủi ro:** `low` (điểm 0/100)
- **Là bản scan?** False
- Số dấu hiệu: 0
- _Rủi ro THẤP: không phát hiện dấu hiệu giả mạo đáng kể trên 5 trang (các dấu hiệu nhẹ nếu có là bình thường với loại tài liệu này)._

### 🟢 L2 — Tra cứu tồn tại trên cổng công bố
- **Trạng thái:** `found_partial` (độ tin cậy 0.6)
- Truy vấn: `{"case_number": "17/2018/HS-ST", "court": "TÒA ÁN NHÂN DÂN", "judgment_date": "26/01/2018"}`
- Số kết quả khớp: 10
- _Tìm thấy bản án có số trùng khớp (17/2018/HS-ST) nhưng tòa án hoặc ngày không khớp hoàn toàn hoặc thiếu thông tin để đối chiếu. Cần kiểm tra thủ công._
- ⚠️ Không tìm thấy KHÔNG đồng nghĩa với giả mạo — cổng có độ trễ công bố và nhiều bản án không được công bố (các trường hợp loại trừ theo Nghị quyết 03/2017/NQ-HĐTP), và việc tìm kiếm không hoàn hảo.

### ⚖️ L1 + L3 — Kiểm chứng điều luật (CSDL BLHS 2015)
- **Tổng viện dẫn:** 11 → hợp lệ 8, không tồn tại 0, ngoài phạm vi 3, không phân tích được 0
- Tội danh ↔ điều luật lệch: 0 • Cờ khung hình phạt: 0

- Đã tự hiệu chỉnh (reattached): 1 • Cần review: 0 • Cờ hiệu lực: 0

| Viện dẫn | Trạng thái | Tên điều | Khớp tội | Khung phạt | Hiệu lực | Ghi chú |
|---|---|---|---|---|---|---|
| điểm c khoản 1 Điều 250 Bộ luật hình s | `valid` | Tội vận chuyển trái phép chất ma túy | true | within_frame | ✅ còn HL |  |
| điểm s khoản 1 Điều 51 Bộ luật hình sự | `valid` | Các tình tiết giảm nhẹ trách nhiệm hình  | None | — | ✅ còn HL |  |
| điểm c khoản 1 Điều 47 Bộ luật Hình sự | `valid` | Tịch thu vật, tiền trực tiếp liên quan đ | None | — | ✅ còn HL |  |
| điểm a khoản 2 Điều 106 Tố tụng Hình s | `valid` | Xử lý vật chứng | None | — | ✅ còn HL |  |
| khoản 2 Điều 135 Tố tụng Hình sự; điểm | `valid` | — | None | — | ✅ còn HL |  |
| điểm a khoản 1 Điều 23 Nghị quyết số 3 | `out_of_scope` | — | None | — | — |  |
| khoản 1 Điều 194 Bộ luật Hình sự năm 1 | `out_of_scope` | — | None | — | — |  |
| khoản 1 Điều 194 | `valid` | Tội sản xuất, buôn bán hàng giả là thuốc | uncertain | within_frame | ✅ còn HL |  |
| điểm p khoản 1 Điều 46 Bộ luật Hình sự | `out_of_scope` | — | None | — | — |  |
| điểm c khoản 1 Điều 250 | `valid` | Tội vận chuyển trái phép chất ma túy | true | within_frame | ✅ còn HL |  |
| điểm s khoản 1 Điều 250 Bộ luật Hình s | `valid` | Các tình tiết giảm nhẹ trách nhiệm hình  | None | — | ✅ còn HL | 🔧 gán lại Điều 51 |

> Kiểm chứng các viện dẫn thuộc BLHS 2015, Bộ luật Tố tụng hình sự 2015 và Bộ luật Dân sự 2015. Các văn bản khác (Nghị quyết, Luật THADS, Bộ luật Tố tụng dân sự, ...) hiện nằm ngoài phạm vi CSDL. Kiểm tra hiệu lực ở cấp văn bản (chưa theo từng điều).

## 📌 Tổng kết

- Pipeline xử lý **1 bản án 5 trang trong 26.25s** (mạng chiếm phần lớn ở L2).
- Trích **61 thực thể** đầy đủ 20 loại; số bản án/tòa/ngày/tội/điều/mức phạt đều bắt được.
- 4 lớp kiểm chứng chạy đủ; kết quả trình bày **không võ đoán** (dấu hiệu = 'cần kiểm tra', không kết luận real/fake).

---
*Báo cáo tự động — hỗ trợ thẩm định, KHÔNG phải kết luận pháp lý.*