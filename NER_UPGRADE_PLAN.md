# Kế hoạch nâng cấp NER v2 — Hành vi · Tình tiết · Quan hệ

> Module: `legal_ner` (Phân tích bản án) — Ưu tiên #1 của roadmap.
> Mục tiêu tổng: chuyển NER từ **"danh sách thực thể tĩnh"** sang **"bức tranh sự
> kiện có quan hệ"** của phiên tòa, để tầng phân tích đánh giá *"mức án có hợp lý
> với tình tiết không"* **có căn cứ thật**.

Hiện trạng: 20 loại thực thể tĩnh (ai, tội gì, điều nào, phạt bao nhiêu), model
`xlm-roberta-base`, sơ đồ BIO, huấn luyện bằng **weak-labeling (regex)**.
Điểm mù: hành vi/tình tiết "động" của phiên tòa, và **quan hệ** giữa các thực thể
(vụ nhiều bị cáo/nhiều tội).

---

## TRỤ CỘT A — Bổ sung các loại thực thể "động"

Thêm các nhãn mới (đặt tên UPPER_SNAKE, đồng bộ `config.py`):

| Nhóm | Nhãn đề xuất | Ý nghĩa | Ví dụ |
|---|---|---|---|
| **Người tiến hành tố tụng** | `JUDGE` | Thẩm phán / chủ tọa | "Thẩm phán - Chủ tọa: Bà Trần Thị K" |
| | `ASSESSOR` | Hội thẩm nhân dân | "Các Hội thẩm nhân dân: Ông…" |
| | `PROSECUTOR` | Kiểm sát viên | "Kiểm sát viên: Ông Cao Tấn N" |
| | `CLERK` | Thư ký phiên tòa | "Thư ký: Bà Nguyễn Thị T" |
| | `LAWYER` | Luật sư / người bào chữa | "Người bào chữa: Luật sư…" |
| | `WITNESS` | Người làm chứng | "Người làm chứng: Anh Đặng Phú V" |
| **Hành vi / trạng thái tại tòa** | `COURT_BEHAVIOR` | Trạng thái tham gia, hành vi tố tụng | "có mặt", "vắng mặt", "kháng cáo", "xin giảm nhẹ", "thay đổi lời khai" |
| **Tình tiết** | `MITIGATING_FACTOR` | Tình tiết **giảm nhẹ** (Điều 51 BLHS) | "thành khẩn khai báo", "ăn năn hối cải", "phạm tội lần đầu", "tự nguyện bồi thường" |
| | `AGGRAVATING_FACTOR` | Tình tiết **tăng nặng** (Điều 52 BLHS) | "tái phạm nguy hiểm", "phạm tội có tổ chức", "phạm tội nhiều lần" |
| **Hành vi phạm tội** | `CRIMINAL_ACT` | Diễn biến hành vi (nâng cấp `VIOLATION_ACT` đang thưa) | "mua 300.000đ một liều heroin, giấu trong xe…" |
| **Định lượng & tang vật** | `QUANTITY` | Khối lượng/trọng lượng/số lượng | "0,1488 gam heroin", "trọng lượng 0,1072g" |
| | `EVIDENCE_ITEM` | Tang vật / vật chứng | "01 xe mô tô Wave", "01 điện thoại Mobell" |

**Lưu ý phân định (tránh chồng nhãn):**
- "thành khẩn khai báo / ăn năn hối cải" → **`MITIGATING_FACTOR`** (vì là tình tiết
  pháp lý), KHÔNG phải `COURT_BEHAVIOR`.
- "có mặt / vắng mặt / kháng cáo / xin giảm nhẹ" → **`COURT_BEHAVIOR`** (trạng thái
  tố tụng, không phải tình tiết định khung).
- Quy tắc ưu tiên khi chồng: tình tiết (Điều 51/52) > hành vi tố tụng chung.

**Vì sao quan trọng:** `MITIGATING_FACTOR` / `AGGRAVATING_FACTOR` là **đầu vào trực
tiếp** cho tầng phân tích "mức án hợp lý" — hiện đang thiếu nên LLM phải suy đoán.

---

## TRỤ CỘT B — Đổi cách huấn luyện: gán nhãn thủ công + Active Learning

Weak-labeling (regex) chỉ bắt được khuôn mẫu cứng → bỏ sót cách diễn đạt đa dạng
(vd `VIOLATION_ACT` hiện rất thưa). Chuyển sang **dữ liệu vàng do người gán**, kết
hợp **active learning** để giảm công sức:

**Quy trình vòng lặp active learning:**
1. Dùng model hiện tại **dự đoán (gợi ý nhãn)** trên bản án chưa gán.
2. Người gán **chỉ sửa chỗ sai** (nhanh hơn gán từ đầu) — qua `gold/annotate.py`.
3. Ưu tiên chọn bản án model **"không chắc"** (uncertainty sampling) hoặc chứa nhãn
   mới còn ít dữ liệu → gán trước, hiệu quả cao nhất.
4. Thêm vào tập gold → retrain → lặp lại.

**Chiến lược lai (khuyến nghị):**
- Với tình tiết **Điều 51/52** (danh mục đóng theo luật): vẫn dùng **gazetteer +
  pattern** để seed nhãn nhanh, rồi người chỉ kiểm/sửa.
- Với hành vi tự do (`CRIMINAL_ACT`, `COURT_BEHAVIOR`): **gán tay là chính** vì diễn
  đạt đa dạng, regex không phủ nổi.

**Hạ tầng gán nhãn:**
- Mở rộng `gold/annotate.py` để (a) nạp gợi ý của model, (b) hiển thị nhãn mới,
  (c) ghi `gold.jsonl` chuẩn BIO.
- (Tùy chọn) dùng công cụ ngoài như **Label Studio / doccano** cho UX gán nhãn tốt.

---

## TRỤ CỘT C — Nâng từ NER lên Trích xuất Quan hệ / Sự kiện (Relation/Event Extraction)

NER hiện trả **danh sách phẳng** → không biết hình phạt nào của bị cáo nào, điều
luật nào cho tội nào. Vụ **nhiều bị cáo / nhiều tội** là điểm mù hoàn toàn.

**Mục tiêu:** xuất ra **bản ghi theo từng bị cáo**:
```
Bị cáo A ─┬─ tội danh: "Vận chuyển trái phép chất ma túy"
          ├─ điều luật: điểm c khoản 1 Điều 250 BLHS 2015
          ├─ hình phạt: 02 năm tù
          ├─ tình tiết giảm nhẹ: [thành khẩn khai báo, ăn năn hối cải]
          └─ tình tiết tăng nặng: []
```

**Các quan hệ cần trích:**
- `DEFENDANT` —*thực hiện*→ `CRIME`
- `CRIME` —*theo*→ `ARTICLE`/`CLAUSE`/`POINT`/`LAW_NAME`
- `DEFENDANT` —*bị tuyên*→ `PENALTY`
- `DEFENDANT` —*có*→ `MITIGATING_FACTOR` / `AGGRAVATING_FACTOR`
- `CRIME` —*gây ra bởi*→ `CRIMINAL_ACT`

**3 hướng triển khai (từ rẻ → mạnh):**
1. **Heuristic theo cấu trúc** (làm trước): gom thực thể theo **đoạn/mục** (mỗi bị
   cáo thường có một khối "Xử phạt bị cáo … …") + theo khoảng cách vị trí. Đủ tốt
   cho vụ 1 bị cáo và nhiều vụ đơn giản.
2. **LLM structuring** (trung gian, tận dụng sẵn Module 1): đưa danh sách thực thể
   + đoạn văn cho LLM, yêu cầu ghép thành JSON theo từng bị cáo. Nhanh làm, linh
   hoạt, không cần train.
3. **Model trích quan hệ** (mạnh nhất, về sau): huấn luyện một mô hình relation
   classification trên cặp thực thể. Cần dữ liệu gán quan hệ.

---

## TODO theo giai đoạn (checklist)

### GĐ 0 — Thiết kế & dữ liệu nền
- [ ] Chốt danh sách nhãn mới ở Trụ cột A + quy tắc phân định chồng nhãn.
- [ ] Trích nguyên văn **Điều 51 & Điều 52 BLHS 2015** từ `data/lawdb/blhs2015.db`
      → xây **gazetteer tình tiết** (giảm nhẹ / tăng nặng).
- [ ] Thu thập thêm bản án đa dạng (nhiều loại tội, nhiều bị cáo) qua `crawler/`.

### GĐ 1 — Mở rộng nhãn & seed dữ liệu
- [ ] Thêm nhãn mới vào [`config.py`](config.py) (`ENTITY_TYPES`).
- [ ] Viết gazetteer + pattern trong [`labeling/patterns.py`](labeling/patterns.py)
      cho tình tiết (Điều 51/52) + `JUDGE/PROSECUTOR/CLERK/ASSESSOR/WITNESS` (mấy
      cái này có từ neo rõ: "Thẩm phán", "Kiểm sát viên"… → regex tốt).
- [ ] Chạy [`labeling/weak_label.py`](labeling/weak_label.py) → sinh nhãn seed.

### GĐ 2 — Gán nhãn vàng (active learning)
- [ ] Nâng cấp [`gold/annotate.py`](gold/annotate.py): nạp gợi ý model + nhãn mới.
- [ ] Gán/sửa tập gold ~80–150 bản án, tập trung `CRIMINAL_ACT`, `COURT_BEHAVIOR`,
      tình tiết (các cách diễn đạt không theo khuôn).
- [ ] Cập nhật [`training/eval_gold.py`](training/eval_gold.py): F1 **theo từng nhãn**.

### GĐ 3 — Huấn luyện lại
- [ ] Retrain `xlm-roberta-base` schema mới ([`training/train.py`](training/train.py)).
- [ ] **Chống hồi quy:** F1 nhãn cũ (bị cáo/điều luật/hình phạt) không được giảm.
- [ ] Lưu checkpoint mới `data/models/.../final` (thay sau khi đạt chỉ tiêu).

### GĐ 4 — Chuẩn hóa tình tiết + Quan hệ
- [ ] Normalizer: ánh xạ tình tiết → đúng **điểm/khoản Điều 51/52** + phân loại
      "theo luật" vs "tòa tự nhận".
- [ ] Trích quan hệ theo **Hướng 1 (heuristic)** rồi **Hướng 2 (LLM structuring)**
      → bản ghi JSON theo từng bị cáo.

### GĐ 5 — Tích hợp API & Frontend
- [ ] Nhãn mới tự chảy qua `/extract`, `/verify` (thêm vào `ENTITY_TYPES` + Entity).
- [ ] Cập nhật `lib/types.ts` `ENTITY_LABELS_VI` + màu/nhóm `EntityHighlight.tsx`.
- [ ] UI: **"Bảng tình tiết"** (giảm nhẹ ✅ / tăng nặng ⚠) + **thẻ theo từng bị cáo**.

### GĐ 6 — Nâng cấp phân tích (dùng tình tiết)
- [ ] Đưa danh sách tình tiết + bản ghi theo bị cáo vào **prompt phân tích chuyên
      sâu** → đánh giá "mức án tương xứng tình tiết" có căn cứ.
- [ ] (Tùy chọn) **Lớp thẩm định 5 — Đối chiếu tình tiết:** tình tiết tòa áp dụng
      có hợp lệ theo Điều 51/52 + ảnh hưởng đúng hướng đến mức án.

### GĐ 7 — Đánh giá & tài liệu
- [ ] Test end-to-end trên bản án nhiều tình tiết / nhiều bị cáo; đo accuracy.
- [ ] Cập nhật `INTEGRATION_VN.md` + báo cáo/slide với chỉ số NER v2.

---

## Tiêu chí nghiệm thu (gợi ý)
- F1 ≥ 0.80 cho `MITIGATING_FACTOR` / `AGGRAVATING_FACTOR` trên tập gold.
- Không hồi quy: F1 các nhãn cũ giảm ≤ 1–2 điểm.
- Vụ nhiều bị cáo: ghép đúng (bị cáo ↔ tội ↔ hình phạt) ≥ 90% trên tập kiểm thử.
- Tầng phân tích chuyên sâu nêu được tình tiết cụ thể đã trích (không suy đoán).

## Thứ tự ưu tiên thực hiện (vòng giá trị)
1. **Tình tiết giảm nhẹ/tăng nặng** (GĐ 1–3, dùng gazetteer Điều 51/52) — giá trị
   cao nhất, làm nhanh nhất vì danh mục đóng.
2. **Đưa vào phân tích** (GĐ 6) — thấy tác dụng ngay.
3. **Quan hệ theo bị cáo** (GĐ 4, Hướng 2 LLM) — mở khóa vụ nhiều bị cáo.
4. Mở rộng `CRIMINAL_ACT`, `COURT_BEHAVIOR`, người tiến hành tố tụng (GĐ 2 gán tay).

## Tham chiếu file
- Cấu hình nhãn: `config.py` · Pattern weak-label: `labeling/patterns.py`,
  `labeling/weak_label.py` · Gán nhãn vàng: `gold/annotate.py`, `gold/gold.jsonl`
- Huấn luyện/đánh giá: `training/train.py`, `training/eval_gold.py`, `training/infer.py`
- CSDL luật (Điều 51/52): `data/lawdb/blhs2015.db`
- API: `api/service.py` (`_infer_and_group`), `api/schemas.py`
- Frontend: `legal-chat-ui/lib/types.ts`, `components/EntityHighlight.tsx`,
  `components/BanAnAnalyzer.tsx`

## Rủi ro & lưu ý
- **Chồng nhãn** tình tiết ↔ hành vi tố tụng → cần quy tắc ưu tiên rõ (đã nêu).
- **Mất cân bằng dữ liệu** (tăng nặng hiếm hơn giảm nhẹ) → cân nhắc oversampling.
- **Lỗi OCR** ở bản scan lan vào NER → ưu tiên test trên bản có lớp chữ trước.
- Đổi `ENTITY_TYPES` làm **đổi số nhãn** → BẮT BUỘC retrain (không load model cũ
  với schema mới được).
