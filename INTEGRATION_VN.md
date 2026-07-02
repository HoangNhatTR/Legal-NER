# Module 2 — Legal NER (Phân tích bản án) — Ghi chú tích hợp vào ProjectGenAI_2

Module này được **clone từ** `https://github.com/minhquoctran2604/legal-docs`
(thư mục `legal_ner/`) và gộp vào dự án Module 1 (RAG pháp luật) theo kiểu
**service riêng, đặt chung repo**.

## Nó làm gì
Nhận **PDF bản án** → trả về **thực thể có cấu trúc** (NER 20 loại: số bản án,
tòa án, bị cáo, tội danh, điều/khoản/điểm, hình phạt…) + **xác minh sơ bộ**:
- Layer 4 — dấu hiệu chỉnh sửa PDF (forgery, dựa metadata)
- Layer 2 — tra cứu tồn tại trên cổng `congbobanan.toaan.gov.vn` (cần mạng)
- Layer 1/3 — đối chiếu viện dẫn & khung hình phạt với BLHS 2015 (offline)

Model: `xlm-roberta-base` fine-tuned **v3r-full** (31 nhãn, full fine-tune,
micro-F1 ~0.970 — xem `NER_MODEL_REPORT.md`), nằm ở
`data/models/legal-ner-v3r-full/final/model.safetensors` (1.1GB). Bộ nhãn được
đọc trực tiếp từ `config.json` của checkpoint nên giải mã luôn khớp model. Suy
luận **section-aware** (tách theo mục bản án) — xem `corpus/sectioner.py`.
(Model cũ `legal-ner-combined` 20 nhãn / micro-F1 0.925 đã được thay thế.)

## Quan hệ với Module 1
- **Service tách biệt.** Module 1 (RAG) chạy port **8000**; module này chạy
  port **8100**. Hai tiến trình độc lập, không chia sẻ code.
- Dùng **chung venv** `Project2\Chatbot` với Module 1.
- Import **tương đối** (`from api import ...`, `from corpus ...`) → **bắt buộc
  chạy từ trong thư mục `legal_ner/`**.

## Chạy trên Windows (đường đã kiểm chứng)
```powershell
cd ProjectGenAI_2\legal_ner
.\run_api_windows.ps1            # -> http://127.0.0.1:8100/docs
```
Gọi thử (PDF có text-layer — **truyền `extractor=pymupdf`**):
```powershell
curl.exe -F "file=@../bản án demo/document (1).pdf" "http://localhost:8100/extract?extractor=pymupdf"
curl.exe -F "file=@../bản án demo/document (1).pdf" "http://localhost:8100/verify?extractor=pymupdf&check_existence=false"
```

## opendataloader-pdf — ĐÃ BẬT trên Windows (2026-06-13)
- Extractor mặc định **`opendataloader`** đã chạy: đã cài JDK 21 portable
  (`C:\Users\HP\jdk\jdk-21.0.11+10` — Java hệ thống là 8, quá cũ) + `pip install
  opendataloader-pdf` (2.4.7). `run_api_windows.ps1` tự set
  `LEGAL_NER_ODL_JAVA_HOME` trỏ JDK này; `/health` báo `java_ok: true`.
- Verify: `/extract?extractor=opendataloader` → **61 thực thể** (ODL) vs 59 (pymupdf)
  trên demo, case_meta đúng, warning "text extracted via opendataloader-digital".
- Frontend (lib/legalNer.ts) mặc định gọi `extractor=opendataloader`; `pymupdf`
  là fallback nếu Java lỗi.

## OCR bản án SCAN/ảnh — ĐÃ BẬT + TỰ ĐỘNG (2026-06-14)

**Tự động:** frontend mặc định `allow_ocr=true`. PDF có lớp chữ vẫn dùng
opendataloader (nhanh, OCR không chạy); chỉ bản SCAN mới kích hoạt OCR. `_ocr_scan`
(service.py) là scan-path chung cho cả 2 extractor.

**2 backend OCR (tự chọn theo env):**
1. **OCR.space (đám mây, NHANH — mặc định nếu có key)** — `api/ocr_cloud.py`:
   render trang -> JPEG -> gọi OCR.space **Engine 2** SONG SONG. Đặt env
   `LEGAL_NER_OCRSPACE_KEY` (free 25k lượt/tháng tại https://ocr.space/ocrapi;
   `run_api_windows.ps1` để sẵn key demo "helloworld" — nên thay bằng key riêng).
   ✅ Verify: scan 10 trang = **49s** (vs EasyOCR 494s), case `20/2026/HS-ST` +
   "sơ thẩm" ĐÚNG (EasyOCR sai thành HS-SI), tiếng Việt chuẩn dấu thanh.
2. **EasyOCR (offline, fallback)** — `api/ocr_easyocr.py`: PyMuPDF render -> EasyOCR
   (vi,en). Không cần key/mạng nhưng CHẬM (~50s/trang CPU) + vài lỗi glyph.
   Tự dùng khi KHÔNG có OCR.space key, hoặc OCR.space lỗi/hết hạn mức.

⚠ easyocr nâng torch 2.11->2.12 (NER vẫn OK). OCR vẫn là request đồng bộ (browser
fetch không timeout); muốn UX tốt hơn dùng async job queue `/jobs/verify`.

## Dependencies đã thêm vào `requirements.txt` của Module 1
`PyMuPDF` (fitz), `transformers`, `pikepdf` (forgery). Hai gói `datasets`,
`seqeval` chỉ cần khi **train/eval lại** model NER (không cần để chạy API).

## Dữ liệu lớn (không commit)
`data/models/**` (model 1.1GB) đã bị `.gitignore` của Module 1 bỏ qua (rule
`models/`). lawdb (`data/lawdb/*.db`, ~11MB) vẫn nằm trong thư mục để tra cứu
BLHS 2015 offline. Lấy lại model gốc bằng `git lfs pull` từ repo nguồn nếu mất.

## Kiểm thử đã chạy (2026-06-13, demo `document (1).pdf`)
- `/extract` (pymupdf): 5 trang, 10853 ký tự, **59 thực thể / 16 loại**;
  case_meta = `17/2018/HS-ST · hình sự · sơ thẩm` ✓
- `/verify`: forgery=low(0); citation 11 viện dẫn → 8 hợp lệ, 3 ngoài phạm vi ✓

## Phân tích so sánh với LLM+RAG — `POST /analyze` (ĐÃ XONG)

Endpoint mới `POST /analyze` (api/analyze.py) cho **báo cáo đối chiếu** bản án với
kho luật lớn của Module 1 — KHÔNG đi qua `validate_document` (tool đó là bộ chấm
thể thức, hay báo nhầm "thiếu yếu tố" + bỏ qua câu hỏi).

- Input (JSON): `{case_meta, entities_grouped, rag_model}` (lấy từ /extract hoặc
  /verify — KHÔNG upload lại PDF).
- Cách làm: dựng **1 câu hỏi tình huống tư vấn NGẮN GỌN** từ tội danh + trích dẫn
  áp dụng (điểm/khoản/Điều của tội danh) + mức tuyên, rồi hỏi Module 1
  `/v1/chat/completions`. Câu ngắn dạng tư vấn → router đưa vào `legal` (retrieve),
  LLM trả lời bám nội dung luật từ corpus. (Câu dài/dày "Căn cứ…" bị router đẩy
  sang validate → tránh.)
- Output: `{cited_articles, law_lookup, analysis}` (analysis = markdown báo cáo).
- Env: `LEGAL_NER_RAG_URL` (mặc định `http://localhost:8000`), `LEGAL_NER_RAG_KEY`.
- ⚠ Cần **Module 1 (RAG :8000) đang chạy**. Mỗi lần ~1-3 phút (1 lượt gọi LLM).

**Bản STREAM `POST /analyze/stream`** (UX): trả `text/event-stream` (SSE kiểu
OpenAI) — frontend hiện đáp án dần thay vì chờ. Phát 1 `status` chunk ngay đầu
("Đang tra cứu…") để báo còn sống. ⚠ RAG full tốn ~100s cho router+truy xuất
TRƯỚC khi sinh chữ nên token về gần cuối; heartbeat + spinner đỡ phần chờ.

**Frontend (tab Phân tích bản án):**
- Nút "Phân tích & so sánh" gọi `/analyze/stream` (rag_model=legal-ai-full).
- "Văn bản gốc (tô màu thực thể)": `EntityHighlight.tsx` tô 20 loại thực thể theo
  offset — giải thích NER, tăng tin cậy.
- "Xuất báo cáo (.md)": tải báo cáo Markdown (vụ án + thực thể + thẩm định + AI).

## Phân tích CHUYÊN SÂU bản chất hành vi — `POST /analyze/deep/stream`

Khác tầng "đối chiếu pháp lý" (hình thức: citation/khung). Tầng này đánh giá BẢN
CHẤT: **định tội** đúng chưa (vd "vận chuyển" vs "tàng trữ" ma túy), **mức án hợp
lý** không, **điểm bất hợp lý**. Nút amber **"Phân tích chuyên sâu bản chất hành vi"**.

- 2 bước: (1) tóm tắt diễn biến hành vi từ mục "NỘI DUNG VỤ ÁN"; (2) hỏi RAG.
- ⚠ Router Module 1 (LLM) **không tất định** — câu hỏi pháp lý chuyên sâu dễ bị
  hiểu nhầm thành "kiểm tra văn bản" (~50%). Khắc phục: hỏi bằng **ngôn ngữ tư vấn
  đời thường** + **guard retry tối đa 3 lần**. Đã test **3/3** thành công, ~40-72s.
- Là phân tích **HỖ TRỢ/diễn giải** (LLM có thể sai về định tội) — có disclaimer
  trong UI; KHÁC phần "hình thức" vốn kiểm chứng được.

## Endpoints Module 2 (tóm tắt)
`/extract` `/verify` (NER + thẩm định) · `/analyze` (đối chiếu, sync+guard) ·
`/analyze/stream` (đối chiếu, token-stream) · `/analyze/deep/stream` (chuyên sâu,
guard+retry) · `/jobs/*` (async OCR). Frontend: EntityHighlight + export .md.
- Frontend: nút "Phân tích & so sánh" trong tab Phân tích bản án gọi endpoint này
  với `rag_model=legal-ai-full` (retrieval điều luật chính xác hơn graph).

**Kiểm thử (demo, mode full):** routing đúng (không validate/chào); khung điểm c
khoản 1 Điều 250 = **02–07 năm** (nguồn corpus 100/2015/QH13); mức 02 năm tù nằm
trong khung; tội danh khớp Điều 250. ✓

## Kiểm thử đã chạy (2026-06-13)
- `/extract` **opendataloader**: 5 trang, 10852 ký tự, **61 thực thể** (pymupdf 59);
  case_meta = `17/2018/HS-ST · hình sự · sơ thẩm` ✓
- `/verify`: forgery=low(0); citation 11 viện dẫn → 8 hợp lệ, 3 ngoài phạm vi ✓
- `/analyze` (legal-ai-full): báo cáo đối chiếu đúng khung/tội danh, có trích dẫn ✓
