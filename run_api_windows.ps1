# Khởi chạy Module 2 (Legal NER — phân tích bản án) trên Windows.
#
# Module này là SERVICE RIÊNG, chạy song song với Module 1 (RAG, port 8000).
# Mặc định trên Windows dùng extractor=pymupdf (chỉ cần PyMuPDF, KHÔNG cần Java/OCR).
#
# Dùng:
#   cd ProjectGenAI_2\legal_ner
#   .\run_api_windows.ps1                 # port 8100
#   .\run_api_windows.ps1 -Port 8200      # đổi port
#
# Lưu ý:
#  - Phải chạy TỪ thư mục legal_ner\ (import tương đối: api, corpus, training, verify).
#  - Backend opendataloader (mặc định trong code) cần JDK 11+ và OCR server — KHÔNG có
#    sẵn trên Windows. Khi gọi API hãy truyền ?extractor=pymupdf cho PDF có text-layer.
#  - Bản án scan (ảnh) cần OCR -> chưa hỗ trợ trên Windows (cần cấu hình thêm).

param(
    [int]$Port = 8100
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

# Python của venv dùng chung với Module 1 (Project2\Chatbot)
$py = Join-Path $here "..\..\Chatbot\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Khong tim thay venv python: $py"
}

$env:PYTHONIOENCODING = "utf-8"

# --- opendataloader-pdf (extractor mac dinh): can JDK 11+ ---
# JDK 21 portable da cai o thu muc user (Java he thong la 8, qua cu).
$jdk = "C:\Users\HP\jdk\jdk-21.0.11+10"
if (Test-Path $jdk) {
    $env:LEGAL_NER_ODL_JAVA_HOME = $jdk
    $env:JAVA_HOME = $jdk
    $env:PATH = "$jdk\bin;$env:PATH"
    Write-Host "[legal_ner] JAVA_HOME = $jdk (opendataloader digital)" -ForegroundColor Green
} else {
    Write-Host "[legal_ner] CANH BAO: khong thay JDK 21 tai $jdk -> opendataloader se loi, dung ?extractor=pymupdf" -ForegroundColor Yellow
}
# Chua dung OCR server hybrid cho ban scan -> tat auto-start.
$env:LEGAL_NER_ODL_HYBRID_AUTOSTART = "0"

# --- OCR ban an SCAN/anh ---
# Mac dinh dung EasyOCR offline (cham ~50s/trang CPU, KHONG can key).
# NHANH hon nhieu (~5s/trang) bang OCR.space (Engine 2 doc tieng Viet rat tot):
# dat LEGAL_NER_OCRSPACE_KEY. Key demo "helloworld" chay duoc nhung BI GIOI HAN —
# lay free key (25.000 luot/thang) tai https://ocr.space/ocrapi roi thay vao day.
# Neu OCR.space loi/het han muc -> tu dong lui ve EasyOCR.
$env:LEGAL_NER_OCRSPACE_KEY = "K83356731288957"

Write-Host "[legal_ner] Khoi chay tren http://127.0.0.1:$Port (Swagger: /docs)" -ForegroundColor Green
Write-Host "[legal_ner] Extractor mac dinh: opendataloader (PDF text-layer). Ban scan: chua ho tro." -ForegroundColor Yellow

& $py -m uvicorn api.main:app --host 0.0.0.0 --port $Port
