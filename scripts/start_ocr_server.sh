#!/usr/bin/env bash
# Start the opendataloader-pdf hybrid OCR server (docling/EasyOCR) for the
# Legal-NER API's scanned-PDF path.
#
#   * Forces CPU OCR (CUDA_VISIBLE_DEVICES="") — EasyOCR OOMs on the shared GPU.
#   * Points ODL's Java bridge at the portable JDK 21 (system java is 8, too old).
#   * --force-ocr so every page is OCR'd (these are scans); --ocr-lang vi,en.
#
# First run loads the EasyOCR models and can take ~1-2 min before /health is OK.
# For production, run this under a supervisor (systemd) — see api/README.md.
#
# Env overrides (all optional):
#   LEGAL_NER_ODL_JAVA_HOME       (default /home/tts/jdk/jdk-21.0.11+10)
#   LEGAL_NER_ODL_HYBRID_HOST     (default 127.0.0.1)
#   LEGAL_NER_ODL_HYBRID_PORT     (default 5002)
#   LEGAL_NER_ODL_HYBRID_OCR_LANG (default vi,en)
set -euo pipefail

VENV_BIN="/home/tts/AI/AIHoang/HoangEnv/bin"
JAVA_HOME_DIR="${LEGAL_NER_ODL_JAVA_HOME:-/home/tts/jdk/jdk-21.0.11+10}"
HOST="${LEGAL_NER_ODL_HYBRID_HOST:-127.0.0.1}"
PORT="${LEGAL_NER_ODL_HYBRID_PORT:-5002}"
OCR_LANG="${LEGAL_NER_ODL_HYBRID_OCR_LANG:-vi,en}"

export JAVA_HOME="$JAVA_HOME_DIR"
export PATH="$JAVA_HOME/bin:$VENV_BIN:$PATH"
# CPU OCR — EasyOCR OOMs on the shared GPU.
export CUDA_VISIBLE_DEVICES=""

echo "[ocr-server] JAVA_HOME=$JAVA_HOME"
echo "[ocr-server] CUDA_VISIBLE_DEVICES='(empty -> CPU OCR)'"
echo "[ocr-server] binding $HOST:$PORT  --force-ocr  --ocr-lang $OCR_LANG"
echo "[ocr-server] first run loads EasyOCR models (~1-2 min); GET /health when ready"

exec "$VENV_BIN/opendataloader-pdf-hybrid" \
    --host "$HOST" \
    --port "$PORT" \
    --force-ocr \
    --ocr-lang "$OCR_LANG" \
    --device cpu
