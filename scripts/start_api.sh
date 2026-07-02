#!/usr/bin/env bash
# Start the Legal-NER FastAPI app with the opendataloader (ODL) backend as the
# DEFAULT extractor. Exports the portable JDK 21 on JAVA_HOME/PATH so digital
# ODL extraction works from a cold start (system java is 8, too old for ODL).
#
# Scanned PDFs additionally need the hybrid OCR server. The API will AUTO-START
# it on the first scan request (see api/odl_adapter.ensure_hybrid_server), but
# for production run scripts/start_ocr_server.sh under a supervisor instead and
# set LEGAL_NER_ODL_HYBRID_AUTOSTART=0.
#
# Env overrides (all optional):
#   LEGAL_NER_ODL_JAVA_HOME (default /home/tts/jdk/jdk-21.0.11+10)
#   LEGAL_NER_API_HOST      (default 0.0.0.0)
#   LEGAL_NER_API_PORT      (default 8100)
set -euo pipefail

PROJECT_DIR="/home/tts/AI/AIHoang/Legal/legal_ner"
VENV_BIN="/home/tts/AI/AIHoang/HoangEnv/bin"
JAVA_HOME_DIR="${LEGAL_NER_ODL_JAVA_HOME:-/home/tts/jdk/jdk-21.0.11+10}"
HOST="${LEGAL_NER_API_HOST:-0.0.0.0}"
PORT="${LEGAL_NER_API_PORT:-8100}"

export JAVA_HOME="$JAVA_HOME_DIR"
export PATH="$JAVA_HOME/bin:$VENV_BIN:$PATH"
# the API process itself never runs EasyOCR (the OCR server does), so this is a
# harmless default that also keeps the shared GPU free if ODL probes CUDA.
export LEGAL_NER_ODL_JAVA_HOME="$JAVA_HOME_DIR"

cd "$PROJECT_DIR"
echo "[api] JAVA_HOME=$JAVA_HOME"
echo "[api] uvicorn api.main:app on $HOST:$PORT (default extractor: opendataloader)"

exec "$VENV_BIN/uvicorn" api.main:app --host "$HOST" --port "$PORT"
