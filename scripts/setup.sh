#!/usr/bin/env bash
# setup.sh — one-shot, self-contained setup: venv + deps + models.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
MODELS=models
PARAKEET="$MODELS/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

echo "==> venv"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q

mkdir -p "$MODELS"

if [ ! -f "$PARAKEET/encoder.int8.onnx" ]; then
  echo "==> downloading parakeet-tdt-0.6b (~570 MB)"
  curl -L -o "$MODELS/parakeet.tar.bz2" \
    "$BASE/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
  tar xjf "$MODELS/parakeet.tar.bz2" -C "$MODELS"
  rm -f "$MODELS/parakeet.tar.bz2"
fi

if [ ! -f "$MODELS/silero_vad.onnx" ]; then
  echo "==> downloading silero VAD"
  curl -L -o "$MODELS/silero_vad.onnx" "$BASE/silero_vad.onnx"
fi

echo "==> done. run with: scripts/run.sh"
