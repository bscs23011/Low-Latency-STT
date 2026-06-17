#!/usr/bin/env bash
# run.sh — start the streaming STT server on 0.0.0.0:8000 (LAN-accessible).
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
IP=$(ip -4 addr show scope global 2>/dev/null | awk '/inet/{print $2}' | cut -d/ -f1 | head -1)
echo "Whisperwave STT  ->  http://${IP:-localhost}:${PORT}"
echo "Open that URL from any device on the same network."
exec .venv/bin/python server/app.py
