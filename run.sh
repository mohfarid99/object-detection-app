#!/usr/bin/env bash
# Start the YOLO12 detection web app (creates .venv and installs deps on first run).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  echo "==> Creating virtualenv (.venv)"
  "$PY" -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
echo "==> http://${HOST}:${PORT}"
exec ./.venv/bin/uvicorn backend.app:app --host "$HOST" --port "$PORT" "$@"
