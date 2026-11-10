#!/bin/sh
cd "$(dirname "$0")"
if [ -x .venv/bin/python3 ]; then PY=.venv/bin/python3; else PY=python3; fi
exec "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-3000}"
