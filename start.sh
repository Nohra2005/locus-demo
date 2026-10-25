#!/usr/bin/env bash
# Starts the Locus demo stack inside one container (see Dockerfile).
set -euo pipefail
APP=/home/user/app
cd "$APP"

echo "[start] qdrant"
QDRANT__STORAGE__STORAGE_PATH=/tmp/qdrant/storage \
QDRANT__STORAGE__SNAPSHOTS_PATH=/tmp/qdrant/snapshots \
QDRANT__SERVICE__HTTP_PORT=6333 \
QDRANT__SERVICE__GRPC_PORT=6334 \
QDRANT__TELEMETRY_DISABLED=true \
QDRANT__LOG_LEVEL=WARN \
  qdrant --disable-telemetry &

echo "[start] loading catalog"
python scripts/load_catalog.py

echo "[start] visual engine"
(cd visual_engine && uvicorn main:app --host 127.0.0.1 --port 8001 --log-level warning) &

export PYTHONPATH="$APP/shared:${PYTHONPATH:-}"

if [ -n "${GOOGLE_API_KEY:-}" ] || [ -n "${GROQ_API_KEY:-}" ]; then
  echo "[start] attribute tagger (free-tier LLM chain)"
  (cd attribute_tagger && uvicorn demo_main:app --host 127.0.0.1 --port 8004 --log-level warning) &
  export TAGGER_HOST=http://127.0.0.1:8004
else
  echo "[start] no LLM key set: attribute tagger and judge are off, visual search still works"
fi

echo "[start] waiting for visual engine to load models"
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8001/health >/dev/null && break
  sleep 2
done

echo "[start] gateway on :7860"
cd gateway
exec uvicorn demo_app:app --host 0.0.0.0 --port 7860 --log-level warning
