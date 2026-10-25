# Locus demo: the whole search stack in one container, for Hugging Face Spaces.
#
# Runs four processes (see start.sh):
#   qdrant           vector database, catalog loaded from data/ at startup   :6333
#   visual_engine    YOLO detection + Fashion-CLIP embeddings                :8001
#   attribute_tagger Gemini attribute extraction (only if OPENROUTER_API_KEY) :8004
#   gateway          FastAPI orchestration + the React frontend              :7860
#
# The production setup (Kubernetes, Prometheus, Grafana, MLflow, LoRA retraining)
# lives in k8s/, monitoring/ and mlops/. It is not needed to try the search.

# ── Stage 1: build the React frontend ─────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Qdrant binary (same major version as qdrant-client 1.10) ────────
FROM qdrant/qdrant:v1.10.1 AS qdrant

# ── Stage 3: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libunwind8 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as user 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only torch first, so nothing pulls the CUDA wheels
RUN pip install --no-cache-dir numpy \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0+cpu torchvision==0.21.0+cpu \
 && pip install --no-cache-dir ultralytics==8.3.2

COPY visual_engine/requirements.txt    /tmp/req-visual.txt
COPY gateway/requirements.txt          /tmp/req-gateway.txt
COPY attribute_tagger/requirements.txt /tmp/req-tagger.txt
RUN pip install --no-cache-dir -r /tmp/req-visual.txt -r /tmp/req-gateway.txt -r /tmp/req-tagger.txt

COPY --from=qdrant /qdrant/qdrant /usr/local/bin/qdrant

COPY --chown=user . .
COPY --chown=user --from=frontend /frontend/dist ./gateway/frontend/dist

USER user

# Download the model weights at build time so a cold start only loads them
RUN cd visual_engine && python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('Bingsu/adetailer', 'deepfashion2_yolov8s-seg.pt'); \
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('patrickjohncyh/fashion-clip'); \
CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip'); \
from ultralytics import YOLOWorld; YOLOWorld('yolov8s-worldv2.pt').set_classes(['shoe'])"

ENV QDRANT_HOST=127.0.0.1 \
    QDRANT_PORT=6333 \
    VISUAL_HOST=http://127.0.0.1:8001 \
    GATEWAY_BASE_URL=http://127.0.0.1:7860 \
    GOLDEN_DATASET_PATH=/home/user/app/mlops/golden_dataset.json \
    GOLDEN_IMAGES_DIR=/home/user/app/mlops/golden_images \
    USERS_FILE=/home/user/app/gateway/users.json \
    DEMO_MODE=1

EXPOSE 7860
CMD ["bash", "start.sh"]
