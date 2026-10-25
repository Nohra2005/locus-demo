FROM python:3.11-slim

WORKDIR /app

# System deps — cached apt lists, no layer bloat
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0

# numpy first so torch's dep-resolver finds it already satisfied
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install numpy

# torch + torchvision (~700 MB) — pip wheel cache persists across rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 --retries 5 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0+cpu torchvision==0.21.0+cpu

# ultralytics (pulls scipy ~38 MB) — own layer so torch cache is never busted
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ultralytics==8.3.2

# App deps — only re-runs when requirements.txt changes
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
