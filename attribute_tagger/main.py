import time
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

import tagger

app = FastAPI(title="Locus Attribute Tagger", version="1.0.0")
Instrumentator().instrument(app).expose(app)

tagger_requests = Counter(
    "locus_tagger_requests_total",
    "Total attribute tagging requests",
)
tagger_failures = Counter(
    "locus_tagger_failures_total",
    "Tagging requests that returned empty attributes (Gemini failure or parse error)",
)
tagger_latency = Histogram(
    "locus_tagger_latency_seconds",
    "End-to-end latency of the Gemini VLM attribute extraction call",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
)


class TagRequest(BaseModel):
    image_b64: str
    category: str = ""


class TagResponse(BaseModel):
    colors: Optional[list[str]] = None
    style: Optional[str] = None
    silhouette: Optional[str] = None
    occasion: Optional[list[str]] = None
    pattern: Optional[str] = None
    material_feel: Optional[str] = None
    trend_tags: Optional[list[str]] = None


@app.get("/")
def root():
    return {"status": "online", "service": "Locus Attribute Tagger"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/tag", response_model=TagResponse)
def tag(body: TagRequest):
    tagger_requests.inc()
    t0 = time.perf_counter()
    attrs = tagger.tag_image(body.image_b64, body.category)
    tagger_latency.observe(time.perf_counter() - t0)
    if not attrs:
        tagger_failures.inc()
    return attrs
