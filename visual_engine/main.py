import asyncio
import io
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from PIL import Image, ImageDraw
from pydantic import BaseModel
from typing import Any, Optional

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram

# ---------------------------------------------------------------------------
# Deferred model loading — uvicorn binds to :8001 (and exposes /metrics)
# BEFORE the heavy CLIP/YOLO models load.  Prometheus sees the service as UP
# immediately; endpoints return 503 while loading, then work normally.
# ---------------------------------------------------------------------------

visualizer = None   # set by _load_models() on startup


def _load_models():
    global visualizer
    from vectorizer import LocusVisualizer
    visualizer = LocusVisualizer()


def _require_visualizer():
    if visualizer is None:
        raise HTTPException(
            status_code=503,
            detail="Visual engine loading — models not ready yet, retry in ~60s",
        )
    return visualizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run blocking model load in a thread pool so the event loop stays free
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _load_models)
    yield


app = FastAPI(title="Locus Visual Engine", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

locus_detections_histogram = Histogram(
    "locus_detections_per_request",
    "Number of YOLO boxes returned per /detect call",
    buckets=[0, 1, 2, 3, 4, 5, 6, 8, 10, 15],
)

locus_clip_confidence = Histogram(
    "locus_clip_confidence",
    "Top CLIP category confidence score per /vectorize call",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
)


# ── Response models ────────────────────────────────────────────────────────────

class DetectionResponse(BaseModel):
    detections: list[dict]
    image_width: int
    image_height: int


class VectorizeResponse(BaseModel):
    filename: Optional[str]
    vector: list[float]
    category: Optional[str]
    category_confidence: Any
    debug_image: Optional[str]


class TextEmbeddingResponse(BaseModel):
    embedding: list[float]


class ReloadAdapterResponse(BaseModel):
    success: bool
    error: Optional[str] = None


@app.get("/")
def read_root():
    return {
        "status":  "ready" if visualizer is not None else "loading",
        "service": "locus visual engine",
    }


@app.get("/health")
def health():
    if visualizer is None:
        raise HTTPException(status_code=503, detail="loading")
    return {"status": "ready"}


# ── Search path ────────────────────────────────────────────────────────────────

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """Search time. Returns all YOLO boxes for user to select from."""
    v = _require_visualizer()
    image_data = await file.read()
    detections, img_width, img_height = await asyncio.to_thread(
        v.detect_objects, image_data
    )
    locus_detections_histogram.observe(len(detections))
    return {
        "detections":   detections,
        "image_width":  img_width,
        "image_height": img_height,
    }


@app.post("/vectorize", response_model=VectorizeResponse)
async def vectorize(
    file:        UploadFile = File(...),
    yolo_label:  str        = Form(""),
    darken:      str        = Form("false"),
    query:       str        = Form("false"),
    style_hint:  str        = Form(""),
):
    """
    Search time. Expects ALREADY CROPPED image bytes — gateway crops before calling.
    Pass query=true when classifying a user-drawn crop (enables person-wearing prompts).
    Pass style_hint with a descriptive phrase for accessories to enable hybrid vector mixing.
    """
    v = _require_visualizer()
    image_data    = await file.read()
    should_darken = darken.lower() == "true"
    query_mode    = query.lower() == "true"

    vector, category, confidence, debug_img = await asyncio.to_thread(
        v.process_image,
        image_bytes=image_data,
        yolo_label=yolo_label,
        darken=should_darken,
        query_mode=query_mode,
        style_hint=style_hint,
    )

    if not vector:
        return {"error": "failed to process image"}

    if isinstance(confidence, dict) and confidence:
        locus_clip_confidence.observe(max(confidence.values()))

    return {
        "filename":            file.filename,
        "vector":              vector,
        "category":            category,
        "category_confidence": confidence,
        "debug_image":         debug_img,
    }


class TextQuery(BaseModel):
    text: str


@app.post("/vectorize-text", response_model=TextEmbeddingResponse)
async def vectorize_text(body: TextQuery):
    """Encode a text string with CLIP's text encoder. Used by /refine in the gateway."""
    v = _require_visualizer()
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    embedding = await asyncio.to_thread(v.encode_text, body.text.strip())
    return {"embedding": embedding}


# ── Index path ─────────────────────────────────────────────────────────────────

@app.post("/index-image")
async def index_image(
    file:  UploadFile = File(...),
    title: str        = Form(""),
):
    """
    Index time only. Returns vector + category + box_source,
    or {"skipped": true} with reason.
    """
    v = _require_visualizer()
    image_data = await file.read()
    result = await asyncio.to_thread(
        v.index_product, image_data, title=title
    )
    return result


# ── Debug index (dashboard preview — does NOT write to Qdrant) ────────────────

@app.post("/debug-index")
async def debug_index(
    file:  UploadFile = File(...),
    title: str        = Form(""),
):
    """
    Same logic as /index-image but returns an annotated image showing:
      - All YOLO detections as grey dashed boxes with label + score
      - The selected crop box in gold (successfully classified)
      - Red border only if skipped

    Does NOT write to Qdrant.
    """
    v = _require_visualizer()
    image_data = await file.read()
    result     = await asyncio.to_thread(v.index_product, image_data, title=title)

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image.thumbnail((600, 600))
        scale_x = image.width  / Image.open(io.BytesIO(image_data)).width
        scale_y = image.height / Image.open(io.BytesIO(image_data)).height
        W, H    = image.size
        draw    = ImageDraw.Draw(image)

        all_detections = result.get("all_detections", [])
        selected_bbox  = result.get("selected_bbox", None)

        # Draw all non-selected boxes in grey
        for det in all_detections:
            bx1, by1, bx2, by2 = det["bbox"]
            bx1 = int(bx1 * scale_x); by1 = int(by1 * scale_y)
            bx2 = int(bx2 * scale_x); by2 = int(by2 * scale_y)

            # Skip if this is the selected box — we'll draw it in gold below
            if selected_bbox:
                sx1, sy1, sx2, sy2 = [
                    int(selected_bbox[0] * scale_x),
                    int(selected_bbox[1] * scale_y),
                    int(selected_bbox[2] * scale_x),
                    int(selected_bbox[3] * scale_y),
                ]
                if abs(bx1 - sx1) < 5 and abs(by1 - sy1) < 5:
                    continue

            draw.rectangle([bx1, by1, bx2, by2], outline="#555555", width=2)
            lbl = f"{det.get('search_label','')} {det.get('score', 0):.2f}"
            draw.rectangle([bx1, by1 - 16, bx1 + len(lbl) * 6, by1], fill="#333333")
            draw.text((bx1 + 2, by1 - 14), lbl, fill="#aaaaaa")

        if result.get("skipped"):
            # Red border for skipped products
            for i in range(4):
                draw.rectangle([i, i, W - i, H - i], outline="#c97070")
            draw.rectangle([0, H - 28, W, H], fill="#c97070")
            draw.text((6, H - 21), result.get("skip_reason", "skipped"), fill="#ffffff")

        elif selected_bbox:
            # Gold box for the selected crop
            sx1 = int(selected_bbox[0] * scale_x)
            sy1 = int(selected_bbox[1] * scale_y)
            sx2 = int(selected_bbox[2] * scale_x)
            sy2 = int(selected_bbox[3] * scale_y)

            for i in range(3):
                draw.rectangle([sx1 + i, sy1 + i, sx2 - i, sy2 - i], outline="#c9a96e")

            # Gold label showing category + box_source
            cat     = result.get("category", "")
            box_src = result.get("box_source", "")
            lbl     = f"✓ {cat}  [{box_src}]"
            draw.rectangle([sx1, sy2, sx1 + len(lbl) * 7, sy2 + 18], fill="#c9a96e")
            draw.text((sx1 + 3, sy2 + 3), lbl, fill="#000000")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        result["debug_image"] = base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"[DEBUG-INDEX] Image annotation failed: {e}")
        result["debug_image"] = None

    return result


# ── Classify text (debug) ──────────────────────────────────────────────────────

class ClassifyTextRequest(BaseModel):
    title: str


@app.post("/classify-text")
async def classify_text(req: ClassifyTextRequest):
    v = _require_visualizer()
    category, confidence = v.classify_text(req.title)
    return {"category": category, "confidence": confidence}


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTER HOT-RELOAD
# Called by promote_model.py after copying a new LoRA adapter to disk.
# No container restart needed — swaps the CLIP vision encoder in-place.
# ══════════════════════════════════════════════════════════════════════════════

class ReloadAdapterRequest(BaseModel):
    adapter_path:           str
    visual_projection_path: str = ""


@app.post("/reload-adapter", response_model=ReloadAdapterResponse)
async def reload_adapter(req: ReloadAdapterRequest):
    """
    Hot-swap the LoRA adapter on the running visual engine.
    Loads a fresh base model + new adapter, then atomically replaces
    self.clip_model so in-flight requests are not disrupted.
    Takes ~30s while the base model reloads from the HuggingFace cache.
    """
    v = _require_visualizer()
    result = await asyncio.to_thread(
        v.reload_adapter,
        adapter_path=req.adapter_path,
        visual_projection_path=req.visual_projection_path,
    )
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result
