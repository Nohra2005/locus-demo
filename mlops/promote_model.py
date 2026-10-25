"""
promote_model.py — Evaluate a new LoRA adapter and promote/rollback.

Evaluation uses a Gemini judge benchmark:
  1. Load old model (currently deployed adapter, or base fashion-clip if none)
  2. Load new model (new LoRA adapter being evaluated)
  3. For each model, build an in-memory catalog index (stratified, 50 items/category)
     so both models search in their own embedding space — fair comparison.
  4. Run 20 balanced golden queries → cosine-search in-memory index → judge top-3
  5. Promote if new_avg_score > old_avg_score + PROMOTION_DELTA

Promotion:
  - Copies adapter to visual_engine/models/lora_adapter/
  - Copies visual_projection.pt to visual_engine/models/
  - Calls POST /reload-adapter on visual engine (hot reload, no restart)
  - Calls POST /reindex on gateway to re-embed catalog with new model
  - Registers adapter in MLflow Model Registry
"""

import base64
import io
import json
import os
import re
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import mlflow
import requests
import torch

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL           = os.getenv("QDRANT_URL")
QDRANT_API_KEY       = os.getenv("QDRANT_API_KEY")
QDRANT_HOST          = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT          = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME      = "locus_items"
MLFLOW_URI           = os.getenv("MLFLOW_TRACKING_URI", "http://20.240.203.22:5000")
GATEWAY_URL          = os.getenv("GATEWAY_URL",   "http://localhost:8000")
VISUAL_ENGINE_URL    = os.getenv("VISUAL_HOST",   "http://localhost:8001")
OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY", "")

SCRIPT_DIR           = Path(__file__).parent
GOLDEN_DATASET_PATH  = SCRIPT_DIR / "golden_dataset.json"
GOLDEN_IMAGES_DIR    = SCRIPT_DIR / "golden_images"
VISUAL_ENGINE_MODELS = Path(os.getenv(
    "VISUAL_ENGINE_MODELS",
    SCRIPT_DIR.parent / "visual_engine" / "models"
))
VISUAL_ENGINE_MODELS_INTERNAL = os.getenv(
    "VISUAL_ENGINE_MODELS_INTERNAL",
    str(VISUAL_ENGINE_MODELS),
)
MODEL_NAME      = "patrickjohncyh/fashion-clip"
EVAL_TOP_K      = 3
PROMOTION_DELTA = 0.02   # new_avg must exceed old_avg by this margin

# ── 20 balanced evaluation queries (2 per major category, 1 for smaller ones) ─
EVAL_QUERY_NAMES = [
    # dress (2)
    "white floral wrap midi dress",
    "black satin slip midi dress",
    # top (2)
    "white ribbed halter neck crop top",
    "black off shoulder ruched top",
    # shoes (2)
    "white leather chunky sneakers",
    "tan suede ankle boots",
    # pants (2)
    "blue high waisted straight leg jeans",
    "Beige Linen Wide leg trousers",
    # jacket (2)
    "camel longline wool coat",
    "brown leather biker jacket",
    # sweater (2)
    "cream chunky knit oversized sweater",
    "grey zip up sweater",
    # bag (2)
    "black shoulder bag",
    "tan leather tote bag",
    # skirt (1)
    "Black satin midi bias skirt",
    # hat (1)
    "beige wide brim straw hat",
    # leggings (1)
    "blue high waisted seamless leggings",
    # jumpsuit (1)
    "black wide leg sleeveless jumpsuit",
    # top extra
    "white oversized shirt",
    # skirt second entry
    "brown mini skirt",
]
# ─────────────────────────────────────────────────────────────────────────────


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _fetch_image_bytes(url: str) -> bytes:
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
        return base64.b64decode(b64)
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Locus-Evaluator/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def _embed_with_model(model, processor, pil_image) -> list[float]:
    """Embed a single PIL image. Returns L2-normalised 512-dim list."""
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        vision_out     = model.vision_model(**inputs)
        image_features = model.visual_projection(vision_out.pooler_output)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features[0].tolist()


# ── Judge ─────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = (
    "You are a fashion visual similarity expert.\n"
    "You will be shown two clothing images: first the QUERY image, then a RESULT image.\n"
    "Rate how visually similar the result is to the query on a continuous scale from 0.00 to 1.00.\n"
    "Use the full range — do not snap to round values. Fine distinctions matter.\n"
    "Reference points (interpolate freely between them):\n"
    "  1.00 = identical item (same product, same photo)\n"
    "  0.92 = near-identical: same garment type, same colour family, same silhouette and fit — only model pose, lighting, or minor brand details differ\n"
    "  0.80 = very similar: same garment type and colour family, slightly different cut detail or wash\n"
    "  0.65 = clearly similar: same category and colour family but noticeably different cut or silhouette\n"
    "  0.50 = partially similar: same broad category but different colour OR clearly different cut\n"
    "  0.30 = same category only, clearly different style or design\n"
    "  0.10 = loosely related (e.g. both outerwear but very different garments)\n"
    "  0.00 = completely unrelated\n"
    "Important: when both images show the same garment type, same colour family, and same fit/silhouette, score at least 0.88 even if the model, background, or exact shade differs slightly.\n"
    "For patterned garments (floral, striped, printed): if both items share the same base colour AND the same pattern type, treat pattern variation as a minor difference only.\n"
    "Consider all visual attributes: category, colour, fabric texture, silhouette, length, fit, and detailing.\n"
    "Respond with ONLY the numeric score. No explanation."
)

_OPENROUTER_MODEL   = "google/gemini-2.0-flash-001"
_OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

_rate_lock  = threading.Lock()
_rate_state: dict = {"last_call": 0.0, "backoff_until": 0.0, "min_gap": 7.0}


def _acquire_slot() -> None:
    """Block until a rate-limit slot is available, then claim it."""
    while True:
        with _rate_lock:
            now = time.monotonic()
            backoff_wait = _rate_state["backoff_until"] - now
            if backoff_wait > 0:
                sleep_for = backoff_wait
            else:
                gap = _rate_state["last_call"] + _rate_state["min_gap"] - now
                if gap <= 0:
                    _rate_state["last_call"] = time.monotonic()
                    return
                sleep_for = gap
        time.sleep(sleep_for)  # outside lock — parallel workers sleep concurrently


def _judge_pair(query_b64: str, result_url: str, api_key: str) -> float | None:
    """Score one (query, result) pair via Gemini. Returns 0.0–1.0 or None on failure."""
    import httpx

    result_bytes = None

    # Serve golden-dataset images from the git-tracked local copy (avoids VM 404s in CI)
    if "/golden-dataset/images/" in result_url:
        filename = result_url.split("/golden-dataset/images/")[-1].strip("/")
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        candidates = list(GOLDEN_IMAGES_DIR.glob(f"{stem}*.jpeg"))
        if candidates:
            try:
                result_bytes = candidates[0].read_bytes()
            except Exception:
                pass

    if result_bytes is None:
        try:
            with httpx.Client(timeout=15.0) as c:
                r = c.get(result_url)
                r.raise_for_status()
                result_bytes = r.content
        except Exception as e:
            print(f"  [JUDGE] Could not fetch result image {result_url[:60]}: {e}")
            return None

    result_b64 = base64.b64encode(result_bytes).decode("utf-8")

    payload = {
        "model": _OPENROUTER_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _JUDGE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{result_b64}"}},
            ],
        }],
        "max_tokens": 50,
        "temperature": 0.0,
    }

    for attempt in range(3):
        _acquire_slot()
        try:
            import httpx as _httpx
            with _httpx.Client(timeout=60.0) as c:
                resp = c.post(
                    _OPENROUTER_API_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            if resp.status_code == 429:
                try:
                    wait = float(resp.headers.get("retry-after", 60))
                except (TypeError, ValueError):
                    wait = 60.0
                with _rate_lock:
                    _rate_state["backoff_until"] = time.monotonic() + wait
                continue  # retry after backoff clears
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r"\d+\.\d+|\d+", content)
            if match is None:
                return None
            return float(match.group())
        except Exception as e:
            if attempt < 2:
                time.sleep(min(5.0 * (2 ** attempt), 30.0))
                continue
            print(f"  [JUDGE] Error after 3 attempts: {e}")
            return None
    return None


# ── Catalog index helpers ─────────────────────────────────────────────────────

def _fetch_catalog_sample(n_per_category: int = 50, seed: int = 42) -> list[dict]:
    """
    Stratified catalog sample: n_per_category items per eval category.

    A flat random sample leaves rare categories (bag, hat, leggings, jumpsuit)
    with only 1-3 items — top-3 retrieval is meaningless and judge scores crash.
    Stratified sampling guarantees each category has a proper retrieval pool.

    Both models are evaluated against the same downloaded images, each
    embedded in their own space — so the comparison is fair.

    Returns list of {"url": str, "category": str, "pil": Image} dicts.
    """
    from PIL import Image
    from qdrant_client.http import models as qmodels
    import random as _random
    _random.seed(seed)

    with open(GOLDEN_DATASET_PATH) as f:
        golden = json.load(f)
    name_to_entry = {e.get("query_name", ""): e for e in golden}
    eval_categories = sorted({
        name_to_entry[n].get("query_category_tag", "")
        for n in EVAL_QUERY_NAMES
        if n in name_to_entry and name_to_entry[n].get("query_category_tag")
    })
    print(f"[EVAL] Fetching stratified catalog sample "
          f"({n_per_category}/category × {len(eval_categories)} categories)...")

    qdrant = _get_qdrant_client()
    items: list[dict] = []

    for cat in eval_categories:
        raw: list[dict] = []
        offset = None
        target = n_per_category * 4  # over-fetch to absorb broken image URLs
        while len(raw) < target:
            batch, next_offset = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=qmodels.Filter(must=[
                    qmodels.FieldCondition(
                        key="category_tag",
                        match=qmodels.MatchValue(value=cat),
                    )
                ]),
                limit=min(200, target - len(raw)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not batch:
                break
            for pt in batch:
                url = pt.payload.get("image_url", "")
                if url:
                    raw.append({"url": url, "category": cat})
            if next_offset is None:
                break
            offset = next_offset

        _random.shuffle(raw)
        downloaded = 0
        for entry in raw:
            if downloaded >= n_per_category:
                break
            try:
                img_bytes = _fetch_image_bytes(entry["url"])
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                items.append({"url": entry["url"], "category": cat, "pil": pil_img})
                downloaded += 1
            except Exception:
                continue
        print(f"[EVAL]   {cat}: {downloaded} items")

    print(f"[EVAL] Catalog sample ready: {len(items)} total items")
    return items


def _build_catalog_index(model, processor, catalog_items: list[dict]) -> tuple:
    """
    Embed catalog_items with model. Returns (embeddings_tensor, valid_items)
    where embeddings_tensor is L2-normalised (N, 512) float32.
    """
    embeddings: list[list[float]] = []
    valid: list[dict] = []
    for item in catalog_items:
        try:
            emb = _embed_with_model(model, processor, item["pil"])
            embeddings.append(emb)
            valid.append(item)
        except Exception:
            continue
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    emb_tensor = emb_tensor / emb_tensor.norm(p=2, dim=-1, keepdim=True)
    print(f"[EVAL] Catalog index built: {len(valid)} embeddings")
    return emb_tensor, valid


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _load_eval_queries() -> list[dict]:
    """Return the 20 selected balanced queries from the golden dataset."""
    with open(GOLDEN_DATASET_PATH) as f:
        golden = json.load(f)

    name_to_entry = {e.get("query_name", ""): e for e in golden}
    selected = []
    missing  = []
    for name in EVAL_QUERY_NAMES:
        if name in name_to_entry:
            selected.append(name_to_entry[name])
        else:
            missing.append(name)

    if missing:
        print(f"[EVAL] WARNING: {len(missing)} eval query name(s) not in golden dataset: {missing}")

    print(f"[EVAL] Loaded {len(selected)} evaluation queries across "
          f"{len(set(e.get('query_category_tag') for e in selected))} categories")
    return selected


def _run_judge_eval(
    model,
    processor,
    queries: list[dict],
    api_key: str,
    top_k: int = EVAL_TOP_K,
    catalog_embeddings: "torch.Tensor | None" = None,
    catalog_items: "list[dict] | None" = None,
) -> dict[str, float]:
    """
    Embed each query with `model`, retrieve top-k, judge with Gemini.

    Uses in-memory cosine search against catalog_embeddings when provided
    (each model searches in its own embedding space — fair comparison).
    Falls back to Qdrant when no catalog is provided (biased: Qdrant holds
    vectors from the previously deployed model).
    """
    from PIL import Image

    query_avgs: dict[str, float] = {}
    use_memory_index = catalog_embeddings is not None and catalog_items is not None

    if not use_memory_index:
        from qdrant_client.http import models as qmodels
        qdrant = _get_qdrant_client()

    for entry in queries:
        query_name = entry.get("query_name", "?")
        query_url  = entry.get("query_image_url", "")
        category   = entry.get("query_category_tag", "")

        if query_url.startswith("http://localhost:"):
            query_url = query_url.replace("http://localhost:8000", GATEWAY_URL.rstrip("/"), 1)

        try:
            filename   = query_url.split("/")[-1]
            local_path = GOLDEN_IMAGES_DIR / filename
            if local_path.exists():
                img_bytes = local_path.read_bytes()
            else:
                img_bytes = _fetch_image_bytes(query_url)
            pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            vector    = _embed_with_model(model, processor, pil_img)
            query_b64 = base64.b64encode(img_bytes).decode("utf-8")
        except Exception as e:
            print(f"  [EVAL] Skip '{query_name}': could not embed — {e}")
            continue

        result_urls: list[str] = []

        if use_memory_index:
            if category:
                indices = [i for i, it in enumerate(catalog_items) if it["category"] == category]
            else:
                indices = list(range(len(catalog_items)))
            if not indices:
                indices = list(range(len(catalog_items)))

            sub_emb  = catalog_embeddings[indices]
            q_tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
            sims     = torch.mm(q_tensor, sub_emb.T).squeeze(0)
            topk_local = sims.topk(min(top_k, len(indices))).indices.tolist()
            result_urls = [catalog_items[indices[i]]["url"] for i in topk_local]
        else:
            query_filter = None
            if category:
                query_filter = qmodels.Filter(must=[
                    qmodels.FieldCondition(
                        key="category_tag",
                        match=qmodels.MatchValue(value=category),
                    )
                ])
            try:
                result = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_vectors=False,
                    search_params=qmodels.SearchParams(hnsw_ef=256),
                )
                for h in result.points:
                    url = h.payload.get("image_url", "")
                    if not url:
                        continue
                    if url.startswith("http://localhost:"):
                        url = url.replace("http://localhost:8000", GATEWAY_URL.rstrip("/"), 1)
                    result_urls.append(url)
            except Exception as e:
                print(f"  [EVAL] Skip '{query_name}': Qdrant error — {e}")
                continue

        if not result_urls:
            print(f"  [EVAL] Skip '{query_name}': no result image URLs")
            continue

        def _judge_one(url):
            return _judge_pair(query_b64, url, api_key)

        query_scores: list[float] = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            for score in as_completed({pool.submit(_judge_one, u): u for u in result_urls}):
                s = score.result()
                if s is not None:
                    query_scores.append(s)

        if query_scores:
            avg = sum(query_scores) / len(query_scores)
            print(f"  [EVAL] '{query_name}' ({category}): {len(query_scores)} scores, avg={avg:.3f}")
            query_avgs[query_name] = avg
        else:
            print(f"  [EVAL] '{query_name}': all judge calls failed — skipped")

        time.sleep(0.1)

    return query_avgs


def _load_model(adapter_dir: Optional[str] = None, visual_projection_path: Optional[str] = None):
    """Load fashion-clip, optionally applying a LoRA adapter."""
    from transformers import CLIPModel, CLIPProcessor

    model     = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    if adapter_dir and Path(adapter_dir).exists():
        from peft import PeftModel
        print(f"[EVAL] Applying LoRA adapter from {adapter_dir}")
        model.vision_model = PeftModel.from_pretrained(model.vision_model, adapter_dir)
        if visual_projection_path and Path(visual_projection_path).exists():
            proj_state = torch.load(visual_projection_path, map_location="cpu")
            model.visual_projection.load_state_dict(proj_state)
    else:
        print("[EVAL] No adapter — using base fashion-clip")

    model.eval()
    return model, processor


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_with_judge(
    new_adapter_dir:            str,
    new_visual_projection_path: Optional[str] = None,
    api_key:                    str = "",
    top_k:                      int = EVAL_TOP_K,
) -> tuple[float, float]:
    """
    Benchmark old model vs new LoRA adapter on 20 balanced golden queries.

    Each model searches its own in-memory catalog index (same downloaded images,
    embedded separately) — so the comparison is fair regardless of what's in
    the production Qdrant collection.

    Returns (old_avg_score, new_avg_score).
    """
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for judge evaluation")

    queries = _load_eval_queries()

    # Download catalog images once — both models embed the same raw pixels
    catalog_items = _fetch_catalog_sample(n_per_category=500)

    deployed_adapter = VISUAL_ENGINE_MODELS / "lora_adapter"
    deployed_proj    = VISUAL_ENGINE_MODELS / "visual_projection.pt"

    print("\n[EVAL] === Evaluating OLD model ===")
    old_model, old_processor = _load_model(
        adapter_dir            = str(deployed_adapter) if deployed_adapter.exists() else None,
        visual_projection_path = str(deployed_proj)    if deployed_proj.exists()    else None,
    )
    old_emb, old_items = _build_catalog_index(old_model, old_processor, catalog_items)
    old_scores = _run_judge_eval(old_model, old_processor, queries, api_key, top_k,
                                  catalog_embeddings=old_emb, catalog_items=old_items)
    del old_model, old_emb

    with _rate_lock:
        remaining = _rate_state["backoff_until"] - time.monotonic()
    if remaining > 0:
        print(f"[EVAL] Waiting {remaining:.0f}s for rate-limit backoff to clear...")
        time.sleep(remaining + 2)

    print("\n[EVAL] === Evaluating NEW model ===")
    new_model, new_processor = _load_model(
        adapter_dir            = new_adapter_dir,
        visual_projection_path = new_visual_projection_path,
    )
    new_emb, new_items = _build_catalog_index(new_model, new_processor, catalog_items)
    new_scores = _run_judge_eval(new_model, new_processor, queries, api_key, top_k,
                                  catalog_embeddings=new_emb, catalog_items=new_items)
    del new_model, new_emb

    # Only compare on queries where both models produced a score — prevents
    # cherry-picked averages when the rate-limiter drops different query subsets.
    shared = sorted(set(old_scores) & set(new_scores))
    if not shared:
        print("[EVAL] WARNING: no shared queries between old and new — returning (0.0, 0.0)")
        return 0.0, 0.0
    old_avg = sum(old_scores[q] for q in shared) / len(shared)
    new_avg = sum(new_scores[q] for q in shared) / len(shared)
    print(f"[EVAL] Comparison on {len(shared)}/20 shared queries")
    print(f"\n[EVAL] OLD avg={old_avg:.4f}  NEW avg={new_avg:.4f}  delta={new_avg - old_avg:+.4f}")
    return old_avg, new_avg


def promote_adapter(
    adapter_dir:            str,
    visual_projection_path: Optional[str],
    mlflow_run_id:          str,
) -> None:
    """Copy adapter to visual_engine/models/, trigger hot reload and re-indexing."""
    VISUAL_ENGINE_MODELS.mkdir(parents=True, exist_ok=True)
    dest_adapter = VISUAL_ENGINE_MODELS / "lora_adapter"
    dest_proj    = VISUAL_ENGINE_MODELS / "visual_projection.pt"

    if dest_adapter.exists():
        backup = VISUAL_ENGINE_MODELS / "lora_adapter_prev"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(dest_adapter, backup)
        print(f"[PROMOTE] Backed up previous adapter to {backup}")

    if dest_adapter.exists():
        shutil.rmtree(dest_adapter)
    shutil.copytree(adapter_dir, dest_adapter)
    print(f"[PROMOTE] Copied adapter to {dest_adapter}")

    if visual_projection_path and Path(visual_projection_path).exists():
        shutil.copy2(visual_projection_path, dest_proj)
        print(f"[PROMOTE] Copied visual_projection to {dest_proj}")

    print("[PROMOTE] Requesting visual engine adapter reload...")
    internal_adapter = Path(VISUAL_ENGINE_MODELS_INTERNAL) / "lora_adapter"
    internal_proj    = Path(VISUAL_ENGINE_MODELS_INTERNAL) / "visual_projection.pt"
    try:
        resp = requests.post(
            f"{VISUAL_ENGINE_URL}/reload-adapter",
            json={
                "adapter_path":           str(internal_adapter),
                "visual_projection_path": str(internal_proj),
            },
            timeout=60,
        )
        resp.raise_for_status()
        print(f"[PROMOTE] Visual engine reloaded: {resp.json()}")
    except Exception as e:
        print(f"[PROMOTE] WARNING: Could not hot-reload visual engine: {e}")
        print("          Restart visual_engine container to apply new adapter.")

    print("[PROMOTE] Triggering catalog re-indexing...")
    try:
        resp = requests.post(f"{GATEWAY_URL}/reindex", timeout=10)
        resp.raise_for_status()
        print(f"[PROMOTE] Re-indexing started: {resp.json()}")
    except Exception as e:
        print(f"[PROMOTE] WARNING: Could not trigger re-indexing: {e}")
        print("          Run POST /reindex manually to update catalog vectors.")

    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        client    = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{mlflow_run_id}/lora_adapter"
        mlflow.log_artifacts(str(dest_adapter), artifact_path="lora_adapter")
        mv = mlflow.register_model(model_uri=model_uri, name="locus_lora_adapter")
        client.set_registered_model_alias(
            name    = "locus_lora_adapter",
            alias   = "production",
            version = mv.version,
        )
        print(f"[PROMOTE] Registered as MLflow model version {mv.version} (alias: production)")
    except Exception as e:
        print(f"[PROMOTE] MLflow registration skipped: {e}")
