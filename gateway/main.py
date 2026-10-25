import asyncio
import base64
import hashlib
import io
import json as _json
import os
import pathlib
import random
import uuid
from datetime import datetime
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter, Histogram
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

import time as _time_module

from judge import run_judge
from auth import router as auth_router, verify_token, optional_token, _load_users, pwd_context, get_store_maps_urls, get_store_locations

# ── Per-search judge score store ──────────────────────────────────────────────
# search_id → {product_id: float_score}  written incrementally by run_judge.
# Entries are cleaned up after _JUDGE_TTL seconds so memory doesn't grow forever.
_judge_scores:     dict[str, dict[str, float]] = {}
_judge_timestamps: dict[str, float]            = {}
_JUDGE_TTL = 300  # 5 minutes

LOW_SCORE_THRESHOLD           = 0.40   # Gemini judge score below this in top-3 → low-quality flag (clothing)
LOW_SCORE_THRESHOLD_ACCESSORY = 0.10   # Accessories have fewer catalog items; only flag clearly broken items
LOW_SCORE_FLAGS_PATH = "/app/low_score_flags.json"

# Style hint phrases passed to visual_engine /vectorize for accessory queries.
# Used to mix CLIP text embeddings into the query vector (hybrid visual+text).
_SHOE_STYLE_HINTS = {
    "sneaker": "sneaker trainer running shoe athletic casual footwear",
    "boot":    "ankle boot chelsea boot combat boot knee high boot",
    "heel":    "high heel stiletto pump court shoe kitten heel formal",
    "sandal":  "sandal flat mule loafer ballet flat espadrille open toe",
    "other":   "shoe footwear",
}
_BAG_HINT = "handbag shoulder bag tote crossbody clutch purse"
_HAT_HINT = "hat cap headwear beret fedora bucket hat beanie"


def _shoe_style_from_name(name: str) -> str:
    lwr = name.lower()
    if any(k in lwr for k in ("sneaker", "trainer", "running", "platform", "wedge", "clog", "chunky")):
        return "sneaker"
    if any(k in lwr for k in ("boot", "bottine", "botte")):
        return "boot"
    if any(k in lwr for k in ("heel", "pump", "stiletto", "kitten")):
        return "heel"
    if any(k in lwr for k in ("sandal", "loafer", "flat", "mule", "espadrille", "ballet")):
        return "sandal"
    return "other"


def _cleanup_judge_scores() -> None:
    cutoff = _time_module.monotonic() - _JUDGE_TTL
    stale  = [sid for sid, ts in _judge_timestamps.items() if ts < cutoff]
    for sid in stale:
        _judge_scores.pop(sid, None)
        _judge_timestamps.pop(sid, None)


# ── Per-search attribute cache ────────────────────────────────────────────────
# search_id → attribute dict (None while Gemini call is in-flight)
# search_id → original matches list (stored immediately for /refine "visual" mode)
_attribute_cache:   dict[str, dict | None] = {}
_original_results:  dict[str, list]        = {}
_attr_timestamps:   dict[str, float]       = {}
_ATTR_TTL = 600  # 10 minutes


def _cleanup_attribute_cache() -> None:
    cutoff = _time_module.monotonic() - _ATTR_TTL
    stale  = [sid for sid, ts in _attr_timestamps.items() if ts < cutoff]
    for sid in stale:
        _attribute_cache.pop(sid, None)
        _original_results.pop(sid, None)
        _attr_timestamps.pop(sid, None)


def _read_low_score_flags() -> list:
    try:
        with open(LOW_SCORE_FLAGS_PATH) as f:
            return _json.load(f)
    except Exception:
        return []


def _write_low_score_flags(flags: list) -> None:
    try:
        with open(LOW_SCORE_FLAGS_PATH, "w") as f:
            _json.dump(flags, f, indent=2)
    except Exception as exc:
        print(f"[LOW_SCORE] Failed to write flags file: {exc}")


async def _audit_corrupt_items(
    suspicious_pids: dict,
    pid_counts: dict,
    top3_matches: list,
    scores_dict: dict,
) -> None:
    """
    Waits for judge scores then checks if a product that occupies ≥2 slots in the
    top 3 raw CLIP results also scores low (< 40% of the best top-3 judge score AND
    < 0.35 absolute). If so, increments corrupt_flag_count in Qdrant. A second flag
    triggers deletion of all Qdrant points for that product.
    """
    top3_pids   = [m["product_id"] for m in top3_matches]
    suspect_set = set(suspicious_pids)

    deadline = _time_module.monotonic() + 90
    while _time_module.monotonic() < deadline:
        if all(pid in scores_dict for pid in suspect_set):
            break
        await asyncio.sleep(5)

    top3_scores = [scores_dict[pid] for pid in top3_pids if pid in scores_dict]
    if not top3_scores:
        print("[CORRUPT] No judge scores returned for top 3 — skipping corrupt audit")
        return

    max_top3 = max(top3_scores)

    for pid, raw_point_ids in suspicious_pids.items():
        count       = pid_counts[pid]
        judge_score = scores_dict.get(pid)

        if judge_score is None:
            print(f"[CORRUPT] {pid[:60]} appeared {count}x in top-3 raw but was not scored — skipping")
            continue

        if not (judge_score < max_top3 * 0.4 and judge_score < 0.35):
            print(f"[CORRUPT] {pid[:60]} appeared {count}x in top-3 raw, "
                  f"judge={judge_score:.3f} vs max={max_top3:.3f} — score not low enough, no action")
            continue

        print(f"[CORRUPT] SUSPICIOUS — {pid[:60]} appeared {count}x in top-3 raw, "
              f"judge={judge_score:.3f} vs max={max_top3:.3f}")

        try:
            scroll_hits, _ = await asyncio.to_thread(
                client.scroll,
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="product_id", match=models.MatchValue(value=pid))
                ]),
                limit=50,
                with_payload=True,
            )
        except Exception as exc:
            print(f"[CORRUPT] Scroll failed for {pid[:60]}: {exc}")
            continue

        if not scroll_hits:
            continue

        current_flags = max((p.payload.get("corrupt_flag_count", 0) for p in scroll_hits), default=0)
        new_flags     = current_flags + 1
        all_ids       = [p.id for p in scroll_hits]

        if new_flags >= 2:
            try:
                await asyncio.to_thread(
                    client.delete,
                    collection_name=COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=all_ids),
                )
                print(f"[CORRUPT] DELETED {len(all_ids)} point(s) for {pid[:60]} "
                      f"(flag #{new_flags} — second offense)")
            except Exception as exc:
                print(f"[CORRUPT] Delete failed for {pid[:60]}: {exc}")
        else:
            try:
                await asyncio.to_thread(
                    client.set_payload,
                    collection_name=COLLECTION_NAME,
                    payload={"corrupt_flag_count": new_flags},
                    points=all_ids,
                )
                print(f"[CORRUPT] FLAGGED {len(all_ids)} point(s) for {pid[:60]} "
                      f"(corrupt_flag_count → {new_flags}; will delete on next occurrence)")
            except Exception as exc:
                print(f"[CORRUPT] Set payload failed for {pid[:60]}: {exc}")


_ACCESSORY_CATEGORIES_SET = {"shoes", "bag", "hat"}


async def _audit_low_score_top3(
    top3_matches: list,
    scores_dict: dict,
    search_category: str = "",
) -> None:
    """
    After Gemini judge scores arrive, checks each of the top-3 results.
    If any scores below LOW_SCORE_THRESHOLD it is flagged with low_score_flag=True
    in Qdrant (hidden from future searches) and recorded in the admin flags JSON.

    Metadata is read directly from the match dict passed in — the same dict that
    run_judge used to assign scores — so the judge score is always linked to the
    correct item regardless of any reranking that may have occurred.

    Accessories (shoes/bag/hat) use a much lower threshold (LOW_SCORE_THRESHOLD_ACCESSORY)
    because CLIP retrieval is noisier for accessories — a cross-style shoe match (e.g.
    sneaker returned for a heel query) scores 0.3-0.4 with Gemini but is still a valid
    catalog item. Flagging it would shrink the already-small accessory pool.
    """
    is_accessory  = search_category in _ACCESSORY_CATEGORIES_SET
    flag_threshold = LOW_SCORE_THRESHOLD_ACCESSORY if is_accessory else LOW_SCORE_THRESHOLD

    top3_pids = [m["product_id"] for m in top3_matches]
    deadline  = _time_module.monotonic() + 90
    while _time_module.monotonic() < deadline:
        if all(pid in scores_dict for pid in top3_pids):
            break
        await asyncio.sleep(5)

    existing_flags = _read_low_score_flags()
    flagged_pids   = {f["product_id"] for f in existing_flags}
    new_flags      = list(existing_flags)

    for match in top3_matches:
        pid   = match["product_id"]
        score = scores_dict.get(pid)
        if score is None:
            continue
        if score >= flag_threshold:
            continue
        if pid in flagged_pids:
            print(f"[LOW_SCORE] {pid[:60]} already flagged (score={score:.3f}) — skipping")
            continue

        print(f"[LOW_SCORE] FLAGGING top-3 match '{match.get('name', '')}' "
              f"({pid[:60]}) judge={score:.3f} < {flag_threshold}")

        try:
            scroll_hits, _ = await asyncio.to_thread(
                client.scroll,
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="product_id", match=models.MatchValue(value=pid))
                ]),
                limit=50,
                with_payload=True,
            )
        except Exception as exc:
            print(f"[LOW_SCORE] Scroll failed for {pid[:60]}: {exc}")
            continue

        if not scroll_hits:
            continue

        all_ids = [p.id for p in scroll_hits]
        try:
            await asyncio.to_thread(
                client.set_payload,
                collection_name=COLLECTION_NAME,
                payload={"low_score_flag": True},
                points=all_ids,
            )
        except Exception as exc:
            print(f"[LOW_SCORE] Set payload failed for {pid[:60]}: {exc}")
            continue

        new_flags.append({
            "product_id":  pid,
            "image_url":   match.get("image_url", ""),
            "name":        match.get("name", ""),
            "store_name":  match.get("store_name", ""),
            "judge_score": score,
            "flagged_at":  datetime.utcnow().isoformat(),
        })
        flagged_pids.add(pid)

    if len(new_flags) > len(existing_flags):
        _write_low_score_flags(new_flags)


async def _run_tagger(search_id: str, image_b64: str, category: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=20.0) as _tc:
            resp = await _tc.post(
                f"{TAGGER_HOST}/tag",
                json={"image_b64": image_b64, "category": category},
            )
        _attribute_cache[search_id] = resp.json() if resp.status_code == 200 else {}
    except Exception as _te:
        print(f"[TAGGER] Failed for search_id={search_id}: {_te.__class__.__name__}: {_te}")
        _attribute_cache[search_id] = {}
    finally:
        _attr_timestamps[search_id] = _time_module.monotonic()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
Instrumentator().instrument(app).expose(app)

# ── Custom Prometheus metrics ─────────────────────────────────────────────────

# Qdrant collection sizes (refreshed every 60s by background task)
locus_qdrant_items = Gauge(
    "locus_qdrant_items_total",
    "Number of points in each Qdrant collection",
    ["collection"],
)

# Search events
locus_searches = Counter(
    "locus_searches_total",
    "Total search requests by detected category",
    ["category"],
)

# Feedback star ratings (runtime counter, incremented on each /feedback POST)
locus_feedback_stars = Counter(
    "locus_feedback_stars_total",
    "Feedback events by star rating and training signal",
    ["stars", "signal"],
)

# Rating distribution gauges — read from Qdrant every 60s, reflect true totals
locus_rating_by_star = Gauge(
    "locus_rating_by_star_total",
    "Ratings per star level from locus_feedback Qdrant collection",
    ["stars", "source"],  # source: all | user | judge
)
locus_rating_by_source = Gauge(
    "locus_rating_by_source_total",
    "Total ratings by source (user vs auto_judge) from locus_feedback",
    ["source"],
)
locus_rating_avg = Gauge(
    "locus_rating_avg_score",
    "Average star rating across all locus_feedback records",
)
locus_rating_by_signal = Gauge(
    "locus_rating_by_signal_total",
    "Total ratings per training signal from locus_feedback",
    ["signal"],
)

# Link health monitor timing
locus_link_monitor_last_run = Gauge(
    "locus_link_monitor_last_run_timestamp",
    "Unix timestamp of the last link health check run",
)
locus_link_monitor_next_run = Gauge(
    "locus_link_monitor_next_run_timestamp",
    "Unix timestamp of the next scheduled link health check run",
)

# Admin / operational metrics (refreshed every 30s by _refresh_admin_metrics)
locus_active_sessions = Gauge(
    "locus_active_sessions",
    "Live search sessions (judge + attribute caches currently held in memory)",
)
locus_corrupt_items = Gauge(
    "locus_corrupt_items_total",
    "Products with corrupt_flag_count >= 1 in locus_items collection",
)
locus_low_score_flags_metric = Gauge(
    "locus_low_score_flags_total",
    "Products with low_score_flag=True in locus_items collection",
)

# User engagement tracking
locus_result_clicks = Counter(
    "locus_result_clicks_total",
    "Number of result card clicks by display position (1-indexed)",
    ["position"],
)
locus_results_time_on_page = Histogram(
    "locus_results_time_on_page_seconds",
    "Seconds a user spends on the results page before navigating away",
    buckets=[5, 10, 20, 30, 60, 120, 300, 600],
)

# Store & retailer analytics
locus_store_impressions = Counter(
    "locus_store_impressions_total",
    "Times a product from a store appeared in search results",
    ["store"],
)
locus_store_item_impressions = Counter(
    "locus_store_item_impressions_total",
    "Times a specific item appeared in search results",
    ["store", "item_name"],
)
locus_store_result_clicks = Counter(
    "locus_store_result_clicks_total",
    "Times a result card from a store was clicked to open the detail sheet",
    ["store"],
)
locus_store_website_clicks = Counter(
    "locus_store_website_clicks_total",
    "Times a user clicked the View on store website link",
    ["store"],
)
locus_store_directions_clicks = Counter(
    "locus_store_directions_clicks_total",
    "Times a user clicked Get directions for a store",
    ["store"],
)

# Per-store inventory gauges — refreshed every 60s from Qdrant (not from impressions)
locus_store_items = Gauge(
    "locus_store_items_total",
    "Number of indexed items per store in locus_items",
    ["store"],
)
locus_store_category_items = Gauge(
    "locus_store_category_items_total",
    "Number of indexed items per store per category in locus_items",
    ["store", "category"],
)

# Store registry metrics — refreshed every 60s from users.json
locus_store_info = Gauge(
    "locus_store_info",
    "Store registration info (value=1); metadata carried as labels",
    ["store_name", "email", "mall", "phone", "store_id"],
)
locus_stores_registered = Gauge(
    "locus_stores_registered_total",
    "Total number of registered store accounts",
)
locus_stores_missing_info = Gauge(
    "locus_stores_missing_info_total",
    "Registered stores missing store_name, phone, or email in users.json",
)
locus_link_monitor_broken = Gauge(
    "locus_link_monitor_broken_total",
    "Product URLs that returned 404 or failed in the last link health check",
)
_known_store_label_sets: set[tuple] = set()  # tracks label combos so deleted stores can be zeroed

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_UPLOAD_BYTES:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=413,
                content={"detail": "Payload too large. Maximum upload size is 10 MB."},
            )
    return await call_next(request)

VISUAL_URL          = os.getenv("VISUAL_HOST",    "http://visual_engine:8001")
TAGGER_HOST         = os.getenv("TAGGER_HOST",    "")
QDRANT_URL          = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY      = os.getenv("QDRANT_API_KEY")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
GATEWAY_BASE_URL    = os.getenv("GATEWAY_BASE_URL", "http://localhost:8000")
QDRANT_HOST         = os.getenv("QDRANT_HOST",    "qdrant")
QDRANT_PORT         = int(os.getenv("QDRANT_PORT", 6333))
# Experiment flag: apply background removal to accessory (shoes/bag/hat) query crops.
COLLECTION_NAME     = "locus_items"
SKIPPED_COLLECTION  = "locus_skipped"
FEEDBACK_COLLECTION = "locus_feedback"
GOLDEN_DATASET_PATH      = os.getenv("GOLDEN_DATASET_PATH", "/mlops/golden_dataset.json")
GOLDEN_IMAGES_DIR        = pathlib.Path(os.getenv("GOLDEN_IMAGES_DIR", "/mlops/golden_images"))
LINK_HEALTH_REPORT       = pathlib.Path("/app/link_health_report.json")
LINK_MONITOR_INTERVAL    = 432000  # 5 days in seconds (matches docker-compose sleep)

GOLDEN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/golden-dataset/images", StaticFiles(directory=str(GOLDEN_IMAGES_DIR)), name="golden_images")
if pathlib.Path("frontend/dist/assets").exists():
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="spa_assets")

if QDRANT_URL:
    print(f"[QDRANT] Connecting to cloud: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
elif os.getenv("QDRANT_HOST"):
    print(f"[QDRANT] Connecting to local: {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
else:
    raise RuntimeError(
        "QDRANT_URL is not set (or is empty). "
        "Set QDRANT_URL=https://<cluster>.qdrant.io:6333 in your .env file."
    )


def _compute_rating_gauges():
    """Synchronous helper: scroll locus_feedback and update all rating Prometheus gauges."""
    all_records = []
    cursor = None
    while True:
        batch, next_cursor = client.scroll(
            collection_name=FEEDBACK_COLLECTION,
            limit=500, offset=cursor,
            with_payload=True, with_vectors=False,
        )
        if not batch:
            break
        all_records.extend(batch)
        if next_cursor is None:
            break
        cursor = next_cursor

    records = [pt.payload for pt in all_records]

    by_star       = {str(i): 0 for i in range(1, 6)}
    by_star_user  = {str(i): 0 for i in range(1, 6)}
    by_star_judge = {str(i): 0 for i in range(1, 6)}

    for r in records:
        star   = r.get("rating", 0)
        source = r.get("source", "user")
        if 1 <= star <= 5:
            by_star[str(star)] += 1
            if source != "user":
                by_star_judge[str(star)] += 1
            else:
                by_star_user[str(star)] += 1

    for s in range(1, 6):
        locus_rating_by_star.labels(stars=str(s), source="all").set(by_star[str(s)])
        locus_rating_by_star.labels(stars=str(s), source="user").set(by_star_user[str(s)])
        locus_rating_by_star.labels(stars=str(s), source="judge").set(by_star_judge[str(s)])

    rated = [r["rating"] for r in records if r.get("rating")]
    locus_rating_avg.set(round(sum(rated) / len(rated), 2) if rated else 0.0)

    positives = sum(1 for r in records if r.get("training_signal") == "positive")
    negatives = sum(1 for r in records if r.get("training_signal") == "negative")
    neutrals  = sum(1 for r in records if r.get("training_signal") == "neutral")
    locus_rating_by_signal.labels(signal="positive").set(positives)
    locus_rating_by_signal.labels(signal="negative").set(negatives)
    locus_rating_by_signal.labels(signal="neutral").set(neutrals)

    user_count  = sum(1 for r in records if r.get("source") == "user")
    judge_count = len(records) - user_count
    locus_rating_by_source.labels(source="user").set(user_count)
    locus_rating_by_source.labels(source="auto_judge").set(judge_count)


def _compute_store_gauges():
    """Synchronous helper: count items per store and per store+category directly from Qdrant."""
    store_counts: dict[str, int] = {}
    store_cat_counts: dict[tuple[str, str], int] = {}
    cursor = None
    while True:
        batch, next_cursor = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=cursor,
            with_payload=["store_name", "category_tag"],
            with_vectors=False,
        )
        if not batch:
            break
        for pt in batch:
            store = (pt.payload.get("store_name") or "").strip()
            cat   = (pt.payload.get("category_tag") or "uncategorized").strip() or "uncategorized"
            if not store or store == "golden_dataset":
                continue
            store_counts[store] = store_counts.get(store, 0) + 1
            store_cat_counts[(store, cat)] = store_cat_counts.get((store, cat), 0) + 1
        if next_cursor is None:
            break
        cursor = next_cursor
    for store, count in store_counts.items():
        locus_store_items.labels(store=store).set(count)
        locus_store_website_clicks.labels(store=store)   # initialize time series at 0 if never clicked
        locus_store_result_clicks.labels(store=store)    # initialize time series at 0 if never clicked
        locus_store_impressions.labels(store=store)       # initialize so 0/0 = NaN (no data) not clicks/1
    for (store, cat), count in store_cat_counts.items():
        locus_store_category_items.labels(store=store, category=cat).set(count)


async def _refresh_qdrant_metrics():
    """Background task: refresh Qdrant collection counts + rating gauges every 60s."""
    while True:
        for col in [COLLECTION_NAME, SKIPPED_COLLECTION, FEEDBACK_COLLECTION]:
            try:
                info = client.get_collection(col)
                locus_qdrant_items.labels(collection=col).set(info.points_count or 0)
            except Exception as e:
                print(f"[METRICS] Could not refresh count for {col}: {e}")
        try:
            await asyncio.to_thread(_compute_rating_gauges)
        except Exception as e:
            print(f"[METRICS] Could not refresh rating gauges: {e}")
        try:
            await asyncio.to_thread(_compute_store_gauges)
        except Exception as e:
            print(f"[METRICS] Could not refresh store gauges: {e}")
        await asyncio.sleep(60)


async def _refresh_link_monitor_metrics():
    """Background task: update link health monitor timing metrics every 60s."""
    from datetime import timezone
    while True:
        try:
            if LINK_HEALTH_REPORT.exists():
                with open(LINK_HEALTH_REPORT) as f:
                    import json as _link_json
                    report = _link_json.load(f)
                run_at_str = report.get("run_at")
                if run_at_str:
                    run_at = datetime.fromisoformat(run_at_str)
                    if run_at.tzinfo is None:
                        run_at = run_at.replace(tzinfo=timezone.utc)
                    last_ts = run_at.timestamp()
                    next_ts = last_ts + LINK_MONITOR_INTERVAL
                    locus_link_monitor_last_run.set(last_ts * 1000)
                    locus_link_monitor_next_run.set(next_ts * 1000)
                locus_link_monitor_broken.set(report.get("broken_found", 0))
        except Exception as e:
            print(f"[METRICS] Could not refresh link monitor metrics: {e}")
        await asyncio.sleep(60)


async def _refresh_admin_metrics():
    """Background task: update session count and corrupt items every 30s."""
    while True:
        locus_active_sessions.set(len(_judge_scores) + len(_attribute_cache))
        try:
            hits, _ = await asyncio.to_thread(
                client.scroll,
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="corrupt_flag_count",
                        range=models.Range(gte=1),
                    )
                ]),
                limit=1000,
                with_payload=False,
                with_vectors=False,
            )
            locus_corrupt_items.set(len(hits))
        except Exception:
            pass
        try:
            low_hits, _ = await asyncio.to_thread(
                client.scroll,
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="low_score_flag",
                        match=models.MatchValue(value=True),
                    )
                ]),
                limit=1000,
                with_payload=False,
                with_vectors=False,
            )
            locus_low_score_flags_metric.set(len(low_hits))
        except Exception:
            pass
        try:
            _refresh_store_registry_metrics()
        except Exception:
            pass
        await asyncio.sleep(30)


def _refresh_store_registry_metrics() -> None:
    """Refresh locus_store_info and locus_stores_registered from users.json."""
    global _known_store_label_sets
    users = _load_users()
    current: set[tuple] = set()
    for email, u in users.items():
        labels = (
            u.get("store_name", ""),
            email,
            u.get("mall", ""),
            u.get("phone", ""),
            u.get("store_id", ""),
        )
        locus_store_info.labels(
            store_name=labels[0],
            email=labels[1],
            mall=labels[2],
            phone=labels[3],
            store_id=labels[4],
        ).set(1)
        current.add(labels)
    for stale in _known_store_label_sets - current:
        try:
            locus_store_info.labels(
                store_name=stale[0], email=stale[1],
                mall=stale[2], phone=stale[3], store_id=stale[4],
            ).set(0)
        except Exception:
            pass
    _known_store_label_sets = current
    locus_stores_registered.set(len(users))
    missing = sum(
        1 for u in users.values()
        if not u.get("store_name") or not u.get("email") or not u.get("phone")
    )
    locus_stores_missing_info.set(missing)


@app.on_event("startup")
async def startup_metrics():
    asyncio.create_task(_refresh_qdrant_metrics())
    asyncio.create_task(_refresh_link_monitor_metrics())
    asyncio.create_task(_refresh_admin_metrics())


def _init_qdrant_collections():
    # ── Main collection ───────────────────────────────────────────────────────
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    for field in ("category_tag", "store_name", "product_id", "box_source", "shoe_style"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema="keyword",
            )
        except Exception:
            pass
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="is_golden",
            field_schema=models.PayloadSchemaType.BOOL,
        )
    except Exception:
        pass
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="broken",
            field_schema=models.PayloadSchemaType.BOOL,
        )
    except Exception:
        pass
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="low_score_flag",
            field_schema=models.PayloadSchemaType.BOOL,
        )
    except Exception:
        pass

    # ── Skipped collection ────────────────────────────────────────────────────
    if not client.collection_exists(collection_name=SKIPPED_COLLECTION):
        client.create_collection(
            collection_name=SKIPPED_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
    for field in ("store_name", "skip_reason", "product_id"):
        try:
            client.create_payload_index(
                collection_name=SKIPPED_COLLECTION,
                field_name=field,
                field_schema="keyword",
            )
        except Exception:
            pass

    # ── Feedback collection ───────────────────────────────────────────────────
    # Dummy 1-dim vectors — we never do vector search here.
    # The training script reads this collection to build (result_id, rating) pairs,
    # then looks up result vectors from locus_items for contrastive fine-tuning.
    if not client.collection_exists(collection_name=FEEDBACK_COLLECTION):
        client.create_collection(
            collection_name=FEEDBACK_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
    for field in ("category", "store_name", "training_signal", "result_product_id", "source"):
        try:
            client.create_payload_index(
                collection_name=FEEDBACK_COLLECTION,
                field_name=field,
                field_schema="keyword",
            )
        except Exception:
            pass
    try:
        client.create_payload_index(
            collection_name=FEEDBACK_COLLECTION,
            field_name="rating",
            field_schema="integer",
        )
    except Exception:
        pass


@app.on_event("startup")
def startup_event():
    if not TAGGER_HOST:
        print("[WARNING] TAGGER_HOST is not set — attribute tagging will be silently disabled. "
              "Set TAGGER_HOST=http://attribute_tagger:8004 to enable it.")
    import time as _time
    for attempt in range(5):
        try:
            _init_qdrant_collections()
            break
        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt
                print(f"[STARTUP] Qdrant not reachable ({e}), retry {attempt + 1}/5 in {wait}s...")
                _time.sleep(wait)
            else:
                print(f"[STARTUP] WARNING: Could not initialize Qdrant collections after 5 attempts: {e}")


@app.get("/")
async def read_root():
    return FileResponse("frontend/dist/index.html")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _crop_image_bytes(image_bytes: bytes, x1: float, y1: float, x2: float, y2: float) -> bytes:
    img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img.size
    x1c = max(0, int(x1)); y1c = max(0, int(y1))
    x2c = min(W, int(x2)); y2c = min(H, int(y2))
    crop = img.crop((x1c, y1c, x2c, y2c))
    buf  = io.BytesIO()
    crop.save(buf, format="JPEG")
    return buf.getvalue()


def _store_skipped(
    product_id: str,
    name: str,
    image_url: str,
    store: str,
    mall: str,
    price: str,
    skip_reason: str,
):
    """Upsert a skipped product into locus_skipped collection."""
    try:
        client.upsert(
            collection_name=SKIPPED_COLLECTION,
            points=[
                PointStruct(
                    id     = str(uuid.uuid5(uuid.NAMESPACE_URL, f"skipped::{image_url or product_id}")),
                    vector = [0.0],
                    payload = {
                        "product_id":  product_id,
                        "name":        name,
                        "image_url":   image_url,
                        "store_name":  store,
                        "mall_name":   mall,
                        "price":       price,
                        "skip_reason": skip_reason,
                        "skipped_at":  datetime.utcnow().isoformat(),
                    }
                )
            ]
        )
    except Exception as e:
        print(f"[SKIPPED] Failed to store skipped product '{name}': {e}")


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK  (1–5 stars, stored to locus_feedback for contrastive fine-tuning)
# ══════════════════════════════════════════════════════════════════════════════
#
# Training signal mapping:
#   5 stars → strong positive pair  (weight 1.0)  pull embeddings closer
#   4 stars → mild positive pair    (weight 0.8)
#   3 stars → neutral               (weight 0.0)  skip during training
#   2 stars → mild negative pair    (weight 0.4)  push embeddings apart
#   1 star  → strong negative pair  (weight 1.0)
#
# The training script (mlops/train.py) reads locus_feedback, looks up each
# result_product_id in locus_items to retrieve its 512-dim vector, then uses
# (vector, rating) pairs to fine-tune the CLIP projection layer.
# ══════════════════════════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    result_product_id: str           # product_id from locus_items payload
    result_image_url:  str = ""      # for human-readable audit
    result_name:       str = ""      # for human-readable audit
    store_name:        str = ""
    category:          str = ""
    rating:            int           # 1–5 stars
    source:            str = "user"  # "user" | "auto_judge"


def _store_feedback_direct(payload: dict) -> None:
    """Write judge feedback directly to Qdrant — avoids HTTP roundtrip during shutdown."""
    rating = payload["rating"]
    if rating >= 4:
        signal, weight = "positive", 0.6 + (rating - 4) * 0.4
    elif rating == 3:
        signal, weight = "neutral", 0.0
    else:
        signal, weight = "negative", 0.4 + (2 - rating) * 0.6
    locus_feedback_stars.labels(stars=str(rating), signal=signal).inc()
    client.upsert(
        collection_name=FEEDBACK_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.0],
                payload={
                    **payload,
                    "training_signal": signal,
                    "weight":          round(weight, 2),
                    "timestamp":       datetime.utcnow().isoformat(),
                },
            )
        ],
    )
    print(f"[FEEDBACK] {rating}★ '{payload.get('result_name','')}' ({payload.get('category','')}) → {signal} w={weight:.2f}")


@app.post("/feedback")
async def receive_feedback(req: FeedbackRequest):
    """
    Store a 1-5 star rating for a search result.

    Called by the frontend after a user rates a result card.
    Each record is an (item, rating) pair that the training script
    uses to build contrastive learning pairs.
    """
    if not 1 <= req.rating <= 5:
        raise HTTPException(status_code=400, detail="rating must be between 1 and 5")

    if not req.result_product_id:
        raise HTTPException(status_code=400, detail="result_product_id is required")

    # Pre-compute training signal so the training script doesn't have to
    if req.rating >= 4:
        training_signal = "positive"
        weight          = 0.6 + (req.rating - 4) * 0.4   # 4→0.6, 5→1.0
    elif req.rating == 3:
        training_signal = "neutral"
        weight          = 0.0
    else:
        training_signal = "negative"
        weight          = 0.4 + (2 - req.rating) * 0.6   # 2→0.4, 1→1.0
    locus_feedback_stars.labels(stars=str(req.rating), signal=training_signal).inc()

    try:
        client.upsert(
            collection_name=FEEDBACK_COLLECTION,
            points=[
                PointStruct(
                    id     = str(uuid.uuid4()),   # unique per feedback event
                    vector = [0.0],               # dummy — never searched by vector
                    payload = {
                        "result_product_id": req.result_product_id,
                        "result_image_url":  req.result_image_url,
                        "result_name":       req.result_name,
                        "store_name":        req.store_name,
                        "category":          req.category,
                        "rating":            req.rating,
                        "training_signal":   training_signal,
                        "weight":            round(weight, 2),
                        "timestamp":         datetime.utcnow().isoformat(),
                        "source":            req.source,
                    }
                )
            ]
        )
        print(f"[FEEDBACK] {req.rating}★ '{req.result_name}' ({req.category}) → {training_signal} w={weight:.2f}")
        return {"status": "stored", "training_signal": training_signal, "weight": round(weight, 2)}

    except Exception as e:
        print(f"[FEEDBACK] Failed to store: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {e}")


@app.get("/feedback")
async def get_feedback(
    training_signal: str = "",   # filter: positive | negative | neutral
    category:        str = "",
    source:          str = "",   # filter: user | auto_judge
    min_rating:      int = 1,
    max_rating:      int = 5,
    limit:           int = 500,
):
    """
    Read feedback records for the training script.
    Returns (result_product_id, rating, training_signal, weight) pairs.

    Usage in mlops/train.py:
        resp = requests.get("http://gateway:8000/feedback?training_signal=positive")
        pairs = resp.json()["records"]
    """
    must_conditions = []

    if training_signal:
        must_conditions.append(models.FieldCondition(
            key="training_signal", match=models.MatchValue(value=training_signal)
        ))
    if category:
        must_conditions.append(models.FieldCondition(
            key="category", match=models.MatchValue(value=category)
        ))
    if source:
        must_conditions.append(models.FieldCondition(
            key="source", match=models.MatchValue(value=source)
        ))
    if min_rating > 1 or max_rating < 5:
        must_conditions.append(models.FieldCondition(
            key="rating",
            range=models.Range(gte=min_rating, lte=max_rating),
        ))

    scroll_filter = models.Filter(must=must_conditions) if must_conditions else None

    all_records = []
    cursor      = None
    while True:
        batch, next_cursor = client.scroll(
            collection_name=FEEDBACK_COLLECTION,
            scroll_filter=scroll_filter,
            limit=250,
            offset=cursor,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        all_records.extend(batch)
        if next_cursor is None:
            break
        cursor = next_cursor

    records = [pt.payload for pt in all_records]

    # Summary stats — useful for checking if there are enough pairs to train
    positives = sum(1 for r in records if r.get("training_signal") == "positive")
    negatives = sum(1 for r in records if r.get("training_signal") == "negative")
    neutrals  = sum(1 for r in records if r.get("training_signal") == "neutral")

    return {
        "total":    len(records),
        "summary":  {"positive": positives, "negative": negatives, "neutral": neutrals},
        "ready_to_train": (positives + negatives) >= 50,  # training threshold from roadmap
        "records":  records[:limit],
    }


@app.get("/rating-stats")
async def get_rating_stats():
    """Aggregate star-rating distribution for the admin dashboard."""
    all_records = []
    cursor = None
    while True:
        batch, next_cursor = client.scroll(
            collection_name=FEEDBACK_COLLECTION,
            limit=500,
            offset=cursor,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        all_records.extend(batch)
        if next_cursor is None:
            break
        cursor = next_cursor

    records = [pt.payload for pt in all_records]
    total = len(records)

    by_star       = {str(i): 0 for i in range(1, 6)}
    by_star_user  = {str(i): 0 for i in range(1, 6)}
    by_star_judge = {str(i): 0 for i in range(1, 6)}

    def is_judge(source: str) -> bool:
        return source != "user"

    for r in records:
        star   = r.get("rating", 0)
        source = r.get("source", "user")
        if 1 <= star <= 5:
            by_star[str(star)] += 1
            if is_judge(source):
                by_star_judge[str(star)] += 1
            else:
                by_star_user[str(star)] += 1

    rated      = [r["rating"] for r in records if r.get("rating")]
    avg_rating = round(sum(rated) / len(rated), 2) if rated else 0.0

    positives = sum(1 for r in records if r.get("training_signal") == "positive")
    negatives = sum(1 for r in records if r.get("training_signal") == "negative")
    neutrals  = sum(1 for r in records if r.get("training_signal") == "neutral")

    user_count  = sum(1 for r in records if not is_judge(r.get("source", "user")))
    judge_count = sum(1 for r in records if is_judge(r.get("source", "user")))

    return {
        "total":          total,
        "avg_rating":     avg_rating,
        "by_star":        by_star,
        "by_star_user":   by_star_user,
        "by_star_judge":  by_star_judge,
        "by_source":      {"user": user_count, "auto_judge": judge_count},
        "by_signal":      {"positive": positives, "negative": negatives, "neutral": neutrals},
        "ready_to_train": (positives + negatives) >= 50,
    }


# ── User engagement tracking ───────────────────────────────────────────────────

class ImpressionItem(BaseModel):
    store:     str = ""
    item_name: str = ""


class TrackEventRequest(BaseModel):
    event_type:       str                    # "result_click" | "results_exit" | "impressions" | "website_click" | "directions_click"
    search_id:        str             = ""
    position:         int             = 0    # 1-indexed display position (result_click only)
    duration_seconds: float           = 0.0  # seconds on results page (results_exit only)
    store:            str             = ""   # store name for click/conversion events
    item_name:        str             = ""   # item name for result_click
    impressions:      list[ImpressionItem] = []  # batch list for impressions event


@app.post("/track-event")
async def track_event(req: TrackEventRequest):
    if req.event_type == "result_click" and req.position > 0:
        locus_result_clicks.labels(position=str(req.position)).inc()
        if req.store:
            locus_store_result_clicks.labels(store=req.store).inc()

    elif req.event_type == "results_exit" and req.duration_seconds > 0:
        locus_results_time_on_page.observe(req.duration_seconds)

    elif req.event_type == "impressions" and req.impressions:
        for imp in req.impressions:
            store = imp.store or "unknown"
            locus_store_impressions.labels(store=store).inc()
            if imp.item_name:
                locus_store_item_impressions.labels(store=store, item_name=imp.item_name).inc()

    elif req.event_type == "website_click" and req.store:
        locus_store_website_clicks.labels(store=req.store).inc()

    elif req.event_type == "directions_click" and req.store:
        locus_store_directions_clicks.labels(store=req.store).inc()

    return {"status": "ok"}


# ── Detect ─────────────────────────────────────────────────────────────────────

@app.post("/detect")
@limiter.limit("5/minute")
async def detect_items(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file")
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{VISUAL_URL}/detect",
                files={"file": (file.filename, image_bytes, file.content_type)},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Visual engine unavailable")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    status = {"gateway": "ready", "visual_engine": "not_ready", "qdrant": "not_ready"}
    try:
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(f"{VISUAL_URL}/", timeout=3.0)
            if resp.status_code == 200:
                status["visual_engine"] = "ready"
    except Exception:
        status["visual_engine"] = "loading"
    try:
        client.get_collections()
        status["qdrant"] = "ready"
    except Exception:
        status["qdrant"] = "loading"
    return {"ready": all(v == "ready" for v in status.values()), "services": status}


@app.get("/stores/locations")
async def stores_locations():
    """Return {store_name: [lat, lng]} for all stores that have pinned coordinates."""
    return get_store_locations()


# ── Index Stats ────────────────────────────────────────────────────────────────

@app.get("/index-stats")
async def index_stats():
    try:
        source_counts: dict[str, int] = {}
        seen_product_ids: set         = set()
        total_points                  = 0
        offset                        = None

        while True:
            results, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not results:
                break
            for pt in results:
                total_points += 1
                p = pt.payload
                if p.get("is_dark") is True:
                    continue
                product_id = p.get("product_id", str(pt.id))
                if product_id in seen_product_ids:
                    continue
                seen_product_ids.add(product_id)
                src = p.get("box_source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
            if next_offset is None:
                break
            offset = next_offset

        tier_summary = {
            "exact": 0, "alias": 0, "best_available": 0,
            "full_image": 0, "unknown": 0, "other": 0,
        }
        for src, count in source_counts.items():
            if src.endswith("_exact"):            tier_summary["exact"] += count
            elif src.endswith("_alias"):          tier_summary["alias"] += count
            elif src.endswith("_best_available"): tier_summary["best_available"] += count
            elif src == "full_image":             tier_summary["full_image"] += count
            elif src == "unknown":                tier_summary["unknown"] += count
            else:                                 tier_summary["other"] += count

        return {
            "total_points":      total_points,
            "unique_products":   len(seen_product_ids),
            "by_box_source_raw": source_counts,
            "by_tier":           tier_summary,
            "reindex_required":  (tier_summary["full_image"] + tier_summary["unknown"] + tier_summary["best_available"]) > 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats query failed: {e}")


# ── Store Catalogue ────────────────────────────────────────────────────────────

@app.get("/store-catalogue")
async def store_catalogue(store_name: str, limit: int = 100, offset: int = 0):
    store_filter = models.Filter(
        must=[models.FieldCondition(
            key="store_name",
            match=models.MatchValue(value=store_name)
        )]
    )
    all_points = []
    next_offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=store_filter,
            limit=1000,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(batch)
        if next_offset is None:
            break
    seen_products = {}
    for point in all_points:
        p          = point.payload
        product_id = p.get("product_id", str(point.id))
        if product_id not in seen_products:
            seen_products[product_id] = {
                "id":           str(point.id),
                "name":         p.get("name", ""),
                "price":        p.get("price", ""),
                "category_tag": p.get("category_tag", ""),
                "image_url":    p.get("image_url", ""),
                "store_name":   p.get("store_name", ""),
                "mall_name":    p.get("mall_name", ""),
            }
    unique_products = list(seen_products.values())
    total           = len(unique_products)
    paginated       = unique_products[offset: offset + limit]
    return {"products": paginated, "total": total, "offset": offset, "limit": limit}


@app.get("/discover")
async def discover(limit: int = 15):
    """
    Return random product samples for the home screen discover feed.
    Three buckets: trending (all categories), women-oriented, men-oriented.
    Items with low_score_flag=True are excluded.
    """
    WOMEN_CATS = ["dress", "skirt", "jumpsuit", "bag", "sports_bra", "leggings", "top"]
    MEN_CATS   = ["pants", "jacket", "sweater", "hat", "shorts", "shoes", "top"]
    ALL_CATS   = ["dress", "skirt", "jumpsuit", "bag", "pants", "jacket", "sweater",
                  "hat", "shorts", "shoes", "top", "sports_bra", "leggings"]

    async def _sample_cats(categories: list, n: int) -> list:
        pool: list[dict] = []
        per_cat = max(3, (n * 3) // len(categories))
        for cat in categories:
            try:
                hits, _ = await asyncio.to_thread(
                    client.scroll,
                    collection_name=COLLECTION_NAME,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(key="category_tag", match=models.MatchValue(value=cat)),
                        ],
                        must_not=[
                            models.FieldCondition(key="low_score_flag",  match=models.MatchValue(value=True)),
                            models.FieldCondition(key="store_name",      match=models.MatchValue(value="golden_dataset")),
                            models.FieldCondition(key="broken",          match=models.MatchValue(value=True)),
                        ],
                    ),
                    limit=60,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                continue
            for point in random.sample(hits, min(per_cat, len(hits))):
                p = point.payload
                if not p.get("image_url"):
                    continue
                pool.append({
                    "product_id": p.get("product_id", str(point.id)),
                    "name":       p.get("name", ""),
                    "price":      p.get("price", ""),
                    "category":   p.get("category_tag", cat),
                    "image_url":  p.get("image_url", ""),
                    "store":      p.get("store_name", ""),
                    "mall":       p.get("mall_name", ""),
                })
        random.shuffle(pool)
        seen: set = set()
        out: list = []
        for item in pool:
            pid = item["product_id"]
            if pid not in seen:
                seen.add(pid)
                out.append(item)
            if len(out) >= n:
                break
        return out

    trending, women, men = await asyncio.gather(
        _sample_cats(ALL_CATS,   limit),
        _sample_cats(WOMEN_CATS, limit),
        _sample_cats(MEN_CATS,   limit),
    )
    return {"trending": trending, "women": women, "men": men}


@app.delete("/store-catalogue/item/{item_id}")
async def delete_catalogue_item(item_id: str, payload=Depends(verify_token)):
    """Delete a catalogue item. Only the owning store may delete its own items."""
    store_name = _load_users().get(payload["sub"], {}).get("store_name", "")
    try:
        # Verify the point belongs to the authenticated store before deleting.
        points = await asyncio.to_thread(
            client.retrieve,
            collection_name=COLLECTION_NAME,
            ids=[item_id],
            with_payload=True,
        )
        if points:
            item_store = points[0].payload.get("store_name", "")
            if item_store and item_store != store_name:
                raise HTTPException(status_code=403, detail="You can only delete items from your own store")
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=[item_id]),
        )
        return {"status": "deleted", "id": item_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/store-stats")
async def store_stats(payload=Depends(verify_token)):
    """Return aggregate stats for the authenticated store."""
    # Read store_name from DB so it stays correct after a profile rename.
    users = _load_users()
    u = users.get(payload["sub"])
    if not u:
        raise HTTPException(status_code=404, detail="Store account not found")
    store_name = u["store_name"]
    try:
        store_filter = models.Filter(must=[
            models.FieldCondition(key="store_name", match=models.MatchValue(value=store_name))
        ])
        all_points = []
        next_offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=store_filter,
                limit=1000,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(batch)
            if next_offset is None:
                break
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    seen: dict[str, dict] = {}
    category_counts: dict[str, int] = {}
    rating_sum = 0
    rating_count = 0
    week_ago = datetime.utcnow().timestamp() - 7 * 86400

    for point in all_points:
        p          = point.payload
        product_id = p.get("product_id", str(point.id))
        if product_id in seen:
            continue
        seen[product_id] = {
            "id":           str(point.id),
            "name":         p.get("name", ""),
            "price":        p.get("price", ""),
            "category_tag": p.get("category_tag", ""),
            "image_url":    p.get("image_url", ""),
            "updated_at":   p.get("updated_at", ""),
        }
        cat = p.get("category_tag") or "other"
        category_counts[cat] = category_counts.get(cat, 0) + 1
        rc = p.get("rating_count", 0) or 0
        rs = p.get("rating_sum", 0) or 0
        rating_count += rc
        rating_sum   += rs

    total       = len(seen)
    avg_rating  = round(rating_sum / rating_count, 2) if rating_count else None

    recent = sorted(
        seen.values(),
        key=lambda x: x.get("updated_at", ""),
        reverse=True,
    )[:10]

    return {
        "store_name":      store_name,
        "total_products":  total,
        "avg_rating":      avg_rating,
        "rating_count":    rating_count,
        "categories":      category_counts,
        "recent_products": recent,
    }


# ── Classify crop (query-time: user drew a box, CLIP predicts category) ────────

@app.post("/classify-crop")
async def classify_crop(
    file: UploadFile = File(...),
    x1:   float      = Form(0),
    y1:   float      = Form(0),
    x2:   float      = Form(0),
    y2:   float      = Form(0),
):
    """
    Crops the uploaded image to the given bbox and runs CLIP classification.
    Returns the predicted category and all per-category scores.
    Called by the frontend after the user draws a bounding box, before searching.
    """
    image_bytes = await file.read()
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file")
    crop_bytes  = _crop_image_bytes(image_bytes, x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else image_bytes

    try:
        async with httpx.AsyncClient() as http:
            vis_response = await http.post(
                f"{VISUAL_URL}/vectorize",
                files={"file": (file.filename, crop_bytes, "image/jpeg")},
                data={"yolo_label": "", "darken": "false", "query": "true"},
                timeout=60.0,
            )
            vis_response.raise_for_status()
            vis_data = vis_response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Visual engine unavailable")

    all_scores = vis_data.get("category_confidence", {})
    category   = vis_data.get("category")
    confidence = all_scores.get(category, 0.0) if isinstance(all_scores, dict) else 0.0

    return {
        "category":   category,
        "confidence": round(confidence, 3),
        "all_scores": all_scores,
    }


# ── Search ─────────────────────────────────────────────────────────────────────

@app.post("/search")
@limiter.limit("10/minute")
async def search_items(
    request:            Request,
    background_tasks:   BackgroundTasks,
    file:               UploadFile = File(...),
    x1:                 float      = Form(0),
    y1:                 float      = Form(0),
    x2:                 float      = Form(0),
    y2:                 float      = Form(0),
    search_label:       str        = Form(""),
    shoe_style:         str        = Form(""),   # optional sub-type for shoes (sneaker|boot|heel|sandal)
    nearby_stores:      str        = Form(""),   # comma-separated store names to restrict results by radius
    include_golden:     bool       = False,
    skip_judge:         bool       = Form(False),
):
    image_bytes = await file.read()

    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file")

    has_bbox = x2 > x1 and y2 > y1
    if has_bbox:
        crop_bytes = _crop_image_bytes(image_bytes, x1, y1, x2, y2)
        print(f"[SEARCH] Cropped to bbox ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    else:
        crop_bytes = image_bytes
        print("[SEARCH] No bbox — using full image")

    # Compute style hint for accessories so the visual engine can apply hybrid vector mixing.
    _style_hint = ""
    _effective_search = search_label or ""
    if _effective_search == "shoes":
        _style_hint = _SHOE_STYLE_HINTS.get(shoe_style or "other", "shoe footwear")
    elif _effective_search == "bag":
        _style_hint = _BAG_HINT
    elif _effective_search == "hat":
        _style_hint = _HAT_HINT
    if _style_hint:
        print(f"[SEARCH] style_hint='{_style_hint[:60]}' for category='{_effective_search}'")

    try:
        async with httpx.AsyncClient() as http:
            vis_response = await http.post(
                f"{VISUAL_URL}/vectorize",
                files={"file": (file.filename, crop_bytes, "image/jpeg")},
                data={"yolo_label": search_label, "darken": "false", "style_hint": _style_hint},
                timeout=60.0,
            )
            vis_response.raise_for_status()
            vis_data = vis_response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Visual engine unavailable")

    vector              = vis_data.get("vector")
    detected_category   = vis_data.get("category")
    category_confidence = vis_data.get("category_confidence", {})
    processed_image     = vis_data.get("debug_image")

    if not vector:
        return JSONResponse({"matches": [], "detected_category": None, "warning": "Image could not be embedded — no fashion content detected"})

    locus_searches.labels(category=detected_category or "unknown").inc()

    mismatch_warning = None
    if search_label and detected_category and search_label != detected_category:
        top_conf = max(category_confidence.values()) if isinstance(category_confidence, dict) else 0.0
        mismatch_warning = (
            f"Selected box category '{search_label}' differs from CLIP detection "
            f"'{detected_category}' (conf={top_conf:.2f}). "
            f"Searching in '{search_label}' category as selected."
        )
        print(f"[SEARCH] WARNING: {mismatch_warning}")

    # ── Accessory uncertainty fallback ────────────────────────────────────────
    # fashion-CLIP is biased toward clothing and frequently scores accessories
    # (hat, bag, shoes) near zero even when the query image is clearly an
    # accessory.  When the sum of accessory scores exceeds a low threshold,
    # drop the category filter so accessory items are reachable in results.
    _ACCESSORY_LABELS = {"hat", "bag", "shoes"}
    _CLOTHING_LABELS  = {
        "top", "sports_bra", "pants", "leggings", "shorts",
        "skirt", "dress", "sweater", "jacket", "jumpsuit",
    }
    _scores           = category_confidence if isinstance(category_confidence, dict) else {}
    _accessory_signal = sum(_scores.get(k, 0.0) for k in _ACCESSORY_LABELS)

    skip_filter_for_accessory = (
        not search_label                          # user has not selected a box
        and detected_category in _CLOTHING_LABELS # CLIP said clothing
        and _accessory_signal > 0.005             # but accessory signal is non-trivial
    )
    if skip_filter_for_accessory:
        print(f"[SEARCH] Accessory uncertainty (accessory_signal={_accessory_signal:.4f}) "
              f"— dropping category filter, returning unfiltered results")

    # ── Dress / skirt ambiguity fallback ──────────────────────────────────────
    # CLIP (fashion-ViT) frequently classifies slip/midi/satin dresses as skirts
    # because it fixates on the lower half of the garment.  When CLIP says skirt
    # but the dress score is competitive (≥ 70 % of the skirt score), widen the
    # Qdrant filter to match both categories so indexed dresses stay reachable.
    _dress_score = _scores.get("dress", 0.0)
    _skirt_score = _scores.get("skirt", 0.0)
    widen_dress_skirt = (
        not search_label                                      # user has not selected a box
        and detected_category == "skirt"                      # CLIP said skirt
        and _skirt_score > 0
        and _dress_score / _skirt_score >= 0.70               # dress is competitive
    )
    if widen_dress_skirt:
        print(f"[SEARCH] Dress/skirt ambiguity (dress={_dress_score:.4f} skirt={_skirt_score:.4f}) "
              f"— widening category filter to dress+skirt")
    # ─────────────────────────────────────────────────────────────────────────

    query_filter    = None
    effective_label = None if skip_filter_for_accessory else (search_label or detected_category)

    must_conditions = []
    if effective_label:
        if widen_dress_skirt:
            must_conditions.append(models.FieldCondition(
                key="category_tag",
                match=models.MatchAny(any=["dress", "skirt"])
            ))
        else:
            must_conditions.append(models.FieldCondition(
                key="category_tag",
                match=models.MatchValue(value=effective_label)
            ))
        # Shoe sub-type filter: when both category=shoes and a shoe_style are
        # provided, narrow the search to that sub-type (sneaker/boot/heel/sandal).
        # Falls back gracefully to shoes-only if shoe_style is missing or "other".
        if effective_label == "shoes" and shoe_style and shoe_style != "other":
            must_conditions.append(models.FieldCondition(
                key="shoe_style",
                match=models.MatchValue(value=shoe_style)
            ))
            print(f"[SEARCH] Shoe sub-filter: shoe_style='{shoe_style}'")

    # Do NOT pre-filter Qdrant to nearby stores. Instead we search globally for the
    # top-N most visually similar items and let the frontend filter by radius
    # client-side. This ensures results always appear even when nearby stores have
    # a thin catalogue, and makes the radius slider on the results page interactive.
    store_filter = [s.strip() for s in nearby_stores.split(",") if s.strip()]
    if store_filter:
        print(f"[SEARCH] Radius hint: {len(store_filter)} nearby stores (client-side filtered)")

    # Exclude golden dataset items in normal searches; include them when toggled
    must_not_conditions = []
    if not include_golden:
        must_not_conditions.append(models.FieldCondition(
            key="store_name",
            match=models.MatchValue(value="golden_dataset")
        ))
    # Exclude items with broken image URLs (hidden until repaired)
    must_not_conditions.append(models.FieldCondition(
        key="broken",
        match=models.MatchValue(value=True)
    ))
    # Exclude items flagged as low-quality by Gemini judge (admin-reviewable)
    must_not_conditions.append(models.FieldCondition(
        key="low_score_flag",
        match=models.MatchValue(value=True)
    ))

    if must_conditions or must_not_conditions:
        query_filter = models.Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None,
        )

    # Build the fallback filter (shoes-only, no sub-type) for graceful degradation
    fallback_filter = query_filter
    shoe_style_active = (
        effective_label == "shoes"
        and shoe_style
        and shoe_style != "other"
        and query_filter is not None
    )
    if shoe_style_active:
        # Fallback = shoes-only without shoe_style sub-filter
        fallback_filter = models.Filter(
            must=[models.FieldCondition(
                key="category_tag",
                match=models.MatchValue(value="shoes")
            )],
            must_not=must_not_conditions if must_not_conditions else None,
        )

    raw_results = []
    for _attempt in range(3):
        try:
            raw_results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                query_filter=query_filter,
                limit=100,
                search_params=models.SearchParams(hnsw_ef=512),
            )
            break
        except Exception as _e:
            if _attempt == 2:
                _qdrant_target = QDRANT_URL or f"{QDRANT_HOST}:{QDRANT_PORT}"
                print(f"[SEARCH] Qdrant unreachable after 3 attempts (target={_qdrant_target}): {_e}")
                raise
            import time as _time
            print(f"[SEARCH] Qdrant error (attempt {_attempt+1}/3): {_e} — retrying in 2s")
            _time.sleep(2)

    # Graceful fallback: if shoe_style sub-filter produced 0 results (catalog not
    # yet re-indexed with shoe_style), retry with shoes-only filter.
    if shoe_style_active and len(raw_results) == 0:
        print(f"[SEARCH] shoe_style='{shoe_style}' returned 0 results — falling back to shoes-only filter")
        raw_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            query_filter=fallback_filter,
            limit=100,
            search_params=models.SearchParams(hnsw_ef=512),
        )

    # ── Corrupt image detection ───────────────────────────────────────────────
    # Count how many times each product_id appears in the top 3 raw CLIP hits.
    # A product taking ≥2 of those slots has duplicate vectors in Qdrant and is
    # a candidate for flagging. Confirmation via judge score happens in background.
    _top3_raw      = raw_results[:3]
    _pid_counts: dict    = {}
    _pid_raw_points: dict = {}
    for _hit in _top3_raw:
        _pid = (_hit.payload.get("product_id")
                or _hit.payload.get("image_url")
                or str(_hit.id))
        _pid_counts[_pid]  = _pid_counts.get(_pid, 0) + 1
        _pid_raw_points.setdefault(_pid, []).append(_hit.id)
    _suspicious_pids = {
        pid: pts for pid, pts in _pid_raw_points.items() if _pid_counts[pid] >= 2
    }
    if _suspicious_pids:
        print(f"[CORRUPT] Detected {len(_suspicious_pids)} suspicious product(s) in top-3 raw hits — "
              f"will confirm with judge scores: {list(_suspicious_pids)[:2]}")

    best_per_product = {}
    for hit in raw_results:
        product_id = hit.payload.get("product_id") or hit.payload.get("image_url") or str(hit.id)
        if product_id not in best_per_product or hit.score > best_per_product[product_id]["score"]:
            best_per_product[product_id] = {
                "name":       hit.payload.get("name", "Unknown"),
                "store_name": hit.payload.get("store_name", "Unknown"),
                "store":      hit.payload.get("store_name", ""),
                "mall_name":  hit.payload.get("mall_name", "Unknown"),
                "price":      hit.payload.get("price", ""),
                "score":      round(hit.score, 3),
                "image_url":  hit.payload.get("image_url", ""),
                # product_id is the key the feedback endpoint needs
                "product_id": product_id,
            }

    matches = sorted(best_per_product.values(), key=lambda x: x["score"], reverse=True)[:15]

    maps_urls = get_store_maps_urls()
    for m in matches:
        m["maps_url"] = maps_urls.get(m.get("store_name", ""), "")

    search_id = str(uuid.uuid4())[:12]

    # ── Attribute tagger: extract visual attributes in background ─────────────
    _cleanup_attribute_cache()
    _original_results[search_id]  = matches
    _attribute_cache[search_id]   = None  # pending
    _attr_timestamps[search_id]   = _time_module.monotonic()
    if TAGGER_HOST and detected_category and matches:
        crop_b64 = base64.b64encode(crop_bytes).decode("utf-8")
        background_tasks.add_task(_run_tagger, search_id, crop_b64, detected_category)

    scores_dict  = {}

    if (OPENROUTER_API_KEY or GOOGLE_API_KEY) and not skip_judge and not include_golden:
        _cleanup_judge_scores()
        _judge_scores[search_id]     = scores_dict
        _judge_timestamps[search_id] = _time_module.monotonic()
        background_tasks.add_task(
            run_judge,
            crop_bytes,
            matches,
            GATEWAY_BASE_URL,
            OPENROUTER_API_KEY,
            scores_dict,
            GOOGLE_API_KEY,
            _store_feedback_direct,
        )
        if _suspicious_pids:
            background_tasks.add_task(
                _audit_corrupt_items,
                _suspicious_pids,
                _pid_counts,
                matches[:3],
                scores_dict,
            )
        # Always audit top-3 for low Gemini scores; flagged items are hidden from
        # future searches and surfaced in the admin dashboard for review.
        # Passing matches[:3] directly ensures judge scores stay linked to the
        # correct item metadata regardless of any reranking.
        background_tasks.add_task(
            _audit_low_score_top3,
            matches[:3],
            scores_dict,
            effective_label or detected_category or "",
        )

    return {
        "matches":                   matches,
        "search_id":                 search_id,
        "debug_image":               processed_image,
        "detected_category":         detected_category,
        "category_confidence":       category_confidence,
        "category_mismatch_warning": mismatch_warning,
    }


# ── Judge score polling ────────────────────────────────────────────────────────

@app.get("/judge-scores/{search_id}")
async def get_judge_scores(search_id: str):
    """
    Returns the judge scores collected so far for a given search.
    Returns {product_id: float} — only entries scored so far are included.
    Frontend polls this every 3s after receiving search results.
    """
    return _judge_scores.get(search_id, {})


# ── Attribute polling ──────────────────────────────────────────────────────────

@app.get("/search/{search_id}/attributes")
async def get_attributes(search_id: str):
    """
    Returns the visual attributes extracted by the attribute_tagger for a search.
    Frontend polls after receiving search results. Responds with status=pending
    while Gemini is still running, status=ready once attributes arrive.
    """
    if search_id not in _attribute_cache:
        return {"status": "not_found"}
    attrs = _attribute_cache[search_id]
    if attrs is None:
        return {"status": "pending"}
    return {"status": "ready", "attributes": attrs}


# ── Result refinement ──────────────────────────────────────────────────────────

class RefineRequest(BaseModel):
    search_id: str
    mode: str        # "style" | "color" | "visual"
    attributes: dict
    category: str


@app.post("/refine")
async def refine_results(body: RefineRequest):
    """
    Re-queries Qdrant using a CLIP text embedding derived from detected attributes.

    mode="visual"  → returns the original CLIP-ranked results (no re-query)
    mode="style"   → text = "{style} {silhouette} {category}" → text-guided Qdrant search
    mode="color"   → text = "{primary_color} {category} clothing" → text-guided Qdrant search
    """
    if body.mode == "visual":
        return {"results": _original_results.get(body.search_id, []), "mode": "visual"}

    if body.mode == "style":
        style     = body.attributes.get("style", "")
        silhouette = body.attributes.get("silhouette", "")
        text = " ".join(filter(None, [style, silhouette, body.category])).strip()
    elif body.mode == "color":
        colors = body.attributes.get("colors", [])
        primary = colors[0] if colors else ""
        text = " ".join(filter(None, [primary, body.category, "clothing"])).strip()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {body.mode!r}")

    if not text:
        raise HTTPException(status_code=400, detail="Could not build text query from attributes")

    # Get CLIP text embedding from visual_engine
    try:
        async with httpx.AsyncClient(timeout=10.0) as _vc:
            vec_resp = await _vc.post(
                f"{VISUAL_URL}/vectorize-text",
                json={"text": text},
            )
        vec_resp.raise_for_status()
        text_vector = vec_resp.json()["embedding"]
    except Exception as _ve:
        raise HTTPException(status_code=502, detail=f"visual_engine /vectorize-text failed: {_ve}")

    # Build category filter (same logic as /search)
    must_conditions = []
    if body.category:
        if body.category in ("dress", "skirt"):
            must_conditions.append(models.FieldCondition(
                key="category_tag",
                match=models.MatchAny(any=["dress", "skirt"]),
            ))
        else:
            must_conditions.append(models.FieldCondition(
                key="category_tag",
                match=models.MatchValue(value=body.category),
            ))

    refine_filter = models.Filter(
        must=must_conditions or None,
        must_not=[
            models.FieldCondition(key="broken",         match=models.MatchValue(value=True)),
            models.FieldCondition(key="store_name",     match=models.MatchValue(value="golden_dataset")),
            models.FieldCondition(key="low_score_flag", match=models.MatchValue(value=True)),
        ],
    )

    raw = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=text_vector,
        query_filter=refine_filter,
        limit=50,
    )

    best_per_product: dict = {}
    for hit in raw:
        pid = hit.payload.get("product_id") or hit.payload.get("image_url") or str(hit.id)
        if pid not in best_per_product or hit.score > best_per_product[pid]["score"]:
            best_per_product[pid] = {
                "name":       hit.payload.get("name", "Unknown"),
                "store_name": hit.payload.get("store_name", "Unknown"),
                "store":      hit.payload.get("store_name", ""),
                "mall_name":  hit.payload.get("mall_name", "Unknown"),
                "price":      hit.payload.get("price", ""),
                "score":      round(hit.score, 3),
                "image_url":  hit.payload.get("image_url", ""),
                "product_id": pid,
            }

    results = sorted(best_per_product.values(), key=lambda x: x["score"], reverse=True)[:25]
    maps_urls = get_store_maps_urls()
    for r in results:
        r["maps_url"] = maps_urls.get(r.get("store_name", ""), "")
    return {"results": results, "mode": body.mode, "query_text": text}


# ══════════════════════════════════════════════════════════════════════════════
# SKIPPED PRODUCTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/skipped-products")
async def get_skipped_products(
    store_name:  str = "",
    skip_reason: str = "",
    limit:       int = 50,
    offset:      int = 0,
):
    must_conditions = []
    if store_name:
        must_conditions.append(models.FieldCondition(
            key="store_name", match=models.MatchValue(value=store_name)
        ))
    if skip_reason:
        must_conditions.append(models.FieldCondition(
            key="skip_reason", match=models.MatchValue(value=skip_reason)
        ))

    scroll_filter = models.Filter(must=must_conditions) if must_conditions else None

    all_results = []
    cursor      = None
    while True:
        batch, next_cursor = client.scroll(
            collection_name=SKIPPED_COLLECTION,
            scroll_filter=scroll_filter,
            limit=250,
            offset=cursor,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        all_results.extend(batch)
        if next_cursor is None:
            break
        cursor = next_cursor

    total     = len(all_results)
    paginated = all_results[offset: offset + limit]
    products  = [pt.payload | {"point_id": str(pt.id)} for pt in paginated]

    return {"products": products, "total": total, "offset": offset, "limit": limit}


@app.delete("/skipped-products/{point_id}")
async def delete_skipped_product(point_id: str):
    try:
        client.delete(
            collection_name=SKIPPED_COLLECTION,
            points_selector=models.PointIdsList(points=[point_id]),
        )
        return {"status": "deleted", "id": point_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# LOW-SCORE FLAGS  (admin review panel)
# ══════════════════════════════════════════════════════════════════════════════

class LowScoreFlagRequest(BaseModel):
    product_id: str


class ManualFlagRequest(BaseModel):
    product_id: str
    name:       str = ""
    store_name: str = ""
    image_url:  str = ""


@app.post("/admin/flag-item")
async def admin_flag_item(body: ManualFlagRequest):
    """Manually suppress a search result. Sets low_score_flag=True in Qdrant and records it in the admin flags JSON."""
    pid = body.product_id
    existing_flags = _read_low_score_flags()
    if any(f["product_id"] == pid for f in existing_flags):
        return {"status": "already_flagged", "product_id": pid}

    try:
        scroll_hits, _ = await asyncio.to_thread(
            client.scroll,
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="product_id", match=models.MatchValue(value=pid))
            ]),
            limit=50,
            with_payload=False,
        )
        if not scroll_hits:
            raise HTTPException(status_code=404, detail="product_id not found in Qdrant")
        await asyncio.to_thread(
            client.set_payload,
            collection_name=COLLECTION_NAME,
            payload={"low_score_flag": True},
            points=[p.id for p in scroll_hits],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    new_entry = {
        "product_id": pid,
        "image_url":  body.image_url,
        "name":       body.name,
        "store_name": body.store_name,
        "judge_score": None,
        "flagged_at": datetime.utcnow().isoformat(),
        "manual":     True,
    }
    _write_low_score_flags(existing_flags + [new_entry])
    print(f"[LOW_SCORE] MANUALLY FLAGGED '{body.name}' ({pid[:60]}) by admin")
    return {"status": "flagged", "product_id": pid}


@app.get("/low-score-flags")
async def get_low_score_flags():
    """Return all products currently flagged for low Gemini judge score."""
    return {"flags": _read_low_score_flags()}


@app.post("/low-score-flags/dismiss")
async def dismiss_low_score_flag(body: LowScoreFlagRequest):
    """Un-flag a product: restore it to search results and remove it from the admin list."""
    pid = body.product_id
    try:
        scroll_hits, _ = await asyncio.to_thread(
            client.scroll,
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="product_id", match=models.MatchValue(value=pid))
            ]),
            limit=50,
            with_payload=False,
        )
        if scroll_hits:
            await asyncio.to_thread(
                client.set_payload,
                collection_name=COLLECTION_NAME,
                payload={"low_score_flag": False},
                points=[p.id for p in scroll_hits],
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    flags = _read_low_score_flags()
    _write_low_score_flags([f for f in flags if f.get("product_id") != pid])
    print(f"[LOW_SCORE] DISMISSED flag for product {pid[:60]}")
    return {"status": "dismissed", "product_id": pid}


@app.post("/low-score-flags/confirm-remove")
async def confirm_remove_flagged_item(body: LowScoreFlagRequest):
    """Permanently delete a low-score-flagged product from Qdrant and the admin list."""
    pid = body.product_id
    try:
        scroll_hits, _ = await asyncio.to_thread(
            client.scroll,
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="product_id", match=models.MatchValue(value=pid))
            ]),
            limit=50,
            with_payload=False,
        )
        if scroll_hits:
            await asyncio.to_thread(
                client.delete,
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=[p.id for p in scroll_hits]),
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    flags = _read_low_score_flags()
    _write_low_score_flags([f for f in flags if f.get("product_id") != pid])
    print(f"[LOW_SCORE] DELETED product {pid[:60]} by admin request")
    return {"status": "deleted", "product_id": pid}


# ══════════════════════════════════════════════════════════════════════════════
# SUPER-ADMIN — STORE MANAGEMENT
# Protected by X-Admin-Key header (ADMIN_API_KEY env var).
# ══════════════════════════════════════════════════════════════════════════════

def _require_admin(request: Request) -> None:
    key = request.headers.get("X-Admin-Key", "")
    if not key or key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin access required — provide X-Admin-Key header")


@app.get("/admin/stores")
async def admin_list_stores(request: Request):
    """Return all registered stores with account info (no passwords)."""
    _require_admin(request)
    users = _load_users()
    stores = []
    for email, u in users.items():
        stores.append({
            "store_id":   u.get("store_id", ""),
            "store_name": u.get("store_name", ""),
            "email":      email,
            "mall":       u.get("mall", ""),
            "phone":      u.get("phone", ""),
            "created_at": u.get("created_at", ""),
        })
    stores.sort(key=lambda s: s["created_at"], reverse=True)
    return {"stores": stores, "total": len(stores)}


@app.delete("/admin/stores/{store_id}")
async def admin_delete_store(store_id: str, request: Request):
    """Remove a store account. Does not delete Qdrant products."""
    _require_admin(request)
    from auth import _save_users
    users = _load_users()
    target_email = next((e for e, u in users.items() if u.get("store_id") == store_id), None)
    if not target_email:
        raise HTTPException(404, "Store not found")
    del users[target_email]
    _save_users(users)
    _refresh_store_registry_metrics()
    print(f"[ADMIN] Deleted store account {target_email} (id={store_id})")
    return {"status": "deleted", "store_id": store_id, "email": target_email}


# ══════════════════════════════════════════════════════════════════════════════
# /add-bulk-batch
# ══════════════════════════════════════════════════════════════════════════════

class BulkBatchRequest(BaseModel):
    items: list[dict]


@app.post("/add-bulk-batch")
async def add_bulk_batch(batch: BulkBatchRequest):
    semaphore = asyncio.Semaphore(2)

    async def index_one(raw: dict):
        name   = raw.get("name", "Product")
        store  = raw.get("store", "")
        mall   = raw.get("mall", "")
        price  = raw.get("price", "")

        image_urls = raw.get("image_urls") or []
        if not image_urls and raw.get("image_url"):
            image_urls = [raw["image_url"]]
        if not image_urls:
            return {"status": "failed", "item": name, "error": "no image URL"}

        img_url    = image_urls[0]
        product_id = raw.get("product_id") or str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"{name}::{store}"
        ))

        async with semaphore:
            try:
                async with httpx.AsyncClient() as http:
                    img_resp = await http.get(img_url, timeout=15.0, follow_redirects=True)
                    img_resp.raise_for_status()
                    img_bytes    = img_resp.content
                    content_type = img_resp.headers.get("content-type", "image/jpeg")

                    idx_resp = await http.post(
                        f"{VISUAL_URL}/index-image",
                        files={"file": ("product.jpg", img_bytes, content_type)},
                        data={"title": name},
                        timeout=90.0,
                    )
                    idx_resp.raise_for_status()
                    idx_data = idx_resp.json()

                    if idx_data.get("skipped"):
                        reason = idx_data.get("skip_reason", "unknown")
                        print(f"[BATCH] Skipped '{name}': {reason}")
                        _store_skipped(
                            product_id  = product_id,
                            name        = name,
                            image_url   = img_url,
                            store       = store,
                            mall        = mall,
                            price       = price,
                            skip_reason = reason,
                        )
                        return {"status": "skipped", "item": name, "reason": reason}

                    vector_normal  = idx_data["vector_normal"]
                    final_category = idx_data.get("category", "unknown")
                    box_source     = idx_data.get("box_source", "unknown")
                    shoe_style     = idx_data.get("shoe_style")

                    batch_payload = {
                        "name":         name,
                        "store_name":   store,
                        "mall_name":    mall,
                        "image_url":    img_url,
                        "category_tag": final_category,
                        "price":        price,
                        "product_id":   product_id,
                        "box_source":   box_source,
                        "updated_at":   datetime.utcnow().isoformat(),
                    }
                    if shoe_style:
                        batch_payload["shoe_style"] = shoe_style
                    if raw.get("is_golden"):
                        batch_payload["is_golden"] = True

                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[PointStruct(
                            id      = str(uuid.uuid5(uuid.NAMESPACE_URL, img_url)),
                            vector  = vector_normal,
                            payload = batch_payload,
                        )]
                    )
                return {"status": "ok", "item": name}

            except Exception as e:
                print(f"[BATCH] Failed: {name} — {e}")
                return {"status": "failed", "item": name, "error": str(e)}

    results = await asyncio.gather(*[index_one(raw) for raw in batch.items])
    success = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed  = [r for r in results if r["status"] == "failed"]

    return {"success": success, "skipped": skipped, "total": len(batch.items), "failed": failed}


# ══════════════════════════════════════════════════════════════════════════════
# /scrape
# ══════════════════════════════════════════════════════════════════════════════

class ScrapeRequest(BaseModel):
    url:          str
    max_products: int = 0


@app.post("/scrape")
async def scrape_store(req: ScrapeRequest):
    req_headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control":   "no-cache",
        "Referer":         "https://www.google.com/",
    }

    parsed   = urlparse(req.url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as http:
        shopify_products = await _try_shopify_api(http, req.url, base_url, req.max_products, req_headers)
        if shopify_products:
            return {"products": shopify_products, "total_found": len(shopify_products), "source_url": req.url, "strategy": "shopify_api"}

        try:
            page_resp = await http.get(req.url, headers=req_headers)
            if page_resp.status_code == 429:
                return {"products": [], "total_found": 0, "source_url": req.url, "strategy": "rate_limited", "error": "Site is rate-limiting. Use the Chrome extension scraper instead."}
            if page_resp.status_code not in (200, 301, 302):
                raise HTTPException(status_code=400, detail=f"Site returned HTTP {page_resp.status_code}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot reach URL: {e}")

        soup     = BeautifulSoup(page_resp.text, "lxml")
        products = []

        for script_tag in soup.find_all("script", type="application/ld+json"):
            try:
                data  = _json.loads(script_tag.string or "")
                items = data if isinstance(data, list) else [data]
                for obj in items:
                    if obj.get("@type") == "ItemList":
                        for element in obj.get("itemListElement", []):
                            _extract_ld_product(element.get("item", element), products)
                    elif obj.get("@type") in ("Product", "product"):
                        _extract_ld_product(obj, products)
                    elif "@graph" in obj:
                        for node in obj["@graph"]:
                            if node.get("@type") == "Product":
                                _extract_ld_product(node, products)
            except Exception:
                continue

        if not products:
            og_image = soup.find("meta", property="og:image")
            og_title = soup.find("meta", property="og:title")
            og_price = soup.find("meta", property="product:price:amount")
            if og_image and og_image.get("content"):
                products.append({
                    "name":       (og_title["content"] if og_title else "Product").strip(),
                    "image_url":  og_image["content"].strip(),
                    "image_urls": [og_image["content"].strip()],
                    "price":      (og_price["content"] if og_price else ""),
                })

        if not products:
            products = _scrape_product_cards(soup, req.url)

    seen, unique = set(), []
    for p in products:
        key = p.get("image_url", "") or (p.get("image_urls") or [""])[0]
        if key and key not in seen and p.get("name"):
            seen.add(key)
            unique.append(p)

    return {"products": unique, "total_found": len(unique), "source_url": req.url, "strategy": "html_fallback"}


async def _try_shopify_api(http, original_url, base_url, max_products, headers):
    path  = urlparse(original_url).path
    parts = [p for p in path.split("/") if p]

    base_endpoints = []
    if len(parts) >= 2 and parts[0] == "collections":
        base_endpoints.append(f"{base_url}/collections/{parts[1]}/products.json")
    base_endpoints.append(f"{base_url}/products.json")

    for base_endpoint in base_endpoints:
        try:
            all_products = []
            page         = 1

            while True:
                api_url = f"{base_endpoint}?limit=250&page={page}"
                resp    = await http.get(api_url, headers=headers, timeout=15.0)
                print(f"[SCRAPE] {api_url} → {resp.status_code} ({len(resp.content)} bytes)")

                if resp.status_code != 200:
                    break

                try:
                    import gzip as _gzip
                    content = resp.content
                    if content[:2] == b'\x1f\x8b':
                        content = _gzip.decompress(content)
                    data = _json.loads(content)
                except Exception:
                    print(f"[SCRAPE] Could not parse JSON: {resp.content[:50]}")
                    break

                raw_products = data.get("products", [])
                if not raw_products:
                    break

                for p in raw_products:
                    name = p.get("title", "").strip()
                    if not name:
                        continue
                    images     = p.get("images", [])
                    image_urls = [img.get("src", "") for img in images if img.get("src")]
                    image_url  = image_urls[0] if image_urls else ""
                    if not image_url:
                        continue
                    price    = ""
                    variants = p.get("variants", [])
                    if variants:
                        pv = variants[0].get("price", "")
                        if pv:
                            price = f"USD {pv}"
                    all_products.append({"name": name, "image_url": image_url, "image_urls": image_urls, "price": price})

                if max_products and len(all_products) >= max_products:
                    all_products = all_products[:max_products]
                    break
                if len(raw_products) < 250:
                    break
                page += 1

            if all_products:
                return all_products

        except Exception as e:
            print(f"[SCRAPE] Shopify API failed: {e}")
            continue

    return []


def _extract_ld_product(obj: dict, products: list):
    name = obj.get("name", "").strip()
    if not name:
        return
    image = obj.get("image", "")
    if isinstance(image, list): image = image[0] if image else ""
    if isinstance(image, dict): image = image.get("url", "")
    price  = ""
    offers = obj.get("offers", {})
    if isinstance(offers, list): offers = offers[0] if offers else {}
    if isinstance(offers, dict):
        pv = str(offers.get("price", ""))
        cu = offers.get("priceCurrency", "")
        if pv and cu:
            price = f"{cu} {pv}"
    if name and image:
        s = str(image).strip()
        products.append({"name": name, "image_url": s, "image_urls": [s], "price": price})


def _scrape_product_cards(soup: BeautifulSoup, base_url: str) -> list:
    products = []
    for card in soup.select("[class*='product'], [class*='item'], article")[:50]:
        img = card.find("img")
        if not img:
            continue
        src = img.get("src") or img.get("data-src", "")
        if not src or src.startswith("data:"):
            continue
        if not src.startswith("http"):
            src = base_url.rstrip("/") + "/" + src.lstrip("/")
        title_el = card.find(["h2", "h3", "h4", "a"])
        name     = title_el.get_text(strip=True) if title_el else "Product"
        if len(name) > 120:
            continue
        products.append({"name": name, "image_url": src, "image_urls": [src], "price": ""})
    return products

# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET  — read/write golden_dataset.json from the gateway
# ══════════════════════════════════════════════════════════════════════════════

def _local_golden_url(url: str) -> bool:
    """Return True if url points to our local golden images directory."""
    return url.startswith(GATEWAY_BASE_URL + "/golden-dataset/images/")


def _delete_local_images(urls: list[str], surviving_urls: set[str]) -> int:
    """Delete image files for urls that are local and not referenced by any surviving entry."""
    deleted = 0
    for url in urls:
        if not _local_golden_url(url) or url in surviving_urls:
            continue
        filename = url.rsplit("/", 1)[-1]
        path = GOLDEN_IMAGES_DIR / filename
        try:
            path.unlink(missing_ok=True)
            deleted += 1
        except Exception as e:
            print(f"[IMAGES] Warning: could not delete {path}: {e}")
    return deleted


def _mirror_image(url: str) -> str:
    """Download url and save to GOLDEN_IMAGES_DIR. Returns a stable gateway-relative URL.
    If the URL is already a local golden-dataset URL, returns it unchanged.
    On any download failure, returns the original URL as fallback."""
    if url.startswith(GATEWAY_BASE_URL + "/golden-dataset/images/"):
        return url
    if url.startswith("data:"):
        # base64 data URI — decode and save
        try:
            header, b64 = url.split(",", 1)
            ext = "jpg" if "jpeg" in header else header.split("/")[-1].split(";")[0]
            img_bytes = __import__("base64").b64decode(b64)
        except Exception:
            return url
    else:
        try:
            resp = httpx.get(url, timeout=15.0, follow_redirects=True)
            resp.raise_for_status()
            img_bytes = resp.content
            ct = resp.headers.get("content-type", "image/jpeg")
            ext = ct.split("/")[-1].split(";")[0] or "jpg"
        except Exception as e:
            print(f"[MIRROR] Could not download {url}: {e}")
            return url
    filename = hashlib.sha1(url.encode()).hexdigest()[:20] + "." + ext
    dest = GOLDEN_IMAGES_DIR / filename
    try:
        if not dest.exists():
            dest.write_bytes(img_bytes)
            print(f"[MIRROR] Saved {filename}")
    except Exception as e:
        print(f"[MIRROR] Failed to save {filename}: {e}")
    return f"{GATEWAY_BASE_URL}/golden-dataset/images/{filename}"


def _load_golden() -> list:
    if not os.path.exists(GOLDEN_DATASET_PATH):
        return []
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return _json.load(f)


def _save_golden(data: list):
    with open(GOLDEN_DATASET_PATH, "w", encoding="utf-8") as f:
        _json.dump(data, f, indent=2, ensure_ascii=False)


@app.get("/golden-dataset")
def get_golden_dataset():
    """Return the full golden_dataset.json content."""
    return {"entries": _load_golden()}


class GoldenEntryRequest(BaseModel):
    query_image_url: str          # URL or base64 data URI for the query image
    query_name:      str
    query_category:  str = ""
    relevant: list[dict]          # [{url, name}] — exactly 5 items expected
    replace: bool = False         # if True, remove existing entry with same query_name first


@app.post("/golden-dataset/entry")
async def add_golden_entry(req: GoldenEntryRequest):
    """
    Index the relevant images into locus_items and append the entry
    to golden_dataset.json.
    """
    if not req.query_name:
        raise HTTPException(status_code=400, detail="query_name is required")
    if not req.relevant:
        raise HTTPException(status_code=400, detail="relevant list cannot be empty")

    # Index relevant images via /add-bulk-batch logic
    batch_items = []
    for item in req.relevant:
        url  = item.get("url", "").strip()
        name = item.get("name", "").strip() or req.query_name
        if not url:
            continue
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
        batch_items.append({
            "name":       name,
            "image_url":  url,
            "store":      "golden_dataset",
            "mall":       "golden_dataset",
            "price":      "",
            "product_id": pid,
            "is_golden":  True,
        })

    if not batch_items:
        raise HTTPException(status_code=400, detail="No valid relevant image URLs")

    # Reuse the add_bulk_batch handler
    from fastapi import Request as _Req
    class _FakeBatch:
        items = batch_items
    bulk_result = await add_bulk_batch(_FakeBatch())

    indexed_pids = []
    for item in batch_items:
        # Only include items that were successfully indexed (not failed)
        failed_names = {f["item"] for f in bulk_result.get("failed", [])}
        if item["name"] not in failed_names:
            indexed_pids.append(item["product_id"])

    relevant_info = [
        {
            "product_id": item["product_id"],
            "name":       item["name"],
            "image_url":  _mirror_image(item["image_url"]),
            "store_name": "golden_dataset",
        }
        for item in batch_items
        if item["product_id"] in indexed_pids
    ]

    entry = {
        "query_image_url":      _mirror_image(req.query_image_url),
        "query_name":           req.query_name,
        "query_category_tag":   req.query_category,
        "relevant_product_ids": indexed_pids,
        "relevant_info":        relevant_info,
        "source":               "manual_frontend",
        "n_relevant":           len(indexed_pids),
        "annotated_by":         "frontend",
        "created_at":           datetime.utcnow().isoformat() + "Z",
    }

    dataset = _load_golden()
    if req.replace:
        old_entry = next((e for e in dataset if e.get("query_name") == req.query_name), None)
        if old_entry:
            # Delete by payload.product_id filter — point IDs in Qdrant use a
            # different hash than product_id (no "golden::" prefix), so PointIdsList
            # won't match. FilterSelector on the payload field always works.
            surviving_pids = {
                info["product_id"]
                for e in [e for e in dataset if e.get("query_name") != req.query_name]
                for info in e.get("relevant_info", [])
                if info.get("product_id")
            }
            old_point_ids = [
                info["product_id"]
                for info in old_entry.get("relevant_info", [])
                if info.get("product_id") and info["product_id"] not in surviving_pids
            ]
            if old_point_ids:
                try:
                    client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[models.FieldCondition(
                                    key="product_id",
                                    match=models.MatchAny(any=old_point_ids),
                                )]
                            )
                        ),
                    )
                    print(f"[REPLACE] Deleted {len(old_point_ids)} old locus_items for '{req.query_name}'")
                except Exception as e:
                    print(f"[REPLACE] Warning: failed to delete old locus_items: {e}")
            # Delete local image files not referenced by the new entry OR other entries.
            # Do this AFTER mirroring so same-URL replacements don't lose their files.
            other_entries = [e for e in dataset if e.get("query_name") != req.query_name]
            new_urls = {entry["query_image_url"]} | {
                i["image_url"] for i in entry.get("relevant_info", []) if i.get("image_url")
            }
            surviving_urls = new_urls | {
                u for e in other_entries
                for u in ([e.get("query_image_url", "")] +
                           [i["image_url"] for i in e.get("relevant_info", []) if i.get("image_url")])
            }
            old_urls = [old_entry.get("query_image_url", "")] + [
                i["image_url"] for i in old_entry.get("relevant_info", []) if i.get("image_url")
            ]
            _delete_local_images(old_urls, surviving_urls)
        dataset = [e for e in dataset if e.get("query_name") != req.query_name]
    dataset.append(entry)
    _save_golden(dataset)

    # Regenerate the report for just this entry — server-side, so it works
    # regardless of which HTML version the browser has open.
    import asyncio as _asyncio, sys as _sys, pathlib as _pl
    _script = _pl.Path("/mlops/visualize_golden_dataset.py")
    report_regenerated = False
    if _script.exists():
        _cmd = [
            _sys.executable, str(_script),
            "--skip-groq",
            "--gateway", "http://localhost:8000",
        ]
        _proc = await _asyncio.create_subprocess_exec(
            *_cmd,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        await _proc.communicate()
        report_regenerated = True

    return {
        "status":             "replaced" if req.replace else "added",
        "entry":              entry,
        "indexed":            bulk_result.get("success", 0),
        "skipped":            bulk_result.get("skipped", 0),
        "failed":             bulk_result.get("failed", []),
        "report_regenerated": report_regenerated,
    }


class DeleteEntryRequest(BaseModel):
    query_name: str
    created_at: str = ""   # disambiguates duplicates; if blank, deletes the first match


@app.delete("/golden-dataset/entry")
async def delete_golden_entry(req: DeleteEntryRequest):
    """Remove one entry from golden_dataset.json and delete its Qdrant points.
    If created_at is provided it targets that specific entry; otherwise the first match."""
    if not req.query_name:
        raise HTTPException(status_code=400, detail="query_name is required")

    dataset = _load_golden()

    # Find the one entry to delete
    if req.created_at:
        target = next(
            (e for e in dataset
             if e.get("query_name") == req.query_name and e.get("created_at") == req.created_at),
            None,
        )
    else:
        target = next((e for e in dataset if e.get("query_name") == req.query_name), None)

    if target is None:
        raise HTTPException(status_code=404, detail=f"Entry '{req.query_name}' not found")

    # Only delete points that are not referenced by any OTHER remaining entry.
    # Use stored product_id — re-hashing image_url is wrong because image_url
    # is now the local mirror URL, not the original URL used at index time.
    surviving = [e for e in dataset if e is not target]
    surviving_pids = {
        info["product_id"]
        for e in surviving
        for info in e.get("relevant_info", [])
        if info.get("product_id")
    }
    point_ids = [
        info["product_id"]
        for info in target.get("relevant_info", [])
        if info.get("product_id") and info["product_id"] not in surviving_pids
    ]
    if point_ids:
        try:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(
                            key="product_id",
                            match=models.MatchAny(any=point_ids),
                        )]
                    )
                ),
            )
            print(f"[DELETE] Removed {len(point_ids)} locus_items for '{req.query_name}'")
        except Exception as e:
            print(f"[DELETE] Warning: failed to delete locus_items: {e}")

    # Delete local image files not referenced by any surviving entry
    surviving_urls = {
        u for e in surviving
        for u in ([e.get("query_image_url", "")] +
                   [i["image_url"] for i in e.get("relevant_info", []) if i.get("image_url")])
    }
    old_urls = [target.get("query_image_url", "")] + [
        i["image_url"] for i in target.get("relevant_info", []) if i.get("image_url")
    ]
    images_removed = _delete_local_images(old_urls, surviving_urls)

    _save_golden(surviving)
    return {"deleted": req.query_name, "points_removed": len(point_ids), "images_removed": images_removed}


@app.post("/golden-dataset/migrate-images")
async def migrate_golden_images():
    """Download all external image URLs in golden_dataset.json to the local golden_images dir
    and rewrite the URLs in place. Safe to run multiple times (already-local URLs are skipped)."""
    dataset = _load_golden()
    updated = 0
    for entry in dataset:
        if entry.get("query_image_url") and not _local_golden_url(entry["query_image_url"]):
            entry["query_image_url"] = _mirror_image(entry["query_image_url"])
            updated += 1
        for info in entry.get("relevant_info", []):
            if info.get("image_url") and not _local_golden_url(info["image_url"]):
                info["image_url"] = _mirror_image(info["image_url"])
                updated += 1
    _save_golden(dataset)
    print(f"[MIGRATE] Mirrored {updated} images to {GOLDEN_IMAGES_DIR}")
    return {"mirrored": updated, "entries": len(dataset)}


@app.post("/golden-dataset/wipe")
async def wipe_golden_dataset():
    """Completely wipe all golden dataset data:
    - All golden Qdrant points from locus_items
    - golden_dataset.json reset to []
    - All local golden images deleted
    - results_cache.json deleted
    """
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(
                    key="store_name",
                    match=models.MatchValue(value="golden_dataset"),
                )]
            )
        ),
    )

    _save_golden([])

    images_removed = 0
    if GOLDEN_IMAGES_DIR.exists():
        for f in GOLDEN_IMAGES_DIR.iterdir():
            if f.is_file():
                f.unlink()
                images_removed += 1

    cache_path = pathlib.Path("/mlops/results_cache.json")
    cache_removed = cache_path.exists()
    cache_path.unlink(missing_ok=True)

    print(f"[WIPE] Done — images={images_removed}, cache={cache_removed}")
    return {"images_removed": images_removed, "cache_removed": cache_removed, "status": "wiped"}


@app.post("/golden-dataset/rebuild")
async def rebuild_golden_dataset():
    """
    Nuclear reset for golden dataset Qdrant entries:
      1. Delete ALL store_name='golden_dataset' points from locus_items.
      2. Re-index every relevant image from golden_dataset.json using the
         stored product_id as the Qdrant point ID (fixes historic ID mismatch).
    Images are read directly from disk — no HTTP round-trip to StaticFiles.
    Safe to run multiple times (idempotent upsert).
    """
    dataset = _load_golden()

    # ── Step 1: wipe all golden points ────────────────────────────────────────
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(
                    key="store_name",
                    match=models.MatchValue(value="golden_dataset"),
                )]
            )
        ),
    )
    print("[REBUILD] Wiped all golden_dataset points from locus_items")

    # ── Step 2: re-index ──────────────────────────────────────────────────────
    indexed = 0
    failed  = 0
    async with httpx.AsyncClient() as http:
        for entry in dataset:
            cat = entry.get("query_category_tag", "")
            for info in entry.get("relevant_info", []):
                pid       = info.get("product_id")
                img_url   = info.get("image_url", "")
                item_name = info.get("name", entry.get("query_name", "golden"))
                if not pid or not img_url:
                    continue

                # Read image bytes directly from disk for local golden URLs
                img_bytes = None
                if "/golden-dataset/images/" in img_url:
                    filename = img_url.rsplit("/", 1)[-1]
                    local_path = GOLDEN_IMAGES_DIR / filename
                    if local_path.exists():
                        img_bytes = local_path.read_bytes()
                if img_bytes is None:
                    try:
                        r = await http.get(img_url, timeout=15.0, follow_redirects=True)
                        r.raise_for_status()
                        img_bytes = r.content
                    except Exception as e:
                        print(f"[REBUILD] Cannot fetch {img_url[:60]}: {e}")
                        failed += 1
                        continue

                try:
                    vis = await http.post(
                        f"{VISUAL_URL}/vectorize",
                        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
                        data={"yolo_label": cat, "darken": "false", "style_hint": (
                        _HAT_HINT if cat == "hat" else
                        _BAG_HINT if cat == "bag" else
                        _SHOE_STYLE_HINTS.get(_shoe_style_from_name(item_name)) if cat == "shoes" else
                        ""
                    )},
                        timeout=60.0,
                    )
                    vis.raise_for_status()
                    vis_data = vis.json()
                    vector   = vis_data.get("vector")
                    if not vector:
                        print(f"[REBUILD] No vector returned for {item_name}")
                        failed += 1
                        continue

                    payload = {
                        "name":         item_name,
                        "store_name":   "golden_dataset",
                        "mall_name":    "golden_dataset",
                        "image_url":    img_url,
                        "category_tag": cat,
                        "price":        "",
                        "product_id":   pid,
                        "is_golden":    True,
                    }
                    if cat == "shoes":
                        payload["shoe_style"] = _shoe_style_from_name(item_name)
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[PointStruct(id=pid, vector=vector, payload=payload)]
                    )
                    indexed += 1
                except Exception as e:
                    print(f"[REBUILD] Failed to index {item_name}: {e}")
                    failed += 1

    print(f"[REBUILD] Done — indexed={indexed}, failed={failed}")
    return {"indexed": indexed, "failed": failed, "entries": len(dataset)}


@app.post("/golden-dataset/gc")
async def golden_dataset_gc():
    """Delete any locus_items points with store_name='golden_dataset' whose
    product_id is no longer referenced in golden_dataset.json.
    Safe to run at any time — only removes orphaned points."""
    dataset = _load_golden()

    # Collect every product_id currently in the dataset
    valid_pids = {
        info["product_id"]
        for entry in dataset
        for info in entry.get("relevant_info", [])
        if info.get("product_id")
    }

    # Scroll all golden points from Qdrant, comparing by payload.product_id
    # (point.id uses a different hash than product_id — don't compare those).
    orphan_pids = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="store_name",
                    match=models.MatchValue(value="golden_dataset"),
                )]
            ),
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            pid = (point.payload or {}).get("product_id")
            if pid and pid not in valid_pids:
                orphan_pids.append(pid)
        if next_offset is None:
            break
        offset = next_offset

    if orphan_pids:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(
                        key="product_id",
                        match=models.MatchAny(any=orphan_pids),
                    )]
                )
            ),
        )
        print(f"[GC] Deleted {len(orphan_pids)} orphaned golden points")
    else:
        print("[GC] No orphaned golden points found")

    return {"orphans_removed": len(orphan_pids), "valid_entries": len(valid_pids)}


class RegenerateRequest(BaseModel):
    query_names: list[str] = []


@app.post("/golden-dataset/regenerate")
async def regenerate_golden_report(req: RegenerateRequest = RegenerateRequest()):
    """
    Re-run visualize_golden_dataset.py (--skip-groq) and overwrite the report HTML.
    Pass query_names to only recompute specific entries (others loaded from cache).
    Uses asyncio subprocess so the event loop stays free to serve /search calls
    made by the visualizer script during regeneration.
    """
    import asyncio, sys, pathlib as _pl
    script = _pl.Path("/mlops/visualize_golden_dataset.py")
    if not script.exists():
        raise HTTPException(
            status_code=404,
            detail="Visualizer script not found at /mlops/visualize_golden_dataset.py",
        )
    cmd = [sys.executable, str(script), "--skip-groq", "--gateway", "http://localhost:8000"]
    if req.query_names:
        cmd += ["--only", ",".join(req.query_names)]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Regeneration failed: {stderr.decode()[-600:]}",
        )
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
# /reindex  — re-embed all catalog products with the current visual engine model
# Called by promote_model.py after a new LoRA adapter is promoted.
# Runs as a fire-and-forget background task; returns immediately.
# ══════════════════════════════════════════════════════════════════════════════

_reindex_running = False


@app.post("/reindex")
async def reindex_catalog(background_tasks: BackgroundTasks):
    """
    Re-embed every product in locus_items using the current visual engine model.
    Triggered automatically after LoRA adapter promotion so that all catalog
    vectors reflect the updated embedding space.
    Returns immediately; re-indexing runs in the background.
    """
    global _reindex_running
    if _reindex_running:
        return {"status": "already_running", "message": "A re-index is already in progress"}
    background_tasks.add_task(_run_full_reindex)
    return {"status": "started", "message": "Re-indexing catalog in background"}


_REINDEX_CHECKPOINT = pathlib.Path(__file__).parent / "reindex_checkpoint.json"


def _load_checkpoint() -> set:
    if _REINDEX_CHECKPOINT.exists():
        try:
            return set(_json.loads(_REINDEX_CHECKPOINT.read_text()))
        except Exception:
            pass
    return set()


def _save_checkpoint(done_ids: set) -> None:
    _REINDEX_CHECKPOINT.write_text(_json.dumps(list(done_ids)))


async def _run_full_reindex():
    global _reindex_running
    _reindex_running = True

    done_ids  = _load_checkpoint()
    resumed   = len(done_ids)
    updated   = 0
    failed    = 0
    semaphore = asyncio.Semaphore(15)

    # Scroll all products from locus_items
    all_products = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        all_products.extend(batch)
        if next_offset is None:
            break
        offset = next_offset

    pending = [pt for pt in all_products if str(pt.id) not in done_ids]
    print(f"[REINDEX] Starting — {len(all_products)} total, {resumed} already done, {len(pending)} remaining")

    async def reembed_one(pt):
        nonlocal updated, failed
        p       = pt.payload
        img_url = p.get("image_url", "")
        name    = p.get("name", "")
        if not img_url:
            return
        async with semaphore:
            try:
                async with httpx.AsyncClient() as http:
                    img_resp = await http.get(img_url, timeout=15.0, follow_redirects=True)
                    if img_resp.status_code == 404:
                        client.delete(collection_name=COLLECTION_NAME, points_selector=[pt.id])
                        print(f"[REINDEX] Deleted 404 '{name}' ({img_url[:80]})")
                        failed += 1
                        done_ids.add(str(pt.id))
                        return
                    img_resp.raise_for_status()
                    idx_resp = await http.post(
                        f"{VISUAL_URL}/index-image",
                        files={"file": ("product.jpg", img_resp.content, "image/jpeg")},
                        data={"title": name},
                        timeout=90.0,
                    )
                    idx_resp.raise_for_status()
                    idx_data = idx_resp.json()

                if idx_data.get("skipped"):
                    done_ids.add(str(pt.id))
                    return

                new_payload = dict(p)
                new_payload["category_tag"] = idx_data.get("category", p.get("category_tag", ""))
                new_payload["box_source"]   = idx_data.get("box_source", p.get("box_source", ""))
                if idx_data.get("shoe_style"):
                    new_payload["shoe_style"] = idx_data["shoe_style"]

                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(
                        id      = pt.id,
                        vector  = idx_data["vector_normal"],
                        payload = new_payload,
                    )]
                )
                updated += 1
                done_ids.add(str(pt.id))
                done = updated + failed
                if done % 100 == 0:
                    _save_checkpoint(done_ids)
                    print(f"[REINDEX] {done + resumed}/{len(all_products)} processed — updated={updated} failed={failed}")
            except Exception as e:
                failed += 1
                print(f"[REINDEX] Failed '{name}': {e}")

    await asyncio.gather(*[reembed_one(pt) for pt in pending])
    _reindex_running = False
    _REINDEX_CHECKPOINT.unlink(missing_ok=True)
    print(f"[REINDEX] Done — updated={updated} failed={failed} total={len(all_products)}")


@app.get("/reindex/status")
async def reindex_status():
    return {"running": _reindex_running}


# ══════════════════════════════════════════════════════════════════════════════
# /trigger-retrain — kick off the LoRA retraining pipeline on demand
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/trigger-retrain")
async def trigger_retrain(background_tasks: BackgroundTasks, force: bool = False):
    """
    Trigger the LoRA retraining pipeline asynchronously via subprocess.
    Pass ?force=true to skip threshold checks.
    The pipeline logs progress to MLflow; monitor via Grafana.
    """
    background_tasks.add_task(_run_retrain_subprocess, force)
    return {"status": "triggered", "force": force,
            "message": "Retraining pipeline started — monitor progress in Grafana"}


async def _run_retrain_subprocess(force: bool):
    import sys
    cmd = [sys.executable, "/mlops/retrain_clip.py"]
    if force:
        cmd.append("--force")
    print(f"[RETRAIN] Triggering pipeline: {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    print(f"[RETRAIN] Pipeline finished (exit={proc.returncode}):\n{stdout.decode()[-2000:]}")


# ══════════════════════════════════════════════════════════════════════════════
# /trigger-link-check — force-run the link health monitor pipeline immediately
# ══════════════════════════════════════════════════════════════════════════════

_link_check_running = False


@app.post("/trigger-link-check")
@app.get("/trigger-link-check")
async def trigger_link_check(background_tasks: BackgroundTasks):
    """
    Force-run the link health monitor (repair_broken_links.py) immediately.
    Accessible via GET so a browser click works from the Grafana button.
    """
    global _link_check_running
    if _link_check_running:
        return {"status": "already_running",
                "message": "Link health check is already in progress"}
    background_tasks.add_task(_run_link_check_subprocess)
    return {"status": "triggered",
            "message": "Link health check started — report will appear at mlops/link_health_report.json"}


async def _run_link_check_subprocess():
    global _link_check_running
    import sys, os as _os
    _link_check_running = True
    try:
        cmd = [sys.executable, "/app/repair_broken_links.py"]
        env = _os.environ.copy()
        env.setdefault("VISUAL_URL", _os.getenv("VISUAL_HOST", "http://visual-engine:8001"))
        env.setdefault("GATEWAY_URL", _os.getenv("GATEWAY_BASE_URL", "http://gateway:8000"))
        print(f"[LINK CHECK] Triggering pipeline: {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        print(f"[LINK CHECK] Pipeline finished (exit={proc.returncode}):\n{stdout.decode()[-2000:]}")
        # Immediately refresh link monitor metrics — don't wait for the 60s background loop
        try:
            if LINK_HEALTH_REPORT.exists():
                import json as _jjson
                from datetime import timezone as _tz
                with open(LINK_HEALTH_REPORT) as _f:
                    _rpt = _jjson.load(_f)
                _run_at_str = _rpt.get("run_at")
                if _run_at_str:
                    _run_at = datetime.fromisoformat(_run_at_str)
                    if _run_at.tzinfo is None:
                        _run_at = _run_at.replace(tzinfo=_tz.utc)
                    _last_ts = _run_at.timestamp()
                    locus_link_monitor_last_run.set(_last_ts * 1000)
                    locus_link_monitor_next_run.set((_last_ts + LINK_MONITOR_INTERVAL) * 1000)
                locus_link_monitor_broken.set(_rpt.get("broken_found", 0))
        except Exception as _me:
            print(f"[LINK CHECK] Could not refresh metrics immediately: {_me}")
    finally:
        _link_check_running = False


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    static = pathlib.Path("frontend/dist") / full_path
    if static.is_file():
        return FileResponse(str(static))
    return FileResponse("frontend/dist/index.html")
