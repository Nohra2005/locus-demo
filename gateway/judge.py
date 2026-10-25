"""
OpenRouter judge — fires per search, stores feedback with source="auto_judge".

Entry point: run_judge(query_image_bytes, results, gateway_base_url, openrouter_api_key, scores_out)
Called as a FastAPI BackgroundTask from the /search endpoint.
No imports from gateway.main — no circular dependencies.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

logger = logging.getLogger(__name__)

# Bump when the scoring prompt changes; logged to MLflow via retrain_clip.py.
JUDGE_PROMPT_VERSION = "v3"

OPENROUTER_MODEL   = "google/gemini-2.0-flash-001"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash-001:generateContent"
)

# ── Rate limiter ───────────────────────────────────────────────────────────────
# 2s minimum gap → ~30 RPM, well within OpenRouter's free-tier ceiling.

_provider_lock  = threading.Lock()
_provider_state: dict = {"last_call": 0.0, "backoff_until": 0.0, "min_gap": 0.5}

# ── Judge recency cache ────────────────────────────────────────────────────────
# Prevents the same product being judged more than once per TTL window.
# Without this, every search re-judges the same popular products and floods
# locus_feedback with duplicate entries, biasing LoRA training.

_JUDGE_CACHE: dict[str, tuple[float, float | None]] = {}  # key → (timestamp, score)
_JUDGE_CACHE_TTL = float(os.getenv("JUDGE_CACHE_TTL_HOURS", "24")) * 3600
_JUDGE_CACHE_LOCK = threading.Lock()


def _is_recently_judged(key: str) -> tuple[bool, float | None]:
    """Return (True, cached_score) if recently judged; (False, None) and register key otherwise."""
    now = time.monotonic()
    with _JUDGE_CACHE_LOCK:
        if len(_JUDGE_CACHE) > 10_000:
            cutoff = now - _JUDGE_CACHE_TTL
            for k in [k for k, (t, _) in _JUDGE_CACHE.items() if t < cutoff]:
                del _JUDGE_CACHE[k]
        entry = _JUDGE_CACHE.get(key)
        if entry is not None and now - entry[0] < _JUDGE_CACHE_TTL:
            return True, entry[1]
        _JUDGE_CACHE[key] = (now, None)
        return False, None


def _set_judge_cache_score(key: str, score: float) -> None:
    with _JUDGE_CACHE_LOCK:
        entry = _JUDGE_CACHE.get(key)
        if entry is not None:
            _JUDGE_CACHE[key] = (entry[0], score)


def _acquire_slot() -> bool:
    """Block until we can call OpenRouter. Returns False if inside a backoff window."""
    while True:
        with _provider_lock:
            now = time.monotonic()
            if now < _provider_state["backoff_until"]:
                return False
            wait = _provider_state["last_call"] + _provider_state["min_gap"] - now
            if wait <= 0:
                _provider_state["last_call"] = time.monotonic()
                return True
        # Sleep outside the lock so other workers can check concurrently
        time.sleep(wait)


def _set_backoff(retry_after: float) -> None:
    with _provider_lock:
        _provider_state["backoff_until"] = time.monotonic() + retry_after
    logger.warning(f"judge: [openrouter] rate-limit backoff for {retry_after:.0f}s")


JUDGE_PROMPT = (
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
    "For patterned garments (floral, striped, printed): if both items share the same base colour AND the same pattern type (e.g. both are floral), treat pattern variation (different flowers, scale, density) as a minor difference only — do NOT drop below 0.80 for this alone. Score 0.88+ if garment type, base colour, and silhouette all match.\n"
    "Consider all visual attributes: category, colour, fabric texture, silhouette, length, fit, and detailing.\n"
    "Respond with ONLY the numeric score. No explanation."
)


def score_to_stars(score: float) -> int:
    if score >= 0.85:
        return 5
    elif score >= 0.70:
        return 4
    elif score >= 0.50:
        return 3
    elif score >= 0.30:
        return 2
    else:
        return 1


def fetch_image_as_base64(url: str) -> str | None:
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        logger.warning(f"judge: failed to fetch image {url}: {e}")
        return None


# ── Provider implementation ────────────────────────────────────────────────────

def _judge_pair_openrouter(query_image_b64: str, result_image_url: str, api_key: str) -> float | None:
    result_b64 = fetch_image_as_base64(result_image_url)
    if result_b64 is None:
        return None

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": JUDGE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{query_image_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{result_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 50,
        "temperature": 0.0,
    }

    for attempt in range(3):
        if not _acquire_slot():
            logger.info("judge: [openrouter] skipping — inside backoff window")
            return None
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    OPENROUTER_API_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )

            if resp.status_code == 429:
                retry_after_raw = resp.headers.get("retry-after")
                try:
                    wait = float(retry_after_raw)
                except (TypeError, ValueError):
                    wait = 60.0
                _set_backoff(wait)
                return None

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r"\d+\.\d+|\d+", content)
            if match is None:
                logger.warning(f"judge: [openrouter] could not parse score: {content!r}")
                return None
            return float(match.group())

        except httpx.HTTPStatusError as e:
            if attempt < 2:
                time.sleep(min(5.0 * (2 ** attempt), 30.0))
                continue
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text
            logger.warning(f"judge: [openrouter] HTTP error {e.response.status_code} after 3 attempts: {body}")
            return None
        except Exception as e:
            logger.warning(f"judge: [openrouter] error: {e}")
            return None

    return None


# ── Fallback provider: Google Gemini REST ─────────────────────────────────────

def _judge_pair_gemini_direct(query_b64: str, result_image_url: str, api_key: str) -> float | None:
    """Call Gemini 2.0 Flash directly via Google's REST API (no OpenRouter)."""
    result_b64 = fetch_image_as_base64(result_image_url)
    if result_b64 is None:
        return None

    payload = {
        "contents": [{
            "parts": [
                {"text": JUDGE_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": query_b64}},
                {"inline_data": {"mime_type": "image/jpeg", "data": result_b64}},
            ]
        }],
        "generationConfig": {"maxOutputTokens": 50, "temperature": 0.0},
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{GEMINI_API_URL}?key={api_key}", json=payload)
        resp.raise_for_status()
        content = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        match = re.search(r"\d+\.\d+|\d+", content)
        if match is None:
            logger.warning(f"judge: [gemini-direct] could not parse score: {content!r}")
            return None
        return float(match.group())
    except Exception as e:
        logger.warning(f"judge: [gemini-direct] error: {e}")
        return None


# ── Entry point ────────────────────────────────────────────────────────────────

def run_judge(
    query_image_bytes: bytes,
    results: list[dict],
    gateway_base_url: str,
    openrouter_api_key: str,
    scores_out: dict | None = None,
    google_api_key: str = "",
    feedback_fn: Callable[[dict], None] | None = None,
) -> None:
    if not openrouter_api_key and not google_api_key:
        logger.warning("judge: no OPENROUTER_API_KEY or GOOGLE_API_KEY configured, skipping")
        return

    query_b64 = base64.b64encode(query_image_bytes).decode("utf-8")
    logger.info(f"judge: starting — {len(results)} result(s), model={OPENROUTER_MODEL}, workers=5")

    def _judge_one(result: dict) -> None:
        result_image_url = result.get("image_url", "")
        result_name      = result.get("name", "")
        product_id       = result.get("product_id", result_image_url)
        key              = product_id or result_image_url

        if not result_image_url:
            logger.info(f"judge: skipping '{result_name}' — no image_url")
            return

        recently_judged, cached_score = _is_recently_judged(key)
        if recently_judged:
            logger.info(f"judge: skipping '{result_name}' — recently judged (cache hit)")
            if scores_out is not None and key and cached_score is not None:
                scores_out[key] = cached_score
            return

        score    = None
        provider = "none"

        # Primary: OpenRouter
        if openrouter_api_key:
            logger.info(f"judge: [openrouter] scoring '{result_name}' ({result_image_url[:80]})")
            score = _judge_pair_openrouter(query_b64, result_image_url, openrouter_api_key)
            if score is not None:
                provider = "openrouter"

        # Fallback 1: Google Gemini REST
        if score is None and google_api_key:
            logger.info(f"judge: [gemini-direct] scoring '{result_name}' (openrouter unavailable)")
            score = _judge_pair_gemini_direct(query_b64, result_image_url, google_api_key)
            if score is not None:
                provider = "gemini-direct"

        # Fallback 2: CLIP cosine score (always available, zero cost)
        if score is None:
            score    = round(result.get("score", 0.0), 3)
            provider = "clip-fallback"
            logger.info(f"judge: [clip-fallback] using CLIP score={score:.3f} for '{result_name}'")

        stars = score_to_stars(score)
        logger.info(f"judge: [{provider}] '{result_name}' → score={score:.2f} stars={stars} key={key[:60]}")

        if scores_out is not None and key:
            scores_out[key] = round(score, 3)
        if key:
            _set_judge_cache_score(key, round(score, 3))

        feedback_payload = {
            "result_product_id": product_id,
            "result_image_url":  result_image_url,
            "result_name":       result_name,
            "store_name":        result.get("store_name", ""),
            "category":          result.get("category_tag", ""),
            "rating":            stars,
            "source":            f"auto_judge_{provider}",
        }

        if feedback_fn is not None:
            try:
                feedback_fn(feedback_payload)
            except Exception as e:
                logger.warning(f"judge: failed to store feedback for '{result_name}': {e}")
        else:
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(f"{gateway_base_url}/feedback", json=feedback_payload)
                    resp.raise_for_status()
            except Exception as e:
                logger.warning(f"judge: failed to post feedback for '{result_name}': {e}")

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(_judge_one, r) for r in results]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.warning(f"judge: unexpected error in worker: {e}")
