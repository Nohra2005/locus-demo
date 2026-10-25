"""
Attribute tagger — extracts structured visual attributes from a cropped clothing image.

Provider chain (same pattern as judge.py):
  1. OpenRouter (Gemini 2.0 Flash) — primary, rate-limited to 30 RPM
  2. Google Gemini REST             — fallback when OpenRouter 429/504s
  3. {}                             — graceful degradation, never raises

Returns a validated dict: colors, style, silhouette, occasion, pattern,
material_feel, trend_tags, _confidence. Returns {} on total failure.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)

# Bump when the prompt changes; log in MLflow so each run records which prompt was active.
TAGGER_PROMPT_VERSION = "v5"

OPENROUTER_MODEL   = "google/gemini-2.0-flash-001"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash-001:generateContent"
)

# ── HTTP client (singleton — avoids TCP+TLS overhead per call) ────────────────
_http = httpx.Client(timeout=20.0)

# ── Rate limiter (mirrors judge.py) ──────────────────────────────────────────
# 2 s minimum gap → ~30 RPM, well within OpenRouter's free-tier ceiling.

_lock:  threading.Lock = threading.Lock()
_state: dict = {"last_call": 0.0, "backoff_until": 0.0, "min_gap": 2.0}


def _acquire_slot() -> bool:
    """Block until we can call OpenRouter. Returns False if inside backoff window."""
    with _lock:
        now = time.monotonic()
        if now < _state["backoff_until"]:
            return False
        wait = _state["last_call"] + _state["min_gap"] - now
        if wait <= 0:
            _state["last_call"] = time.monotonic()
            return True
        # Stamp last_call before releasing so concurrent callers stagger correctly
        _state["last_call"] = time.monotonic() + wait
    time.sleep(wait)  # outside the lock
    return True


def _set_backoff(seconds: float) -> None:
    with _lock:
        _state["backoff_until"] = time.monotonic() + seconds
    logger.warning(f"tagger: [openrouter] rate-limit backoff for {seconds:.0f}s")


# ── Vocabularies ──────────────────────────────────────────────────────────────

VOCAB_STYLE = {
    "casual", "formal", "streetwear", "athletic", "bohemian", "minimalist",
    "preppy", "romantic", "quiet luxury", "edgy", "classic", "resort",
}
VOCAB_PATTERN = {
    "solid", "striped", "floral", "plaid", "animal print", "geometric",
    "abstract", "polka dot", "tie-dye", "logo print",
}
VOCAB_MATERIAL = {
    "cotton", "silk", "denim", "knitwear", "leather", "chiffon",
    "linen", "velvet", "satin", "wool", "synthetic",
}
VOCAB_OCCASION = {
    "office", "casual", "evening", "beach", "workout", "date night",
    "weekend", "formal event",
}

_STYLE_ALIASES = {
    "casual wear": "casual",
    "business casual": "formal",
    "business formal": "formal",
    "smart casual": "formal",
    "sports": "athletic",
    "sport": "athletic",
    "activewear": "athletic",
    "boho": "bohemian",
    "hippie": "bohemian",
    "old money": "quiet luxury",
    "luxury": "quiet luxury",
    "grunge": "edgy",
    "punk": "edgy",
    "preppy classic": "preppy",
    "vacation": "resort",
    "resort wear": "resort",
}
_MATERIAL_ALIASES = {
    "knit": "knitwear",
    "knitted": "knitwear",
    "faux leather": "leather",
    "suede": "leather",
    "polyester": "synthetic",
    "nylon": "synthetic",
    "spandex": "synthetic",
    "jersey": "synthetic",
    "chiffon fabric": "chiffon",
    "satin fabric": "satin",
}
_PATTERN_ALIASES = {
    "plain": "solid",
    "no pattern": "solid",
    "stripes": "striped",
    "floral print": "floral",
    "checkered": "plaid",
    "check": "plaid",
    "leopard": "animal print",
    "zebra": "animal print",
    "snake print": "animal print",
    "dots": "polka dot",
    "polka dots": "polka dot",
    "logo": "logo print",
    "branded": "logo print",
}

# ── Category-specific prompts ─────────────────────────────────────────────────

_CATEGORY_HINTS: dict[str, str] = {
    "tops": (
        "This is a top/shirt/knitwear. Pay special attention to: "
        "neckline style (crew, v-neck, turtleneck, off-shoulder, halter, square, cowl), "
        "sleeve length (sleeveless, short, 3/4, long), "
        "and fit (cropped, oversized, fitted, relaxed, boxy)."
    ),
    "outerwear": (
        "This is a jacket or coat. Pay special attention to: "
        "collar/lapel style (notch lapel, mandarin, hood, no collar), "
        "closure (single-breasted, double-breasted, zip, snap), "
        "and length (cropped, hip-length, knee-length, long)."
    ),
    "bottoms": (
        "This is a bottom garment (pants/shorts/skirt/leggings). Pay special attention to: "
        "rise (high-rise, mid-rise, low-rise), "
        "leg/silhouette style (straight, wide-leg, slim, flared, tapered, A-line, pleated), "
        "and length (full-length, cropped, midi, mini, short)."
    ),
    "full_body": (
        "This is a dress or jumpsuit. Pay special attention to: "
        "hemline length (mini, midi, maxi, floor-length), "
        "neckline style, sleeve style, "
        "and overall silhouette (A-line, bodycon, wrap, slip, shift, smock, empire)."
    ),
    "shoes": (
        "This is footwear. Adapt your answer accordingly: "
        "For silhouette: describe heel type (flat, block heel, stiletto, wedge, platform, kitten heel) and toe shape (round, pointed, square, open-toe). "
        "For material_feel: describe the upper material (leather, suede, canvas, synthetic, knit). "
        "For occasion: focus on formality. "
        "For style: use the standard style labels. "
        "For pattern: describe the shoe's surface pattern."
    ),
    "accessories": (
        "This is an accessory (bag or hat). Adapt your answer accordingly: "
        "For silhouette: describe the shape/form factor (tote, crossbody, bucket hat, baseball cap, etc.). "
        "For material_feel: describe the main material. "
        "For colors: focus on the body color and any hardware. "
        "For style: use the standard style labels."
    ),
}

# Flat category → hint lookup built once at import time
_CATEGORY_HINT_MAP: dict[str, str] = {
    cat: _CATEGORY_HINTS[group]
    for group, members in [
        ("tops",        {"top", "sweater", "sports_bra"}),
        ("outerwear",   {"jacket"}),
        ("bottoms",     {"pants", "leggings", "shorts", "skirt"}),
        ("full_body",   {"dress", "jumpsuit"}),
        ("shoes",       {"shoes"}),
        ("accessories", {"bag", "hat"}),
    ]
    for cat in members
}


def _category_hint(category: str) -> str:
    return _CATEGORY_HINT_MAP.get(category.strip().lower(), "")


_BASE_PROMPT = """\
You are a fashion attribute extraction system. Analyze the clothing item in this image.

Return ONLY a valid JSON object with exactly these keys:
{{
  "colors": ["list", "of", "1-3", "dominant colors"],
  "style": "one style label",
  "silhouette": "brief shape/cut description",
  "occasion": ["list", "of", "1-2", "occasions"],
  "pattern": "one pattern label",
  "material_feel": "one material impression",
  "trend_tags": ["list", "of", "0-2", "trend tags"]
}}

Guidelines:
- colors: be specific — "navy blue" not "blue", "forest green" not "green". List 1-3 dominant colors.
- style: MUST be exactly one of: casual, formal, streetwear, athletic, bohemian, minimalist, \
preppy, romantic, quiet luxury, edgy, classic, resort
- silhouette: e.g. "fitted blazer", "flowy midi dress", "slim-fit jeans", \
"oversized hoodie", "a-line skirt"
- occasion: MUST be 1-2 values from: office, casual, evening, beach, workout, \
date night, weekend, formal event
- pattern: MUST be exactly one of: solid, striped, floral, plaid, animal print, \
geometric, abstract, polka dot, tie-dye, logo print
- material_feel: MUST be exactly one of: cotton, silk, denim, knitwear, leather, \
chiffon, linen, velvet, satin, wool, synthetic
- trend_tags: optional 0-2 values, e.g. "Y2K", "cottagecore", "dark academia", \
"gorpcore", "clean girl", "coastal grandmother"

{category_hint}
Respond ONLY with valid JSON. No markdown. No explanation. No extra text."""


@lru_cache(maxsize=None)  # at most ~11 distinct categories — cache forever
def _build_prompt(category: str) -> str:
    hint = _category_hint(category)
    hint_block = f"\nCategory-specific guidance: {hint}\n" if hint else ""
    return _BASE_PROMPT.format(category_hint=hint_block)


# ── Output validation ─────────────────────────────────────────────────────────

def _normalise(value: str, aliases: dict[str, str]) -> str | None:
    v = value.strip().lower()
    if v in aliases:
        return aliases[v]
    return None


_CONSTRAINED_FIELDS = [
    ("style",         VOCAB_STYLE,    _STYLE_ALIASES),
    ("pattern",       VOCAB_PATTERN,  _PATTERN_ALIASES),
    ("material_feel", VOCAB_MATERIAL, _MATERIAL_ALIASES),
]

_ALL_FIELDS = ["colors", "style", "silhouette", "occasion", "pattern", "material_feel", "trend_tags"]


def _validate(attrs: dict) -> tuple[dict, float]:
    """
    Validate and normalise each field. Returns (cleaned_attrs, confidence)
    where confidence = fraction of constrained fields already in-vocab before normalisation.
    """
    out = dict(attrs)
    hits = 0

    for field, vocab, aliases in _CONSTRAINED_FIELDS:
        raw = str(attrs.get(field, "")).strip().lower()
        if raw in vocab:
            hits += 1
            out[field] = raw
        else:
            normalised = _normalise(raw, aliases)
            out[field] = normalised if normalised else raw

    colors = attrs.get("colors", [])
    out["colors"] = [str(c).strip().lower() for c in colors if c] if isinstance(colors, list) else []

    occasions = attrs.get("occasion", [])
    out["occasion"] = [str(o).strip().lower() for o in occasions if o] if isinstance(occasions, list) else []

    tags = attrs.get("trend_tags", [])
    out["trend_tags"] = [str(t).strip().lower() for t in tags if t] if isinstance(tags, list) else []

    out["silhouette"] = str(attrs.get("silhouette", "")).strip().lower()

    return out, hits / len(_CONSTRAINED_FIELDS)


# ── JSON parsing helper ───────────────────────────────────────────────────────

def _parse_json(content: str) -> dict | None:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"tagger: JSON parse failed: {e} — raw: {content!r}")
        return None


# ── Provider 1: OpenRouter ────────────────────────────────────────────────────

def _tag_openrouter(image_b64: str, prompt: str, api_key: str) -> dict | None:
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 300,
        "temperature": 0.0,
    }

    for attempt in range(3):
        if not _acquire_slot():
            logger.info("tagger: [openrouter] skipping — inside backoff window")
            return None
        try:
            resp = _http.post(
                OPENROUTER_API_URL,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )

            if resp.status_code == 429:
                try:
                    wait = float(resp.headers.get("retry-after", 60.0))
                except (TypeError, ValueError):
                    wait = 60.0
                _set_backoff(wait)
                return None

            if resp.status_code == 504:
                if attempt < 2:
                    wait = min(5.0 * (2 ** attempt), 30.0)
                    logger.warning(f"tagger: [openrouter] 504 on attempt {attempt+1}, retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                else:
                    logger.warning("tagger: [openrouter] 504 on attempt 3, giving up")
                    return None

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _parse_json(content)

        except httpx.HTTPStatusError as e:
            if attempt < 2:
                time.sleep(min(5.0 * (2 ** attempt), 30.0))
                continue
            logger.warning(f"tagger: [openrouter] HTTP {e.response.status_code} after 3 attempts")
            return None
        except Exception as e:
            logger.warning(f"tagger: [openrouter] error: {e}")
            return None

    return None


# ── Provider 2: Google Gemini REST ────────────────────────────────────────────

def _tag_gemini_direct(image_b64: str, prompt: str, api_key: str) -> dict | None:
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
            ]
        }],
        "generationConfig": {"maxOutputTokens": 300, "temperature": 0.0},
    }
    try:
        resp = _http.post(f"{GEMINI_API_URL}?key={api_key}", json=payload)
        resp.raise_for_status()
        content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_json(content)
    except Exception as e:
        logger.warning(f"tagger: [gemini-direct] error: {e}")
        return None


# ── Entry point ───────────────────────────────────────────────────────────────

def tag_image(image_b64: str, category: str) -> dict:
    """
    Tags the image using the provider chain: OpenRouter → Gemini direct → {}.
    Returns a validated, normalised attribute dict with _confidence score.
    Never raises.
    """
    prompt = _build_prompt(category.strip().lower())
    raw: dict | None = None
    provider = None

    if OPENROUTER_API_KEY:
        raw = _tag_openrouter(image_b64, prompt, OPENROUTER_API_KEY)
        if raw is not None:
            provider = "openrouter"

    if raw is None and GOOGLE_API_KEY:
        logger.info("tagger: [gemini-direct] OpenRouter unavailable, using direct API")
        raw = _tag_gemini_direct(image_b64, prompt, GOOGLE_API_KEY)
        if raw is not None:
            provider = "gemini-direct"

    if raw is None:
        logger.warning(f"tagger: all providers failed for category={category}")
        return {}

    attrs, confidence = _validate(raw)
    attrs["_confidence"] = round(confidence, 2)
    logger.info(f"tagger: [{provider}] category={category} confidence={confidence:.2f}")
    return attrs
