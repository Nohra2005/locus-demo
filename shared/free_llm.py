"""
Free-tier vision LLM calls for the public demo: Gemini first, Groq as fallback.

Free tiers have low per-minute and per-day limits, so every provider here has:
  - a per-minute cap and a per-day cap, counted in this process
  - a backoff window after a 429, using the provider's Retry-After when given
  - a maximum number of images per request (Groq's free vision model takes 3)

If a provider is over budget, backing off, or fails, the next one is tried.
If none can answer, callers fall back to non-LLM behaviour (the CLIP score
for the judge, empty attributes for the tagger). Search never waits on this.

Environment:
  GOOGLE_API_KEY   Gemini key (AI Studio)          GEMINI_MODEL  default gemini-flash-lite-latest
  GROQ_API_KEY     Groq key                         GROQ_MODEL    default qwen/qwen3.8-27b
  LLM_BUDGET_SHARE fraction of each daily cap this process may use (default 0.5,
                   because the judge and the tagger run as two processes)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("free_llm")

_SHARE = float(os.getenv("LLM_BUDGET_SHARE", "0.5"))


@dataclass
class Provider:
    name: str
    api_key: str
    model: str
    max_images: int
    rpm: int
    rpd: int
    _minute: deque = field(default_factory=deque)
    _day: deque = field(default_factory=deque)
    _backoff_until: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def try_reserve(self) -> bool:
        """Take one request from the budget, or return False without waiting."""
        now = time.time()
        with self._lock:
            if now < self._backoff_until:
                return False
            for q, window in ((self._minute, 60), (self._day, 86400)):
                while q and now - q[0] > window:
                    q.popleft()
            if len(self._minute) >= self.rpm or len(self._day) >= self.rpd:
                return False
            self._minute.append(now)
            self._day.append(now)
            return True

    def back_off(self, seconds: float) -> None:
        with self._lock:
            self._backoff_until = time.time() + seconds
        logger.warning(f"[{self.name}] rate limited, pausing for {seconds:.0f}s")

    def status(self) -> dict:
        with self._lock:
            return {
                "model": self.model,
                "used_last_minute": len(self._minute), "rpm": self.rpm,
                "used_last_day": len(self._day), "rpd": self.rpd,
                "backing_off": time.time() < self._backoff_until,
            }


def _build_providers() -> list[Provider]:
    out = []
    if os.getenv("GOOGLE_API_KEY"):
        out.append(Provider(
            "gemini", os.environ["GOOGLE_API_KEY"],
            os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            max_images=int(os.getenv("GEMINI_MAX_IMAGES", "6")),
            rpm=int(os.getenv("GEMINI_RPM", "8")),
            rpd=int(int(os.getenv("GEMINI_RPD", "450")) * _SHARE),
        ))
    if os.getenv("GROQ_API_KEY"):
        out.append(Provider(
            "groq", os.environ["GROQ_API_KEY"],
            os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            max_images=int(os.getenv("GROQ_MAX_IMAGES", "3")),
            rpm=int(os.getenv("GROQ_RPM", "20")),
            rpd=int(int(os.getenv("GROQ_RPD", "900")) * _SHARE),
        ))
    return out


PROVIDERS = _build_providers()
_http = httpx.Client(timeout=45.0)


def enabled() -> bool:
    return bool(PROVIDERS)


def status() -> dict:
    return {p.name: p.status() for p in PROVIDERS}


def _retry_after(resp: httpx.Response, default: float = 60.0) -> float:
    try:
        return float(resp.headers.get("retry-after", default))
    except ValueError:
        return default


def _call_gemini(p: Provider, prompt: str, images_b64: list[str], max_tokens: int) -> str | None:
    parts = [{"text": prompt}] + [
        {"inline_data": {"mime_type": "image/jpeg", "data": b}} for b in images_b64
    ]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{p.model}:generateContent"
    resp = _http.post(url, params={"key": p.api_key}, json={
        "contents": [{"parts": parts}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0},
    })
    if resp.status_code == 429:
        p.back_off(_retry_after(resp))
        return None
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(p: Provider, prompt: str, images_b64: list[str], max_tokens: int) -> str | None:
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in images_b64
    ]
    resp = _http.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {p.api_key}"},
        json={"model": p.model, "messages": [{"role": "user", "content": content}],
              "max_tokens": max_tokens, "temperature": 0.0},
    )
    if resp.status_code == 429:
        p.back_off(_retry_after(resp))
        return None
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


_CALLS = {"gemini": _call_gemini, "groq": _call_groq}


def ask(build_request, max_tokens: int = 300):
    """
    Try each provider in order.

    build_request(max_images) -> (prompt, images_b64) lets the caller fit the
    request to the provider (for example, judge fewer results on Groq).
    Returns (text, provider_name) or (None, None).
    """
    for p in PROVIDERS:
        if not p.try_reserve():
            continue
        prompt, images = build_request(p.max_images)
        if not images:
            continue
        try:
            text = _CALLS[p.name](p, prompt, images, max_tokens)
            if text:
                return text, p.name
        except Exception as e:
            logger.warning(f"[{p.name}] request failed: {e}")
    return None, None
