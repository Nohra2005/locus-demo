"""
Batched LLM judge for the free-tier demo.

The original judge (judge.py) makes one LLM call per search result, which is
fine on a paid key but uses a free daily quota in a few searches. This version
scores the top results of a search in ONE call: the query image plus up to N
result images, answered with a JSON list of scores. Results that do not fit in
the call, or that arrive when every provider is over budget, get the CLIP
similarity score, exactly like judge.py's own fallback.

It keeps judge.py's interface, cache, prompt scale and feedback format, so the
gateway, the low-score audit and the LoRA training data work unchanged.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import free_llm
from judge import (
    _is_recently_judged,
    _set_judge_cache_score,
    fetch_image_as_base64,
    score_to_stars,
)

logger = logging.getLogger(__name__)

JUDGE_TOP_N = int(os.getenv("JUDGE_TOP_N", "5"))

BATCH_PROMPT = (
    "You are a fashion visual similarity expert.\n"
    "The FIRST image is the QUERY. The next {n} images are RESULTS, numbered 1 to {n} in order.\n"
    "For each result, rate how visually similar it is to the query from 0.00 to 1.00:\n"
    "  1.00 identical item; 0.92 same garment type, colour family and silhouette;\n"
    "  0.80 same type and colour, slightly different cut; 0.65 same category and colour, different cut;\n"
    "  0.50 same category, different colour or clearly different cut; 0.30 same category only;\n"
    "  0.10 loosely related; 0.00 unrelated.\n"
    "Use the full range. Consider category, colour, texture, silhouette, length, fit and details.\n"
    "Respond with ONLY a JSON array of {n} numbers, for example [0.81, 0.42].\n"
)


def _parse_scores(text: str, n: int) -> list[float] | None:
    m = re.search(r"\[[^\]]*\]", text)
    if not m:
        return None
    try:
        vals = [float(v) for v in json.loads(m.group())]
    except (ValueError, TypeError):
        return None
    if len(vals) != n:
        return None
    return [min(max(v, 0.0), 1.0) for v in vals]


def run_judge(
    query_image_bytes: bytes,
    results: list[dict],
    gateway_base_url: str,
    openrouter_api_key: str = "",
    scores_out: dict | None = None,
    google_api_key: str = "",
    feedback_fn: Callable[[dict], None] | None = None,
) -> None:
    query_b64 = base64.b64encode(query_image_bytes).decode("utf-8")

    # Skip anything judged recently (same 24h cache as judge.py)
    todo = []
    for r in results:
        key = r.get("product_id") or r.get("image_url", "")
        if not key or not r.get("image_url"):
            continue
        cached, score = _is_recently_judged(key)
        if cached:
            if scores_out is not None and score is not None:
                scores_out[key] = score
            continue
        todo.append((key, r))

    if not todo:
        return

    head = todo[:JUDGE_TOP_N]
    with ThreadPoolExecutor(max_workers=5) as pool:
        imgs = list(pool.map(lambda kr: fetch_image_as_base64(kr[1]["image_url"]), head))
    usable = [(kr, img) for kr, img in zip(head, imgs) if img]

    llm_scores: dict[str, float] = {}
    provider = None
    if usable and free_llm.enabled():
        sent: list = []

        def build(max_images: int):
            sent.clear()
            sent.extend(usable[: max(0, max_images - 1)])
            n = len(sent)
            return BATCH_PROMPT.format(n=n), [query_b64] + [img for _, img in sent]

        text, provider = free_llm.ask(build, max_tokens=80)
        if text:
            scores = _parse_scores(text, len(sent))
            if scores:
                llm_scores = {kr[0]: s for (kr, _), s in zip(sent, scores)}
            else:
                logger.warning(f"judge: [{provider}] could not parse scores from {text!r}")
                provider = None

    for key, r in todo:
        if key in llm_scores:
            score, source = llm_scores[key], provider
        else:
            score, source = round(r.get("score", 0.0), 3), "clip-fallback"
        score = round(score, 3)
        if scores_out is not None:
            scores_out[key] = score
        _set_judge_cache_score(key, score)

        payload = {
            "result_product_id": r.get("product_id", r["image_url"]),
            "result_image_url":  r["image_url"],
            "result_name":       r.get("name", ""),
            "store_name":        r.get("store_name", ""),
            "category":          r.get("category_tag", ""),
            "rating":            score_to_stars(score),
            "source":            f"auto_judge_{source}",
        }
        if feedback_fn is not None:
            try:
                feedback_fn(payload)
            except Exception as e:
                logger.warning(f"judge: failed to store feedback: {e}")

    logger.info(f"judge: {len(llm_scores)} scored by {provider or 'none'}, "
                f"{len(todo) - len(llm_scores)} by CLIP fallback")
