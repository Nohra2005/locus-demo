"""
ci_eval.py — Judge-based CI quality gate.

Calls /search on every entry in golden_dataset.json that has a local image,
judges the top-3 results per query with Gemini via OpenRouter, and compares
the overall average against a stored baseline.

Baseline workflow:
    # Run once on a known-good state to record the baseline:
    python mlops/ci_eval.py --gateway-url http://localhost:8000 --set-baseline

    # Every subsequent CI run compares against that baseline:
    python mlops/ci_eval.py --gateway-url http://localhost:8000

The baseline is stored in mlops/ci_baseline.json and should be committed.
CI fails if the current score drops more than REGRESSION_TOLERANCE below baseline.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import threading
from pathlib import Path

import requests

SCRIPT_DIR          = Path(__file__).parent
GOLDEN_DATASET_PATH = SCRIPT_DIR / "golden_dataset.json"
GOLDEN_IMAGES_DIR   = SCRIPT_DIR / "golden_images"
BASELINE_PATH       = SCRIPT_DIR / "ci_baseline.json"
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
_MODEL              = "google/gemini-2.0-flash-001"

# How many points below baseline triggers a failure (e.g. 0.04 = 4 percentage points)
REGRESSION_TOLERANCE = 0.04

# /search is rate-limited to 10/minute; 7s between calls keeps us at ~8/min
SEARCH_INTERVAL = 7.0
# Minimum fraction of eligible queries that must be evaluated for the result to be valid
MIN_COVERAGE = 0.60


def _shoe_style_from_name(name: str) -> str:
    lower = name.lower()
    if any(k in lower for k in ("sneaker", "trainer", "running", "platform", "wedge", "clog", "chunky")):
        return "sneaker"
    if any(k in lower for k in ("boot", "bottine", "botte")):
        return "boot"
    if any(k in lower for k in ("heel", "pump", "stiletto", "kitten")):
        return "heel"
    if any(k in lower for k in ("sandal", "loafer", "flat", "mule", "espadrille", "ballet")):
        return "sandal"
    return "other"


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
    "For patterned garments (floral, striped, printed): if both items share the same base colour AND the same pattern type (e.g. both are floral), treat pattern variation (different flowers, scale, density) as a minor difference only — do NOT drop below 0.80 for this alone. Score 0.88+ if garment type, base colour, and silhouette all match.\n"
    "Consider all visual attributes: category, colour, fabric texture, silhouette, length, fit, and detailing.\n"
    "Respond with ONLY the numeric score. No explanation."
)

_lock      = threading.Lock()
_last_call = 0.0


def _judge(query_b64: str, result_url: str, gateway_url: str) -> float | None:
    global _last_call

    result_url = result_url.replace("http://localhost:8000", gateway_url)

    # Image fetch is a one-time attempt — failure is a missing image, not a rate-limit issue
    try:
        r = requests.get(result_url, timeout=10)
        r.raise_for_status()
        result_b64 = base64.b64encode(r.content).decode()
    except Exception as e:
        print(f"    [WARN] Could not fetch result image: {e}")
        return None

    payload = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": _JUDGE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{result_b64}"}},
        ]}],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        with _lock:
            gap = _last_call + 2.0 - time.monotonic()
            if gap > 0:
                time.sleep(gap)
            _last_call = time.monotonic()
        try:
            r = requests.post(
                _OPENROUTER_URL,
                json=payload,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=30,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\d+\.\d+|\d+", text)
            return float(m.group()) if m else None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 5.0 * (attempt + 1)  # 5s, then 10s
                print(f"    [RETRY {attempt + 1}] judge failed ({e}), waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                print(f"    [WARN] Judge call failed after {MAX_RETRIES} attempts: {e}")
                return None


def run_eval(gateway_url: str) -> tuple[float, int, dict[str, float]]:
    """Returns (overall_avg, n_evaluated, {category: avg_score})."""
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text())
    all_scores: list[float] = []
    category_scores: dict[str, list[float]] = {}
    evaluated = 0

    for entry in dataset:
        query_name = entry.get("query_name", "")
        img_url    = entry.get("query_image_url", "")
        fname      = img_url.rsplit("/", 1)[-1]
        img_path   = GOLDEN_IMAGES_DIR / fname
        category   = entry.get("query_category_tag", "")

        if not img_path.exists():
            print(f"  [SKIP] local image not found: {fname}")
            continue

        image_bytes = img_path.read_bytes()
        query_b64   = base64.b64encode(image_bytes).decode()

        search_data = {"search_label": category} if category else {}
        if category == "shoes":
            style = _shoe_style_from_name(query_name)
            if style and style != "other":
                search_data["shoe_style"] = style
        bbox = entry.get("query_bbox")
        if bbox:
            search_data.update({
                "x1": bbox["x1"], "y1": bbox["y1"],
                "x2": bbox["x2"], "y2": bbox["y2"],
            })

        search_resp = None
        for attempt in range(3):
            time.sleep(SEARCH_INTERVAL)
            try:
                r = requests.post(
                    f"{gateway_url}/search",
                    files={"file": ("q.jpg", io.BytesIO(image_bytes), "image/jpeg")},
                    data=search_data,
                    timeout=60,
                )
                if r.status_code == 429:
                    wait = 30.0 * (attempt + 1)
                    print(f"  [RETRY {attempt+1}] 429 rate-limit on '{query_name}', waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                search_resp = r
                break
            except Exception as e:
                print(f"  [FAIL] /search error for '{query_name}': {e}")
                break

        if search_resp is None:
            continue
        matches = search_resp.json().get("matches", [])[:3]

        if not matches:
            print(f"  [WARN] no results for '{query_name}'")
            continue

        scores: list[float] = []
        total_calls = 0
        for m in matches:
            result_img_url = m.get("image_url", "")
            if not result_img_url:
                continue
            total_calls += 1
            score = _judge(query_b64, result_img_url, gateway_url)
            if score is not None:
                scores.append(score)

        if scores:
            avg = sum(scores) / len(scores)
            success_note = ""
            if total_calls > 0 and len(scores) / total_calls < 0.5:
                success_note = f"  [WARN: only {len(scores)}/{total_calls} judge calls succeeded]"
            print(f"  '{query_name}' [{category}]: {len(scores)}/{total_calls} scores, avg={avg:.3f}{success_note}")
            all_scores.extend(scores)
            category_scores.setdefault(category, []).extend(scores)
            evaluated += 1
        else:
            print(f"  [WARN] all judge calls failed for '{query_name}'")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    per_category = {cat: sum(v) / len(v) for cat, v in category_scores.items()}
    return overall, evaluated, per_category


def main():
    parser = argparse.ArgumentParser(description="CI judge-based quality gate for Locus")
    parser.add_argument("--gateway-url",  default="http://localhost:8000")
    parser.add_argument("--set-baseline", action="store_true",
                        help="Record current score as the new baseline and exit 0")
    parser.add_argument("--tolerance",    type=float, default=REGRESSION_TOLERANCE,
                        help="Max allowed drop below baseline before CI fails (default 0.04)")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("[ERROR] OPENROUTER_API_KEY is not set")
        sys.exit(1)

    n_queries = sum(
        1 for e in json.loads(GOLDEN_DATASET_PATH.read_text())
        if (GOLDEN_IMAGES_DIR / e.get("query_image_url", "").rsplit("/", 1)[-1]).exists()
    )
    print(f"[CI EVAL] Gateway:  {args.gateway_url}")
    print(f"[CI EVAL] Queries:  {n_queries} (all golden dataset entries with local images)\n")

    score, evaluated, per_category = run_eval(args.gateway_url)

    print(f"\n[CI EVAL] Evaluated {evaluated}/{n_queries} queries")
    print(f"[CI EVAL] Overall avg judge score: {score:.4f}")

    if evaluated == 0:
        print("[FAIL] no queries could be evaluated")
        sys.exit(1)

    coverage = evaluated / n_queries if n_queries > 0 else 0.0
    if coverage < MIN_COVERAGE:
        print(f"[FAIL] only {evaluated}/{n_queries} queries evaluated ({coverage:.0%}) — "
              f"minimum required is {MIN_COVERAGE:.0%}. Check for rate-limiting or service errors.")
        sys.exit(1)

    if args.set_baseline:
        if coverage < MIN_COVERAGE:
            print(f"[ERROR] Only {evaluated}/{n_queries} queries evaluated ({coverage:.0%}). "
                  f"Baseline not saved — minimum coverage {MIN_COVERAGE:.0%} required.")
            sys.exit(1)
        baseline_doc = {
            "score":        round(score, 4),
            "n_queries":    evaluated,
            "model":        _MODEL,
            "per_category": {cat: round(v, 4) for cat, v in sorted(per_category.items())},
        }
        BASELINE_PATH.write_text(json.dumps(baseline_doc, indent=2) + "\n", encoding="utf-8")
        print(f"\n[CI EVAL] Per-category (new baseline):")
        for cat, avg in sorted(per_category.items()):
            print(f"           {cat:12s}  {avg:.3f}")
        print(f"\n[OK] Baseline saved to {BASELINE_PATH.name}: {score:.4f} (n={evaluated})")
        sys.exit(0)

    # Load baseline
    if not BASELINE_PATH.exists():
        print("\n[WARN] No baseline file found. Run with --set-baseline first.")
        print("       Falling back to fixed threshold 0.65.")
        threshold = 0.65
        baseline  = None
        baseline_per_cat: dict[str, float] = {}
    else:
        baseline_data    = json.loads(BASELINE_PATH.read_text())
        baseline         = baseline_data["score"]
        baseline_per_cat = baseline_data.get("per_category", {})
        threshold        = round(baseline - args.tolerance, 4)
        print(f"\n[CI EVAL] Baseline: {baseline:.4f}  tolerance: -{args.tolerance:.2f}  threshold: {threshold:.4f}")

    print(f"\n[CI EVAL] Per-category vs baseline:")
    for cat, avg in sorted(per_category.items()):
        b     = baseline_per_cat.get(cat)
        delta = f"  (delta {avg - b:+.3f})" if b is not None else ""
        print(f"           {cat:12s}  {avg:.3f}{delta}")

    if score >= threshold:
        print(f"[PASS] {score:.4f} >= {threshold:.4f}")
        sys.exit(0)
    else:
        drop = baseline - score
        print(f"[FAIL] {score:.4f} < {threshold:.4f}  (dropped {drop:.4f} from baseline)")
        sys.exit(1)


if __name__ == "__main__":
    main()
