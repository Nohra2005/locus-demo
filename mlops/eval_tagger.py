"""
eval_tagger.py — validate tagger improvements.

Part 1 (no API): unit-tests _validate() normalisation logic.
Part 2 (API):    old generic prompt vs new category-aware prompt on all golden images.
                 Uses OpenRouter (configurable model) with Gemini direct as fallback.

Usage:
    python mlops/eval_tagger.py                              # default model (gpt-4o)
    python mlops/eval_tagger.py --model openai/gpt-4o        # explicit model
    python mlops/eval_tagger.py --offline                    # Part 1 only
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


ROOT       = Path(__file__).parent.parent
IMAGES_DIR = ROOT / "mlops" / "golden_images"
DATASET    = ROOT / "mlops" / "golden_dataset.json"

sys.path.insert(0, str(ROOT / "attribute_tagger"))
from tagger import (  # noqa: E402
    _validate, _build_prompt, _parse_json, _tag_gemini_direct,
    VOCAB_STYLE, VOCAB_PATTERN, VOCAB_MATERIAL,
    OPENROUTER_API_KEY, GOOGLE_API_KEY,
)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_EVAL_MODEL = "openai/gpt-4o"

# Generic prompt used as the "old" baseline — no category-specific guidance.
OLD_PROMPT_FOR_EVAL = """\
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
- colors: e.g. "navy blue", "ivory", "forest green", "hot pink", "camel"
- style: one of casual, formal, streetwear, athletic, bohemian, minimalist, preppy, \
romantic, quiet luxury, edgy, classic, resort
- silhouette: e.g. "fitted blazer", "flowy midi dress", "slim-fit jeans", \
"oversized hoodie", "a-line skirt"
- occasion: e.g. "office", "casual", "evening", "beach", "workout", "date night", \
"weekend", "formal event"
- pattern: one of solid, striped, floral, plaid, animal print, geometric, \
abstract, polka dot, tie-dye, logo print
- material_feel: one of cotton, silk, denim, knitwear, leather, chiffon, \
linen, velvet, satin, wool, synthetic
- trend_tags: e.g. "old money", "Y2K", "cottagecore", "quiet luxury", "dark academia", \
"coastal grandmother", "clean girl", "gorpcore"

Respond ONLY with valid JSON. No markdown. No explanation. No extra text."""

CONSTRAINED_FIELDS = {
    "style":         VOCAB_STYLE,
    "pattern":       VOCAB_PATTERN,
    "material_feel": VOCAB_MATERIAL,
}

ALL_FIELDS = ["colors", "style", "silhouette", "occasion", "pattern", "material_feel", "trend_tags"]

# ── Part 1: validation unit tests ─────────────────────────────────────────────

MOCK_RESPONSES = [
    {
        "desc": "style alias: 'casual wear' -> 'casual'",
        "input": {"colors": ["white"], "style": "casual wear", "silhouette": "fitted tee",
                  "occasion": ["casual"], "pattern": "solid", "material_feel": "cotton", "trend_tags": []},
        "expected_style": "casual",
    },
    {
        "desc": "material alias: 'knit' -> 'knitwear'",
        "input": {"colors": ["cream"], "style": "classic", "silhouette": "oversized sweater",
                  "occasion": ["casual"], "pattern": "solid", "material_feel": "knit", "trend_tags": []},
        "expected_material": "knitwear",
    },
    {
        "desc": "pattern alias: 'plain' -> 'solid'",
        "input": {"colors": ["navy"], "style": "minimalist", "silhouette": "slim trousers",
                  "occasion": ["office"], "pattern": "plain", "material_feel": "wool", "trend_tags": []},
        "expected_pattern": "solid",
    },
    {
        "desc": "all fields in vocab — confidence should be 1.0",
        "input": {"colors": ["black"], "style": "edgy", "silhouette": "moto jacket",
                  "occasion": ["casual", "evening"], "pattern": "solid",
                  "material_feel": "leather", "trend_tags": ["Y2K"]},
        "expected_confidence": 1.0,
    },
    {
        "desc": "mixed case normalisation",
        "input": {"colors": ["Navy Blue"], "style": "Casual", "silhouette": "straight leg",
                  "occasion": ["Casual"], "pattern": "Solid", "material_feel": "Denim",
                  "trend_tags": ["Old Money"]},
        "expected_style": "casual",
        "expected_pattern": "solid",
        "expected_material": "denim",
    },
    {
        "desc": "unknown style kept as-is (no silent drop)",
        "input": {"colors": ["red"], "style": "futuristic", "silhouette": "structured jacket",
                  "occasion": ["evening"], "pattern": "geometric", "material_feel": "synthetic",
                  "trend_tags": []},
        "expected_style": "futuristic",
    },
]


def run_validation_tests() -> int:
    print("=" * 60)
    print("PART 1 - Validation / normalisation unit tests (no API)")
    print("=" * 60)
    failures = 0
    for tc in MOCK_RESPONSES:
        out, confidence = _validate(tc["input"])
        passed = True
        notes = []
        if "expected_style" in tc and out["style"] != tc["expected_style"]:
            passed = False
            notes.append(f"style: got {out['style']!r}, want {tc['expected_style']!r}")
        if "expected_material" in tc and out["material_feel"] != tc["expected_material"]:
            passed = False
            notes.append(f"material_feel: got {out['material_feel']!r}, want {tc['expected_material']!r}")
        if "expected_pattern" in tc and out["pattern"] != tc["expected_pattern"]:
            passed = False
            notes.append(f"pattern: got {out['pattern']!r}, want {tc['expected_pattern']!r}")
        if "expected_confidence" in tc and abs(confidence - tc["expected_confidence"]) > 0.01:
            passed = False
            notes.append(f"confidence: got {confidence:.2f}, want {tc['expected_confidence']:.2f}")
        if not passed:
            failures += 1
        print(f"  [{'PASS' if passed else 'FAIL'}] {tc['desc']}")
        for n in notes:
            print(f"         ! {n}")
    print(f"\nResult: {len(MOCK_RESPONSES) - failures}/{len(MOCK_RESPONSES)} passed")
    return failures


# ── Part 2: API call ──────────────────────────────────────────────────────────

def _openrouter_call(image_b64: str, prompt: str, model: str) -> dict | None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                OPENROUTER_API_URL,
                json=payload,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            )
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("retry-after", 10))
            print(f"    [openrouter] 429 — waiting {retry_after}s...")
            time.sleep(retry_after)
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    OPENROUTER_API_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _parse_json(content)
    except Exception as e:
        print(f"    [openrouter] error: {e}")
        return None


def _provider_call(image_b64: str, prompt: str, model: str) -> dict | None:
    """OpenRouter (configurable model) first, Gemini direct fallback."""
    if OPENROUTER_API_KEY:
        print(f"    [{model}] calling...")
        result = _openrouter_call(image_b64, prompt, model)
        if result is not None:
            print(f"    [{model}] OK")
            return result
        print(f"    [{model}] failed, falling back to gemini")
    if GOOGLE_API_KEY:
        print("    [gemini] calling...")
        result = _tag_gemini_direct(image_b64, prompt, GOOGLE_API_KEY)
        if result is not None:
            print("    [gemini] OK")
        else:
            print("    [gemini] failed")
        return result
    return None


def score(attrs: dict) -> dict:
    if not attrs:
        return {"vocab_hit_rate": 0.0, "completeness": 0.0, "color_specificity": 0.0}
    hits = sum(1 for f, v in CONSTRAINED_FIELDS.items() if str(attrs.get(f,"")).strip().lower() in v)
    present = sum(1 for f in ALL_FIELDS if attrs.get(f) not in (None, [], ""))
    colors = attrs.get("colors", [])
    color_spec = (sum(len(str(c).split()) for c in colors) / len(colors)) if isinstance(colors, list) and colors else 0.0
    return {
        "vocab_hit_rate":    round(hits / len(CONSTRAINED_FIELDS), 3),
        "completeness":      round(present / len(ALL_FIELDS), 3),
        "color_specificity": round(color_spec, 3),
    }


def run_api_comparison(model: str = DEFAULT_EVAL_MODEL) -> None:
    print()
    print("=" * 60)
    print(f"PART 2 - OLD vs NEW prompt  |  model: {model}")
    print("=" * 60)

    if not OPENROUTER_API_KEY and not GOOGLE_API_KEY:
        print("No API keys set - skipping.")
        return

    dataset = json.loads(DATASET.read_text())
    samples: list[dict] = []
    per_cat: dict[str, int] = {}
    for entry in dataset:
        cat   = entry["query_category_tag"]
        fname = entry["query_image_url"].split("/")[-1]
        if (IMAGES_DIR / fname).exists() and per_cat.get(cat, 0) < 5:
            samples.append({"category": cat, "name": entry.get("query_name", fname), "file": fname})
            per_cat[cat] = per_cat.get(cat, 0) + 1

    print(f"Running on {len(samples)} images ({len(per_cat)} categories)...\n")
    results = []

    for i, s in enumerate(samples):
        cat, name, fname = s["category"], s["name"], s["file"]
        print(f"[{i+1:02d}/{len(samples)}] {cat:12s} | {name[:45]}")
        image_b64 = base64.b64encode((IMAGES_DIR / fname).read_bytes()).decode()

        old_raw = _provider_call(image_b64, OLD_PROMPT_FOR_EVAL, model)
        time.sleep(2)
        new_raw = _provider_call(image_b64, _build_prompt(cat), model)
        if new_raw:
            new_validated, _ = _validate(new_raw)
        else:
            new_validated = None
        time.sleep(2)

        s_old = score(old_raw or {})
        s_new = score(new_validated or {})
        results.append({"category": cat, "name": name, "old": s_old, "new": s_new,
                        "old_raw": old_raw, "new_raw": new_validated})

        print(f"         OLD vocab={s_old['vocab_hit_rate']:.2f} complete={s_old['completeness']:.2f} color_spec={s_old['color_specificity']:.2f}")
        print(f"         NEW vocab={s_new['vocab_hit_rate']:.2f} complete={s_new['completeness']:.2f} color_spec={s_new['color_specificity']:.2f}")
        # Show silhouette field specifically for non-clothing categories
        if cat in ("shoes", "bag", "hat") and old_raw and new_validated:
            print(f"         silhouette OLD: {old_raw.get('silhouette','')}")
            print(f"         silhouette NEW: {new_validated.get('silhouette','')}")
        print()

    if not results:
        return

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    metrics = ["vocab_hit_rate", "completeness", "color_specificity"]
    print("=" * 65)
    print(f"{'METRIC':<22} {'OLD':>8} {'NEW':>8} {'DELTA':>8} {'WIN?':>6}")
    print("=" * 65)
    for m in metrics:
        old_avg = avg([r["old"][m] for r in results])
        new_avg = avg([r["new"][m] for r in results])
        delta   = new_avg - old_avg
        win     = "NEW +" if delta > 0.01 else ("TIE" if abs(delta) <= 0.01 else "OLD +")
        print(f"{m:<22} {old_avg:>8.3f} {new_avg:>8.3f} {delta:>+8.3f} {win:>6}")
    print("=" * 65)

    print("\nPer-category vocab_hit_rate delta (NEW - OLD):")
    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["new"]["vocab_hit_rate"] - r["old"]["vocab_hit_rate"])
    for cat, deltas in sorted(by_cat.items()):
        d = avg(deltas)
        bar = ("+" * int(d * 10)) if d > 0 else ("-" * int(abs(d) * 10))
        print(f"  {cat:<14} {d:>+.3f}  {bar}")

    out_path = ROOT / "mlops" / "tagger_eval_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    offline = "--offline" in sys.argv
    model = DEFAULT_EVAL_MODEL
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--model" and i < len(sys.argv) - 1:
            model = sys.argv[i + 1]

    failures = run_validation_tests()
    if not offline:
        run_api_comparison(model)
    sys.exit(0 if failures == 0 else 1)
