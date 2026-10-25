"""
tune_tagger_prompts.py — Attribute tagger prompt A/B experiment.

Evaluates multiple _BASE_PROMPT variants against the 35-query golden dataset.
For each variant:
  - Runs the tagger on all golden images (via OpenRouter → Gemini direct fallback)
  - Computes vocab_hit_rate, completeness, color_specificity, confidence_avg
  - Runs the gateway /search for each golden query → top 5 results
  - Judges each (query, result) pair with Gemini → gemini_top5_avg
  - Logs all metrics to MLflow on the VM

Judge scores are computed once (search quality is independent of the tagger prompt)
and reused across all variant runs — saves ~7 min of API calls per extra variant.

Usage:
    python mlops/tune_tagger_prompts.py \\
        --gateway-url http://20.240.203.22:30800 \\
        [--variants v4_baseline,v5a_strict_vocab] \\
        [--mlflow-uri http://20.240.203.22:5000] \\
        [--dry-run]    # tagger only, skip judge (fast/cheap check)
        [--offline]    # no API calls, prints prompts and exits
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent
IMAGES_DIR = ROOT / "mlops" / "golden_images"
DATASET    = ROOT / "mlops" / "golden_dataset.json"

# ── load .env ─────────────────────────────────────────────────────────────────

env_path = ROOT / ".env"
if env_path.exists():
    for _line in env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")

DEFAULT_MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://20.240.203.22:5000")
DEFAULT_GATEWAY_URL = "http://20.240.203.22:30800"
EXPERIMENT_NAME     = "tagger_prompt_tuning"

# ── import tagger internals ───────────────────────────────────────────────────

sys.path.insert(0, str(ROOT / "attribute_tagger"))
from tagger import (  # noqa: E402
    _validate,
    _parse_json,
    _tag_openrouter,
    _tag_gemini_direct,
    _category_hint,
    VOCAB_STYLE,
    VOCAB_PATTERN,
    VOCAB_MATERIAL,
    OPENROUTER_MODEL,
)

# ── import judge internals ────────────────────────────────────────────────────

sys.path.insert(0, str(ROOT / "gateway"))
from judge import (  # noqa: E402
    _judge_pair_openrouter,
    _judge_pair_gemini_direct,
)

# ── prompt variants ───────────────────────────────────────────────────────────
# Every variant uses {category_hint} as the sole .format() placeholder.
# {{ and }} are literal braces for the JSON schema block.

_JSON_SCHEMA = """\
{{
  "colors": ["list", "of", "1-3", "dominant colors"],
  "style": "one style label",
  "silhouette": "brief shape/cut description",
  "occasion": ["list", "of", "1-2", "occasions"],
  "pattern": "one pattern label",
  "material_feel": "one material impression",
  "trend_tags": ["list", "of", "0-2", "trend tags"]
}}"""

_HEADER = (
    "You are a fashion attribute extraction system. "
    "Analyze the clothing item in this image.\n\n"
    f"Return ONLY a valid JSON object with exactly these keys:\n{_JSON_SCHEMA}\n"
)

_FOOTER = "{category_hint}\nRespond ONLY with valid JSON. No markdown. No explanation. No extra text."

PROMPT_VARIANTS: dict[str, str] = {

    # ── v4_baseline: current production prompt (control) ──────────────────────
    "v4_baseline": (
        _HEADER
        + """
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

"""
        + _FOOTER
    ),

    # ── v5a_strict_vocab: exhaustive MUST-use lists for all constrained fields ─
    "v5a_strict_vocab": (
        _HEADER
        + """
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

"""
        + _FOOTER
    ),

    # ── v5b_negative_examples: explicit alias avoidance block ─────────────────
    "v5b_negative_examples": (
        _HEADER
        + """
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

Do not use these common incorrect values (use the alternative shown instead):
- "knit" or "knitted" → use "knitwear"
- "plain" or "no pattern" → use "solid"
- "faux leather" or "suede" → use "leather"
- "polyester", "nylon", or "spandex" → use "synthetic"
- "boho" or "hippie" → use "bohemian"
- "activewear" or "sports" → use "athletic"
- "old money" or "luxury" → use "quiet luxury"
- "business casual" or "smart casual" → use "formal"

"""
        + _FOOTER
    ),

    # ── v5c_priority_rules: disambiguation rules for ambiguous cases ───────────
    "v5c_priority_rules": (
        _HEADER
        + """
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

Disambiguation rules when two values seem equally valid:
- style: prefer "athletic" over "streetwear" when workout/gym use is primary; \
prefer "quiet luxury" over "minimalist" when premium materials are evident
- style: prefer "formal" over "classic" when the item is clearly for office or evening events
- pattern: use "solid" only when there is truly no visible pattern, even subtle texture
- colors: always use compound color names ("navy blue", "forest green") not single words
- occasion: prefer "casual" for everyday basics; "weekend" for relaxed leisure pieces
- material_feel: when uncertain, choose the material that is most visually dominant

"""
        + _FOOTER
    ),

    # ── v5d_rich_silhouette: expanded silhouette examples per category group ───
    "v5d_rich_silhouette": (
        _HEADER
        + """
Guidelines:
- colors: e.g. "navy blue", "ivory", "forest green", "hot pink", "camel"
- style: one of casual, formal, streetwear, athletic, bohemian, minimalist, preppy, \
romantic, quiet luxury, edgy, classic, resort
- silhouette: describe the shape/cut precisely using compound terms. \
Tops — "fitted crew-neck tee", "boxy cropped sweatshirt", "v-neck puff-sleeve blouse", \
"oversized ribbed knit pullover". \
Bottoms — "high-rise wide-leg trousers", "mid-rise straight-leg jeans", \
"midi A-line skirt", "flared yoga leggings", "pleated wide-leg shorts". \
Dresses/jumpsuits — "sleeveless bodycon mini", "wrap midi dress", \
"smocked maxi dress", "wide-leg linen jumpsuit". \
Shoes — "block-heel ankle boot", "pointed-toe stiletto pump", \
"chunky platform sneaker", "strappy flat sandal". \
Bags — "structured top-handle tote", "quilted flap crossbody", "drawstring bucket bag". \
Outerwear — "oversized double-breasted blazer", "cropped zip-up puffer", \
"longline wool trench coat".
- occasion: e.g. "office", "casual", "evening", "beach", "workout", "date night", \
"weekend", "formal event"
- pattern: one of solid, striped, floral, plaid, animal print, geometric, \
abstract, polka dot, tie-dye, logo print
- material_feel: one of cotton, silk, denim, knitwear, leather, chiffon, \
linen, velvet, satin, wool, synthetic
- trend_tags: e.g. "old money", "Y2K", "cottagecore", "quiet luxury", "dark academia", \
"coastal grandmother", "clean girl", "gorpcore"

"""
        + _FOOTER
    ),
}

# ── helpers ───────────────────────────────────────────────────────────────────

ALL_FIELDS        = ["colors", "style", "silhouette", "occasion", "pattern", "material_feel", "trend_tags"]
CONSTRAINED_FIELDS = {"style": VOCAB_STYLE, "pattern": VOCAB_PATTERN, "material_feel": VOCAB_MATERIAL}


def _build_variant_prompt(base_template: str, category: str) -> str:
    hint = _category_hint(category)
    hint_block = f"\nCategory-specific guidance: {hint}\n" if hint else ""
    return base_template.format(category_hint=hint_block)


def _call_tagger(image_b64: str, category: str, base_template: str) -> tuple[dict, float]:
    """Call tagger with a custom base prompt. Returns (validated_attrs, confidence)."""
    prompt = _build_variant_prompt(base_template, category)
    raw: dict | None = None

    if OPENROUTER_API_KEY:
        raw = _tag_openrouter(image_b64, prompt, OPENROUTER_API_KEY)
    if raw is None and GOOGLE_API_KEY:
        raw = _tag_gemini_direct(image_b64, prompt, GOOGLE_API_KEY)

    if not raw:
        return {}, 0.0

    validated, confidence = _validate(raw)
    return validated, confidence


def _tagger_score(validated: dict, confidence: float) -> dict[str, float]:
    hits     = sum(1 for f, v in CONSTRAINED_FIELDS.items()
                   if str(validated.get(f, "")).strip().lower() in v)
    present  = sum(1 for f in ALL_FIELDS if validated.get(f) not in (None, [], ""))
    colors   = validated.get("colors", [])
    color_sp = (sum(len(str(c).split()) for c in colors) / len(colors)
                if isinstance(colors, list) and colors else 0.0)
    return {
        "vocab_hit_rate":    round(hits / len(CONSTRAINED_FIELDS), 4),
        "completeness":      round(present / len(ALL_FIELDS), 4),
        "color_specificity": round(color_sp, 4),
        "confidence":        round(confidence, 4),
    }


def _search_gateway(entry: dict, gateway_url: str, top_k: int = 5) -> list[str]:
    """POST to /search, return up to top_k result image URLs."""
    fname = entry["query_image_url"].split("/")[-1]
    img_path = IMAGES_DIR / fname
    if not img_path.exists():
        print(f"  [search] missing image: {fname}")
        return []

    img_bytes = img_path.read_bytes()
    category  = entry.get("query_category_tag", "")
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{gateway_url.rstrip('/')}/search",
                files={"file": (fname, img_bytes, "image/jpeg")},
                data={"search_label": category, "skip_judge": "true"},
            )
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
        return [m["image_url"] for m in matches[:top_k] if m.get("image_url")]
    except Exception as e:
        print(f"  [search] error for '{entry.get('query_name', fname)}': {e}")
        return []


def _judge_pair(query_b64: str, result_url: str) -> float | None:
    """Judge one (query, result) pair — OpenRouter first, Gemini direct fallback."""
    score = None
    if OPENROUTER_API_KEY:
        score = _judge_pair_openrouter(query_b64, result_url, OPENROUTER_API_KEY)
    if score is None and GOOGLE_API_KEY:
        score = _judge_pair_gemini_direct(query_b64, result_url, GOOGLE_API_KEY)
    return score


def _load_image_b64(entry: dict) -> str | None:
    fname    = entry["query_image_url"].split("/")[-1]
    img_path = IMAGES_DIR / fname
    if not img_path.exists():
        return None
    return base64.b64encode(img_path.read_bytes()).decode("utf-8")


# ── judge phase (computed once, reused across all variants) ───────────────────

def compute_judge_scores(
    dataset: list[dict],
    gateway_url: str,
) -> list[dict[str, Any]]:
    """
    For every golden query: search → top 5 → judge each pair.
    Returns a flat list of {category, query_name, score} dicts.
    """
    rows: list[dict] = []
    total = len(dataset)
    print(f"\n[judge] Evaluating top-5 search results for {total} queries...")

    for i, entry in enumerate(dataset, 1):
        name     = entry.get("query_name", "?")
        category = entry.get("query_category_tag", "")
        print(f"  [{i:02d}/{total}] {category:<14} | {name[:45]}")

        query_b64 = _load_image_b64(entry)
        if query_b64 is None:
            print(f"    [judge] missing local image — skipped")
            continue

        result_urls = _search_gateway(entry, gateway_url, top_k=5)
        if not result_urls:
            print(f"    [judge] no results — skipped")
            continue

        query_scores: list[float] = []
        for url in result_urls:
            score = _judge_pair(query_b64, url)
            if score is not None:
                query_scores.append(score)
                rows.append({"category": category, "query_name": name, "score": score})

        if query_scores:
            avg = mean(query_scores)
            print(f"    [judge] {len(query_scores)} scores, avg={avg:.3f}")
        else:
            print(f"    [judge] all calls failed")

    print(f"[judge] Done. {len(rows)} judge scores collected.")
    return rows


# ── tagger phase (once per variant) ──────────────────────────────────────────

def run_tagger_variant(
    variant_name: str,
    base_template: str,
    dataset: list[dict],
) -> list[dict[str, Any]]:
    """
    Run the tagger with base_template on all golden images.
    Returns a flat list of per-image score dicts.
    """
    rows: list[dict] = []
    total = len(dataset)
    print(f"\n[tagger:{variant_name}] Running on {total} images...")

    for i, entry in enumerate(dataset, 1):
        name     = entry.get("query_name", "?")
        category = entry.get("query_category_tag", "")
        print(f"  [{i:02d}/{total}] {category:<14} | {name[:45]}")

        image_b64 = _load_image_b64(entry)
        if image_b64 is None:
            print(f"    [tagger] missing image — skipped")
            continue

        validated, confidence = _call_tagger(image_b64, category, base_template)
        s = _tagger_score(validated, confidence)
        s["category"]   = category
        s["query_name"] = name
        s["raw"]        = validated
        rows.append(s)

        print(
            f"    vocab={s['vocab_hit_rate']:.2f}  "
            f"complete={s['completeness']:.2f}  "
            f"color_spec={s['color_specificity']:.2f}  "
            f"conf={s['confidence']:.2f}"
        )

    print(f"[tagger:{variant_name}] Done. {len(rows)} images processed.")
    return rows


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_run(
    variant_name:  str,
    base_template: str,
    tagger_rows:   list[dict],
    judge_rows:    list[dict],
    mlflow_uri:    str,
    dry_run:       bool,
) -> None:
    import mlflow  # lazy import — not required for --offline

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    from datetime import datetime
    run_name = f"{variant_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # ── params ────────────────────────────────────────────────────────────
        mlflow.log_param("prompt_variant", variant_name)
        mlflow.log_param("model",          OPENROUTER_MODEL)
        mlflow.log_param("n_queries",      len(tagger_rows))
        mlflow.log_param("dry_run",        dry_run)

        # ── overall tagger metrics ─────────────────────────────────────────────
        if tagger_rows:
            mlflow.log_metric("vocab_hit_rate",    mean(r["vocab_hit_rate"]    for r in tagger_rows))
            mlflow.log_metric("completeness",      mean(r["completeness"]      for r in tagger_rows))
            mlflow.log_metric("color_specificity", mean(r["color_specificity"] for r in tagger_rows))
            mlflow.log_metric("confidence_avg",    mean(r["confidence"]        for r in tagger_rows))

        # ── per-category tagger metrics ────────────────────────────────────────
        by_cat: dict[str, list] = defaultdict(list)
        for r in tagger_rows:
            by_cat[r["category"]].append(r)
        for cat, cat_rows in by_cat.items():
            mlflow.log_metric(f"vocab_hit_rate_{cat}",    mean(r["vocab_hit_rate"] for r in cat_rows))
            mlflow.log_metric(f"completeness_{cat}",      mean(r["completeness"]   for r in cat_rows))
            mlflow.log_metric(f"confidence_avg_{cat}",    mean(r["confidence"]     for r in cat_rows))

        # ── overall judge metrics ──────────────────────────────────────────────
        if judge_rows:
            scores = [r["score"] for r in judge_rows]
            mlflow.log_metric("gemini_top5_avg",       mean(scores))
            mlflow.log_metric("gemini_top5_pass_rate", mean(s >= 0.70 for s in scores))

        # ── per-category judge metrics ─────────────────────────────────────────
            by_cat_j: dict[str, list[float]] = defaultdict(list)
            for r in judge_rows:
                by_cat_j[r["category"]].append(r["score"])
            for cat, cat_scores in by_cat_j.items():
                mlflow.log_metric(f"gemini_top5_avg_{cat}", mean(cat_scores))

        # ── artifact: per-image results ────────────────────────────────────────
        results_payload = {
            "variant": variant_name,
            "tagger":  tagger_rows,
            "judge":   judge_rows,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_results.json", delete=False, prefix=f"{variant_name}_"
        ) as tf:
            json.dump(results_payload, tf, indent=2, default=str)
            tf_path = tf.name
        mlflow.log_artifact(tf_path, artifact_path="results")
        os.unlink(tf_path)

        # ── artifact: full prompt text ─────────────────────────────────────────
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_prompt.txt", delete=False, prefix=f"{variant_name}_",
            encoding="utf-8",
        ) as tf:
            tf.write(base_template)
            tf_path = tf.name
        mlflow.log_artifact(tf_path, artifact_path="prompts")
        os.unlink(tf_path)

    print(f"[mlflow] Run '{run_name}' logged to {mlflow_uri} → experiment '{EXPERIMENT_NAME}'")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(
    results: dict[str, dict],
    judge_rows: list[dict],
) -> None:
    judge_avg = mean(r["score"] for r in judge_rows) if judge_rows else 0.0
    judge_pass = mean(r["score"] >= 0.70 for r in judge_rows) if judge_rows else 0.0

    cols = ["vocab_hit_rate", "completeness", "color_specificity", "confidence_avg"]
    header = f"{'VARIANT':<26} {'VOCAB':>6} {'COMPL':>6} {'C-SPEC':>6} {'CONF':>6} {'JUDGE_AVG':>10} {'JUDGE_PASS':>11}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for variant_name, row in results.items():
        if not row:
            print(f"{variant_name:<26}  (no data)")
            continue
        print(
            f"{variant_name:<26} "
            f"{row.get('vocab_hit_rate', 0):>6.3f} "
            f"{row.get('completeness', 0):>6.3f} "
            f"{row.get('color_specificity', 0):>6.3f} "
            f"{row.get('confidence_avg', 0):>6.3f} "
            f"{judge_avg:>10.3f} "
            f"{judge_pass:>10.1%}"
        )
    print("=" * len(header))
    print("(judge columns are identical across variants — search quality is tagger-independent)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Tagger prompt A/B experiment")
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL,
                        help="Gateway base URL for /search calls")
    parser.add_argument("--mlflow-uri", default=DEFAULT_MLFLOW_URI,
                        help="MLflow tracking URI")
    parser.add_argument("--variants", default="",
                        help="Comma-separated variant keys to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip judge phase — tagger metrics only")
    parser.add_argument("--offline", action="store_true",
                        help="Print prompts and exit, no API calls")
    args = parser.parse_args()

    if args.offline:
        for name, prompt in PROMPT_VARIANTS.items():
            print(f"\n{'='*60}\n{name}\n{'='*60}")
            # Show resolved prompt for the 'tops' category as a sample
            print(_build_variant_prompt(prompt, "tops"))
        return

    if not OPENROUTER_API_KEY and not GOOGLE_API_KEY:
        print("ERROR: neither OPENROUTER_API_KEY nor GOOGLE_API_KEY is set.")
        sys.exit(1)

    dataset: list[dict] = json.loads(DATASET.read_text())

    # Filter to variants requested
    selected: dict[str, str] = {}
    if args.variants:
        for key in args.variants.split(","):
            key = key.strip()
            if key in PROMPT_VARIANTS:
                selected[key] = PROMPT_VARIANTS[key]
            else:
                print(f"WARNING: unknown variant '{key}' — skipped. Available: {list(PROMPT_VARIANTS)}")
    else:
        selected = dict(PROMPT_VARIANTS)

    print(f"\nRunning {len(selected)} variant(s): {list(selected)}")
    print(f"MLflow: {args.mlflow_uri}  |  Gateway: {args.gateway_url}")
    print(f"Dataset: {len(dataset)} queries  |  Images: {IMAGES_DIR}")

    # ── Phase 1: judge scores (once, search-quality baseline) ─────────────────
    judge_rows: list[dict] = []
    if not args.dry_run:
        judge_rows = compute_judge_scores(dataset, args.gateway_url)
    else:
        print("\n[judge] --dry-run set — skipping judge phase.")

    # ── Phase 2: tagger evaluation per variant ─────────────────────────────────
    summary: dict[str, dict] = {}

    for variant_name, base_template in selected.items():
        tagger_rows = run_tagger_variant(variant_name, base_template, dataset)

        aggregate: dict[str, float] = {}
        if tagger_rows:
            for metric in ("vocab_hit_rate", "completeness", "color_specificity", "confidence"):
                aggregate[metric if metric != "confidence" else "confidence_avg"] = mean(
                    r[metric] for r in tagger_rows
                )
        summary[variant_name] = aggregate

        log_run(
            variant_name  = variant_name,
            base_template = base_template,
            tagger_rows   = tagger_rows,
            judge_rows    = judge_rows,
            mlflow_uri    = args.mlflow_uri,
            dry_run       = args.dry_run,
        )

    # ── Phase 3: summary table ─────────────────────────────────────────────────
    print_summary(summary, judge_rows)

    # Save summary JSON alongside the dataset for reference
    out_path = ROOT / "mlops" / "tagger_prompt_tuning_summary.json"
    out_path.write_text(json.dumps({
        "variants": summary,
        "judge_avg": mean(r["score"] for r in judge_rows) if judge_rows else None,
        "judge_pass_rate": mean(r["score"] >= 0.70 for r in judge_rows) if judge_rows else None,
    }, indent=2))
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
