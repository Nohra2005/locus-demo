"""
Judge calibration script.

Evaluates the AI judge against known-similar pairs from the golden dataset.
Does NOT use the search pipeline — pairs are taken directly from golden_dataset.json.

Usage:
    python mlops/calibrate_judge.py
    python mlops/calibrate_judge.py --limit 5
    python mlops/calibrate_judge.py --mlflow

Requires OPENROUTER_API_KEY env var.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

# Allow importing from gateway/
sys.path.insert(0, str(Path(__file__).parent.parent / "gateway"))
from judge import JUDGE_PROMPT, OPENROUTER_MODEL, OPENROUTER_API_URL  # noqa: E402

import httpx  # noqa: E402

GOLDEN_DATASET      = Path(__file__).parent / "golden_dataset.json"
GOLDEN_IMAGES       = Path(__file__).parent / "golden_images"
CALIBRATION_RESULTS = Path(__file__).parent / "calibration_results.json"

# Terms that suggest a query item is visually generic
_GENERIC_TERMS = {"white", "black", "basic", "plain", "simple"}

PASS_THRESHOLD = 0.70
WARN_THRESHOLD = 0.50


def _url_to_local(image_url: str) -> Path:
    filename = image_url.rsplit("/", 1)[-1]
    return GOLDEN_IMAGES / filename


def _load_image_b64(path: Path) -> str | None:
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _call_judge(query_b64: str, result_b64: str, api_key: str) -> float | None:
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": JUDGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{result_b64}"}},
                ],
            }
        ],
        "max_tokens": 50,
        "temperature": 0.0,
    }

    for attempt in range(3):
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    OPENROUTER_API_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("retry-after", 60))
                print(f"  [429] rate limited, waiting {retry_after:.0f}s …")
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            import re
            content = resp.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\d+\.\d+|\d+", content)
            return float(m.group()) if m else None
        except Exception as e:
            if attempt < 2:
                time.sleep(5.0 * (2 ** attempt))
            else:
                print(f"  [ERROR] {e}")
                return None

    return None


def _is_generic(query_name: str) -> bool:
    words = set(query_name.lower().split())
    if words & _GENERIC_TERMS:
        return True
    if len(query_name.split()) <= 2:
        return True
    return False


def run_calibration(api_key: str, limit: int | None, use_mlflow: bool) -> None:
    entries = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))
    if limit:
        entries = entries[:limit]

    all_scores: list[float] = []
    results: list[dict] = []

    # Rate limiting: 2s between calls
    _last_call = [0.0]

    def _throttle():
        gap = 2.0 - (time.monotonic() - _last_call[0])
        if gap > 0:
            time.sleep(gap)
        _last_call[0] = time.monotonic()

    for i, entry in enumerate(entries, 1):
        name     = entry["query_name"]
        generic  = _is_generic(name)
        tag      = " [GENERIC]" if generic else ""
        category = entry.get("query_category_tag", "?")

        print(f"\n{'='*60}")
        print(f"[{i}/{len(entries)}] {name} ({category}){tag}")

        query_path = _url_to_local(entry["query_image_url"])
        query_b64  = _load_image_b64(query_path)
        if query_b64 is None:
            print("  Skipping — query image missing")
            continue

        pair_scores: list[float] = []
        scores_by_id: dict[str, float] = {}

        for rel in entry.get("relevant_info", []):
            rel_name  = rel.get("name", "?")
            rel_pid   = rel.get("product_id", "")
            rel_path  = _url_to_local(rel["image_url"])
            rel_b64   = _load_image_b64(rel_path)
            if rel_b64 is None:
                continue

            _throttle()
            score = _call_judge(query_b64, rel_b64, api_key)

            if score is None:
                print(f"  ✗  {rel_name[:50]} → [no score]")
                continue

            flag = " ⚠ LOW" if score < WARN_THRESHOLD else ""
            print(f"  {'✓' if score >= PASS_THRESHOLD else '~'}  {rel_name[:50]} → {score:.2f}{flag}")
            pair_scores.append(score)
            if rel_pid:
                scores_by_id[rel_pid] = round(score, 3)

        if not pair_scores:
            continue

        avg   = sum(pair_scores) / len(pair_scores)
        label = "PASS" if avg >= PASS_THRESHOLD else ("WARN" if avg >= WARN_THRESHOLD else "FAIL")
        print(f"  → avg={avg:.2f}  [{label}]  ({len(pair_scores)}/5 scored)")

        all_scores.extend(pair_scores)
        results.append({
            "query_name":    name,
            "category":      category,
            "generic":       generic,
            "avg_score":     round(avg, 3),
            "pair_scores":   [round(s, 3) for s in pair_scores],
            "scores_by_id":  scores_by_id,
            "label":         label,
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    if not all_scores:
        print("\nNo scores collected.")
        return

    overall_avg   = sum(all_scores) / len(all_scores)
    pct_pass      = 100 * sum(1 for s in all_scores if s >= PASS_THRESHOLD) / len(all_scores)
    fail_queries  = [r for r in results if r["label"] == "FAIL"]
    warn_queries  = [r for r in results if r["label"] == "WARN"]
    generic_fails = [r for r in results if r["generic"] and r["label"] in ("FAIL", "WARN")]

    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"  Queries evaluated : {len(results)}")
    print(f"  Pairs scored      : {len(all_scores)}")
    print(f"  Overall avg score : {overall_avg:.3f}")
    print(f"  Pairs ≥ 0.80      : {pct_pass:.1f}%")
    print(f"  PASS / WARN / FAIL: {sum(1 for r in results if r['label']=='PASS')} / {len(warn_queries)} / {len(fail_queries)}")

    if fail_queries or warn_queries:
        print("\n  Queries needing attention (FAIL/WARN):")
        for r in sorted(fail_queries + warn_queries, key=lambda x: x["avg_score"]):
            g = " [GENERIC]" if r["generic"] else ""
            print(f"    [{r['label']}] {r['query_name']}{g}  avg={r['avg_score']:.2f}")

    if generic_fails:
        print("\n  Generic items that also fail — strong re-annotation candidates:")
        for r in generic_fails:
            print(f"    {r['query_name']}  avg={r['avg_score']:.2f}")

    if pct_pass < 70:
        print("\n  ⚠  Calibration rate < 70% — consider tightening JUDGE_PROMPT anchor points.")

    CALIBRATION_RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {CALIBRATION_RESULTS}")

    if use_mlflow:
        _log_mlflow(results, overall_avg, pct_pass)


def _log_mlflow(results: list[dict], overall_avg: float, pct_pass: float) -> None:
    try:
        import mlflow
        mlflow.set_experiment("judge_calibration")
        with mlflow.start_run():
            mlflow.log_metric("overall_avg_score", round(overall_avg, 4))
            mlflow.log_metric("pct_pairs_passing", round(pct_pass, 2))
            mlflow.log_metric("n_queries", len(results))
            mlflow.log_metric("n_fail", sum(1 for r in results if r["label"] == "FAIL"))
            mlflow.log_metric("n_warn", sum(1 for r in results if r["label"] == "WARN"))
            mlflow.log_metric("n_pass", sum(1 for r in results if r["label"] == "PASS"))
            tmp = Path("calibration_results.json")
            tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(tmp))
            tmp.unlink()
        print("\n  MLflow run logged to experiment 'judge_calibration'.")
    except Exception as e:
        print(f"\n  [MLflow error] {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the AI judge against the golden dataset")
    parser.add_argument("--limit",  type=int, default=None, help="Run on first N queries only")
    parser.add_argument("--mlflow", action="store_true",    help="Log results to MLflow")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("Error: OPENROUTER_API_KEY env var not set.")

    run_calibration(api_key=api_key, limit=args.limit, use_mlflow=args.mlflow)
