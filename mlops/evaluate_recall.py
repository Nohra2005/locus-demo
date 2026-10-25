"""
Locus Recall Evaluation
=======================
Measures Recall@K against the golden dataset.

Usage:
    python mlops/evaluate_recall.py [--gateway-url URL] [--k 5,10,25] [--mlflow]

No LLM calls — pure set intersection against known ground truth.
Exits with code 0 on success, code 1 if fewer than 25 queries evaluated.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import warnings

import requests

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
GOLDEN_IMAGES_DIR   = os.path.join(os.path.dirname(__file__), "golden_images")
_GOLDEN_URL_PREFIX  = "/golden-dataset/images/"


def load_golden_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _image_bytes_from_entry(entry: dict) -> bytes | None:
    url = entry.get("query_image_url", "")
    if url.startswith("data:"):
        # base64 data URI
        try:
            header, b64 = url.split(",", 1)
            return base64.b64decode(b64)
        except Exception as e:
            logger.warning(f"Failed to decode data URI for '{entry.get('query_name')}': {e}")
            return None
    if _GOLDEN_URL_PREFIX in url:
        # Local golden-dataset image — read directly from disk, no HTTP round-trip
        filename   = url.rsplit("/", 1)[-1]
        local_path = os.path.join(GOLDEN_IMAGES_DIR, filename)
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                return f.read()
        logger.warning(f"Local image not found on disk: {local_path}")
        return None
    # Plain URL — download it
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.warning(f"Failed to download query image for '{entry.get('query_name')}': {e}")
        return None


def search(image_bytes: bytes, gateway_url: str, category_tag: str = "") -> list[str]:
    """
    Call POST /search (with include_golden=true) and return product_id list.
    Returns empty list on error.
    """
    files  = {"file": ("query.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    data   = {}
    if category_tag:
        data["search_label"] = category_tag

    try:
        resp = requests.post(
            f"{gateway_url}/search?include_golden=true",
            files=files,
            data=data,
            timeout=30,
        )
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
        return [m["product_id"] for m in matches if m.get("product_id")]
    except Exception as e:
        logger.warning(f"Search request failed: {e}")
        return []


def evaluate(gateway_url: str, ks: list[int]) -> dict:
    dataset = load_golden_dataset(GOLDEN_DATASET_PATH)
    n_queries = len(dataset)
    total_ground_truth = sum(len(e.get("relevant_product_ids", [])) for e in dataset)

    hits = {k: 0 for k in ks}
    per_query = {k: [] for k in ks}
    evaluated = 0

    for entry in dataset:
        name       = entry.get("query_name", "?")
        ground_ids = set(entry.get("relevant_product_ids", []))

        if not ground_ids:
            logger.warning(f"Skipping '{name}' — no relevant_product_ids")
            continue

        image_bytes = _image_bytes_from_entry(entry)
        if image_bytes is None:
            logger.warning(f"Skipping '{name}' — could not load query image")
            for k in ks:
                per_query[k].append({"name": name, "hits": 0, "total": len(ground_ids), "ok": False})
            continue

        category_tag = entry.get("query_category_tag", "")
        result_ids   = search(image_bytes, gateway_url, category_tag)
        evaluated   += 1

        for k in ks:
            top_k     = set(result_ids[:k])
            n_hits    = len(ground_ids & top_k)
            hits[k]  += n_hits
            per_query[k].append({
                "name":  name,
                "hits":  n_hits,
                "total": len(ground_ids),
                "ok":    n_hits >= len(ground_ids) // 2,
            })

    return {
        "n_queries":           n_queries,
        "evaluated":           evaluated,
        "total_ground_truth":  total_ground_truth,
        "hits":                hits,
        "per_query":           per_query,
        "ks":                  ks,
    }


def print_report(r: dict):
    ks                 = r["ks"]
    n_queries          = r["n_queries"]
    total_ground_truth = r["total_ground_truth"]
    evaluated          = r["evaluated"]

    print("=" * 60)
    print("Locus Recall Evaluation")
    print(f"Queries: {n_queries}   Ground-truth items: {total_ground_truth}")
    print("=" * 60)

    for k in ks:
        h   = r["hits"][k]
        pct = 100.0 * h / total_ground_truth if total_ground_truth else 0.0
        print(f"Recall@{k:<3}: {h:>4} / {total_ground_truth}  ({pct:.1f}%)")

    print("-" * 60)
    k0 = ks[0]
    print(f"Per-query breakdown (Recall@{k0}):")
    for pq in r["per_query"][k0]:
        label = "[PASS]" if pq["ok"] else "[FAIL]"
        print(f"  {label} {pq['name']} — {pq['hits']}/{pq['total']} hits")
    print("=" * 60)

    if evaluated < n_queries:
        warnings.warn(f"Only {evaluated}/{n_queries} queries were evaluated successfully.")


def log_mlflow(r: dict, run_name: str | None = None):
    import mlflow
    ks   = r["ks"]
    gt   = r["total_ground_truth"]

    mlflow.set_experiment("locus_recall_eval")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("queries_evaluated", r["evaluated"])
        if run_name:
            mlflow.set_tag("run_name", run_name)
        for k in ks:
            h   = r["hits"][k]
            pct = h / gt if gt else 0.0
            mlflow.log_metric(f"recall_at_{k}", pct)
            mlflow.log_metric(f"hits_at_{k}",   h)
        mlflow.log_metric("queries_evaluated", r["evaluated"])
    print(f"[MLflow] Metrics logged to experiment 'locus_recall_eval' (run: {run_name}).")


def main():
    parser = argparse.ArgumentParser(description="Locus Recall@K evaluation")
    parser.add_argument("--gateway-url", default="http://localhost:8000",
                        help="Gateway base URL (default: http://localhost:8000)")
    parser.add_argument("--k", default="5,10,25",
                        help="Comma-separated K values (default: 5,10,25)")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log results to MLflow under experiment locus_recall_eval")
    parser.add_argument("--run-name", default=None,
                        help="MLflow run name (e.g. 'baseline' or 'remove_bg')")
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.k.split(",")]

    results = evaluate(args.gateway_url, ks)
    print_report(results)

    if args.mlflow:
        log_mlflow(results, run_name=args.run_name)

    if results["evaluated"] < 25:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
