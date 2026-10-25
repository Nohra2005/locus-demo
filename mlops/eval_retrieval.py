"""
eval_retrieval.py — Comprehensive ranking evaluation for Locus.

Reports Recall@K, Precision@K, MRR, NDCG@K, rank distribution, and per-category
breakdowns. Uses actual relevant_product_ids from golden_dataset — no LLM/API cost.

Usage:
    python mlops/eval_retrieval.py                          # default gateway, K=1,3,5,10,15
    python mlops/eval_retrieval.py --gateway-url http://...
    python mlops/eval_retrieval.py --k 1,3,5,10 --ndcg-k 5
    python mlops/eval_retrieval.py --json results.json      # also save raw JSON
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import requests

SCRIPT_DIR          = Path(__file__).parent
GOLDEN_DATASET_PATH = SCRIPT_DIR / "golden_dataset.json"
GOLDEN_IMAGES_DIR   = SCRIPT_DIR / "golden_images"
_GOLDEN_URL_PREFIX  = "/golden-dataset/images/"


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _load_image(entry: dict) -> bytes | None:
    url  = entry.get("query_image_url", "")
    fname = url.rsplit("/", 1)[-1]
    path  = GOLDEN_IMAGES_DIR / fname
    return path.read_bytes() if path.exists() else None


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


def _search(image_bytes: bytes, gateway_url: str, category: str, shoe_style: str) -> list[str]:
    files = {"file": ("q.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    data  = {"search_label": category, "skip_judge": "true"}
    if shoe_style and shoe_style != "other":
        data["shoe_style"] = shoe_style
    try:
        r = requests.post(
            f"{gateway_url}/search",
            files=files,
            data=data,
            params={"include_golden": "true"},
            timeout=60,
        )
        r.raise_for_status()
        return [m["product_id"] for m in r.json().get("matches", []) if m.get("product_id")]
    except Exception as e:
        print(f"   [FAIL] {e}")
        return []


# ── Ranking metrics ───────────────────────────────────────────────────────────

def _recall_at_k(result_ids: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(result_ids[:k])) / len(relevant) if relevant else 0.0


def _precision_at_k(result_ids: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(result_ids[:k])) / k if k else 0.0


def _mrr(result_ids: list[str], relevant: set[str]) -> float:
    for i, pid in enumerate(result_ids):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _ndcg_at_k(result_ids: list[str], relevant: set[str], k: int) -> float:
    rels  = [1.0 if pid in relevant else 0.0 for pid in result_ids[:k]]
    ideal = sorted(rels, reverse=True)
    idcg  = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0


def _first_hit(result_ids: list[str], relevant: set[str]) -> int | None:
    for i, pid in enumerate(result_ids):
        if pid in relevant:
            return i + 1
    return None


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run(gateway_url: str, ks: list[int], ndcg_k: int, delay: float = 0.0) -> dict:
    dataset  = json.loads(GOLDEN_DATASET_PATH.read_text())
    per_query: list[dict] = []
    cat_data: dict[str, list[dict]] = {}
    skipped = 0

    print(f"[EVAL] Gateway: {gateway_url}")
    print(f"[EVAL] Dataset: {len(dataset)} entries\n")

    for i, entry in enumerate(dataset):
        name     = entry.get("query_name", "?")
        cat      = entry.get("query_category_tag", "unknown")
        rel_ids  = set(entry.get("relevant_product_ids", []))

        if not rel_ids:
            skipped += 1
            continue

        image_bytes = _load_image(entry)
        if image_bytes is None:
            print(f"  [{i+1:02d}] SKIP (no image): {name}")
            skipped += 1
            continue

        shoe_style = _shoe_style_from_name(name) if cat == "shoes" else ""
        print(f"  [{i+1:02d}] {cat:12s} | {name[:48]}", end="", flush=True)

        result_ids = _search(image_bytes, gateway_url, cat, shoe_style)
        if delay > 0:
            time.sleep(delay)

        first_rank = _first_hit(result_ids, rel_ids)
        mrr_score  = _mrr(result_ids, rel_ids)
        ndcg_score = _ndcg_at_k(result_ids, rel_ids, ndcg_k)
        recalls    = {k: _recall_at_k(result_ids, rel_ids, k) for k in ks}
        precisions = {k: _precision_at_k(result_ids, rel_ids, k) for k in ks}

        rank_str = f"rank {first_rank}" if first_rank else "MISS"
        print(f"  R@5={recalls.get(5, 0):.2f}  MRR={mrr_score:.3f}  {rank_str}")

        row = {
            "name":       name,
            "category":   cat,
            "n_relevant": len(rel_ids),
            "recall":     recalls,
            "precision":  precisions,
            "mrr":        mrr_score,
            "ndcg":       ndcg_score,
            "first_rank": first_rank,
            "n_results":  len(result_ids),
        }
        per_query.append(row)
        cat_data.setdefault(cat, []).append(row)

    return {
        "gateway_url": gateway_url,
        "n_dataset":   len(dataset),
        "evaluated":   len(per_query),
        "skipped":     skipped,
        "ks":          ks,
        "ndcg_k":      ndcg_k,
        "per_query":   per_query,
        "cat_data":    cat_data,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _avg(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def print_report(data: dict) -> None:
    ks      = data["ks"]
    ndcg_k  = data["ndcg_k"]
    results = data["per_query"]
    cats    = data["cat_data"]
    n_eval  = data["evaluated"]

    W = 72
    print("\n" + "=" * W)
    print(f"  LOCUS RETRIEVAL REPORT  |  {n_eval} queries evaluated  ({data['skipped']} skipped)")
    print("=" * W)

    # ── Overall table ─────────────────────────────────────────────────────────
    print(f"\n{'METRIC':<16}", end="")
    for k in ks:
        print(f"   @{k:<3}", end="")
    print()
    print("-" * (16 + 7 * len(ks)))

    for metric, label in [("recall", "Recall"), ("precision", "Precision")]:
        print(f"{label:<16}", end="")
        for k in ks:
            v = _avg([r[metric].get(k, 0.0) for r in results])
            print(f"  {v:>5.3f}", end="")
        print()

    mrr_avg  = _avg([r["mrr"]  for r in results])
    ndcg_avg = _avg([r["ndcg"] for r in results])
    print(f"\n{'MRR':<16}  {mrr_avg:.4f}   (higher is better; perfect = 1.000)")
    print(f"{'NDCG@'+str(ndcg_k):<16}  {ndcg_avg:.4f}   (higher is better; perfect = 1.000)")

    hits_at5  = sum(1 for r in results if r["recall"].get(5, 0) > 0)
    success_r = hits_at5 / n_eval if n_eval else 0.0
    print(f"\nSuccess rate (>=1 hit in top 5):  {hits_at5}/{n_eval}  ({success_r:.1%})")

    # ── Rank distribution ─────────────────────────────────────────────────────
    ranks  = [r["first_rank"] for r in results if r["first_rank"] is not None]
    misses = sum(1 for r in results if r["first_rank"] is None)
    print(f"\nFirst relevant item rank distribution  ({len(ranks)} found, {misses} not found):")
    buckets = [(1, 1, "#"), (2, 2, "#"), (3, 3, "#"), (4, 5, " "), (6, 10, " "), (11, 15, " "), (16, 9999, "!")]
    for lo, hi, marker in buckets:
        count  = sum(1 for r in ranks if lo <= r <= hi)
        bar    = marker * count
        label  = f"rank {lo}" if lo == hi else f"rank {lo}-{hi}" if hi < 9999 else f"rank {lo}+"
        pct    = count / n_eval if n_eval else 0.0
        print(f"  {label:<14}  {count:>2}  ({pct:>5.1%})  {bar}")

    # ── Per-category breakdown ────────────────────────────────────────────────
    show_ks = ks[:min(3, len(ks))]  # show up to @1, @3, @5 in category table
    header  = f"\n{'CATEGORY':<14}"
    for k in show_ks:
        header += f"  R@{k:<2}"
    header += f"  {'MRR':>5}  NDCG@{ndcg_k}  {'1st<=5':>6}  N"
    print(header)
    print("-" * (14 + 6 * len(show_ks) + 5 + 8 + 6 + 4))

    for cat in sorted(cats.keys()):
        rows    = cats[cat]
        line    = f"{cat:<14}"
        for k in show_ks:
            line += f"  {_avg([r['recall'].get(k, 0.0) for r in rows]):>4.3f}"
        mrr_c   = _avg([r["mrr"]  for r in rows])
        ndcg_c  = _avg([r["ndcg"] for r in rows])
        hit5_c  = sum(1 for r in rows if (r["first_rank"] or 999) <= 5)
        line   += f"  {mrr_c:>5.3f}  {ndcg_c:.4f}  {hit5_c:>2}/{len(rows):>2}  {len(rows):>2}"
        print(line)

    # ── Worst queries ─────────────────────────────────────────────────────────
    worst = sorted(results, key=lambda r: (r["mrr"], r["recall"].get(5, 0)))[:5]
    print(f"\nBottom 5 queries by MRR:")
    for r in worst:
        rank_str = f"rank {r['first_rank']}" if r["first_rank"] else "NOT FOUND"
        print(f"  MRR={r['mrr']:.3f}  [{r['category']:10s}]  {r['name'][:48]}  ({rank_str})")

    # ── Best queries ──────────────────────────────────────────────────────────
    best = sorted(results, key=lambda r: (-r["mrr"], -r["recall"].get(5, 0)))[:3]
    print(f"\nTop 3 queries by MRR:")
    for r in best:
        rank_str = f"rank {r['first_rank']}" if r["first_rank"] else "NOT FOUND"
        print(f"  MRR={r['mrr']:.3f}  [{r['category']:10s}]  {r['name'][:48]}  ({rank_str})")

    print("=" * W)
    print(f"\nKey numbers: Recall@5={_avg([r['recall'].get(5,0) for r in results]):.4f}  "
          f"MRR={mrr_avg:.4f}  NDCG@{ndcg_k}={ndcg_avg:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Locus comprehensive ranking evaluation")
    parser.add_argument("--gateway-url", default="http://localhost:8000")
    parser.add_argument("--k",           default="1,3,5,10,15",
                        help="Comma-separated K values (default: 1,3,5,10,15)")
    parser.add_argument("--ndcg-k",      type=int, default=5,
                        help="K for NDCG computation (default: 5)")
    parser.add_argument("--json",        metavar="FILE", default=None,
                        help="Also write raw results as JSON to this file")
    parser.add_argument("--delay",       type=float, default=0.0,
                        help="Seconds to sleep between queries (use 7 to stay under 10/min rate limit)")
    args = parser.parse_args()

    ks   = [int(x.strip()) for x in args.k.split(",")]
    data = run(args.gateway_url, ks, args.ndcg_k, delay=args.delay)

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"\n[EVAL] Raw results saved to {out}")

    print_report(data)

    evaluated = data["evaluated"]
    if evaluated == 0:
        print("[FAIL] No queries evaluated")
        sys.exit(1)

    recall5 = _avg([r["recall"].get(5, 0.0) for r in data["per_query"]])
    mrr     = _avg([r["mrr"] for r in data["per_query"]])
    print(f"\n[EVAL] Done. Recall@5={recall5:.4f}  MRR={mrr:.4f}  ({evaluated} queries)")
    sys.exit(0)


if __name__ == "__main__":
    main()
