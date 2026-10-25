"""
Golden Dataset Builder — Google Lens Workflow
==============================================
Builds the golden dataset for MLFlow evaluation by indexing Google Lens results
into Qdrant under store_name="golden_dataset".

Workflow (repeat 20-30 times):
  1. You find a fashion image on Google → copy its URL
  2. You open it in Google Lens → pick 5 visually similar items → copy their URLs
  3. This script indexes those 5 images into Qdrant (store = "golden_dataset")
  4. It saves the mapping {query image → 5 product_ids} to golden_dataset.json

Usage:
  cd mlops
  python build_golden_from_lens.py

  # Use a different gateway:
  GATEWAY_URL=http://localhost:8000 python build_golden_from_lens.py

Requirements:
  pip install requests
"""

import argparse
import json
import os
import uuid
from datetime import datetime

import requests

# ─── Configuration ────────────────────────────────────────────────────────────
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
OUTPUT_FILE = "golden_dataset.json"
STORE_NAME  = "golden_dataset"
MALL_NAME   = "golden_dataset"
TARGET      = 30   # total queries to build (20-30 is sufficient)
# ──────────────────────────────────────────────────────────────────────────────


def compute_product_id(image_url: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"golden::{image_url}"))


def index_images(items: list) -> dict:
    resp = requests.post(
        f"{GATEWAY_URL}/add-bulk-batch",
        json={"items": items},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def load_dataset() -> list:
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return []


def save_dataset(data: list):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  💾  Saved {len(data)} entries → {OUTPUT_FILE}")


def check_gateway():
    try:
        resp = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        data = resp.json()
        if data.get("ready"):
            print(f"  ✅  Gateway ready   ({GATEWAY_URL})")
        else:
            services  = data.get("services", {})
            not_ready = [k for k, v in services.items() if v != "ready"]
            print(f"  ⚠️  Gateway not fully ready — {not_ready} still loading.")
    except Exception as e:
        print(f"  ❌  Cannot reach gateway at {GATEWAY_URL}: {e}")
        raise SystemExit(1)


def prompt_url(label: str) -> str:
    while True:
        val = input(f"  {label}: ").strip()
        if val:
            return val
        print("  ⚠️  URL cannot be empty. Try again.")


def prompt_name(label: str, fallback: str) -> str:
    val = input(f"  {label} [{fallback}]: ").strip()
    return val if val else fallback


def build_set(set_num: int) -> dict | None:
    print()
    print(f"  ┌── Set {set_num}/{TARGET} " + "─" * 45)
    print(f"  │")
    print(f"  │  Step 1: find a fashion item on Google → copy image URL")
    print(f"  │  Step 2: open it in Google Lens → pick 5 similar items → copy their URLs")
    print(f"  │")

    query_url = input("  │  Query image URL (or 'quit'): ").strip()
    if query_url.lower() in ("quit", "q", "exit"):
        return None

    query_name = prompt_name(
        "  │  Query item name (e.g. 'blue denim jacket')",
        f"query_{set_num}"
    )
    query_category = prompt_name(
        "  │  Category (top/pants/jacket/dress/shorts/skirt/sweater/shoes/bag/hat/jumpsuit/leggings/sports_bra)",
        ""
    )

    print(f"  │")
    print(f"  │  Now enter the 5 Google Lens matches:")
    print(f"  │")

    relevant_pairs = []
    for i in range(5):
        url  = prompt_url(f"  │  Relevant {i+1} URL")
        name = prompt_name(
            f"  │  Relevant {i+1} name (e.g. 'red floral midi dress')",
            f"{query_category or 'item'}_{set_num}_{i+1}"
        )
        relevant_pairs.append((name, url))

    print(f"  └" + "─" * 55)

    print(f"\n  🔄  Indexing {len(relevant_pairs)} images into Qdrant…")

    batch_items = [
        {
            "name"      : name,
            "image_url" : url,
            "store"     : STORE_NAME,
            "mall"      : MALL_NAME,
            "price"     : "",
            "product_id": compute_product_id(url),
            "is_golden" : True,
        }
        for name, url in relevant_pairs
    ]

    try:
        result = index_images(batch_items)
    except requests.HTTPError as e:
        print(f"  ❌  Gateway error: {e}")
        return None
    except Exception as e:
        print(f"  ❌  Unexpected error: {e}")
        return None

    ok      = result.get("success", 0)
    skipped = result.get("skipped", 0)
    failed  = result.get("failed", [])

    print(f"  ✅  {ok} indexed  |  {skipped} skipped  |  {len(failed)} failed")

    failed_names  = {f["item"] for f in failed}
    accepted_pairs = [(n, u) for n, u in relevant_pairs if n not in failed_names]

    if len(accepted_pairs) < 3:
        print(f"\n  ⚠️  Only {len(accepted_pairs)} images indexed successfully (need at least 3).")
        retry = input("  Skip this set and continue? (y/n): ").strip().lower()
        if retry == "y":
            return None

    relevant_product_ids = [compute_product_id(url) for _, url in accepted_pairs]
    relevant_info = [
        {
            "product_id": compute_product_id(url),
            "name"      : name,
            "image_url" : url,
            "store_name": STORE_NAME,
        }
        for name, url in accepted_pairs
    ]

    return {
        "query_image_url"     : query_url,
        "query_name"          : query_name,
        "query_category_tag"  : query_category,
        "relevant_product_ids": relevant_product_ids,
        "relevant_info"       : relevant_info,
        "source"              : "google_lens",
        "n_relevant"          : len(accepted_pairs),
        "annotated_by"        : "marc",
        "created_at"          : datetime.utcnow().isoformat() + "Z",
    }


def main():
    print()
    print("═" * 62)
    print("  🏗   Locus Golden Dataset Builder — Google Lens Workflow")
    print("═" * 62)
    print(f"  Gateway    : {GATEWAY_URL}")
    print(f"  Output     : {OUTPUT_FILE}")
    print(f"  Target     : {TARGET} queries")
    print(f"  Store name : {STORE_NAME}")
    print()

    check_gateway()
    print()

    dataset = load_dataset()
    print(f"  Resuming from {len(dataset)} existing entries.\n")

    while len(dataset) < TARGET:
        set_num = len(dataset) + 1
        entry   = build_set(set_num)

        if entry is None:
            if input("\n  Continue to next set? (y/n): ").strip().lower() != "y":
                break
            continue

        dataset.append(entry)
        save_dataset(dataset)
        print(f"\n  Progress: {len(dataset)}/{TARGET}")

        if len(dataset) >= TARGET:
            print(f"\n  🎉  Golden dataset complete! ({TARGET}/{TARGET})")
            break

    print()
    print("═" * 62)
    n          = len(dataset)
    categories = {}
    for e in dataset:
        cat = e.get("query_category_tag", "unknown") or "unknown"
        categories[cat] = categories.get(cat, 0) + 1

    print(f"  ✅  Golden dataset: {n} queries saved to {OUTPUT_FILE}")
    print()
    print("  Category breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {cat:<15} {count:2d}  {bar}")
    print()
    print("  Next step:")
    print("    python evaluate_mlflow.py")
    print("═" * 62)
    print()


def backfill_is_golden():
    """One-off migration: set is_golden=True on all existing golden_dataset points."""
    from qdrant_client import QdrantClient
    from dotenv import load_dotenv
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=30)
    else:
        client = QdrantClient(host="localhost", port=6333, timeout=30)

    from qdrant_client.http import models

    updated = 0
    cursor  = None
    print("Backfilling is_golden=True on all golden_dataset points…")
    while True:
        batch, next_cursor = client.scroll(
            collection_name="locus_items",
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="store_name",
                match=models.MatchValue(value="golden_dataset"),
            )]),
            limit=250,
            offset=cursor,
            with_payload=False,
            with_vectors=False,
        )
        if not batch:
            break
        ids = [pt.id for pt in batch]
        client.set_payload(
            collection_name="locus_items",
            payload={"is_golden": True},
            points=ids,
        )
        updated += len(ids)
        print(f"  Updated {updated} points…")
        if next_cursor is None:
            break
        cursor = next_cursor

    print(f"Done. Set is_golden=True on {updated} points.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true",
                        help="Backfill is_golden=True on all existing golden_dataset points")
    args = parser.parse_args()

    if args.backfill:
        backfill_is_golden()
    else:
        main()