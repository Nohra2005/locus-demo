"""
add_hat_queries.py — expand hat coverage in golden_dataset.json.

Hat queries are currently only 2 (wide-brim straw + black cap), which is too
few for reliable per-category scoring. This script:

  1. Scrolls Qdrant for all hat products
  2. Picks candidates for three new query types:
       beanie / knit hat, fedora / beret, patterned or coloured cap
  3. Downloads the query image to mlops/golden_images/
  4. Finds the 5 nearest Qdrant neighbours by vector for auto-annotation
  5. Prints the JSON entries to paste into golden_dataset.json

Review the printed images before committing: run with --commit to write
directly to golden_dataset.json after you are satisfied.

Usage:
    # Set env vars first (Windows Git Bash):
    export $(cat .env | tr -d '\\r' | grep -v '#' | xargs)

    # Preview what would be added:
    python mlops/add_hat_queries.py

    # Write to golden_dataset.json:
    python mlops/add_hat_queries.py --commit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

SCRIPT_DIR          = Path(__file__).parent
GOLDEN_DATASET_PATH = SCRIPT_DIR / "golden_dataset.json"
GOLDEN_IMAGES_DIR   = SCRIPT_DIR / "golden_images"
COLLECTION          = "locus_items"
GATEWAY_URL         = "http://localhost:8000"

# Keywords that map a hat product name to a target query type.
# The first type whose keywords match wins.
_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("beanie",  ["beanie", "knit hat", "knit cap", "bobble", "pom pom", "slouch"]),
    ("fedora",  ["fedora", "beret", "trilby", "panama", "boater", "cloche"]),
    ("cap",     ["cap", "baseball", "trucker", "dad cap", "snapback", "bucket hat",
                 "bucket", "fisherman"]),
]

# Query types we still need to add (not already covered by existing golden entries)
_ALREADY_COVERED = {"wide brim straw", "black cap"}


def _classify_hat(name: str) -> str | None:
    lower = name.lower()
    for type_name, keywords in _TYPE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return type_name
    return None


def _download_image(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  [WARN] download failed for {url}: {e}")
        return False


def _find_neighbors(client: QdrantClient, vector: list[float], exclude_id: str,
                    limit: int = 5) -> list[dict]:
    response = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=models.Filter(must=[
            models.FieldCondition(
                key="category_tag",
                match=models.MatchValue(value="hat"),
            )
        ]),
        limit=limit + 1,  # +1 in case the query item itself appears
        with_payload=True,
    )
    results = response.points
    neighbours = []
    for r in results:
        if str(r.id) == exclude_id:
            continue
        payload = r.payload or {}
        img_url = payload.get("image_url", "")
        name    = payload.get("name", str(r.id))
        neighbours.append({
            "product_id": str(r.id),
            "name":       name,
            "image_url":  img_url,
            "store_name": payload.get("store_name", ""),
        })
        if len(neighbours) == limit:
            break
    return neighbours


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true",
                        help="Write new entries to golden_dataset.json (default: dry-run preview)")
    parser.add_argument("--gateway-url", default=GATEWAY_URL)
    args = parser.parse_args()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        print("[ERROR] QDRANT_URL env var not set")
        sys.exit(1)

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=60)

    # Scroll all hats
    print("Scrolling Qdrant for hat products…")
    all_hats: list[dict] = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(
                    key="category_tag",
                    match=models.MatchValue(value="hat"),
                )
            ]),
            with_vectors=True,
            with_payload=True,
            limit=200,
            offset=offset,
        )
        if not results:
            break
        for p in results:
            payload = p.payload or {}
            vec     = p.vector
            if isinstance(vec, dict):
                vec = vec.get("", vec.get("default", None))
            all_hats.append({
                "id":        str(p.id),
                "name":      payload.get("name", ""),
                "image_url": payload.get("image_url", ""),
                "vector":    vec,
            })
        if next_offset is None:
            break
        offset = next_offset

    print(f"Found {len(all_hats)} hat products in Qdrant\n")

    # Classify each hat and group by type
    by_type: dict[str, list[dict]] = {}
    for hat in all_hats:
        t = _classify_hat(hat["name"])
        if t:
            by_type.setdefault(t, []).append(hat)

    for t, items in by_type.items():
        print(f"  {t}: {len(items)} products")
    print()

    # Load existing golden dataset to avoid duplicating query images
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    existing_query_urls = {e["query_image_url"] for e in dataset}
    existing_relevant_ids = {
        pid
        for e in dataset
        for pid in e.get("relevant_product_ids", [])
    }

    GOLDEN_IMAGES_DIR.mkdir(exist_ok=True)

    new_entries: list[dict] = []

    for type_name in ["beanie", "fedora", "cap"]:
        candidates = by_type.get(type_name, [])
        if not candidates:
            print(f"[SKIP] no '{type_name}' products found in Qdrant")
            continue

        # Prefer products not already in relevant_ids (truly new queries)
        fresh = [c for c in candidates if c["id"] not in existing_relevant_ids]
        pool  = fresh if fresh else candidates

        # Pick the first candidate with a downloadable image
        chosen = None
        for candidate in pool:
            img_url = candidate["image_url"]
            if not img_url:
                continue
            # Derive a filename from the URL tail (matches gateway /golden-dataset/images/ pattern)
            fname = img_url.rsplit("/", 1)[-1]
            if not fname.endswith((".jpg", ".jpeg", ".png", ".webp")):
                fname = candidate["id"].replace("-", "")[:20] + ".jpeg"
            dest = GOLDEN_IMAGES_DIR / fname
            if dest.exists() or _download_image(img_url, dest):
                chosen = {"hat": candidate, "img_path": dest, "fname": fname}
                break

        if not chosen:
            print(f"[SKIP] could not download any image for type '{type_name}'")
            continue

        hat      = chosen["hat"]
        img_path = chosen["img_path"]
        fname    = chosen["fname"]

        # Resolve the query image URL (must be reachable via gateway /golden-dataset/images/)
        query_img_url = f"{args.gateway_url}/golden-dataset/images/{fname}"

        # Find 5 nearest hat neighbours for auto-annotation
        if hat["vector"] is None:
            print(f"[SKIP] '{hat['name']}' has no stored vector")
            continue

        neighbours = _find_neighbors(client, hat["vector"], hat["id"], limit=5)
        if not neighbours:
            print(f"[SKIP] no neighbours found for '{hat['name']}'")
            continue

        # Build the golden dataset entry
        relevant_ids  = [hat["id"]] + [n["product_id"] for n in neighbours[:4]]
        relevant_info = [
            {
                "product_id": hat["id"],
                "name":       hat["name"],
                "image_url":  hat["image_url"],
                "store_name": "qdrant",
            }
        ] + [
            {
                "product_id": n["product_id"],
                "name":       n["name"],
                "image_url":  n["image_url"],
                "store_name": n["store_name"],
            }
            for n in neighbours[:4]
        ]

        entry = {
            "query_image_url":      query_img_url,
            "query_name":           hat["name"],
            "query_category_tag":   "hat",
            "relevant_product_ids": relevant_ids,
            "relevant_info":        relevant_info,
            "source":               "add_hat_queries",
            "n_relevant":           len(relevant_ids),
            "annotated_by":         "vector_neighbors",
            "created_at":           datetime.now(timezone.utc).isoformat(),
        }
        new_entries.append(entry)

        print(f"[{type_name.upper()}] '{hat['name']}'")
        print(f"  query image : {img_path.name}")
        print(f"  neighbours  : {[n['name'][:40] for n in neighbours[:4]]}")

    if not new_entries:
        print("\nNo new hat entries generated. Check Qdrant hat products.")
        sys.exit(0)

    print(f"\n{'=' * 60}")
    print("New entries to add to golden_dataset.json:")
    print("=" * 60)
    print(json.dumps(new_entries, indent=2))

    if args.commit:
        dataset.extend(new_entries)
        GOLDEN_DATASET_PATH.write_text(
            json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n[OK] Added {len(new_entries)} hat entries to {GOLDEN_DATASET_PATH.name}")
        print(f"     Total queries now: {len(dataset)}")
        hat_count = sum(1 for e in dataset if e.get("query_category_tag") == "hat")
        print(f"     Hat queries: {hat_count}")
    else:
        print(f"\nDry run — review the entries above, then re-run with --commit to apply.")
        print(f"Also verify the downloaded images in: {GOLDEN_IMAGES_DIR}")


if __name__ == "__main__":
    main()
