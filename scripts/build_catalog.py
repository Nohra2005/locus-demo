"""
Rebuild the Locus catalog from the stores' own websites.

Runs against a local Locus stack (the demo container started with DEMO_MODE=0):
  1. For each store in data/stores.json, calls the gateway's /scrape endpoint
     (Shopify products.json when available, HTML fallback otherwise).
  2. Sends the products to /add-bulk-batch, which downloads each photo,
     detects the garment, embeds it with Fashion-CLIP and stores it in Qdrant.
  3. Re-indexes the golden-dataset products (is_golden=True) so the
     evaluation scripts in mlops/ still find their relevant items.

Product IDs are uuid5(name::store), the same rule the gateway uses, so a
product keeps its ID across rebuilds as long as its name and store match.

Then run scripts/export_catalog.py to write data/catalog.jsonl.gz.

Usage:
    GATEWAY_URL=http://localhost:7860 python scripts/build_catalog.py [--max-per-store N]
"""
import argparse
import json
import os
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
GATEWAY = os.getenv("GATEWAY_URL", "http://localhost:7860").rstrip("/")
BATCH = 25


def post(path, payload, timeout=600):
    for attempt in range(3):
        try:
            r = requests.post(f"{GATEWAY}{path}", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  {path} failed ({e}), retry {attempt + 1}/3")
            time.sleep(5 * (attempt + 1))
    return None


def index(items, label):
    ok = skipped = failed = 0
    for i in range(0, len(items), BATCH):
        res = post("/add-bulk-batch", {"items": items[i:i + BATCH]})
        if res:
            ok += res.get("success", 0)
            skipped += res.get("skipped", 0)
            failed += len(res.get("failed", []))
        else:
            failed += len(items[i:i + BATCH])
        print(f"  {label}: {min(i + BATCH, len(items))}/{len(items)} sent "
              f"(indexed {ok}, skipped {skipped}, failed {failed})", flush=True)
    return ok, skipped, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-store", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    stores = json.loads((ROOT / "data" / "stores.json").read_text())["stores"]
    summary = []

    for s in stores:
        print(f"\n== {s['store']} ({s['url']})")
        res = post("/scrape", {"url": s["url"], "max_products": args.max_per_store})
        products = (res or {}).get("products", [])
        if args.max_per_store:
            products = products[: args.max_per_store]
        print(f"  scraped {len(products)} products (strategy: {(res or {}).get('strategy')})")
        items = [{
            "name": p.get("name", ""),
            "image_urls": p.get("image_urls") or [p.get("image_url", "")],
            "price": p.get("price", ""),
            "store": s["store"],
            "mall": s.get("mall", ""),
        } for p in products if p.get("name")]
        summary.append((s["store"], len(products), *index(items, s["store"])))

    golden = json.loads((ROOT / "mlops" / "golden_dataset.json").read_text())
    seen, items = set(), []
    for q in golden:
        for r in q.get("relevant_info", []):
            url = r.get("image_url", "")
            if r["product_id"] in seen or not url:
                continue
            seen.add(r["product_id"])
            store = r.get("store_name", "golden_dataset")
            if "/golden-dataset/images/" in url:
                # photos stored in the repo, served by the gateway
                url = f"{GATEWAY}/golden-dataset/images/{url.rsplit('/', 1)[-1]}"
            # Store products (e.g. Forever21) are real catalog items: re-adding them
            # here only pins the product_id the golden set expects. The point ID is
            # uuid5(image_url), so this overwrites the scraped copy, no duplicate.
            items.append({
                "name": r["name"], "store": store, "image_urls": [url],
                "product_id": r["product_id"], "is_golden": store == "golden_dataset",
            })
    print(f"\n== golden dataset ({len(items)} products)")
    summary.append(("golden dataset", len(items), *index(items, "golden")))

    print("\nstore | scraped | indexed | skipped | failed")
    for row in summary:
        print(" | ".join(str(x) for x in row))
    if sum(r[2] for r in summary) == 0:
        sys.exit("Nothing was indexed.")


if __name__ == "__main__":
    main()
