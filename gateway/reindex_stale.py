"""
reindex_stale.py  —  Phase 5.1

Two-pass re-indexer:

Pass 1 — locus_items: re-indexes products with box_source = "full_image" or
         "unknown" (indexed before the Phase 2 crop fix).
         Also DELETES products that are now classified as not_fashion
         (socks, underwear, swimwear etc. that should never have been indexed).

Pass 2 — locus_skipped: retries products that previously got no_consensus.
         These never made it into locus_items so Pass 1 misses them entirely.
         After a whitelist update they may now classify correctly and can be
         promoted to locus_items.

Usage:
    docker compose exec gateway python reindex_stale.py

Environment variables:
    QDRANT_URL      — Qdrant Cloud URL  (optional, falls back to local)
    QDRANT_API_KEY  — Qdrant Cloud key  (optional)
    QDRANT_HOST     — local host        (default: localhost)
    QDRANT_PORT     — local port        (default: 6333)
    VISUAL_URL      — visual engine URL (default: http://visual_engine:8001)
    DRY_RUN         — set to 1 to print what would happen without writing
    CONCURRENCY     — parallel workers  (default: 3)
"""

import asyncio
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL         = os.getenv("QDRANT_URL")
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY")
QDRANT_HOST        = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT        = int(os.getenv("QDRANT_PORT", 6333))
VISUAL_URL         = os.getenv("VISUAL_URL", "http://visual_engine:8001")
COLLECTION_NAME    = "locus_items"
SKIPPED_COLLECTION = "locus_skipped"
DRY_RUN            = os.getenv("DRY_RUN", "0") == "1"
CONCURRENCY        = int(os.getenv("CONCURRENCY", "3"))

STALE_SOURCES = {"full_image", "unknown", None, ""}


def _is_stale(box_source) -> bool:
    """A point is stale if it has no crop (full_image/unknown) or was indexed
    with a best_available fallback — which may have picked a wrong-family box
    (e.g. shoe bbox for a dress) before the family-aware Tier 3 fix."""
    return box_source in STALE_SOURCES or (
        isinstance(box_source, str) and box_source.endswith("_best_available")
    )

if QDRANT_URL:
    print(f"[QDRANT] Cloud: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print(f"[QDRANT] Local: {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

if DRY_RUN:
    print("[DRY RUN] No writes will be made.\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_skipped_collection():
    if not client.collection_exists(SKIPPED_COLLECTION):
        client.create_collection(
            collection_name=SKIPPED_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
        for field in ("store_name", "skip_reason", "product_id"):
            try:
                client.create_payload_index(
                    collection_name=SKIPPED_COLLECTION,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                pass
        print(f"[QDRANT] Created collection '{SKIPPED_COLLECTION}'")


def _store_skipped(product: dict, reason: str):
    """
    Upserts a skipped product into locus_skipped for dashboard review.
    Only stores no_consensus and no_box_found — not_fashion is deleted, not stored.
    """
    if DRY_RUN:
        return
    try:
        img_url    = product.get("image_url", "")
        product_id = product.get("product_id", "")
        key        = img_url or product_id or product.get("name", "unknown")

        client.upsert(
            collection_name=SKIPPED_COLLECTION,
            points=[
                PointStruct(
                    id     = str(uuid.uuid5(uuid.NAMESPACE_URL, f"skipped::{key}")),
                    vector = [0.0],
                    payload = {
                        "product_id":  product_id,
                        "name":        product.get("name", ""),
                        "image_url":   img_url,
                        "store_name":  product.get("store_name", ""),
                        "mall_name":   product.get("mall_name", ""),
                        "price":       product.get("price", ""),
                        "skip_reason": reason,
                        "skipped_at":  datetime.utcnow().isoformat(),
                    }
                )
            ]
        )
    except Exception as e:
        print(f"  [WARN]  Could not store skipped '{product.get('name','')}': {e}")


def _delete_from_collection(collection: str, point_id: str, name: str = ""):
    """Delete a single point from a collection by its point ID."""
    if DRY_RUN or not point_id:
        return
    try:
        client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[point_id]),
        )
    except Exception as e:
        print(f"  [WARN]  Could not delete '{name}' from {collection}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — locus_items (stale box_source)
# ══════════════════════════════════════════════════════════════════════════════

def collect_stale_products() -> list[dict]:
    stale      = {}
    total_seen = 0
    offset     = None

    print("Pass 1 — scanning locus_items for stale vectors...")

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            break

        for pt in results:
            total_seen += 1
            p = pt.payload

            if p.get("is_dark") is True:
                continue

            product_id = p.get("product_id", str(pt.id))
            box_source = p.get("box_source", None)

            if not _is_stale(box_source):
                continue

            if product_id not in stale:
                stale[product_id] = {
                    "point_id":     str(pt.id),
                    "product_id":   product_id,
                    "name":         p.get("name", ""),
                    "store_name":   p.get("store_name", ""),
                    "mall_name":    p.get("mall_name", ""),
                    "image_url":    p.get("image_url", ""),
                    "category_tag": p.get("category_tag", ""),
                    "price":        p.get("price", ""),
                    "box_source":   box_source,
                }

        print(f"  ... scanned {total_seen} points, {len(stale)} stale so far", end="\r")

        if next_offset is None:
            break
        offset = next_offset

    print(f"\n  Scan complete. Total points: {total_seen}, stale products: {len(stale)}")
    return list(stale.values())


async def reindex_one(
    semaphore: asyncio.Semaphore,
    product: dict,
    http: httpx.AsyncClient,
) -> dict:
    async with semaphore:
        name     = product["name"]
        img_url  = product["image_url"]
        old_src  = product["box_source"]
        point_id = product["point_id"]

        if not img_url:
            return {"status": "skipped_no_url", "item": name, "product_id": product["product_id"]}

        try:
            img_resp = await http.get(img_url, timeout=15.0, follow_redirects=True)
            img_resp.raise_for_status()
            img_bytes    = img_resp.content
            content_type = img_resp.headers.get("content-type", "image/jpeg")

            idx_resp = await http.post(
                f"{VISUAL_URL}/index-image",
                files={"file": ("product.jpg", img_bytes, content_type)},
                data={"title": name},
                timeout=90.0,
            )
            idx_resp.raise_for_status()
            idx_data = idx_resp.json()

            if idx_data.get("skipped"):
                reason = idx_data.get("skip_reason", "unknown")

                if reason == "no_box_found":
                    # Keep old vector — stale beats nothing.
                    # Do NOT call _store_skipped: item is already in locus_items
                    # and adding it to locus_skipped causes Pass 2 to double-process it.
                    print(f"  [KEEP]  '{name}' — no box found, keeping old vector")
                    return {
                        "status":     "kept_no_box",
                        "item":       name,
                        "product_id": product["product_id"],
                        "reason":     reason,
                    }

                if reason == "no_consensus":
                    # Store for whitelist review — a keyword fix may rescue it.
                    _store_skipped(product, reason)
                    print(f"  [SKIP]  '{name}' — {reason}")
                    return {
                        "status":     "skipped",
                        "item":       name,
                        "product_id": product["product_id"],
                        "reason":     reason,
                    }

                if reason == "not_fashion":
                    # DELETE from locus_items — socks, underwear, swimwear etc.
                    # should never have been indexed in the first place.
                    if DRY_RUN:
                        print(f"  [DEL?]  '{name}' — not_fashion (dry run, would delete)")
                    else:
                        _delete_from_collection(COLLECTION_NAME, point_id, name)
                        print(f"  [DEL]   '{name}' — not_fashion, deleted from locus_items")
                    return {
                        "status":     "deleted",
                        "item":       name,
                        "product_id": product["product_id"],
                        "reason":     reason,
                    }

                # Any other skip reason — leave as-is
                print(f"  [SKIP]  '{name}' — {reason}")
                return {
                    "status":     "skipped",
                    "item":       name,
                    "product_id": product["product_id"],
                    "reason":     reason,
                }

            # Successfully re-indexed — upsert with clean box_source
            vector_normal  = idx_data["vector_normal"]
            new_category   = idx_data.get("category", product["category_tag"])
            new_box_source = idx_data.get("box_source", "unknown")

            if not DRY_RUN:
                payload = {
                    "name":         name,
                    "store_name":   product["store_name"],
                    "mall_name":    product["mall_name"],
                    "image_url":    img_url,
                    "category_tag": new_category,
                    "price":        product["price"],
                    "product_id":   product["product_id"],
                    "box_source":   new_box_source,
                }
                if idx_data.get("shoe_style"):
                    payload["shoe_style"] = idx_data["shoe_style"]
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(
                        id      = point_id,
                        vector  = vector_normal,
                        payload = payload,
                    )]
                )

            print(f"  [OK]    '{name}'  {old_src} -> {new_box_source}  cat={new_category}")
            return {
                "status":         "ok",
                "item":           name,
                "product_id":     product["product_id"],
                "old_box_source": old_src,
                "new_box_source": new_box_source,
            }

        except Exception as e:
            print(f"  [ERR]   '{name}' — {e}")
            return {
                "status":     "failed",
                "item":       name,
                "product_id": product["product_id"],
                "error":      str(e),
            }


# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — locus_skipped (no_consensus retry)
# ══════════════════════════════════════════════════════════════════════════════

async def reindex_from_skipped(http: httpx.AsyncClient):
    """
    Retry products in locus_skipped with skip_reason = no_consensus OR no_box_found.
    - no_consensus: a whitelist update may now resolve the category.
    - no_box_found: the full-image fallback for accessories may now index them.
    Successfully indexed products are removed from locus_skipped.
    """
    print("\nPass 2 — scanning locus_skipped for no_consensus / no_box_found products...")

    all_skipped = []
    offset      = None

    while True:
        results, next_offset = client.scroll(
            collection_name=SKIPPED_COLLECTION,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            break
        for pt in results:
            p = pt.payload
            if p.get("skip_reason") in ("no_consensus", "no_box_found"):
                all_skipped.append({"point_id": str(pt.id), **p})
        if next_offset is None:
            break
        offset = next_offset

    if not all_skipped:
        print("  No retryable products in locus_skipped — nothing to retry.")
        return

    print(f"  Found {len(all_skipped)} products to retry.\n")

    semaphore  = asyncio.Semaphore(CONCURRENCY)
    ok_count   = 0
    fail_count = 0
    still_skip = 0

    async def retry_one(product: dict):
        nonlocal ok_count, fail_count, still_skip
        async with semaphore:
            name     = product.get("name", "")
            img_url  = product.get("image_url", "")
            point_id = product.get("point_id", "")

            if not img_url:
                return

            try:
                img_resp = await http.get(img_url, timeout=15.0, follow_redirects=True)
                img_resp.raise_for_status()
                img_bytes    = img_resp.content
                content_type = img_resp.headers.get("content-type", "image/jpeg")

                idx_resp = await http.post(
                    f"{VISUAL_URL}/index-image",
                    files={"file": ("product.jpg", img_bytes, content_type)},
                    data={"title": name},
                    timeout=90.0,
                )
                idx_resp.raise_for_status()
                idx_data = idx_resp.json()

                if idx_data.get("skipped"):
                    reason = idx_data.get("skip_reason", "unknown")
                    print(f"  [STILL SKIP] '{name}' — {reason}")
                    still_skip += 1
                    # Update reason if it changed (e.g. no_consensus → no_box_found)
                    if reason != "no_consensus":
                        _store_skipped(product, reason)
                    return

                vector_normal  = idx_data["vector_normal"]
                new_category   = idx_data.get("category", "")
                new_box_source = idx_data.get("box_source", "unknown")

                if not DRY_RUN:
                    payload = {
                        "name":         name,
                        "store_name":   product.get("store_name", ""),
                        "mall_name":    product.get("mall_name", ""),
                        "image_url":    img_url,
                        "category_tag": new_category,
                        "price":        product.get("price", ""),
                        "product_id":   product.get("product_id", ""),
                        "box_source":   new_box_source,
                    }
                    if idx_data.get("shoe_style"):
                        payload["shoe_style"] = idx_data["shoe_style"]
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[PointStruct(
                            id      = str(uuid.uuid5(uuid.NAMESPACE_URL, img_url)),
                            vector  = vector_normal,
                            payload = payload,
                        )]
                    )
                    _delete_from_collection(SKIPPED_COLLECTION, point_id, name)

                print(f"  [OK]    '{name}' → {new_category} ({new_box_source})")
                ok_count += 1

            except Exception as e:
                print(f"  [ERR]   '{name}' — {e}")
                fail_count += 1

    await asyncio.gather(*[retry_one(p) for p in all_skipped])

    print(f"\nPass 2 complete:")
    print(f"  Promoted to locus_items : {ok_count}")
    print(f"  Still skipping          : {still_skip}")
    print(f"  Failed                  : {fail_count}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("LOCUS Re-index — Phase 5.1 (two-pass)")
    print("=" * 60)

    _ensure_skipped_collection()

    # ── Pass 1: stale vectors in locus_items ──────────────────────────────────
    stale = collect_stale_products()
    print(f"\nFound {len(stale)} stale products to re-index.\n")

    if stale:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        async with httpx.AsyncClient() as http:
            tasks   = [reindex_one(semaphore, p, http) for p in stale]
            results = await asyncio.gather(*tasks)

        ok           = [r for r in results if r["status"] == "ok"]
        kept_no_box  = [r for r in results if r["status"] == "kept_no_box"]
        deleted      = [r for r in results if r["status"] == "deleted"]
        skipped      = [r for r in results if r["status"] == "skipped"]
        failed       = [r for r in results if r["status"] == "failed"]
        no_url       = [r for r in results if r["status"] == "skipped_no_url"]
        no_consensus = [r for r in skipped if r.get("reason") == "no_consensus"]

        print("\n" + "=" * 60)
        print("PASS 1 COMPLETE")
        print("=" * 60)
        print(f"  Re-indexed successfully  : {len(ok)}")
        print(f"  Kept (no box found)      : {len(kept_no_box)}")
        print(f"  Deleted (not_fashion)    : {len(deleted)}  <- removed from locus_items")
        print(f"  Skipped (no_consensus)   : {len(no_consensus)}")
        print(f"  No image URL             : {len(no_url)}")
        print(f"  Failed (network/other)   : {len(failed)}")

        if failed:
            print("\nFailed products:")
            for r in failed[:20]:
                print(f"  - {r['item']}: {r.get('error', '?')}")
            if len(failed) > 20:
                print(f"  ... and {len(failed) - 20} more")
    else:
        print("Pass 1: nothing to do — locus_items is clean.\n")

    # ── Pass 2: no_consensus products in locus_skipped ───────────────────────
    async with httpx.AsyncClient() as http:
        await reindex_from_skipped(http)

    if DRY_RUN:
        print("\n[DRY RUN] No writes were made.")

    print("\nRun GET /index-stats to verify reindex_required=false")
    print("Check Whitelist Review tab for remaining skipped products.")


if __name__ == "__main__":
    asyncio.run(main())