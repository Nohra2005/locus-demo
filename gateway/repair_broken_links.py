"""
repair_broken_links.py  —  Link Health Monitor

Three-phase pipeline that runs on a 5-day schedule (via docker-compose link_monitor service).

  Phase 1 — AUDIT   : async HEAD-check every image_url in locus_items.
                       Items whose URLs were broken last run but are healthy
                       now get their broken flag cleared automatically.
  Phase 2 — MARK    : sets broken=True on dead items immediately so they
                       are hidden from search results while recovery runs.
  Phase 3 — RECOVER : scrapes each affected store once, fuzzy-matches the
                       product by name, re-embeds the new image, upserts a
                       new Qdrant point, and deletes the old broken point.
                       Items that cannot be matched are left broken=True for
                       manual review.

Usage:
    docker compose exec gateway python repair_broken_links.py

    # Dry run — prints what would happen, writes nothing
    DRY_RUN=1 docker compose exec gateway python repair_broken_links.py

Environment variables:
    QDRANT_URL       Qdrant Cloud URL          (optional, falls back to local)
    QDRANT_API_KEY   Qdrant Cloud key          (optional)
    QDRANT_HOST      local host                (default: localhost)
    QDRANT_PORT      local port                (default: 6333)
    VISUAL_URL       visual engine base URL    (default: http://visual_engine:8001)
    GATEWAY_URL      gateway base URL          (default: http://gateway:8000)
    DRY_RUN          set to 1 for read-only    (default: 0)
    CONCURRENCY      parallel HEAD workers     (default: 20)

Store URL map:
    Edit gateway/store_urls.json to map store_name → scrape URL.
    Keys must exactly match the store_name values stored in Qdrant.
    Example: {"Zara": "https://www.zara.com/us/en/woman-new-in-l1180.html"}
"""

import asyncio
import difflib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
VISUAL_URL      = os.getenv("VISUAL_URL", "http://visual_engine:8001")
GATEWAY_URL     = os.getenv("GATEWAY_URL", "http://gateway:8000")
COLLECTION_NAME = "locus_items"
DRY_RUN         = os.getenv("DRY_RUN", "0") == "1"
CONCURRENCY     = int(os.getenv("CONCURRENCY", "20"))

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE          = Path(__file__).parent
_STORE_URLS    = _BASE / "store_urls.json"
_REPORT_PATH   = _BASE / "link_health_report.json"

# ── Store URL map ─────────────────────────────────────────────────────────────
def _load_store_urls() -> dict:
    if not _STORE_URLS.exists():
        print(f"[WARN] {_STORE_URLS} not found — recovery will be skipped.")
        print(f"       Create it: {{\"StoreName\": \"https://store.com\", ...}}")
        return {}
    with open(_STORE_URLS) as f:
        return json.load(f)

STORE_URL_MAP: dict[str, str] = _load_store_urls()

# ── Qdrant client ─────────────────────────────────────────────────────────────
if QDRANT_URL:
    print(f"[QDRANT] Cloud: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print(f"[QDRANT] Local: {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

if DRY_RUN:
    print("[DRY RUN] No writes will be made.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — AUDIT
# ══════════════════════════════════════════════════════════════════════════════

async def _head_check(
    sem: asyncio.Semaphore,
    http: httpx.AsyncClient,
    product: dict,
) -> dict:
    """HEAD-checks a single image_url. Adds url_ok + http_status to the dict."""
    async with sem:
        url = product["image_url"]

        # Skip local dev URLs and data URIs — not checkable externally
        if not url or url.startswith("data:") or url.startswith("http://localhost"):
            return {**product, "url_ok": True, "http_status": None}

        try:
            resp = await http.head(url, timeout=10.0, follow_redirects=True)
            ok   = resp.status_code < 400
            if not ok:
                print(f"  [DEAD]  HTTP {resp.status_code}  {url[:80]}")
            return {**product, "url_ok": ok, "http_status": resp.status_code}
        except Exception as exc:
            print(f"  [DEAD]  {type(exc).__name__}: {url[:80]}")
            return {**product, "url_ok": False, "http_status": None, "error": str(exc)}


async def audit_links() -> tuple[list[dict], list[dict], list[dict]]:
    """
    Scrolls all locus_items and HEAD-checks every image_url in parallel.

    Returns:
        healthy          — items whose URL is reachable
        broken           — items whose URL is dead
        recovered_flags  — items previously marked broken=True but now healthy
    """
    print("Phase 1 — Auditing image URLs...\n")

    all_products: list[dict] = []
    offset = None

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
            p = pt.payload
            all_products.append({
                "point_id":       str(pt.id),
                "product_id":     p.get("product_id", str(pt.id)),
                "name":           p.get("name", ""),
                "store_name":     p.get("store_name", ""),
                "mall_name":      p.get("mall_name", ""),
                "image_url":      p.get("image_url", ""),
                "category_tag":   p.get("category_tag", ""),
                "price":          p.get("price", ""),
                "box_source":     p.get("box_source", ""),
                "shoe_style":     p.get("shoe_style"),
                "is_golden":      p.get("is_golden", False),
                "already_broken": p.get("broken", False),
                "broken_count":   p.get("broken_count", 0),
            })

        print(f"  ... loaded {len(all_products)} items", end="\r")

        if next_offset is None:
            break
        offset = next_offset

    total = len(all_products)
    print(f"\n  Loaded {total} items. Running {CONCURRENCY} parallel HEAD checks...")

    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(follow_redirects=True) as http:
        checked = await asyncio.gather(*[
            _head_check(sem, http, p) for p in all_products
        ])

    healthy         = [r for r in checked if     r["url_ok"]]
    broken          = [r for r in checked if not r["url_ok"]]
    recovered_flags = [r for r in healthy  if     r["already_broken"]]

    print(f"\n  Results — healthy: {len(healthy)}  |  broken: {len(broken)}  "
          f"|  self-healed: {len(recovered_flags)}")
    return healthy, broken, recovered_flags


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — MARK / UNMARK
# ══════════════════════════════════════════════════════════════════════════════

def mark_items_broken(broken_items: list[dict]) -> list[dict]:
    """
    First failure  → sets broken=True, broken_count=1, hidden from search.
    Second failure → deletes the point (link is permanently dead).
    Returns the list of items still alive (not deleted), for Phase 3 recovery.
    """
    new_items     = [b for b in broken_items if not b["already_broken"]]
    repeat_items  = [b for b in broken_items if     b["already_broken"]]
    to_delete     = [b for b in repeat_items if b["broken_count"] >= 1 and not b.get("is_golden")]
    to_increment  = [b for b in repeat_items if b["broken_count"] < 1]

    print(f"\nPhase 2 — {len(new_items)} new broken  |  {len(to_delete)} to delete  |  {len(to_increment)} incrementing...")

    if DRY_RUN:
        for b in new_items:
            print(f"  [DRY]   would mark    '{b['name']}'")
        for b in to_increment:
            print(f"  [DRY]   would bump    '{b['name']}' broken_count={b['broken_count']+1}")
        for b in to_delete:
            print(f"  [DRY]   would DELETE  '{b['name']}' (broken_count={b['broken_count']+1})")
        return broken_items

    now = datetime.utcnow().isoformat()

    for item in new_items:
        try:
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"broken": True, "broken_since": now, "broken_count": 1},
                points=[item["point_id"]],
            )
        except Exception as exc:
            print(f"  [WARN]  Could not mark '{item['name']}': {exc}")

    for item in to_increment:
        try:
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"broken_count": item["broken_count"] + 1},
                points=[item["point_id"]],
            )
        except Exception as exc:
            print(f"  [WARN]  Could not increment '{item['name']}': {exc}")

    deleted = 0
    for item in to_delete:
        try:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=[item["point_id"]]),
            )
            print(f"  [DEL]   '{item['name']}' — dead for 2 runs, removed")
            deleted += 1
        except Exception as exc:
            print(f"  [WARN]  Could not delete '{item['name']}': {exc}")

    print(f"  Done — marked={len(new_items)} deleted={deleted}")

    deleted_ids = {b["point_id"] for b in to_delete}
    return [b for b in broken_items if b["point_id"] not in deleted_ids]


def clear_broken_flags(recovered_flag_items: list[dict]) -> None:
    """Removes broken=True from items whose URL is healthy again."""
    if not recovered_flag_items:
        return

    print(f"\n  Clearing broken flag on {len(recovered_flag_items)} self-healed items...")
    if DRY_RUN:
        for r in recovered_flag_items:
            print(f"  [DRY]   would clear flag for '{r['name']}'")
        return

    for item in recovered_flag_items:
        try:
            client.delete_payload(
                collection_name=COLLECTION_NAME,
                keys=["broken", "broken_since"],
                points=[item["point_id"]],
            )
            print(f"  [CLEAR] '{item['name']}'")
        except Exception as exc:
            print(f"  [WARN]  Could not clear flag for '{item['name']}': {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — RECOVER
# ══════════════════════════════════════════════════════════════════════════════

def _fuzzy_match(target: str, candidates: list[dict], threshold: float = 0.75) -> dict | None:
    """
    Fuzzy-matches a product name against a list of scraped products.
    Returns the best match if score >= threshold, else None.
    """
    target_lower = target.lower()
    best_score   = 0.0
    best_match: dict | None = None

    for c in candidates:
        score = difflib.SequenceMatcher(
            None, target_lower, c.get("name", "").lower()
        ).ratio()
        if score > best_score:
            best_score = score
            best_match = c

    return best_match if best_score >= threshold else None


async def _fetch_store_catalog(http: httpx.AsyncClient, store_url: str) -> list[dict]:
    """
    Fetches current products from a store.
    Tries Shopify /products.json first, then falls back to the gateway /scrape endpoint.
    Returns [{name, image_url}, ...].
    """
    from urllib.parse import urlparse
    base = f"{urlparse(store_url).scheme}://{urlparse(store_url).netloc}"

    # ── Shopify API (fast, no HTML parsing) ───────────────────────────────────
    try:
        resp = await http.get(f"{base}/products.json?limit=250", timeout=20.0)
        if resp.status_code == 200:
            raw = resp.json().get("products", [])
            if raw:
                result = []
                for p in raw:
                    name   = p.get("title", "").strip()
                    images = p.get("images", [])
                    if name and images:
                        result.append({"name": name, "image_url": images[0].get("src", "")})
                if result:
                    return result
    except Exception:
        pass

    # ── Gateway /scrape fallback ───────────────────────────────────────────────
    try:
        resp = await http.post(
            f"{GATEWAY_URL}/scrape",
            json={"url": store_url, "max_products": 500},
            timeout=60.0,
        )
        if resp.status_code == 200:
            return resp.json().get("products", [])
    except Exception as exc:
        print(f"  [WARN]  /scrape failed for {store_url}: {exc}")

    return []


async def _recover_one(
    sem: asyncio.Semaphore,
    http: httpx.AsyncClient,
    item: dict,
    store_catalog: dict[str, list[dict]],
) -> dict:
    """
    Tries to find a replacement URL for a single broken item and re-indexes it.
    On success: upserts new Qdrant point, deletes old broken point.
    """
    async with sem:
        name       = item["name"]
        store_name = item["store_name"]

        # No URL mapping for this store
        if store_name not in store_catalog:
            return {"status": "no_store_url", "item": name, "store": store_name}

        match = _fuzzy_match(name, store_catalog[store_name])
        if not match:
            print(f"  [MISS]  No name match for '{name}' in {store_name}")
            return {"status": "no_match", "item": name, "store": store_name}

        new_url = match.get("image_url", "")
        if not new_url:
            return {"status": "no_match", "item": name, "store": store_name}

        if new_url == item["image_url"]:
            # Store returned the same dead URL — nothing we can do
            print(f"  [SAME]  Store returned identical dead URL for '{name}'")
            return {"status": "same_url", "item": name, "store": store_name}

        print(f"  [FOUND] '{name}'  new_url={new_url[:60]}...")

        try:
            # Fetch new image bytes
            img_resp = await http.get(new_url, timeout=15.0, follow_redirects=True)
            img_resp.raise_for_status()
            img_bytes    = img_resp.content
            content_type = img_resp.headers.get("content-type", "image/jpeg")

            # Re-embed via Visual Engine
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
                print(f"  [SKIP]  '{name}' — visual engine: {reason}")
                return {"status": "skipped_by_engine", "item": name,
                        "store": store_name, "reason": reason}

            vector_normal  = idx_data["vector_normal"]
            new_category   = idx_data.get("category", item["category_tag"])
            new_box_source = idx_data.get("box_source", "unknown")
            shoe_style     = idx_data.get("shoe_style")

            new_payload = {
                "name":         name,
                "store_name":   item["store_name"],
                "mall_name":    item["mall_name"],
                "image_url":    new_url,
                "category_tag": new_category,
                "price":        item["price"],
                "product_id":   item["product_id"],
                "box_source":   new_box_source,
            }
            if shoe_style:
                new_payload["shoe_style"] = shoe_style
            if item.get("is_golden"):
                new_payload["is_golden"] = True
            # Note: no broken=True on the new point — it's healthy

            if not DRY_RUN:
                # Upsert new point with new URL (new UUID5)
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(
                        id      = str(uuid.uuid5(uuid.NAMESPACE_URL, new_url)),
                        vector  = vector_normal,
                        payload = new_payload,
                    )],
                )
                # Delete old broken point
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=[item["point_id"]]),
                )

            print(f"  [OK]    '{name}'  recovered, old point deleted")
            return {
                "status":  "recovered",
                "item":    name,
                "store":   store_name,
                "old_url": item["image_url"],
                "new_url": new_url,
            }

        except Exception as exc:
            print(f"  [ERR]   '{name}' — {exc}")
            return {"status": "error", "item": name, "store": store_name, "error": str(exc)}


async def recover_broken(broken_items: list[dict]) -> list[dict]:
    """
    Pre-fetches each affected store's catalog once, then recovers all broken
    items in parallel. One store fetch = many items recovered.
    """
    if not broken_items:
        print("\nPhase 3 — No broken items to recover.")
        return []

    stores_needed = {
        b["store_name"] for b in broken_items if b["store_name"] in STORE_URL_MAP
    }
    unknown_stores = {
        b["store_name"] for b in broken_items if b["store_name"] not in STORE_URL_MAP
    }

    if unknown_stores:
        print(f"\n  [WARN]  No URL for store(s): {', '.join(sorted(unknown_stores))}")
        print(f"          Add them to gateway/store_urls.json to enable recovery.")

    print(f"\nPhase 3 — Fetching catalogs for {len(stores_needed)} store(s)...")

    store_catalog: dict[str, list[dict]] = {}
    async with httpx.AsyncClient(follow_redirects=True) as http:
        for store_name in stores_needed:
            store_url = STORE_URL_MAP[store_name]
            print(f"  Scraping '{store_name}' ({store_url})...")
            products = await _fetch_store_catalog(http, store_url)
            store_catalog[store_name] = products
            print(f"  → {len(products)} products retrieved")

        print(f"\n  Recovering {len(broken_items)} broken items...")
        sem     = asyncio.Semaphore(CONCURRENCY)
        results = await asyncio.gather(*[
            _recover_one(sem, http, item, store_catalog)
            for item in broken_items
        ])

    return list(results)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def _wait_for_qdrant(retries: int = 6) -> None:
    for attempt in range(retries):
        try:
            client.get_collections()
            return
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"[STARTUP] Qdrant not reachable ({e}), retry {attempt + 1}/{retries} in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"Could not connect to Qdrant after {retries} attempts: {e}") from e


async def main() -> None:
    await _wait_for_qdrant()

    start = datetime.utcnow()
    print("=" * 60)
    print("Locus — Link Health Monitor")
    print(f"Started : {start.isoformat()}")
    if DRY_RUN:
        print("Mode    : DRY RUN (no writes)")
    print("=" * 60 + "\n")

    # Phase 1 — audit
    healthy, broken, recovered_flags = await audit_links()

    # Phase 2 — mark / unmark
    surviving_broken = mark_items_broken(broken)
    clear_broken_flags(recovered_flags)

    # Phase 3 — recover (only items not already deleted)
    recovery_results = await recover_broken(surviving_broken)

    recovered     = [r for r in recovery_results if r["status"] == "recovered"]
    unrecoverable = [r for r in recovery_results if r["status"] != "recovered"]
    auto_deleted  = len(broken) - len(surviving_broken)

    # ── Report ────────────────────────────────────────────────────────────────
    end = datetime.utcnow()
    report = {
        "run_at":             start.isoformat(),
        "duration_seconds":   round((end - start).total_seconds(), 1),
        "dry_run":            DRY_RUN,
        "total_checked":      len(healthy) + len(broken),
        "healthy":            len(healthy),
        "broken_found":       len(broken),
        "auto_deleted":       auto_deleted,
        "self_healed_flags":  len(recovered_flags),
        "recovered":          len(recovered),
        "unrecoverable":      len(unrecoverable),
        "broken_detail": [
            {
                "name":        b["name"],
                "store":       b["store_name"],
                "url":         b["image_url"],
                "http_status": b.get("http_status"),
            }
            for b in broken
        ],
        "recovery_detail": recovery_results,
    }

    if not DRY_RUN:
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved → {_REPORT_PATH}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Total checked   : {report['total_checked']}")
    print(f"  Healthy         : {report['healthy']}")
    print(f"  Broken found    : {report['broken_found']}")
    print(f"  Auto-deleted    : {report['auto_deleted']}")
    print(f"  Self-healed     : {report['self_healed_flags']}")
    print(f"  Recovered       : {report['recovered']}")
    print(f"  Unrecoverable   : {report['unrecoverable']}")
    print(f"  Duration        : {report['duration_seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
