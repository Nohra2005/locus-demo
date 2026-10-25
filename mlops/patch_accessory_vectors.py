"""
One-shot script: patch accessory (shoes/bag/hat) vectors in Qdrant with hybrid
visual+text embeddings — WITHOUT a full reindex.

How it works:
  1. Scroll Qdrant for all accessories (category_tag in shoes/bag/hat)
  2. For each, fetch the stored vector + product title from payload
  3. Call visual_engine /vectorize-text to get CLIP text embedding for the title
  4. Mix: 0.7 × existing_visual_vector + 0.3 × text_vector, re-normalize
  5. Upsert the new vector back to Qdrant (payload unchanged)

Run after restarting visual_engine (so the query-side hybrid code is live):
    python mlops/patch_accessory_vectors.py --gateway-url http://localhost:8000

Dry run (prints what would change, writes nothing):
    python mlops/patch_accessory_vectors.py --dry-run

Revert: run with --alpha 0 to restore pure visual vectors.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "locus_items"
ACCESSORY_CATS  = ["shoes", "bag", "hat"]

ALPHA_DEFAULT   = 0.3   # 30% text, 70% visual — matches vectorizer.py _HYBRID_ALPHA


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-9:
        return vec
    return [x / norm for x in vec]


def _mix(vis: list[float], txt: list[float], alpha: float) -> list[float]:
    mixed = [(1.0 - alpha) * v + alpha * t for v, t in zip(vis, txt)]
    return _normalize(mixed)


def _get_text_embedding(visual_url: str, title: str) -> list[float] | None:
    try:
        r = requests.post(
            f"{visual_url}/vectorize-text",
            json={"text": title},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()["embedding"]
    except Exception as e:
        print(f"    [WARN] /vectorize-text failed for '{title[:40]}': {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", default="http://localhost:8000",
                        help="Gateway base URL (used to reach visual_engine indirectly)")
    parser.add_argument("--visual-url",  default="",
                        help="Direct visual_engine URL (overrides gateway-url detection). "
                             "E.g. http://localhost:8001")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                        help=f"Text weight in hybrid mix (default {ALPHA_DEFAULT}). "
                             "Set to 0 to restore pure visual vectors.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing to Qdrant")
    parser.add_argument("--category", default="",
                        help="Limit to one category: shoes | bag | hat (default: all accessories)")
    args = parser.parse_args()

    if not QDRANT_URL:
        print("[ERROR] QDRANT_URL env var is not set")
        sys.exit(1)

    # Resolve visual_engine URL — try direct if given, else probe gateway
    visual_url = args.visual_url.rstrip("/")
    if not visual_url:
        # Guess: gateway is at 8000, visual_engine is at 8001 on same host
        host = args.gateway_url.rstrip("/").rsplit(":", 1)[0]
        visual_url = f"{host}:8001"
        # Quick health check — fall back to gateway proxy path
        try:
            requests.get(f"{visual_url}/", timeout=3).raise_for_status()
        except Exception:
            print(f"[INFO] Direct visual_engine at {visual_url} not reachable — "
                  "using gateway's /vectorize-text proxy route")
            visual_url = args.gateway_url.rstrip("/")

    print(f"[PATCH] visual_engine URL : {visual_url}")
    print(f"[PATCH] Qdrant collection : {COLLECTION_NAME}")
    print(f"[PATCH] Alpha (text weight): {args.alpha}")
    print(f"[PATCH] Dry run           : {args.dry_run}")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    target_cats = [args.category] if args.category else ACCESSORY_CATS

    total_found   = 0
    total_patched = 0
    total_skipped = 0

    for cat in target_cats:
        print(f"\n--- Category: {cat} ---")
        offset = None
        cat_found = 0

        while True:
            results, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="category_tag",
                        match=models.MatchValue(value=cat),
                    )
                ]),
                with_vectors=True,
                with_payload=True,
                limit=100,
                offset=offset,
            )
            if not results:
                break

            for point in results:
                cat_found  += 1
                total_found += 1

                payload = point.payload or {}
                title   = payload.get("name", "") or payload.get("title", "")
                pid     = str(point.id)

                if not title:
                    print(f"  [SKIP] {pid[:12]} — no title in payload")
                    total_skipped += 1
                    continue

                # For shoes, build the title hint using shoe_style from payload
                shoe_style = payload.get("shoe_style", "other") or "other"
                if cat == "shoes":
                    style_hints = {
                        "sneaker": "sneaker trainer running shoe athletic casual footwear",
                        "boot":    "ankle boot chelsea boot combat boot knee high boot",
                        "heel":    "high heel stiletto pump court shoe kitten heel formal",
                        "sandal":  "sandal flat mule loafer ballet flat espadrille open toe",
                        "other":   "shoe footwear",
                    }
                    text_input = f"{style_hints.get(shoe_style, 'shoe footwear')} {title}"
                else:
                    text_input = title

                # Get current visual vector
                existing_vec = point.vector
                if existing_vec is None:
                    print(f"  [SKIP] {pid[:12]} — no vector stored")
                    total_skipped += 1
                    continue
                if isinstance(existing_vec, dict):
                    existing_vec = existing_vec.get("", existing_vec.get("default", None))
                    if existing_vec is None:
                        print(f"  [SKIP] {pid[:12]} — named vectors not supported in this mode")
                        total_skipped += 1
                        continue

                txt_vec = _get_text_embedding(visual_url, text_input)
                if txt_vec is None:
                    total_skipped += 1
                    continue

                if args.alpha == 0:
                    new_vec = _normalize(list(existing_vec))
                else:
                    new_vec = _mix(list(existing_vec), txt_vec, args.alpha)

                print(f"  {'[DRY]' if args.dry_run else '[OK] '} {pid[:12]} "
                      f"shoe_style={shoe_style:8s} '{title[:45]}'")

                if not args.dry_run:
                    for _attempt in range(3):
                        try:
                            client.upsert(
                                collection_name=COLLECTION_NAME,
                                points=[models.PointStruct(
                                    id=point.id,
                                    vector=new_vec,
                                    payload=payload,
                                )],
                            )
                            break
                        except Exception as _ue:
                            if _attempt == 2:
                                print(f"  [FAIL] {pid[:12]} after 3 attempts: {_ue}")
                                total_skipped += 1
                                total_patched -= 1
                            else:
                                time.sleep(2.0 * (_attempt + 1))
                    time.sleep(0.05)  # light throttle — don't hammer Qdrant cloud

                total_patched += 1

            if next_offset is None:
                break
            offset = next_offset

        print(f"  -> {cat}: {cat_found} items processed")

    print(f"\n[PATCH] Done.")
    print(f"  Found   : {total_found}")
    print(f"  Patched : {total_patched}{'  (dry run — nothing written)' if args.dry_run else ''}")
    print(f"  Skipped : {total_skipped}")

    if args.dry_run:
        print("\nRe-run without --dry-run to apply changes.")
    else:
        print("\nNext steps:")
        print("  1. Run: python mlops/dismiss_accessory_flags.py --gateway-url <url>")
        print("  2. Run: python mlops/ci_eval.py --gateway-url <url>")


if __name__ == "__main__":
    main()
