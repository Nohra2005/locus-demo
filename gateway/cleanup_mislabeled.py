"""
cleanup_mislabeled.py

Audits locus_items for products whose stored category_tag contradicts their
product name. Three detection passes run in order, each catching what the
previous misses:

  Pass 1 — Multi-piece set heuristic
    Items whose names contain "set", "2-piece", "bundle", etc. are multi-garment
    products that should never have been indexed as a single category.
    Flagged as reason="multi_piece_set", correct_cat="not_fashion" → deleted.

  Pass 2 — Whitelist token contradiction
    Runs the same UNAMBIGUOUS_TOKEN_MAP logic as the vectorizer. If the whitelist
    gives a confident category that DISAGREES with the stored category_tag, the
    item is flagged. Covers "T-Shirt and Shorts Set stored as dress", "socks
    stored as shoes", etc.
    Flagged as reason="category_mismatch" or "title_is_not_fashion" → deleted.

  Pass 3 — LLM fallback (opt-in)
    For titles with no whitelist signal (Pass 2 returned None), optionally calls
    Groq to ask whether the item is a single wearable fashion product. Catches
    "Sport Set", "Athletic Bundle", brand bundles, accessories sets, etc.
    Enable with LLM_FALLBACK=1. Rate-limited to stay within Groq free tier.
    Flagged as reason="llm_not_fashion" → deleted.

Outputs
-------
- Prints a breakdown table: how many mismatches per (store, stored_cat, correct_cat)
- Prints every flagged item name, store, stored vs. correct category
- Writes flagged.json next to the script for inspection before deletion
- With DRY_RUN=0: deletes all flagged items from locus_items

Usage
-----
Run from the project root (not via docker exec — needs access to visual_engine/clip_labels.py):

    # Load credentials first (once per shell session):
    export $(grep -v '^#' .env | xargs)

    # Audit only (no deletions):
    python gateway/cleanup_mislabeled.py

    # Audit with LLM fallback for ambiguous titles:
    LLM_FALLBACK=1 python gateway/cleanup_mislabeled.py

    # Actually delete:
    DRY_RUN=0 python gateway/cleanup_mislabeled.py

    # Scope to one store and delete:
    STORE_FILTER=mikesport DRY_RUN=0 python gateway/cleanup_mislabeled.py

Environment variables
---------------------
    QDRANT_URL      — Qdrant Cloud URL  (optional, falls back to local)
    QDRANT_API_KEY  — Qdrant Cloud key  (optional)
    QDRANT_HOST     — local host        (default: localhost)
    QDRANT_PORT     — local port        (default: 6333)
    DRY_RUN         — 1 = report only, 0 = delete (default: 1, safe by default)
    STORE_FILTER    — comma-separated store names to scan (default: all stores)
                      e.g. STORE_FILTER=mikesport,intersport
    LLM_FALLBACK    — 1 = use Groq for titles with no whitelist signal (default: 0)
    GROQ_API_KEY    — required when LLM_FALLBACK=1
"""

import os
import re
import sys
import json
import time

# ---------------------------------------------------------------------------
# Bootstrap: make clip_labels importable whether run from gateway/ or project root
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from qdrant_client import QdrantClient
from qdrant_client.http import models

try:
    from clip_labels import UNAMBIGUOUS_TOKEN_MAP
except ImportError:
    # Running from outside the gateway container — try visual_engine path
    import sys
    sys.path.insert(0, os.path.join(_this_dir, "..", "visual_engine"))
    from clip_labels import UNAMBIGUOUS_TOKEN_MAP

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "locus_items"
DRY_RUN         = os.getenv("DRY_RUN", "1") != "0"   # safe default: audit only
STORE_FILTER    = [s.strip() for s in os.getenv("STORE_FILTER", "").split(",") if s.strip()]
LLM_FALLBACK    = os.getenv("LLM_FALLBACK", "0") == "1"
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
OUTPUT_JSON     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flagged.json")

if QDRANT_URL:
    print(f"[QDRANT] Cloud: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print(f"[QDRANT] Local: {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

if DRY_RUN:
    print("[DRY RUN] Audit only — no deletions will be made.")
    print("          Set DRY_RUN=0 to actually delete.\n")
else:
    print("[LIVE]    Deletions ENABLED.\n")

if STORE_FILTER:
    print(f"[FILTER]  Scanning stores: {STORE_FILTER}\n")
else:
    print("[FILTER]  Scanning ALL stores.\n")

if LLM_FALLBACK:
    if not GROQ_API_KEY:
        print("[WARN]    LLM_FALLBACK=1 but GROQ_API_KEY is not set — LLM pass disabled.\n")
        LLM_FALLBACK = False
    else:
        print("[LLM]     Groq fallback enabled for titles with no whitelist signal.\n")


# ── Pass 1: multi-piece set heuristic ────────────────────────────────────────
# These patterns catch "T-Shirt and Shorts Set", "2-Piece Sport Bundle", etc.
# Compiled once at module load for speed during the scroll loop.

_SET_RE = re.compile(
    r'\b('
    r'set|sets|bundle|bundles|kit|kits'          # generic multi-piece words
    r'|combo'
    r'|coord|co-ord|coords|co-ords'               # coord sets
    r'|matching\s+set|lounge\s+set|jogger\s+set'
    r'|tracksuit|track\s*suit'                    # tracksuits = top + bottoms
    r'|two[\s-]piece|2[\s-]piece'
    r'|three[\s-]piece|3[\s-]piece'
    r')\b'
    r'|\b\d+[\s-]*packs?\b',                     # "3 packs", "2-pack" — NOT "Eagle Pack"
    re.IGNORECASE,
)

def is_multi_piece_set(title: str) -> bool:
    """Return True if the title suggests a multi-garment set that should not be
    indexed as a single fashion item."""
    return bool(_SET_RE.search(title))


# ── Pass 3: Groq LLM fallback ─────────────────────────────────────────────────
# Only called for titles where the whitelist gives no signal (correct_cat=None).
# Batches titles and sends them in one prompt to minimise API calls.
# Rate-limited to stay within Groq free tier (~30 RPM).

_LLM_BATCH_SIZE = 40   # titles per Groq call
_LLM_RPM_GAP    = 2.1  # seconds between calls

def classify_titles_llm(titles: list[str]) -> dict[str, bool]:
    """
    Ask Groq whether each title is a single wearable fashion item.
    Returns a dict {title: is_fashion} where is_fashion=False means flag for deletion.
    Titles that error out default to is_fashion=True (safe — don't delete on uncertainty).
    """
    import httpx

    results = {}
    for i in range(0, len(titles), _LLM_BATCH_SIZE):
        batch = titles[i : i + _LLM_BATCH_SIZE]
        numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))
        prompt = (
            "You are a fashion product classifier. For each product name below, "
            "answer only YES or NO:\n"
            "- YES if it is a single wearable fashion item (one garment, one pair of shoes, "
            "one bag, one hat, etc.)\n"
            "- NO if it is a multi-piece set, bundle, kit, non-wearable, or sporting equipment\n\n"
            f"{numbered}\n\n"
            "Reply with exactly one word per line (YES or NO), in the same order, nothing else."
        )
        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": len(batch) * 4,
                },
                timeout=20,
            )
            resp.raise_for_status()
            lines = resp.json()["choices"][0]["message"]["content"].strip().splitlines()
            for j, title in enumerate(batch):
                answer = lines[j].strip().upper() if j < len(lines) else "YES"
                results[title] = (answer != "NO")
        except Exception as e:
            batch_num = i // _LLM_BATCH_SIZE + 1
            print(f"\n  [LLM WARN] Groq batch {batch_num} failed: {e}")
            for title in batch:
                results[title] = True   # default: keep on error

        if i + _LLM_BATCH_SIZE < len(titles):
            time.sleep(_LLM_RPM_GAP)

    return results


# ── Title classifier (whitelist branch only) ─────────────────────────────────

def classify_title_whitelist(title: str):
    """
    Lightweight version of vectorizer._classify_title(), whitelist branch only.
    Returns (category, confidence) where confidence is 1.0 for whitelist hits,
    or (None, 0.0) if no whitelist token matched.

    Only the whitelist branch is used here because:
    - We want high-precision signals to flag deletions
    - Sentence-transformer results are softer and may produce false flags
    - The token map is the same one used at index time
    """
    if not title or len(title.strip()) < 2:
        return None, 0.0

    # Normalize phrases that contain fashion-unrelated tokens but are genuinely
    # fashion items, preventing false not_fashion hits from the token map.
    # e.g. "tie-dye" / "tie dyed" → the word "tie" alone maps to not_fashion (neck tie).
    normalized = re.sub(r'\btie[\s-]dyed?\b', 'tiedye', title, flags=re.IGNORECASE)

    tokens = normalized.lower().replace("-", " ").replace("/", " ").split()

    fashion_hits     = {}
    not_fashion_hits = 0
    last_seen        = {}

    for idx, token in enumerate(tokens):
        cat = UNAMBIGUOUS_TOKEN_MAP.get(token)
        if cat is None:
            continue
        if cat == "not_fashion":
            not_fashion_hits += 1
        else:
            fashion_hits[cat] = fashion_hits.get(cat, 0) + 1
            last_seen[cat]    = idx

    # Bigrams
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    for i, bigram in enumerate(bigrams):
        cat = UNAMBIGUOUS_TOKEN_MAP.get(bigram)
        if cat is None:
            continue
        if cat == "not_fashion":
            not_fashion_hits += 2
        else:
            fashion_hits[cat] = fashion_hits.get(cat, 0) + 2
            last_seen[cat]    = len(tokens) + i

    if fashion_hits:
        best_count = max(fashion_hits.values())
        tied = [c for c, cnt in fashion_hits.items() if cnt == best_count]
        best_cat = tied[0] if len(tied) == 1 else max(tied, key=lambda c: last_seen.get(c, 0))

        if not_fashion_hits >= best_count:
            return "not_fashion", 1.0
        return best_cat, 1.0

    if not_fashion_hits > 0:
        return "not_fashion", 1.0

    return None, 0.0


# ── Scan ─────────────────────────────────────────────────────────────────────

def scan_collection():
    """
    Scrolls through locus_items and runs three detection passes:
      Pass 1 — multi-piece set heuristic  (regex)
      Pass 2 — whitelist token contradiction
      Pass 3 — Groq LLM fallback for titles with no whitelist signal (opt-in)

    Returns a list of flagged dicts, each with:
        point_id, product_id, name, store_name, stored_cat, correct_cat, reason
    """
    flagged          = []
    ambiguous        = []   # items with no whitelist signal → candidates for LLM pass
    total_seen       = 0
    offset           = None

    print("Scanning locus_items…")

    scroll_filter = None
    if STORE_FILTER:
        scroll_filter = models.Filter(
            must=[models.FieldCondition(
                key="store_name",
                match=models.MatchAny(any=STORE_FILTER),
            )]
        )

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            break

        for pt in results:
            total_seen += 1
            p          = pt.payload
            name       = p.get("name", "")
            stored_cat = p.get("category_tag", "")
            store_name = p.get("store_name", "")

            # Skip golden dataset items — curated, not indexed via pipeline
            if p.get("is_golden") or store_name == "golden_dataset":
                continue

            item = {
                "point_id":   str(pt.id),
                "product_id": p.get("product_id", str(pt.id)),
                "name":       name,
                "store_name": store_name,
                "stored_cat": stored_cat,
            }

            # ── Pass 1: multi-piece set heuristic ─────────────────────────
            if is_multi_piece_set(name):
                flagged.append({**item,
                    "correct_cat": "not_fashion",
                    "reason":      "multi_piece_set"})
                continue

            # ── Pass 2: whitelist token contradiction ──────────────────────
            correct_cat, _ = classify_title_whitelist(name)

            if correct_cat == "not_fashion":
                flagged.append({**item,
                    "correct_cat": "not_fashion",
                    "reason":      "title_is_not_fashion"})
                continue

            if correct_cat is not None and correct_cat != stored_cat:
                flagged.append({**item,
                    "correct_cat": correct_cat,
                    "reason":      "category_mismatch"})
                continue

            # ── Pass 3 candidate: no whitelist signal ──────────────────────
            if correct_cat is None and LLM_FALLBACK:
                ambiguous.append(item)

        print(f"  … scanned {total_seen} items, {len(flagged)} flagged so far", end="\r")

        if next_offset is None:
            break
        offset = next_offset

    print(f"\n  Scan complete. {total_seen} items seen, {len(flagged)} flagged by heuristics.")

    # ── Pass 3: LLM fallback ───────────────────────────────────────────────────
    if LLM_FALLBACK and ambiguous:
        print(f"  Running Groq LLM on {len(ambiguous)} ambiguous titles…")
        titles     = [a["name"] for a in ambiguous]
        llm_result = classify_titles_llm(titles)
        llm_flagged = 0
        for item in ambiguous:
            if not llm_result.get(item["name"], True):   # False = not fashion
                flagged.append({**item,
                    "correct_cat": "not_fashion",
                    "reason":      "llm_not_fashion"})
                llm_flagged += 1
        print(f"  LLM pass: {llm_flagged} additional items flagged.")

    print(f"  Total flagged: {len(flagged)}\n")
    return flagged


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(flagged: list):
    if not flagged:
        print("No mismatches found.")
        return

    # ── Summary table: (store, stored_cat → correct_cat, reason) counts ──────
    from collections import Counter
    from itertools import groupby

    summary = Counter(
        (f["store_name"], f["stored_cat"], f["correct_cat"], f["reason"])
        for f in flagged
    )

    print("=" * 84)
    print(f"{'STORE':<24} {'STORED AS':<12} {'CORRECT':<12} {'REASON':<22} {'COUNT':>6}")
    print("-" * 84)
    for (store, stored, correct, reason), count in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"{store:<24} {stored:<12} {correct:<12} {reason:<22} {count:>6}")
    print("=" * 84)
    print()

    # ── Per-item detail ───────────────────────────────────────────────────────
    sorted_flagged = sorted(flagged, key=lambda x: (x["reason"], x["store_name"], x["stored_cat"]))
    for (reason, store, stored), group in groupby(
        sorted_flagged, key=lambda x: (x["reason"], x["store_name"], x["stored_cat"])
    ):
        items = list(group)
        print(f"  [{reason}]  {store}  stored={stored}  ({len(items)} items)")
        for item in items[:20]:
            print(f"    • {item['name'][:80]}")
        if len(items) > 20:
            print(f"    … and {len(items) - 20} more")
        print()

    # ── Write JSON for inspection ─────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(flagged, fh, indent=2, ensure_ascii=False)
    print(f"Full list written to: {OUTPUT_JSON}")


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_flagged(flagged: list):
    if not flagged:
        return

    point_ids = [f["point_id"] for f in flagged]
    print(f"Deleting {len(point_ids)} items from '{COLLECTION_NAME}'…")

    # Qdrant accepts up to 1000 IDs per delete call
    BATCH = 1000
    deleted = 0
    for i in range(0, len(point_ids), BATCH):
        batch = point_ids[i : i + BATCH]
        try:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=batch),
            )
            deleted += len(batch)
            print(f"  … deleted {deleted}/{len(point_ids)}", end="\r")
        except Exception as e:
            print(f"\n  [WARN] Batch delete failed: {e}")

    print(f"\n  Done. {deleted} items removed from '{COLLECTION_NAME}'.")
    print()
    print("NOTE: These items will reappear if the store is re-indexed with the")
    print("      same product data. Run reindex_stale.py afterwards if needed,")
    print("      or fix the root cause (YOLO/CLIP misclassification) first.")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    flagged = scan_collection()
    print_report(flagged)

    if not flagged:
        sys.exit(0)

    if DRY_RUN:
        print(f"DRY RUN complete. {len(flagged)} items would be deleted.")
        print("Set DRY_RUN=0 to delete them.")
    else:
        confirm = input(f"\nAbout to delete {len(flagged)} items. Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            delete_flagged(flagged)
        else:
            print("Aborted.")
