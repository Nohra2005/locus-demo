#!/usr/bin/env python3
"""
repair_golden_dataset.py — Locus Golden Dataset Repair Tool
============================================================
Surgically fixes bad relevant_info slots without rebuilding the
30-entry golden dataset from scratch.

Step 1 — AUDIT   : inspect every slot; classify as OK / BAD / MISSING
Step 2 — REPAIR  : interactive URL-by-URL replacement
Step 3 — REPORT  : cosine similarity before/after, save repair_report.json

No external dependencies beyond qdrant-client and requests
(image dimensions parsed in pure Python via struct — no Pillow needed).

Run from project root or from mlops/:
    python mlops/repair_golden_dataset.py
"""

import base64
import io
import json
import math
import os
import struct
import sys
import time
import uuid
import urllib.request
import urllib.parse
from datetime import datetime

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct

# ── .env loader ───────────────────────────────────────────────────────────────
def _load_dotenv(path: str):
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            # strip quotes AND Windows carriage-return that corrupts cloud URLs
            val = val.strip().strip('"').strip("'").rstrip("\r")
            if key and key not in os.environ:
                os.environ[key] = val

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _cand in [
    os.path.join(_script_dir, "..", ".env"),   # typical: run from mlops/
    os.path.join(os.getcwd(), ".env"),          # run from project root
    ".env",
]:
    if os.path.exists(_cand):
        _load_dotenv(os.path.abspath(_cand))
        break

# ── Configuration ─────────────────────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL",    "").rstrip("\r")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY","").rstrip("\r")
VISUAL_URL     = os.getenv("VISUAL_URL",    "http://localhost:8001")
COLLECTION     = "locus_items"
MIN_DIM        = 400          # minimum acceptable pixel dimension

# Domain lists (mirrors golden_dataset_builder.html)
BAD_DOMAINS = {
    "pinterest.com","pinimg.com","pinterest.co.uk",
    "instagram.com","cdninstagram.com","fbcdn.net","facebook.com",
    "twitter.com","x.com","twimg.com","pbs.twimg.com",
    "tiktok.com","tiktokcdn.com",
    "reddit.com","redd.it","redditmedia.com",
    "youtube.com","youtu.be","ytimg.com",
    "snapchat.com","tumblr.com",
}
GOOD_DOMAINS = {
    "asos-media.com","images.asos-media.com",
    "image.hm.com","lp2.hm.com",
    "static.zara.net","zara.com",
    "img01.ztat.net","img02.ztat.net","mosaic01.ztat.net",
    "st.mngbcn.com","mango.com",
    "cdn.shopify.com","myshopify.com",
    "wixstatic.com","static.wixstatic.com",
    "squarespace-cdn.com","images.squarespace-cdn.com",
    "n.nordstrommedia.com",
    "net-a-porter.com","mrporter.com",
    "img.ssense.com",
    "cdn-images.farfetch-contents.com",
    "is4.revolveassets.com","images.urbndata.com",
    "image.uniqlo.com",
    "static.e-stradivarius.net","static.pullandbear.net","static.bershka.net",
    "i.imgur.com","images.unsplash.com","images.pexels.com",
}

# File paths
for _p in [
    os.path.join(_script_dir, "golden_dataset.json"),
    os.path.join(os.getcwd(), "golden_dataset.json"),
    os.path.join(os.getcwd(), "mlops", "golden_dataset.json"),
]:
    if os.path.exists(_p):
        GOLDEN_PATH = _p
        break
else:
    GOLDEN_PATH = os.path.join(_script_dir, "golden_dataset.json")

REPORT_PATH = os.path.join(os.path.dirname(GOLDEN_PATH), "repair_report.json")

# ══════════════════════════════════════════════════════════════════════════════
# PURE-PYTHON IMAGE SIZE PARSER
# Handles JPEG (all SOF variants), PNG, WebP (VP8/VP8L), GIF — no Pillow.
# ══════════════════════════════════════════════════════════════════════════════

def _image_size(data: bytes):
    """Return (width, height) from raw image bytes, or (0, 0) on failure."""
    try:
        # ── PNG ───────────────────────────────────────────────────────────────
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            w = struct.unpack('>I', data[16:20])[0]
            h = struct.unpack('>I', data[20:24])[0]
            return w, h

        # ── JPEG ──────────────────────────────────────────────────────────────
        if data[:2] == b'\xff\xd8':
            i = 2
            end = len(data)
            while i + 3 < end:
                # skip padding bytes
                while i < end and data[i] == 0xff:
                    i += 1
                marker = data[i]; i += 1
                if marker == 0xd9:            # EOI
                    break
                if marker in (0xc0, 0xc1, 0xc2):   # SOF0/1/2
                    # structure: length(2) precision(1) height(2) width(2)
                    h = struct.unpack('>H', data[i + 3 : i + 5])[0]
                    w = struct.unpack('>H', data[i + 5 : i + 7])[0]
                    return w, h
                seg_len = struct.unpack('>H', data[i : i + 2])[0]
                i += seg_len
            return 0, 0

        # ── WebP ──────────────────────────────────────────────────────────────
        if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            if data[12:16] == b'VP8 ':
                w = struct.unpack('<H', data[26:28])[0] & 0x3fff
                h = struct.unpack('<H', data[28:30])[0] & 0x3fff
                return w, h
            if data[12:16] == b'VP8L':
                n = struct.unpack('<I', data[21:25])[0]
                return (n & 0x3fff) + 1, ((n >> 14) & 0x3fff) + 1
            if data[12:16] == b'VP8X':
                w = (struct.unpack('<I', data[24:28])[0] & 0xffffff) + 1
                h = (struct.unpack('<I', data[27:31])[0] & 0xffffff) + 1
                return w, h

        # ── GIF ───────────────────────────────────────────────────────────────
        if data[:6] in (b'GIF87a', b'GIF89a'):
            w = struct.unpack('<H', data[6:8])[0]
            h = struct.unpack('<H', data[8:10])[0]
            return w, h

    except Exception:
        pass
    return 0, 0


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def p(msg=""):
    print(msg, flush=True)


def cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if (na and nb) else 0.0


def compute_product_id(image_url: str) -> str:
    """
    Mirrors build_golden_from_lens.py:
        uuid5(NAMESPACE_URL, 'golden::' + image_url)
    This is the value stored in payload.product_id.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"golden::{image_url}"))


def compute_point_id(image_url: str) -> str:
    """
    Mirrors gateway add_bulk_batch:
        uuid5(NAMESPACE_URL, image_url)   — no 'golden::' prefix.
    This is the Qdrant point ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, image_url))


def domain_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return ""


def fetch_image(url: str) -> bytes:
    """Download from https:// or decode data: URI."""
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
        return base64.b64decode(b64)
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Locus-Repair/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


def classify_domain(url: str) -> str:
    """Return 'bad' | 'good' | 'risky'."""
    host = domain_of(url)
    for d in BAD_DOMAINS:
        if host == d or host.endswith("." + d):
            return "bad"
    for d in GOOD_DOMAINS:
        if host == d or host.endswith("." + d):
            return "good"
    return "risky"


def validate_and_download(url: str):
    """
    Full validation + download for a replacement URL.
    Returns (ok: bool, reason: str, img_bytes: bytes, size: (w, h)).
    """
    if not url.startswith("https://"):
        return False, "Must start with https://", b"", (0, 0)

    if "encrypted-tbn" in url.lower():
        return False, "Google thumbnail cache (encrypted-tbn) is not allowed", b"", (0, 0)

    cls  = classify_domain(url)
    host = domain_of(url)
    if cls == "bad":
        return (False,
                f"{host} is a blocked CDN (social media — server cannot download images)",
                b"", (0, 0))
    if cls == "risky":
        p(f"  [WARN] Unknown domain: {host} — attempting download anyway")

    try:
        img_bytes = fetch_image(url)
    except Exception as e:
        return False, f"Download failed: {e}", b"", (0, 0)

    w, h = _image_size(img_bytes)
    if w == 0 or h == 0:
        return (False,
                "Could not parse image dimensions — unsupported format or corrupt file",
                b"", (0, 0))

    if min(w, h) < MIN_DIM:
        return (False,
                f"Image too small: {w}×{h}px  (minimum {MIN_DIM}px required)",
                b"", (0, 0))

    return True, "", img_bytes, (w, h)


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def fetch_golden_index(client: QdrantClient) -> dict:
    """
    Single scroll pass — fetches all store_name='golden_dataset' points.
    Returns {product_id: {point_id, vector, category_tag, image_url, name}}.
    """
    index  = {}
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="store_name",
                match=models.MatchValue(value="golden_dataset"),
            )]),
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not batch:
            break
        for pt in batch:
            pid = pt.payload.get("product_id", str(pt.id))
            index[pid] = {
                "point_id":     str(pt.id),
                "vector":       pt.vector,
                "category_tag": pt.payload.get("category_tag", ""),
                "image_url":    pt.payload.get("image_url", ""),
                "name":         pt.payload.get("name", ""),
            }
        if next_offset is None:
            break
        offset = next_offset
    return index


# ── Visual-engine helpers ─────────────────────────────────────────────────────

def call_vectorize(img_bytes: bytes):
    """
    POST to /vectorize. Returns (vector, category, conf_float).
    NOTE: category_confidence from this endpoint is a dict of all label scores,
    not a scalar. We take max(values()) to get a single float.
    """
    resp = requests.post(
        f"{VISUAL_URL}/vectorize",
        files={"file": ("query.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        timeout=60,
    )
    resp.raise_for_status()
    d       = resp.json()
    raw_cf  = d.get("category_confidence", 0.0)
    conf    = max(raw_cf.values()) if isinstance(raw_cf, dict) else float(raw_cf or 0)
    return d.get("vector"), d.get("category"), conf


def call_index_image(img_bytes: bytes, title: str) -> dict:
    """POST to /index-image. Returns full response dict."""
    resp = requests.post(
        f"{VISUAL_URL}/index-image",
        files={"file": ("product.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        data={"title": title},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def run_audit(dataset: list, qdrant_index: dict) -> list:
    """
    Classify every relevant_info slot.

    Status priority:
      BAD     — URL contains 'encrypted-tbn', or image < 400px, or unreachable
      MISSING — product_id absent from Qdrant (implies no vector)
      OK      — URL is fine AND product_id is in Qdrant

    Note: encrypted-tbn slots are flagged BAD without downloading (fast path).
    Non-thumbnail URLs are downloaded to verify resolution.

    Returns list of slot_info dicts ordered by (entry_idx, slot_idx).
    Each dict stores the old_vector and old_point_id for use in later steps.
    """
    p()
    p("=" * 72)
    p("  STEP 1 — AUDIT")
    p("=" * 72)
    p()

    # Header
    entry_w = 30; name_w = 22; domain_w = 22; res_w = 9
    hdr = (f"  {'#':>3}  {'Entry':<{entry_w}}  {'Sl'}  "
           f"{'Name':<{name_w}}  {'Domain':<{domain_w}}  "
           f"{'Res':>{res_w}}  Status")
    p(hdr)
    p("  " + "─" * (len(hdr) - 2))

    all_slots = []

    for ei, entry in enumerate(dataset):
        q_name = entry.get("query_name", f"entry_{ei}")

        for si, info in enumerate(entry.get("relevant_info", [])):
            product_id = info.get("product_id", "")
            name       = info.get("name", "")
            url        = info.get("image_url", "")

            slot = {
                "entry_idx":      ei,
                "slot_idx":       si,
                "entry_name":     q_name,
                "product_id":     product_id,
                "name":           name,
                "old_url":        url,
                "old_resolution": None,   # (w, h) if we could read it
                "old_vector":     None,   # list[float] from Qdrant, if present
                "old_point_id":   None,   # Qdrant UUID, for deletion on repair
                "status":         "OK",
                "bad_reason":     None,
            }

            # ── 1. Pull existing Qdrant data (fast, already fetched) ──────────
            if product_id and product_id in qdrant_index:
                rec = qdrant_index[product_id]
                slot["old_vector"]   = rec["vector"]
                slot["old_point_id"] = rec["point_id"]
            else:
                # Mark MISSING now; may be overridden to BAD by URL checks below
                slot["status"]     = "MISSING"
                slot["bad_reason"] = "not in Qdrant"

            # ── 2. URL / resolution checks ────────────────────────────────────
            if not url:
                slot["status"]     = "BAD"
                slot["bad_reason"] = "no URL"

            elif "encrypted-tbn" in url.lower():
                # Fast-path: we know these are small Google cache thumbnails.
                # Skip the download to avoid 150 unnecessary HTTP requests.
                slot["status"]     = "BAD"
                slot["bad_reason"] = "encrypted-tbn thumbnail"

            else:
                # Download and measure — only runs for non-thumbnail URLs
                try:
                    img_bytes = fetch_image(url)
                    w, h      = _image_size(img_bytes)
                    slot["old_resolution"] = (w, h)
                    if w == 0 or h == 0:
                        slot["status"]     = "BAD"
                        slot["bad_reason"] = "unreadable image"
                    elif min(w, h) < MIN_DIM:
                        slot["status"]     = "BAD"
                        slot["bad_reason"] = f"too small ({w}×{h}px)"
                except Exception:
                    slot["status"]     = "BAD"
                    slot["bad_reason"] = "download failed"

            # ── 3. Print audit row ────────────────────────────────────────────
            host     = domain_of(url) if url else "—"
            host_col = (host[:domain_w - 2] + "..") if len(host) > domain_w else host

            res_str  = "—"
            if slot["old_resolution"]:
                rw, rh  = slot["old_resolution"]
                res_str = f"{rw}×{rh}"

            status_col = slot["status"]
            if slot["bad_reason"] and slot["bad_reason"] != slot["status"].lower():
                status_col += f" ({slot['bad_reason']})"

            e_col = (q_name[:entry_w - 2] + "..") if len(q_name) > entry_w else q_name
            n_col = (name[:name_w - 2] + "..") if len(name) > name_w else name

            p(f"  {ei+1:>3}  {e_col:<{entry_w}}  {si+1:>2}  "
              f"{n_col:<{name_w}}  {host_col:<{domain_w}}  "
              f"{res_str:>{res_w}}  {status_col}")

            all_slots.append(slot)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok      = sum(1 for s in all_slots if s["status"] == "OK")
    n_bad     = sum(1 for s in all_slots if s["status"] == "BAD")
    n_missing = sum(1 for s in all_slots if s["status"] == "MISSING")

    p()
    p("  " + "─" * 50)
    p(f"  Total slots : {len(all_slots)}")
    p(f"  OK          : {n_ok}")
    p(f"  BAD         : {n_bad}")
    p(f"  MISSING     : {n_missing}")
    p()

    p("  Per-entry breakdown:")
    for ei, entry in enumerate(dataset):
        e_slots  = [s for s in all_slots if s["entry_idx"] == ei]
        n_bad_e  = sum(1 for s in e_slots if s["status"] != "OK")
        if n_bad_e == 0:
            flag = "✓ fully OK"
        else:
            flag = f"⚠ {n_bad_e}/{len(e_slots)} need repair"
        q = entry.get("query_name", f"entry_{ei}")
        p(f"    {ei+1:>2}. {q[:52]:<52} {flag}")

    p()

    # ── Query image audit ─────────────────────────────────────────────────────
    p("  " + "─" * 50)
    p("  QUERY IMAGE AUDIT (query_image_url per entry)")
    p("  " + "─" * 50)

    bad_queries = []
    q_hdr = f"  {'#':>3}  {'Entry':<30}  {'Type':<12}  Status"
    p(q_hdr)
    p("  " + "─" * (len(q_hdr) - 2))

    for ei, entry in enumerate(dataset):
        q_name = entry.get("query_name", f"entry_{ei}")
        url    = entry.get("query_image_url", "")

        if url.startswith("data:"):
            kind   = "data-uri"
            status = "OK"
            reason = None
        elif "encrypted-tbn" in url.lower():
            kind   = "http"
            status = "BAD"
            reason = "encrypted-tbn thumbnail"
        elif url.startswith("http"):
            kind = "http"
            # Download and measure
            try:
                img_bytes = fetch_image(url)
                w, h      = _image_size(img_bytes)
                if w == 0 or h == 0:
                    status = "BAD"
                    reason = "unreadable image"
                elif min(w, h) < MIN_DIM:
                    status = "BAD"
                    reason = f"too small ({w}×{h}px)"
                else:
                    status = "OK"
                    reason = None
            except Exception as ex:
                status = "BAD"
                reason = f"download failed: {ex}"
        else:
            kind   = "unknown"
            status = "BAD"
            reason = "no URL or unsupported scheme"

        e_col = (q_name[:28] + "..") if len(q_name) > 30 else q_name
        status_col = status if not reason else f"{status} ({reason})"
        p(f"  {ei+1:>3}  {e_col:<30}  {kind:<12}  {status_col}")

        if status != "OK":
            bad_queries.append({
                "entry_idx":  ei,
                "entry_name": q_name,
                "old_url":    url,
                "status":     status,
                "bad_reason": reason,
            })

    n_q_bad = len(bad_queries)
    p()
    p(f"  Query images OK  : {len(dataset) - n_q_bad}")
    p(f"  Query images BAD : {n_q_bad}")
    p()

    return all_slots, bad_queries


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — INTERACTIVE REPAIR
# ══════════════════════════════════════════════════════════════════════════════

def run_repair(dataset: list, all_slots: list, bad_queries: list,
               client: QdrantClient) -> list:
    """
    Walk through every BAD / MISSING slot and ask the user for a replacement URL.
    Then walks through bad query images and asks for replacements (no Qdrant writes).

    Per slot: maximum 2 URL attempts.
    If the user presses Enter at any prompt → skip slot immediately.
    If a URL fails validation or YOLO skips it → show reason → re-prompt once.

    Returns list of repair_records (only non-OK slots, each with 'action' key).
    """
    bad_slots = [s for s in all_slots if s["status"] in ("BAD", "MISSING")]

    if not bad_slots:
        p("  All 150 slots are OK — nothing to repair.")
        return []

    n_entries = len({s["entry_idx"] for s in bad_slots})
    p()
    p("=" * 72)
    p("  STEP 2 — INTERACTIVE REPAIR")
    p(f"  {len(bad_slots)} slots to repair across {n_entries} entries")
    p("  Press Enter at any prompt to skip that slot.")
    p("=" * 72)

    repair_records = []

    # Group by entry, preserving entry order
    by_entry = {}
    for slot in bad_slots:
        by_entry.setdefault(slot["entry_idx"], []).append(slot)

    for ei, slots in sorted(by_entry.items()):
        entry  = dataset[ei]
        q_name = entry.get("query_name", f"entry_{ei}")
        q_cat  = entry.get("query_category_tag", "")

        p()
        p(f"  {'─'*68}")
        p(f"  Entry {ei+1}/{len(dataset)}: \"{q_name}\"  [{q_cat}]")
        p(f"  {len(slots)} slot(s) need repair")
        p(f"  {'─'*68}")

        for slot in slots:
            si   = slot["slot_idx"]
            name = slot["name"]
            url  = slot["old_url"]

            p()
            p(f"  Slot {si+1}/5 — \"{name}\"")
            p(f"    Status  : {slot['status']}")
            if slot["bad_reason"]:
                p(f"    Reason  : {slot['bad_reason']}")
            if slot["old_resolution"]:
                rw, rh = slot["old_resolution"]
                p(f"    Old res : {rw}×{rh}px")
            if url:
                p(f"    Old URL : {url[:80]}")

            repaired = False

            for attempt in range(2):
                prompt = (
                    "\n  Paste replacement URL (or Enter to skip): "
                    if attempt == 0 else
                    "  Try a different URL (or Enter to skip this slot): "
                )
                try:
                    raw = input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    p("\n  [ABORT] Repair interrupted by user.")
                    # Save partial progress before exiting
                    _save_dataset(dataset)
                    sys.exit(0)

                if not raw:
                    p("  [SKIP] Slot skipped.")
                    break    # exit the attempts loop; repaired stays False

                # ── Validate + download ───────────────────────────────────────
                ok, reason, img_bytes, size = validate_and_download(raw)
                if not ok:
                    p(f"  [REJECT] {reason}")
                    continue     # will re-prompt if attempt == 0, else fall through

                # ── Send to /index-image ──────────────────────────────────────
                p(f"  Sending to /index-image…")
                try:
                    idx = call_index_image(img_bytes, name)
                except Exception as e:
                    p(f"  [ERROR] /index-image call failed: {e}")
                    continue

                if idx.get("skipped"):
                    reason = idx.get("skip_reason", "unknown")
                    p(f"  [YOLO SKIP] Visual engine skipped: {reason}")
                    p(f"             (YOLO found no usable bounding box on this image)")
                    continue

                # ── Success — update Qdrant ───────────────────────────────────
                vector     = idx["vector_normal"]
                category   = idx.get("category") or q_cat
                box_source = idx.get("box_source", "unknown")
                w, h       = size

                # Delete stale point (only if BAD — MISSING has no point to delete)
                if slot["status"] == "BAD" and slot["old_point_id"]:
                    try:
                        client.delete(
                            collection_name=COLLECTION,
                            points_selector=models.PointIdsList(
                                points=[slot["old_point_id"]]
                            ),
                        )
                    except Exception as e:
                        p(f"  [WARN] Could not delete old Qdrant point: {e}")

                # Upsert new point
                # Point ID = uuid5(image_url) — matches gateway convention
                # Payload product_id stays the same as before (golden:: uuid5)
                new_point_id = compute_point_id(raw)
                client.upsert(
                    collection_name=COLLECTION,
                    points=[PointStruct(
                        id      = new_point_id,
                        vector  = vector,
                        payload = {
                            "product_id":   slot["product_id"],
                            "name":         name,
                            "image_url":    raw,
                            "store_name":   "golden_dataset",
                            "mall_name":    "golden_dataset",
                            "category_tag": category,
                            "box_source":   box_source,
                            "price":        "",
                        },
                    )],
                )

                # Update golden_dataset.json immediately (crash-safe)
                dataset[ei]["relevant_info"][si]["image_url"] = raw
                _save_dataset(dataset)

                vec_norm = math.sqrt(sum(x * x for x in vector))
                p(f"  [OK] Slot {si+1} repaired ✓")
                p(f"       New URL    : {raw[:70]}")
                p(f"       New res    : {w}×{h}px")
                p(f"       Category   : {category}")
                p(f"       Box source : {box_source}")
                p(f"       Vector ‖v‖ : {vec_norm:.4f}")

                repair_records.append({
                    **{k: v for k, v in slot.items()
                       if k not in ("old_vector",)},   # keep slot metadata, drop heavy vector
                    "old_vector":     slot["old_vector"],   # re-add explicitly for Step 3
                    "new_url":        raw,
                    "new_resolution": [w, h],
                    "new_vector":     vector,
                    "new_category":   category,
                    "new_box_source": box_source,
                    "action":         "repaired",
                })
                repaired = True
                break
            # end attempt loop

            if not repaired:
                repair_records.append({
                    **slot,
                    "new_url":        None,
                    "new_resolution": None,
                    "new_vector":     None,
                    "new_category":   None,
                    "new_box_source": None,
                    "action":         "skipped",
                })

    # ── Query image repair ────────────────────────────────────────────────────
    if bad_queries:
        p()
        p("=" * 72)
        p("  QUERY IMAGE REPAIR")
        p(f"  {len(bad_queries)} query image(s) need replacement")
        p("  These are the search images used to evaluate recall.")
        p("  Provide a direct product image URL (≥400px). No Qdrant writes needed.")
        p("  Press Enter to skip.")
        p("=" * 72)

        for bq in bad_queries:
            ei     = bq["entry_idx"]
            q_name = bq["entry_name"]
            url    = bq["old_url"]

            p()
            p(f"  Entry {ei+1}/{len(dataset)}: \"{q_name}\"")
            p(f"    Reason  : {bq['bad_reason']}")
            if url and not url.startswith("data:"):
                p(f"    Old URL : {url[:80]}")

            for attempt in range(2):
                prompt = (
                    "\n  Paste replacement query image URL (or Enter to skip): "
                    if attempt == 0 else
                    "  Try a different URL (or Enter to skip): "
                )
                try:
                    raw = input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    p("\n  [ABORT] Repair interrupted.")
                    _save_dataset(dataset)
                    sys.exit(0)

                if not raw:
                    p("  [SKIP] Query image skipped.")
                    break

                ok, reason, img_bytes, size = validate_and_download(raw)
                if not ok:
                    p(f"  [REJECT] {reason}")
                    continue

                # Convert to base64 data URI so it's self-contained
                import mimetypes
                mime = mimetypes.guess_type(raw.split("?")[0])[0] or "image/jpeg"
                b64  = base64.b64encode(img_bytes).decode("ascii")
                data_uri = f"data:{mime};base64,{b64}"

                dataset[ei]["query_image_url"] = data_uri
                _save_dataset(dataset)

                w, h = size
                p(f"  [OK] Query image updated ✓  ({w}×{h}px)")
                break

    return repair_records


def _save_dataset(dataset: list):
    """Persist golden_dataset.json. Called after every successful repair."""
    with open(GOLDEN_PATH, "w") as fh:
        json.dump(dataset, fh, indent=2)
    p(f"  [SAVED] golden_dataset.json updated")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def run_final_report(dataset: list, all_slots: list, repair_records: list):
    """
    Print before/after cosine similarity for every repaired entry,
    then save mlops/repair_report.json.
    """
    n_ok     = sum(1 for s in all_slots if s["status"] == "OK")
    repaired = [r for r in repair_records if r["action"] == "repaired"]
    skipped  = [r for r in repair_records if r["action"] == "skipped"]

    p()
    p("=" * 72)
    p("  STEP 3 — FINAL REPORT")
    p("=" * 72)
    p()
    p(f"  Slots already OK  : {n_ok}")
    p(f"  Slots repaired    : {len(repaired)}")
    p(f"  Slots skipped     : {len(skipped)}")

    # ── Cosine similarity before / after ──────────────────────────────────────
    # Only run for entries that had at least one repair (one /vectorize call per entry)
    sim_results = {}     # (entry_idx, slot_idx) → {old_cosine_sim, new_cosine_sim}

    repaired_by_entry = {}
    for r in repaired:
        repaired_by_entry.setdefault(r["entry_idx"], []).append(r)

    if repaired_by_entry:
        p()
        p(f"  Cosine similarity check  ({len(repaired_by_entry)} entries):")

        for ei, recs in sorted(repaired_by_entry.items()):
            entry  = dataset[ei]
            q_name = entry.get("query_name", f"entry_{ei}")

            try:
                q_bytes = fetch_image(entry["query_image_url"])
                q_vec, _, _ = call_vectorize(q_bytes)
            except Exception as e:
                p(f"\n  [WARN] Could not vectorize query \"{q_name}\": {e}")
                continue
            if not q_vec:
                p(f"\n  [WARN] Empty vector for query \"{q_name}\"")
                continue

            p(f"\n  Entry: \"{q_name}\"")
            for rec in recs:
                key     = (ei, rec["slot_idx"])
                old_sim = (cosine_sim(q_vec, rec["old_vector"])
                           if rec.get("old_vector") else None)
                new_sim = (cosine_sim(q_vec, rec["new_vector"])
                           if rec.get("new_vector") else None)

                old_s = f"{old_sim:.4f}" if old_sim is not None else "N/A (was missing)"
                new_s = f"{new_sim:.4f}" if new_sim is not None else "N/A"

                delta_s = ""
                if old_sim is not None and new_sim is not None:
                    delta  = new_sim - old_sim
                    arrow  = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "→")
                    delta_s = f"   {arrow} Δ={delta:+.4f}"

                p(f"    Slot {rec['slot_idx']+1}: \"{rec['name'][:40]}\"")
                p(f"      Before: {old_s}  →  After: {new_s}{delta_s}")

                sim_results[key] = {
                    "old_cosine_sim": old_sim,
                    "new_cosine_sim": new_sim,
                }

    # ── Build and save repair_report.json ─────────────────────────────────────
    repair_lookup = {(r["entry_idx"], r["slot_idx"]): r for r in repair_records}

    report_slots = []
    for slot in all_slots:
        if slot["status"] == "OK":
            continue
        key  = (slot["entry_idx"], slot["slot_idx"])
        rec  = repair_lookup.get(key, {})
        sims = sim_results.get(key, {})

        report_slots.append({
            "entry":           slot["entry_name"],
            "entry_idx":       slot["entry_idx"],
            "slot_idx":        slot["slot_idx"],
            "product_id":      slot["product_id"],
            "slot_name":       slot["name"],
            "audit_status":    slot["status"],
            "bad_reason":      slot.get("bad_reason"),
            "old_url":         slot["old_url"],
            "old_resolution":  (list(slot["old_resolution"])
                                if slot["old_resolution"] else None),
            "old_cosine_sim":  sims.get("old_cosine_sim"),
            "action":          rec.get("action", "not_processed"),
            "new_url":         rec.get("new_url"),
            "new_resolution":  rec.get("new_resolution"),
            "new_cosine_sim":  sims.get("new_cosine_sim"),
            "new_category":    rec.get("new_category"),
            "new_box_source":  rec.get("new_box_source"),
        })

    report = {
        "run_at":  datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_slots": len(all_slots),
            "ok":          n_ok,
            "bad":         sum(1 for s in all_slots if s["status"] == "BAD"),
            "missing":     sum(1 for s in all_slots if s["status"] == "MISSING"),
            "repaired":    len(repaired),
            "skipped":     len(skipped),
        },
        "slots": report_slots,
    }

    with open(REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)

    p()
    p(f"  Repair report saved → {REPORT_PATH}")
    p()
    p("=" * 72)
    p()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p()
    p("=" * 72)
    p("  LOCUS GOLDEN DATASET REPAIR TOOL")
    p("=" * 72)

    # ── Pre-flight ────────────────────────────────────────────────────────────
    if not QDRANT_URL:
        p("[FATAL] QDRANT_URL is not set — check .env file")
        sys.exit(1)
    if not os.path.exists(GOLDEN_PATH):
        p(f"[FATAL] Golden dataset not found at: {GOLDEN_PATH}")
        sys.exit(1)

    p(f"  Qdrant URL  : {QDRANT_URL}")
    p(f"  Visual URL  : {VISUAL_URL}")
    p(f"  Collection  : {COLLECTION}")
    p(f"  Dataset     : {GOLDEN_PATH}")
    p()

    # ── Connect to Qdrant ─────────────────────────────────────────────────────
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    try:
        info  = client.get_collection(COLLECTION)
        total = (getattr(info, "vectors_count", None)
                 or getattr(info, "points_count", "?"))
        p(f"  Collection '{COLLECTION}': {total} vectors")
    except Exception as e:
        p(f"[FATAL] Cannot reach Qdrant: {e}")
        sys.exit(1)

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(GOLDEN_PATH) as fh:
        dataset = json.load(fh)
    p(f"  Dataset: {len(dataset)} entries")

    p("  Fetching all golden_dataset points from Qdrant (single scroll)…")
    t0           = time.time()
    qdrant_index = fetch_golden_index(client)
    p(f"  → {len(qdrant_index)} points fetched in {time.time() - t0:.1f}s")

    # ── Steps ─────────────────────────────────────────────────────────────────
    all_slots, bad_queries = run_audit(dataset, qdrant_index)
    repair_records         = run_repair(dataset, all_slots, bad_queries, client)
    run_final_report(dataset, all_slots, repair_records)


if __name__ == "__main__":
    main()
