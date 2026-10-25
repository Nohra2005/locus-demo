"""
Golden Dataset Report — Locus
==============================
Generates golden_dataset_report.html showing recall for each golden query.

For each query:
  • Full query image
  • Bounding box drawn on it (what YOLO detected)
  • The cropped image — what CLIP receives
  • Ground-truth relevant products (green border)
  • Top-5 search results with recall@5 hit indicator

Usage:
    cd mlops
    python visualize_golden_dataset.py [--gateway http://localhost:8000]

Output:
    mlops/golden_dataset_report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import re
import sys
import time
import threading

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except ImportError:
    PIL_OK = False

import httpx
import requests

HERE                = pathlib.Path(__file__).parent
GOLDEN              = HERE / "golden_dataset.json"
OUT_HTML            = HERE / "golden_dataset_report.html"
GOLDEN_IMAGES_DIR   = HERE / "golden_images"
CALIBRATION_RESULTS = HERE / "calibration_results.json"

CATEGORIES = [
    "top","pants","jacket","dress","shorts","skirt","sweater",
    "jumpsuit","leggings","sports_bra","shoes","bag","hat",
]


# ── Judge (inline, same model/prompt as gateway/judge.py) ─────────────────────

_JUDGE_MODEL   = "google/gemini-2.0-flash-001"
_JUDGE_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_JUDGE_PROMPT  = (
    "You are a fashion visual similarity expert.\n"
    "You will be shown two clothing images: first the QUERY image, then a RESULT image.\n"
    "Rate how visually similar the result is to the query.\n"
    "Give a score from 0.00 to 1.00 using exactly two decimal places.\n"
    "Anchor points:\n"
    "  1.00 = identical item\n"
    "  0.80 = very similar (same style, colour, silhouette)\n"
    "  0.60 = similar style (same category, close design)\n"
    "  0.40 = same category only\n"
    "  0.20 = loosely related\n"
    "  0.00 = unrelated\n"
    "Respond with ONLY the numeric score. No explanation."
)

_rate_lock = threading.Lock()
_last_call  = [0.0]


def _throttle():
    with _rate_lock:
        gap = 2.0 - (time.monotonic() - _last_call[0])
        if gap > 0:
            time.sleep(gap)
        _last_call[0] = time.monotonic()


def _score_pair(query_b64: str, result_b64: str, api_key: str) -> float | None:
    payload = {
        "model": _JUDGE_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _JUDGE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{result_b64}"}},
        ]}],
        "max_tokens": 50,
        "temperature": 0.0,
    }
    for attempt in range(3):
        _throttle()
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(_JUDGE_API_URL, json=payload,
                                   headers={"Authorization": f"Bearer {api_key}"})
            if resp.status_code == 429:
                wait = float(resp.headers.get("retry-after", 60))
                print(f"    [429] rate limited, waiting {wait:.0f}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\d+\.\d+|\d+", content)
            return float(m.group()) if m else None
        except Exception as e:
            if attempt < 2:
                time.sleep(5.0 * (2 ** attempt))
            else:
                print(f"    [judge error] {e}")
    return None


# ── Image helpers ──────────────────────────────────────────────────────────────

def get_image_bytes(url: str) -> bytes | None:
    try:
        if url.startswith("data:"):
            _, b64 = url.split(",", 1)
            return base64.b64decode(b64)
        # Fast-path: read local golden-dataset images directly from disk (no HTTP)
        if "golden-dataset/images/" in url:
            filename = url.rsplit("/", 1)[-1]
            local_path = GOLDEN_IMAGES_DIR / filename
            if local_path.exists():
                return local_path.read_bytes()
        # Fallback: HTTP fetch for external URLs
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Locus-Visualizer/1.0)"},
            timeout=(5, 15),  # (connect, read) — fail fast on dead hosts
            allow_redirects=True,
        )
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"    [warn] Cannot fetch {url[:60]}: {e}")
        return None


def draw_bbox_on_image(img_bytes: bytes, bbox: list) -> str | None:
    """Draw bounding box on image, return base64 data URI."""
    if not PIL_OK or not bbox or not img_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        lw = max(3, int(min(img.width, img.height) * 0.008))
        draw.rectangle([x1, y1, x2, y2], outline="#f59e0b", width=lw)
        # semi-transparent fill
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d2 = ImageDraw.Draw(overlay)
        d2.rectangle([x1, y1, x2, y2], fill=(245, 158, 11, 35))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"    [warn] bbox draw failed: {e}")
        return None


def crop_image(img_bytes: bytes, bbox: list) -> str | None:
    """Crop to bbox, return data URI."""
    if not PIL_OK or not bbox or not img_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"    [warn] crop failed: {e}")
        return None


def bytes_to_data_uri(img_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()



def make_img(src: str, w: int = 130, h: int = 160, extra: str = "") -> str:
    if not src:
        return f'<div class="no-img" style="width:{w}px;height:{h}px">no img</div>'
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>"
           f"<rect width='{w}' height='{h}' fill='#222'/>"
           f"<text x='50%' y='50%' text-anchor='middle' dy='.35em' "
           f"font-size='11' fill='#666'>no img</text></svg>")
    fb = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return (f'<img src="{src}" referrerpolicy="no-referrer" '
            f'style="width:{w}px;height:{h}px;object-fit:cover;border-radius:6px;{extra}" '
            f'onerror="this.onerror=null;this.src=\'{fb}\'">')


# ── Gateway calls ──────────────────────────────────────────────────────────────

def detect(img_bytes: bytes, gateway: str) -> list:
    try:
        r = requests.post(
            f"{gateway}/detect",
            files={"file": ("q.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            timeout=20,
        )
        r.raise_for_status()
        return r.json().get("detections", [])
    except Exception as e:
        print(f"    [warn] detect failed: {e}")
        return []





def search(img_bytes: bytes, gateway: str, label: str = "", k: int = 25) -> list:
    try:
        data = {"skip_judge": "true"}
        if label:
            data["search_label"] = label
        r = requests.post(
            f"{gateway}/search?include_golden=true",
            files={"file": ("q.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data=data,
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("matches", [])[:k]
    except Exception as e:
        print(f"    [warn] search failed: {e}")
        return []




# ── Select best bbox ───────────────────────────────────────────────────────────

_ALIAS = {
    "top": {"shirt","blouse","t-shirt","tshirt","tee","tank","polo","tunic","button"},
    "pants": {"trousers","jeans","chino","chinos","slacks","flare"},
    "jacket": {"coat","blazer","cardigan","hoodie","sweatshirt"},
    "dress": {"gown","midi","maxi","minidress"},
    "shorts": {"bermuda"},
    "shoes": {"sneaker","boot","heel","sandal","loafer","mule","pump","trainer"},
    "bag": {"purse","tote","backpack","clutch","satchel"},
    "hat": {"cap","beanie","beret","bucket"},
    "leggings": {"tights"},
    "skirt": {"miniskirt","midi skirt"},
}

def _best_bbox(detections: list, category: str) -> dict | None:
    if not detections:
        return None
    for d in detections:
        sl = (d.get("search_label") or "").lower()
        if sl == category:
            return d
    for d in detections:
        sl = (d.get("search_label") or "").lower()
        lb = (d.get("label") or "").lower()
        aliases = _ALIAS.get(category, set())
        if any(a in lb or a in sl for a in aliases):
            return d
    # No match — return None so the full image is used rather than a wrong bbox
    return None


# ── HTML template ──────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0f0e0c; color: #e0d8cc; min-width: 960px; font-size: 13px;
}

.topbar {
  background: #1a1814; border-bottom: 1px solid #2a2620;
  padding: 16px 32px; display: flex; align-items: center; gap: 16px;
}
.topbar h1 { font-size: 18px; font-weight: 700; color: #c9a96e; }
.topbar p  { font-size: 11px; color: #6b6458; flex: 1; }
.topbar .pill {
  background: #2a2620; border: 1px solid #3a3530;
  border-radius: 20px; padding: 4px 12px; font-size: 11px; color: #c9a96e;
}

.summary { background: #161410; margin: 20px 28px 0; border-radius: 10px;
           border: 1px solid #2a2620; overflow-x: auto; }
.summary h2 { padding: 12px 18px; font-size: 12px; letter-spacing: .08em;
              text-transform: uppercase; color: #6b6458;
              border-bottom: 1px solid #2a2620; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { background: #1a1814; padding: 8px 12px; text-align: left;
     font-weight: 600; color: #6b6458; border-bottom: 1px solid #2a2620;
     white-space: nowrap; }
td { padding: 7px 12px; border-bottom: 1px solid #1e1c19;
     vertical-align: middle; }
tr:hover td { background: #1a1814; }
.anchor-link { color: #c9a96e; text-decoration: none; }
.anchor-link:hover { text-decoration: underline; }

.queries { padding: 20px 28px 60px; display: flex; flex-direction: column; gap: 24px; }

.qcard { background: #161410; border: 1px solid #2a2620;
         border-radius: 12px; overflow: hidden; }
.qcard-header {
  display: flex; align-items: center; flex-wrap: wrap; gap: 10px;
  padding: 12px 18px; border-bottom: 1px solid #2a2620;
  background: #1a1814;
}
.qcard-header h3 { font-size: 14px; font-weight: 700; color: #e0d8cc; flex: 1; }
.cat-pill { background: #2a2620; border-radius: 20px; padding: 3px 10px;
            font-size: 10px; color: #6b6458; }

.qcard-body {
  display: grid;
  grid-template-columns: 148px 148px 1fr;
  min-height: 220px;
}
.col-img { padding: 14px 10px; border-right: 1px solid #2a2620;
           display: flex; flex-direction: column; align-items: center; gap: 8px; }
.col-lbl { font-size: 9px; text-transform: uppercase; letter-spacing: .08em;
           color: #4a4540; text-align: center; }
.col-lbl b { display: block; color: #6b6458; font-size: 10px;
             text-transform: none; letter-spacing: 0; margin-top: 2px; }

.col-panels { padding: 14px 18px; display: flex; flex-direction: column; gap: 14px; }
.panel-lbl { font-size: 9px; text-transform: uppercase; letter-spacing: .08em;
             color: #4a4540; margin-bottom: 8px;
             border-bottom: 1px solid #2a2620; padding-bottom: 4px; }

.row { display: flex; gap: 10px; flex-wrap: wrap; }
.ritem { display: flex; flex-direction: column; align-items: center; gap: 4px;
         position: relative; flex-shrink: 0; }
.rank { position: absolute; top: 3px; left: 3px; background: rgba(0,0,0,.7);
        color: #e0d8cc; font-size: 9px; font-weight: 700;
        padding: 2px 5px; border-radius: 4px; z-index: 1; }
.nm { font-size: 10px; color: #6b6458; text-align: center; max-width: 120px;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.gt-lbl { font-size: 9px; font-weight: 700; color: #7aab8a; }

.badge { display: inline-flex; align-items: center; gap: 3px;
         padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.badge small { font-weight: 400; opacity: .8; }
.green  { background: #1a3a25; color: #7aab8a; border: 1px solid #2a4a35; }
.lime   { background: #243520; color: #a3c96e; border: 1px solid #344525; }
.yellow { background: #3a2e10; color: #c9a96e; border: 1px solid #4a3e20; }
.orange { background: #3a2010; color: #c97040; border: 1px solid #4a3020; }
.red    { background: #3a1010; color: #c97070; border: 1px solid #4a2020; }
.gray   { background: #2a2620; color: #6b6458; border: 1px solid #3a3530; }

.hit-yes { color: #7aab8a; font-weight: 700; }
.hit-no  { color: #c97070; }

.meta { font-size: 11px; color: #4a4540; display: flex; flex-wrap: wrap; gap: 8px; }
.meta b { color: #6b6458; }
.no-img { background: #1a1814; display: flex; align-items: center;
          justify-content: center; font-size: 10px; color: #3a3530;
          border-radius: 6px; text-align: center; border: 1px solid #2a2620; }

.warn-banner {
  background: #3a2e10; border-left: 3px solid #c9a96e;
  padding: 8px 16px; font-size: 11px; color: #c9a96e;
}

.recall-pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;
}

.judge-badge {
  font-size: 10px; font-weight: 700; padding: 2px 6px;
  border-radius: 10px; display: inline-block; line-height: 1.4;
}
.judge-pass  { background: #1a3a25; color: #7aab8a; border: 1px solid #2a4a35; }
.judge-warn  { background: #3a2e10; color: #c9a96e; border: 1px solid #4a3e20; }
.judge-fail  { background: #3a1010; color: #c97070; border: 1px solid #4a2020; }
.judge-none  { background: #2a2620; color: #4a4540; border: 1px solid #3a3530; }
.judge-avg-pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;
}
"""


def _load_calibration() -> dict[str, dict]:
    """Return {query_name: {product_id: score, ..., '_avg': float, '_label': str}}."""
    if not CALIBRATION_RESULTS.exists():
        return {}
    try:
        data = json.loads(CALIBRATION_RESULTS.read_text(encoding="utf-8"))
        out = {}
        for entry in data:
            name = entry.get("query_name", "")
            scores = dict(entry.get("scores_by_id", {}))
            scores["_avg"]   = entry.get("avg_score")
            scores["_label"] = entry.get("label", "")
            out[name] = scores
        return out
    except Exception as e:
        print(f"  [warn] Could not load calibration_results.json: {e}")
        return {}


def _judge_badge(score: float | None) -> str:
    if score is None:
        return '<span class="judge-badge judge-none">no score</span>'
    cls = "judge-pass" if score >= 0.80 else ("judge-warn" if score >= 0.65 else "judge-fail")
    return f'<span class="judge-badge {cls}">judge {score:.2f}</span>'


def render(entries: list, results_map: dict, gateway_url: str = "http://localhost:8000") -> str:
    calib = _load_calibration()
    if calib:
        print(f"  Loaded calibration scores for {len(calib)} queries.")
    n = len(entries)
    recalls = [results_map[e["query_name"]]["recall5"] for e in entries
               if e["query_name"] in results_map]
    avg_recall = round(sum(recalls) / len(recalls), 3) if recalls else "—"
    hits_total = sum(results_map[e["query_name"]]["hits5"] for e in entries
                     if e["query_name"] in results_map)
    gt_total   = sum(len(e.get("relevant_product_ids", [])) for e in entries)

    # sort: worst recall first
    entries_sorted = sorted(
        entries,
        key=lambda e: results_map.get(e["query_name"], {}).get("recall5", 1.0)
    )

    # ── summary table ─────────────────────────────────────────────────────────
    sum_rows = []
    for e in entries_sorted:
        name = e.get("query_name", "?")
        cat  = e.get("query_category_tag", "")
        r    = results_map.get(name, {})
        rec5 = r.get("recall5")
        hits = r.get("hits5", 0)
        gt_n = len(e.get("relevant_product_ids", []))
        anchor = name.replace(" ", "_")

        rec_html = "—"
        if rec5 is not None:
            pct = int(rec5 * 100)
            cls = "green" if pct >= 60 else ("yellow" if pct >= 40 else "red")
            rec_html = f'<span class="badge {cls}">{hits}/{gt_n} ({pct}%)</span>'

        tier = r.get("bbox_tier", "—")
        label = r.get("search_label", "—")
        sum_rows.append(
            f"<tr><td><a class='anchor-link' href='#{anchor}'>{name}</a></td>"
            f"<td>{cat}</td><td>{rec_html}</td>"
            f"<td>{tier}</td><td>{label}</td></tr>"
        )

    # ── per-query cards ───────────────────────────────────────────────────────
    cards = []
    for i, e in enumerate(entries_sorted, 1):
        name      = e.get("query_name", "?")
        cat       = e.get("query_category_tag", "")
        query_url = e.get("query_image_url", "")
        gt_prods  = e.get("relevant_info", [])
        gt_ids    = set(e.get("relevant_product_ids", []))
        anchor    = name.replace(" ", "_")
        r         = results_map.get(name, {})
        rec5      = r.get("recall5")
        hits5     = r.get("hits5", 0)
        gt_n      = len(gt_ids)
        bbox_img  = r.get("bbox_img_src")   # query with box drawn
        crop_src  = r.get("crop_src")       # cropped region
        tier      = r.get("bbox_tier", "—")
        label     = r.get("search_label", "")
        bbox         = r.get("bbox")
        gt_bbox_imgs = r.get("gt_bbox_imgs", {})
        result_records = r.get("results", [])
        calib_entry  = calib.get(name, {})

        print(f"  [{i:2d}/{n}] rendering card: {name[:50]}")

        # ── recall pill ───────────────────────────────────────────────────────
        rec_html = "—"
        if rec5 is not None:
            pct = int(rec5 * 100)
            cls = "green" if pct >= 60 else ("yellow" if pct >= 40 else "red")
            rec_html = (f'<span class="recall-pill badge {cls}">'
                        f'Recall@5: {hits5}/{gt_n} ({pct}%)</span>')

        # ── judge avg pill ────────────────────────────────────────────────────
        judge_avg_html = ""
        if calib_entry:
            avg_s = calib_entry.get("_avg")
            lbl_s = calib_entry.get("_label", "")
            if avg_s is not None:
                jcls = "green" if lbl_s == "PASS" else ("yellow" if lbl_s == "WARN" else "red")
                judge_avg_html = (f'<span class="judge-avg-pill badge {jcls}">'
                                  f'Judge avg: {avg_s:.2f} [{lbl_s}]</span>')

        # ── left col: query + bbox ────────────────────────────────────────────
        full_src = query_url if query_url.startswith("data:") else query_url
        bbox_label = (f"bbox: {[int(v) for v in bbox]}" if bbox
                      else "full image (no bbox)")
        left_col = f"""
    <div class="col-img">
      <div class="col-lbl">Full query<br><b>{bbox_label}</b></div>
      {make_img(bbox_img or full_src, 124, 155)}
    </div>"""

        # ── middle col: crop ─────────────────────────────────────────────────
        if crop_src:
            crop_label = "Crop sent to CLIP"
        elif bbox:
            crop_label = "Crop (PIL unavailable)"
        else:
            crop_label = "Full image (no bbox)"
        mid_col = f"""
    <div class="col-img">
      <div class="col-lbl">{crop_label}</div>
      {make_img(crop_src or full_src, 124, 155)}
    </div>"""


        # ── ground-truth items ────────────────────────────────────────────────
        gt_html = ""
        for p in gt_prods:
            nm    = (p.get("name") or "")[:30]
            pid   = p.get("product_id", "")
            img   = gt_bbox_imgs.get(pid) or p.get("image_url", "")
            jscore = calib_entry.get(pid) if calib_entry else None
            gt_html += f"""
      <div class="ritem">
        {make_img(img, 118, 140, extra="outline:2px solid #7aab8a;outline-offset:2px;")}
        <span class="nm" title="{p.get('name','')}">{nm}</span>
        <span class="gt-lbl">ground truth</span>
        {_judge_badge(jscore)}
      </div>"""
        if not gt_html:
            gt_html = '<span style="font-size:11px;color:#4a4540;">No GT images</span>'

        # ── search results ────────────────────────────────────────────────────
        res_html = ""
        for rec in result_records[:5]:
            rank      = rec.get("rank", "?")
            img       = rec.get("bbox_img_src") or rec.get("image_url", "")
            nm        = (rec.get("name") or "")[:30]
            pid       = rec.get("product_id", "")
            is_gt     = pid in gt_ids
            border    = "outline:3px solid #7aab8a;outline-offset:2px;" if is_gt else ""
            gt_lbl    = '<span class="gt-lbl">✓ GT match</span>' if is_gt else ""
            jscore    = rec.get("judge_score")
            res_html += f"""
      <div class="ritem">
        <span class="rank">#{rank}</span>
        {make_img(img, 118, 140, extra=border)}
        {gt_lbl}
        <span class="nm" title="{rec.get('name','')}">{nm}</span>
        {_judge_badge(jscore)}
      </div>"""
        if not res_html:
            res_html = '<span style="font-size:11px;color:#4a4540;">Run the script to populate results.</span>'

        # ── meta ──────────────────────────────────────────────────────────────
        meta_parts = [f"tier: <b>{tier}</b>", f"search_label: <b>{label or '—'}</b>"]
        if bbox:
            meta_parts.append(f"bbox: <b>[{', '.join(str(int(v)) for v in bbox)}]</b>")

        cards.append(f"""
  <div class="qcard" id="{anchor}">
    <div class="qcard-header">
      <h3>{name}</h3>
      <span class="cat-pill">{cat}</span>
      {rec_html}
      {judge_avg_html}
    </div>
    <div class="qcard-body">
      {left_col}
      {mid_col}
      <div class="col-panels">
        <div>
          <div class="panel-lbl">Ground-truth relevant products ({len(gt_prods)} annotated)</div>
          <div class="row">{gt_html}</div>
        </div>
        <div>
          <div class="panel-lbl">Top-5 search results (green border = ground-truth match)</div>
          <div class="row">{res_html}</div>
        </div>
        <div class="meta">{" · ".join(f"<span>{m}</span>" for m in meta_parts)}</div>
      </div>
    </div>
  </div>""")

    cats_options = "".join(
        f'<option value="{c}">{c}</option>' for c in CATEGORIES
    )
    existing_names = [e.get("query_name","") for e in entries_sorted]

    # Build delete options: duplicate names get their date appended so each is selectable
    from collections import Counter as _Counter
    _name_counts = _Counter(e.get("query_name","") for e in entries_sorted)
    delete_options = "".join(
        f'<option value="{e.get("query_name","")}|{e.get("created_at","")}">'
        f'{e.get("query_name","")}{"  (" + e.get("created_at","")[:10] + ")" if _name_counts[e.get("query_name","")] > 1 else ""}'
        f'</option>'
        for e in entries_sorted
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Locus — Golden Dataset Report</title>
  <style>{CSS}

/* ── Add/Replace panel ─────────────────────────────── */
.fab {{
  position: fixed; bottom: 28px; right: 28px; z-index: 500;
  background: #c9a96e; color: #0f0e0c;
  border: none; border-radius: 28px;
  padding: 12px 22px; font-size: 14px; font-weight: 700;
  cursor: pointer; box-shadow: 0 4px 20px rgba(0,0,0,.5);
  transition: transform .15s, box-shadow .15s;
}}
.fab:hover {{ transform: translateY(-2px); box-shadow: 0 6px 24px rgba(0,0,0,.6); }}

.overlay {{
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.7); z-index: 600;
  align-items: center; justify-content: center; padding: 20px;
}}
.overlay.open {{ display: flex; }}

.panel {{
  background: #161410; border: 1px solid #2a2620; border-radius: 16px;
  width: 100%; max-width: 640px; max-height: 90vh;
  overflow-y: auto; padding: 28px;
  display: flex; flex-direction: column; gap: 18px;
}}
.panel h2 {{ font-size: 16px; font-weight: 700; color: #c9a96e; }}
.panel .close-btn {{
  position: absolute; top: 16px; right: 20px;
  background: none; border: none; color: #6b6458;
  font-size: 22px; cursor: pointer; line-height: 1;
}}
.panel label {{ font-size: 11px; color: #6b6458; display: block; margin-bottom: 4px; }}
.panel input, .panel select, .panel textarea {{
  width: 100%; padding: 8px 12px; border-radius: 8px;
  background: #0f0e0c; border: 1px solid #2a2620;
  color: #e0d8cc; font-size: 13px; font-family: inherit; outline: none;
}}
.panel input:focus, .panel select:focus {{ border-color: #c9a96e; }}
.rel-row {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }}
.rel-row img {{ width: 40px; height: 48px; object-fit: cover; border-radius: 6px;
               border: 1px solid #2a2620; flex-shrink: 0; }}
.rel-row .num {{ font-size: 11px; color: #4a4540; width: 18px; flex-shrink: 0; }}
.panel .submit-btn {{
  padding: 10px 24px; border-radius: 20px; font-size: 13px; font-weight: 700;
  background: #c9a96e; color: #0f0e0c; border: none; cursor: pointer; align-self: flex-end;
}}
.panel .submit-btn:disabled {{ opacity: .5; cursor: not-allowed; }}
.msg {{ padding: 10px 14px; border-radius: 8px; font-size: 12px; }}
.msg.ok  {{ background: rgba(122,171,138,.12); color: #7aab8a;
            border: 1px solid #7aab8a; }}
.msg.err {{ background: rgba(201,112,112,.12); color: #c97070;
            border: 1px solid #c97070; }}
.qimg-preview {{
  display: flex; align-items: flex-start; gap: 12px; margin-top: 8px;
}}
.qimg-preview img {{ width: 80px; height: 96px; object-fit: cover;
                     border-radius: 8px; border: 1px solid #2a2620; flex-shrink: 0; }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  </style>
</head>
<body>

<div class="topbar">
  <h1>Locus — Golden Dataset Report</h1>
  <p>{n} queries · sorted worst→best Recall@5</p>
  <span class="pill">Recall@5 avg: {avg_recall}</span>
  <span class="pill">Hits: {hits_total} / {gt_total}</span>
</div>

<div class="summary">
  <h2>Summary</h2>
  <table>
    <thead>
      <tr><th>Query</th><th>Category</th><th>Recall@5</th>
          <th>BBox tier</th><th>Search label</th></tr>
    </thead>
    <tbody>{"".join(sum_rows)}</tbody>
  </table>
</div>

<div class="queries">{"".join(cards)}</div>

<!-- ── Floating action button ─────────────────────── -->
<button class="fab" onclick="openPanel()">+ Add / Replace entry</button>

<!-- ── Add / Replace panel ───────────────────────── -->
<div class="overlay" id="overlay" onclick="maybeClose(event)">
  <div class="panel" style="position:relative;">
    <button class="close-btn" onclick="closePanel()">×</button>
    <h2>Add or replace a golden dataset entry</h2>

    <!-- Mode toggle -->
    <div>
      <label>Mode</label>
      <div style="display:flex;gap:10px;margin-top:4px;">
        <label style="display:flex;align-items:center;gap:6px;color:#e0d8cc;cursor:pointer;">
          <input type="radio" name="mode" value="add" checked onchange="toggleMode(this)"> Add new entry
        </label>
        <label style="display:flex;align-items:center;gap:6px;color:#e0d8cc;cursor:pointer;">
          <input type="radio" name="mode" value="replace" onchange="toggleMode(this)"> Replace existing
        </label>
        <label style="display:flex;align-items:center;gap:6px;color:#e0d8cc;cursor:pointer;">
          <input type="radio" name="mode" value="delete" onchange="toggleMode(this)"> Delete entry
        </label>
      </div>
    </div>

    <!-- Delete: pick entry to delete -->
    <div id="delete-row" style="display:none;">
      <label>Select entry to delete</label>
      <select id="delete-select" onchange="document.getElementById('delete-confirm').style.display=this.value?'':'none'">
        <option value="">— pick a query —</option>
        {delete_options}
      </select>
      <div id="delete-confirm" style="display:none;margin-top:12px;padding:10px;background:#2a1a1a;border:1px solid #7f1d1d;border-radius:6px;font-size:13px;color:#fca5a5;">
        This will permanently remove the entry from golden_dataset.json and delete its indexed images from Qdrant.
      </div>
    </div>

    <!-- Replace: pick existing entry -->
    <div id="replace-row" style="display:none;">
      <label>Select entry to replace</label>
      <select id="replace-select">
        <option value="">— pick a query —</option>
        {"".join(f'<option value="{name}">{name}</option>' for name in existing_names)}
      </select>
    </div>

    <!-- Entry fields (hidden in delete mode) -->
    <div id="entry-fields">

    <!-- Query image -->
    <div>
      <label>Query image *</label>
      <div class="qimg-preview">
        <img id="qpreview" src="" style="display:none;" referrerpolicy="no-referrer">
        <div style="flex:1;display:flex;flex-direction:column;gap:8px;">
          <input id="query-url" type="text" placeholder="Paste image URL or data URI…"
                 oninput="previewQuery()">
          <div style="display:flex;gap:8px;align-items:center;">
            <span style="font-size:11px;color:#4a4540;">or</span>
            <button onclick="document.getElementById('file-in').click()"
                    style="padding:5px 14px;border-radius:20px;font-size:11px;
                           background:#1c1a16;border:1px solid #2a2620;
                           color:#e0d8cc;cursor:pointer;">Upload file</button>
            <input id="file-in" type="file" accept="image/*" style="display:none"
                   onchange="handleFile(this)">
          </div>
        </div>
      </div>
    </div>

    <!-- Name + Category -->
    <div class="pair">
      <div>
        <label>Query name *</label>
        <input id="query-name" type="text" placeholder="e.g. wide leg linen trousers">
      </div>
      <div>
        <label>Category</label>
        <select id="query-cat">
          <option value="">— select —</option>
          {cats_options}
        </select>
      </div>
    </div>

    <!-- Relevant images -->
    <div>
      <label style="margin-bottom:10px;">Ground-truth relevant images (up to 5)</label>
      <div id="rel-rows">
        {"".join(f'''
        <div class="rel-row" id="rrow-{i}">
          <span class="num">{i+1}.</span>
          <img id="rprev-{i}" src="" style="display:none;" referrerpolicy="no-referrer"
               onerror="this.style.display='none'">
          <input type="text" id="rurl-{i}" placeholder="Relevant image {i+1} URL"
                 oninput="previewRel({i})" style="flex:2;">
          <input type="text" id="rname-{i}" placeholder="Name (optional)" style="flex:1;">
        </div>''' for i in range(5))}
      </div>
    </div>

    </div><!-- /entry-fields -->

    <!-- Message -->
    <div id="msg" style="display:none;"></div>

    <button class="submit-btn" id="submit-btn" onclick="submitEntry()">
      Add to dataset
    </button>
  </div>
</div>

<script>
const GATEWAY = "{gateway_url}";
let mode = "add";

function openPanel() {{ document.getElementById("overlay").classList.add("open"); }}
function closePanel() {{ document.getElementById("overlay").classList.remove("open"); }}
function maybeClose(e) {{ if (e.target.id === "overlay") closePanel(); }}

function toggleMode(el) {{
  mode = el.value;
  document.getElementById("replace-row").style.display = mode === "replace" ? "" : "none";
  document.getElementById("delete-row").style.display  = mode === "delete"  ? "" : "none";
  document.getElementById("entry-fields").style.display = mode === "delete" ? "none" : "";
  document.getElementById("delete-confirm").style.display = "none";
  const btnText = mode === "replace" ? "Replace entry" : mode === "delete" ? "Delete entry" : "Add to dataset";
  document.getElementById("submit-btn").textContent = btnText;
  document.getElementById("submit-btn").style.background = mode === "delete" ? "#dc2626" : "";

  if (mode === "replace") {{
    const sel = document.getElementById("replace-select").value;
    if (sel) document.getElementById("query-name").value = sel;
    document.getElementById("replace-select").onchange = function() {{
      document.getElementById("query-name").value = this.value;
    }};
  }}
}}

function previewQuery() {{
  const url = document.getElementById("query-url").value.trim();
  const img = document.getElementById("qpreview");
  img.src = url;
  img.style.display = url ? "" : "none";
}}

function handleFile(input) {{
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {{
    document.getElementById("query-url").value = e.target.result;
    const img = document.getElementById("qpreview");
    img.src = e.target.result;
    img.style.display = "";
  }};
  reader.readAsDataURL(file);
}}

function previewRel(i) {{
  const url = document.getElementById("rurl-" + i).value.trim();
  const img = document.getElementById("rprev-" + i);
  img.src = url;
  img.style.display = url ? "" : "none";
}}

function showMsg(text, type) {{
  const el = document.getElementById("msg");
  el.textContent = text;
  el.className = "msg " + type;
  el.style.display = "";
}}

async function submitEntry() {{
  const btn = document.getElementById("submit-btn");

  // ── Delete mode ──────────────────────────────────────────────────────────────
  if (mode === "delete") {{
    const raw = document.getElementById("delete-select").value;
    if (!raw) {{ showMsg("Select an entry to delete.", "err"); return; }}
    const [queryName, createdAt] = raw.split("|");
    btn.disabled = true;
    btn.textContent = "Deleting…";
    showMsg(`Deleting "${{queryName}}"…`, "ok");
    try {{
      const res = await fetch(GATEWAY + "/golden-dataset/entry", {{
        method: "DELETE",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ query_name: queryName, created_at: createdAt || "" }}),
      }});
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
      showMsg(`✓ Deleted "${{queryName}}" (${{data.points_removed}} images removed). Regenerating report…`, "ok");
      btn.textContent = "Regenerating…";
      try {{
        const regen = await fetch(GATEWAY + "/golden-dataset/regenerate", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{}}),
        }});
        if (!regen.ok) {{
          const rd = await regen.json().catch(() => ({{}}));
          throw new Error(rd.detail || "Regeneration failed");
        }}
      }} catch(regenErr) {{
        showMsg(`✓ Deleted, but report regeneration failed: ${{regenErr.message}}. Re-run visualize_golden_dataset.py manually.`, "err");
        btn.disabled = false;
        btn.textContent = "Delete entry";
        return;
      }}
      showMsg("✓ Done — reloading…", "ok");
      setTimeout(() => window.location.reload(), 800);
    }} catch(e) {{
      showMsg("Error: " + e.message, "err");
      btn.disabled = false;
      btn.textContent = "Delete entry";
    }}
    return;
  }}

  // ── Add / Replace mode ───────────────────────────────────────────────────────
  const queryUrl  = document.getElementById("query-url").value.trim();
  const queryName = document.getElementById("query-name").value.trim();
  const queryCat  = document.getElementById("query-cat").value;
  const relevant  = [];
  for (let i = 0; i < 5; i++) {{
    const url = document.getElementById("rurl-" + i).value.trim();
    const nm  = document.getElementById("rname-" + i).value.trim();
    if (url) relevant.push({{ url, name: nm || queryName }});
  }}

  if (!queryUrl)  {{ showMsg("Query image is required.", "err"); return; }}
  if (!queryName) {{ showMsg("Query name is required.", "err"); return; }}
  if (!relevant.length) {{ showMsg("Add at least one relevant image URL.", "err"); return; }}

  btn.disabled = true;
  btn.textContent = "Indexing…";
  showMsg("Indexing images — this may take 20–60s…", "ok");

  try {{
    const res = await fetch(GATEWAY + "/golden-dataset/entry", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ query_image_url: queryUrl, query_name: queryName,
                              query_category: queryCat, relevant,
                              replace: mode === "replace" }}),
    }});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));

    showMsg(
      `✓ Saved "${{queryName}}" — ${{data.indexed}} indexed, ${{data.skipped}} skipped. Regenerating report…`,
      "ok"
    );
    btn.textContent = "Regenerating…";

    if (!data.report_regenerated) {{
      try {{
        const regen = await fetch(GATEWAY + "/golden-dataset/regenerate", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ query_names: [queryName] }}),
        }});
        if (!regen.ok) {{
          const rd = await regen.json().catch(() => ({{}}));
          throw new Error(rd.detail || "Regeneration failed");
        }}
      }} catch(regenErr) {{
        showMsg(
          `✓ Saved, but report regeneration failed: ${{regenErr.message}}. Reload the page manually after re-running visualize_golden_dataset.py.`,
          "err"
        );
        btn.disabled = false;
        btn.textContent = mode === "replace" ? "Replace entry" : "Add to dataset";
        return;
      }}
    }}

    showMsg("✓ Done — reloading…", "ok");
    window.location.reload();

  }} catch(e) {{
    showMsg("Error: " + e.message, "err");
    btn.disabled = false;
    btn.textContent = mode === "replace" ? "Replace entry" : "Add to dataset";
  }}
}}
</script>

</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://localhost:8000",
                        type=lambda s: s.strip())
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N golden-dataset entries")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        print("OPENROUTER_API_KEY found — judge scores will be shown on search results.")
    else:
        print("No OPENROUTER_API_KEY — search result judge scores will be skipped.")

    if not CALIBRATION_RESULTS.exists():
        print("⚠  calibration_results.json not found — run calibrate_judge.py first to see scores on GT images.")

    if not GOLDEN.exists():
        print(f"ERROR: {GOLDEN} not found"); sys.exit(1)

    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    if args.limit:
        golden = golden[: args.limit]
    print(f"Loaded {len(golden)} entries from golden_dataset.json")

    results_map: dict = {}

    for i, entry in enumerate(golden, 1):
        name    = entry.get("query_name", f"query_{i}")
        cat     = entry.get("query_category_tag", "")
        url     = entry.get("query_image_url", "")
        gt_ids  = set(entry.get("relevant_product_ids", []))
        print(f"\n[{i:2d}/{len(golden)}] {name}")

        img_bytes = get_image_bytes(url)
        if not img_bytes:
            print("  [skip] Cannot load query image")
            results_map[name] = {"recall5": 0, "hits5": 0, "results": []}
            continue

        # Detect bounding box
        detections = detect(img_bytes, args.gateway)
        best = _best_bbox(detections, cat)
        bbox = best["bbox"] if best else None
        tier = ("exact" if best and best.get("search_label") == cat else
                "alias" if best else "full-image")
        sl   = best.get("search_label", "") if best else ""

        # Draw bbox on image
        bbox_img_src = draw_bbox_on_image(img_bytes, bbox) if bbox else None

        # Crop (what CLIP actually sees)
        if bbox:
            crop_bytes = None
            if PIL_OK:
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cropped = img.crop((x1, y1, x2, y2))
                    buf = io.BytesIO()
                    cropped.save(buf, format="JPEG", quality=88)
                    crop_bytes = buf.getvalue()
                except Exception:
                    crop_bytes = None
        else:
            crop_bytes = img_bytes

        crop_src  = bytes_to_data_uri(crop_bytes) if crop_bytes else None

        # Search — always use the full query image so the gateway can run its own
        # YOLO crop pipeline (same as evaluate_recall.py). Sending a pre-cropped
        # image causes YOLO to mis-detect on the tight crop and tanks recall.
        matches = search(img_bytes, args.gateway, cat)
        print(f"  Search returned {len(matches)} results")

        # Recall@5
        top5_ids = {m.get("product_id") for m in matches[:5]}
        hits5    = len(gt_ids & top5_ids)
        recall5  = hits5 / len(gt_ids) if gt_ids else 0.0
        print(f"  Recall@5: {hits5}/{len(gt_ids)} ({recall5*100:.0f}%)")

        query_b64 = base64.b64encode(img_bytes).decode("utf-8")

        result_records = []
        for rank, m in enumerate(matches[:5], 1):
            judge_score = None
            if api_key:
                res_url   = m.get("image_url", "")
                res_bytes = get_image_bytes(res_url) if res_url else None
                if res_bytes:
                    res_b64     = base64.b64encode(res_bytes).decode("utf-8")
                    judge_score = _score_pair(query_b64, res_b64, api_key)
                    lbl = f"{judge_score:.2f}" if judge_score is not None else "none"
                    print(f"    rank #{rank} '{m.get('name','')[:30]}' → judge {lbl}")
            result_records.append({
                "rank":        rank,
                "image_url":   m.get("image_url", ""),
                "name":        m.get("name", ""),
                "product_id":  m.get("product_id", ""),
                "store_name":  m.get("store_name", ""),
                "judge_score": judge_score,
            })

        # Detect + draw bbox on ground-truth images
        gt_bboxes: dict    = {}
        gt_bbox_imgs: dict = {}
        for p in entry.get("relevant_info", []):
            pid = p.get("product_id", "")
            gt_url = p.get("image_url", "")
            if not gt_url:
                continue
            gt_rb = get_image_bytes(gt_url)
            if not gt_rb:
                continue
            gt_dets = detect(gt_rb, args.gateway)
            gt_best = _best_bbox(gt_dets, cat)
            gt_bbox = gt_best["bbox"] if gt_best else None
            gt_bbox_src = draw_bbox_on_image(gt_rb, gt_bbox) if gt_bbox else None
            if gt_bbox:
                gt_bboxes[pid] = gt_bbox
            if gt_bbox_src:
                gt_bbox_imgs[pid] = gt_bbox_src

        results_map[name] = {
            "recall5":        recall5,
            "hits5":          hits5,
            "bbox_tier":      tier,
            "search_label":   sl,
            "bbox":           bbox,
            "bbox_img_src":   bbox_img_src,
            "crop_src":       crop_src,
            "results":        result_records,
            "gt_bboxes":      gt_bboxes,
            "gt_bbox_imgs":   gt_bbox_imgs,
        }

    print("\nRendering HTML report...")
    html = render(golden, results_map, gateway_url=args.gateway)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Done: {OUT_HTML}")


if __name__ == "__main__":
    main()
