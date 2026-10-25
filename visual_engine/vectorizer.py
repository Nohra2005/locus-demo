# =============================================================================
# vectorizer.py
#
# ARCHITECTURE:
#
#   CATEGORY DECISION — title is ground truth, CLIP is fallback only:
#     Signal 1: title classifier (whitelist → sentence-transformers)
#     Signal 2: CLIP image classification on full image
#                ↑ only used when title gives nothing useful
#
#   WHITELIST TITLE = FINAL:
#     If the title contains an unambiguous token (whitelist hit, conf=1.0),
#     that category is returned immediately. CLIP cannot override it.
#     Reason: the image shows context (model wearing multiple items).
#     Only the title tells us what is actually for sale.
#
#   TIE-BREAK:
#     When two categories score equally in the token map, the one whose
#     token appeared LAST in the title wins. Brand names and generic words
#     ("Top Ten", "Women", "Men") appear first; the actual garment word
#     ("Tight", "Jacket", "Dress") appears later.
#
#   WHITELIST OVERRIDES:
#     At startup, vectorizer loads clip_labels.py (static) and then merges
#     whitelist_overrides.json on top. Approved community suggestions become
#     active without editing clip_labels.py. Reload via reload_overrides().
#
#   YOLO IS GEOMETRY ONLY — 3-tier crop selection, no full-image fallback.
#
#   CLIP RELABEL:
#     After NMS, each YOLO box is cropped and re-classified by CLIP.
#     This fixes flat-lay misclassifications where YOLO's "vest" label
#     covers both gilets (jacket) and tank tops (top). CLIP sees the
#     actual crop and decides correctly.
#
#   DARK VECTORS: removed — never queried, wasted 50% storage.
#
# CANONICAL LABELS (13):
#   top, sports_bra, pants, leggings, shorts, skirt, dress,
#   sweater, jacket, shoes, hat, bag, jumpsuit
# =============================================================================

import copy
import io
import base64
import time

try:
    import torch
    from PIL import Image, ImageEnhance
    from transformers import CLIPProcessor, CLIPModel
    from sentence_transformers import SentenceTransformer, util
    from detector_clothing import ClothingDetector
    from detector_accessories import AccessoryDetector
except ImportError:
    pass
from clip_labels import (
    CANONICAL_LABELS,
    CLIP_PROMPTS,
    QUERY_CLIP_PROMPTS,
    LABEL_DESCRIPTIONS,
    UNAMBIGUOUS_TOKEN_MAP,
)

# =============================================================================
# SHOE STYLE — sub-category within the "shoes" collection.
# Derived from YOLO-World prompt labels (detector_accessories.py PROMPTS).
# Stored as a Qdrant payload field so searches can filter within shoes.
#
# Buckets: sneaker | boot | heel | sandal | other
# =============================================================================

def shoe_style_from_label(prompt_label: str) -> str:
    """
    Map a YOLO-World shoe prompt label to a coarse shoe_style bucket.
    Called at index time (from the selected box label) and at search time
    (from the query box label) to enable sub-collection filtering within shoes.
    """
    lbl = prompt_label.lower()
    if any(k in lbl for k in ("sneaker", "trainer", "running shoe", "platform", "wedge", "clog", "slip-on")):
        return "sneaker"
    if "boot" in lbl:
        return "boot"
    if any(k in lbl for k in ("heel", "pump", "stiletto")):
        return "heel"
    if any(k in lbl for k in ("sandal", "loafer", "flat", "mule", "espadrille")):
        return "sandal"
    return "other"


# =============================================================================
# CATEGORY ALIASES
# Maps voted final_category → acceptable YOLO search_label values (Tier 2).
# =============================================================================
_ACCESSORY_CATEGORIES = {"shoes", "hat", "bag"}

# Style hint phrases for CLIP text encoder — used to sharpen accessory query vectors.
# Fashion-CLIP's text space separates these well even when visual space does not.
_SHOE_STYLE_HINTS = {
    "sneaker": "sneaker trainer running shoe athletic casual footwear",
    "boot":    "ankle boot chelsea boot combat boot knee high boot",
    "heel":    "high heel stiletto pump court shoe kitten heel formal",
    "sandal":  "sandal flat mule loafer ballet flat espadrille open toe",
    "other":   "shoe footwear",
}
_BAG_HINT = "handbag shoulder bag tote crossbody clutch purse"
_HAT_HINT = "hat cap headwear beret fedora bucket hat beanie"

_HYBRID_ALPHA = 0.3  # weight for text component in hybrid vector (0 = visual-only)

CATEGORY_ALIASES = {
    "top":        ["top", "shirt"],
    "sports_bra": ["sports_bra", "top", "shirt"],
    "sweater":    ["sweater", "top", "shirt"],
    "jacket":     ["jacket", "coat"],
    "dress":      ["dress", "jumpsuit", "skirt"],
    "jumpsuit":   ["jumpsuit", "dress"],
    "skirt":      ["skirt", "dress"],
    "pants":      ["pants", "trousers", "jeans"],
    "leggings":   ["leggings", "pants", "shorts"],
    "shorts":     ["shorts", "pants"],
    "shoes":      ["shoes", "sneakers", "boots", "sandals"],
    "hat":        ["hat", "cap"],
    "bag":        ["bag", "handbag", "backpack"],
}


# =============================================================================
# NMS HELPERS
# =============================================================================

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _nms(detections, iou_threshold=0.3):
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    suppressed = set()
    _FULL_GARMENT  = {"dress", "skirt"}   # full-length — protect from upper-body suppression
    _PARTIAL_UPPER = {"top", "jacket"}    # upper-body crops that may overlap a dress bbox
    for i, d in enumerate(dets):
        if i in suppressed:
            continue
        kept.append(d)
        for j in range(i + 1, len(dets)):
            if j in suppressed:
                continue
            if _iou(d["bbox"], dets[j]["bbox"]) > iou_threshold:
                # Don't let an upper-body box erase a full-garment box.
                # A vest/top crop of the torso naturally overlaps the full dress bbox —
                # both detections are valid and should survive for downstream selection.
                if (d["search_label"] in _PARTIAL_UPPER
                        and dets[j]["search_label"] in _FULL_GARMENT):
                    continue  # keep both
                suppressed.add(j)
    print(f"  NMS: {len(detections)} -> {len(kept)} detections after suppression")
    return kept


# =============================================================================
# MAIN CLASS
# =============================================================================

class LocusVisualizer:
    @staticmethod
    def _load_clip_with_retry(model_id: str, retries: int = 3, delay: float = 5.0):
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                model = CLIPModel.from_pretrained(model_id)
                try:
                    processor = CLIPProcessor.from_pretrained(model_id)
                except Exception:
                    # Some fine-tuned CLIP models (e.g. patrickjohncyh/fashion-clip) don't
                    # ship a processor_config.json — transformers>=4.48 requires it.
                    # Fall back to the base ViT-B/32 processor, which is identical.
                    print(f"[CLIP] No processor config for '{model_id}', "
                          f"falling back to openai/clip-vit-base-patch32 processor")
                    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                return model, processor
            except Exception as e:
                last_err = e
                print(f"[CLIP] Load attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise RuntimeError(f"Failed to load CLIP model '{model_id}' after {retries} attempts: {last_err}")

    def __init__(self):

        self.clothing_detector  = ClothingDetector()
        self.accessory_detector = AccessoryDetector()

        print("Loading CLIP ViT-B/16...")
        self.clip_model, self.clip_processor = self._load_clip_with_retry("patrickjohncyh/fashion-clip")
        self.clip_labels    = CANONICAL_LABELS

        # Pre-compute CLIP text embeddings for canonical labels.
        # Use component layers directly — get_text_features() return type changed
        # across transformers versions (plain tensor vs BaseModelOutputWithPooling).
        text_inputs = self.clip_processor(
            text=CLIP_PROMPTS, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            text_out           = self.clip_model.text_model(**text_inputs)
            text_features      = self.clip_model.text_projection(text_out.pooler_output)
            self.text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # Pre-compute query-time text embeddings (person-wearing style prompts).
        # Used when classifying user-drawn crops from real-world photos — not
        # clean product images.  Better prompts → better zero-shot accuracy.
        query_text_inputs = self.clip_processor(
            text=QUERY_CLIP_PROMPTS, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            query_text_out           = self.clip_model.text_model(**query_text_inputs)
            query_text_features      = self.clip_model.text_projection(query_text_out.pooler_output)
            self.query_text_features = query_text_features / query_text_features.norm(p=2, dim=-1, keepdim=True)

        print("Loading sentence-transformers (all-MiniLM-L6-v2)...")
        self.st_model         = SentenceTransformer("all-MiniLM-L6-v2")
        self._all_categories  = list(LABEL_DESCRIPTIONS.keys())
        desc_texts            = [LABEL_DESCRIPTIONS[c] for c in self._all_categories]
        self._desc_embeddings = self.st_model.encode(
            desc_texts, convert_to_tensor=True, normalize_embeddings=True
        )

        self.token_map = copy.deepcopy(UNAMBIGUOUS_TOKEN_MAP)

        print("=" * 50)
        print("LOCUS VISUAL ENGINE READY")
        print(f"CLIP labels: {self.clip_labels}")
        print(f"Whitelist tokens: {len(self.token_map)}")
        print("=" * 50)

    # =========================================================================
    # PUBLIC: reload_adapter()
    # Hot-swap the CLIP vision encoder's LoRA adapter without restarting.
    # Called by POST /reload-adapter after promote_model.py copies new weights.
    #
    # Strategy: load a fresh CLIPModel, apply the adapter, swap self.clip_model.
    # In-flight requests during the ~30s load keep using the old model reference,
    # then new requests pick up the new one. No lock needed — Python's GIL
    # makes the final attribute assignment atomic at the interpreter level.
    # =========================================================================
    def reload_adapter(self, adapter_path: str, visual_projection_path: str = "") -> dict:
        from pathlib import Path
        try:
            from peft import PeftModel
        except ImportError:
            return {"success": False, "error": "peft not installed in visual_engine"}

        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            return {"success": False, "error": f"Adapter path not found: {adapter_path}"}

        try:
            print(f"[RELOAD] Loading fresh base model + LoRA adapter from {adapter_path}...")
            new_model     = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            new_model.vision_model = PeftModel.from_pretrained(
                new_model.vision_model, str(adapter_path)
            )

            if visual_projection_path and Path(visual_projection_path).exists():
                proj_state = torch.load(visual_projection_path, map_location="cpu")
                new_model.visual_projection.load_state_dict(proj_state)
                print(f"[RELOAD] Reloaded visual_projection from {visual_projection_path}")

            new_model.eval()

            # Atomic swap — old model stays alive until GC collects it
            self.clip_model = new_model
            print(f"[RELOAD] LoRA adapter hot-swapped successfully")
            return {"success": True, "adapter_path": str(adapter_path)}

        except Exception as e:
            print(f"[RELOAD] Failed to reload adapter: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # PUBLIC: detect_objects() — search time only
    # =========================================================================
    def detect_objects(self, image_bytes):
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H  = image.size

            clothing       = self.clothing_detector.detect(image, self._classify_crop)
            accessories    = self.accessory_detector.detect(image, self._classify_crop)
            all_detections = _nms(clothing + accessories, iou_threshold=0.3)
            # Cap at top-6 by confidence — prevents clutter on busy lifestyle photos
            all_detections = sorted(all_detections, key=lambda d: d["score"], reverse=True)[:6]

            if not all_detections:
                print("detect_objects(): no boxes found — returning empty list")
                return all_detections, W, H

            # ── CLIP relabel each box ─────────────────────────────────────────
            # YOLO finds geometry; CLIP decides what the item actually is.
            # Fixes flat-lay misclassifications — e.g. DeepFashion2 calls every
            # sleeveless garment "vest" (→ jacket), but CLIP on the crop
            # correctly identifies tank tops, halternecks, etc. as "top".
            # Only override when CLIP confidence ≥ 0.52; below that keep YOLO.
            CLIP_RELABEL_THRESHOLD = 0.52
            # dress ↔ skirt are visually indistinguishable to CLIP on a crop
            # (slip/midi dresses look like skirts from the waist down).
            # DeepFashion2 knows garment shape; block CLIP from flipping these.
            _DRESS_SKIRT = {"dress", "skirt"}
            # CLIP sees texture, not length. A knit dress crop looks like "sweater"
            # or "top" to CLIP because it can't tell the garment is dress-length.
            # When DeepFashion2 (the shape expert) detected a dress or skirt,
            # block CLIP from flipping it to a texture-based category.
            _TEXTURE_CATEGORIES = {"sweater", "top"}
            # Fashion-CLIP is systematically biased toward clothing and scores
            # accessories (shoes, bag, hat) near zero even on clear accessory crops.
            # YOLO-World was prompted with rich, specific accessory descriptions
            # ("boot ankle boot chelsea boot combat boot", etc.) — trust it over CLIP.
            # Block CLIP from flipping any YOLO-World accessory detection to clothing.
            _ACCESSORY_LABELS = {"shoes", "bag", "hat"}

            for det in all_detections:
                bx1, by1, bx2, by2 = det["bbox"]
                bx1 = max(0, int(bx1)); by1 = max(0, int(by1))
                bx2 = min(W, int(bx2)); by2 = min(H, int(by2))
                if bx2 <= bx1 or by2 <= by1:
                    continue
                try:
                    crop = image.crop((bx1, by1, bx2, by2))
                    clip_label, clip_conf, _ = self._classify_crop(crop)
                    if clip_label and clip_conf >= CLIP_RELABEL_THRESHOLD:
                        if clip_label != det["search_label"]:
                            if det["search_label"] in _DRESS_SKIRT and clip_label in _DRESS_SKIRT:
                                print(f"  [CLIP-RELABEL BLOCKED] '{det['search_label']}' "
                                      f"→ '{clip_label}' (conf={clip_conf:.2f}) — dress/skirt pair")
                            elif (det.get("source") == "deepfashion2"
                                  and det["search_label"] in _DRESS_SKIRT
                                  and clip_label in _TEXTURE_CATEGORIES):
                                print(f"  [CLIP-RELABEL BLOCKED] DF2 '{det['search_label']}' "
                                      f"→ '{clip_label}' (conf={clip_conf:.2f}) — shape > texture")
                            elif (det.get("source") == "deepfashion2"
                                  and det["search_label"] not in _DRESS_SKIRT
                                  and clip_label in _DRESS_SKIRT):
                                # Mirror of the above: DF2 said shirt/jacket/pants/etc.
                                # CLIP cannot see garment length on a crop and guesses dress.
                                # Trust DeepFashion2 (shape expert) — block top → dress.
                                print(f"  [CLIP-RELABEL BLOCKED] DF2 '{det['search_label']}' "
                                      f"→ '{clip_label}' (conf={clip_conf:.2f}) — shape > CLIP length guess")
                            elif (det.get("source") == "yolo_world"
                                  and det["search_label"] in _ACCESSORY_LABELS
                                  and clip_label not in _ACCESSORY_LABELS):
                                print(f"  [CLIP-RELABEL BLOCKED] YOLO-World '{det['search_label']}' "
                                      f"→ '{clip_label}' (conf={clip_conf:.2f}) — trust YOLO-World for accessories")
                            else:
                                print(f"  [CLIP-RELABEL] '{det['search_label']}' "
                                      f"→ '{clip_label}' (conf={clip_conf:.2f})")
                                det["search_label"] = clip_label
                        det["clip_conf"] = round(clip_conf, 3)
                except Exception:
                    pass  # keep original YOLO label on any error
            # ─────────────────────────────────────────────────────────────────

            print(f"detect_objects(): {len(all_detections)} boxes in {time.time()-t0:.2f}s")
            return all_detections, W, H

        except Exception as e:
            print(f"detect_objects() error: {e}")
            return [], 0, 0

    # =========================================================================
    # PUBLIC: process_image() — search time, pre-cropped bytes
    # =========================================================================
    def process_image(self, image_bytes, yolo_label="", darken=False, query_mode=False, style_hint=""):
        t0 = time.time()
        try:
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            if max(input_image.size) > 512:
                input_image.thumbnail((512, 512))

            clip_input    = ImageEnhance.Brightness(input_image).enhance(0.3) if darken else input_image
            vector, category_tag, category_scores = self._clip_embed(clip_input, yolo_label, query_mode=query_mode)

            # Hybrid vector at query time: mix visual + CLIP text(style_hint) for accessories.
            if yolo_label in _ACCESSORY_CATEGORIES and style_hint and style_hint.strip():
                try:
                    text_vec = torch.tensor(self.encode_text(style_hint.strip()), dtype=torch.float32)
                    vis_vec  = torch.tensor(vector, dtype=torch.float32)
                    mixed    = (1.0 - _HYBRID_ALPHA) * vis_vec + _HYBRID_ALPHA * text_vec
                    mixed    = mixed / mixed.norm(p=2)
                    vector   = mixed.tolist()
                    print(f"[HYBRID-QRY] style_hint='{style_hint[:50]}' mixed (α={_HYBRID_ALPHA})")
                except Exception as _he:
                    print(f"[HYBRID-QRY] fallback to visual-only: {_he}")

            buf = io.BytesIO()
            input_image.save(buf, format="PNG")
            debug_img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            print(f"process_image(darken={darken}) done in {time.time()-t0:.2f}s")
            return vector, category_tag, category_scores, debug_img_b64

        except Exception as e:
            print(f"process_image() error: {e}")
            return None, None, None, None

    # =========================================================================
    # PUBLIC: index_product() — index time + debug
    # =========================================================================
    def index_product(self, image_bytes: bytes, title: str = ""):
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H  = image.size

            # ── Signal 1: title ───────────────────────────────────────────────
            title_cat, title_method, title_conf = self._classify_title(title)
            print(f"[S1-TITLE] '{title}' → '{title_cat}' via {title_method} ({title_conf:.3f})")

            # ── Signal 2: CLIP on full image ──────────────────────────────────
            clip_cat, clip_conf, _ = self._classify_crop(image)
            if clip_conf < 0.45:
                clip_cat = None
            print(f"[S2-CLIP]  '{title}' → '{clip_cat}' ({clip_conf:.3f})")

            # ── Vote ──────────────────────────────────────────────────────────
            final_category = self._vote(
                title_cat, title_conf, clip_cat, clip_conf, title_method
            )

            if final_category is None:
                return {
                    "skipped": True, "skip_reason": "no_consensus",
                    "all_detections": [], "selected_bbox": None,
                }
            if final_category == "not_fashion":
                return {
                    "skipped": True, "skip_reason": "not_fashion",
                    "all_detections": [], "selected_bbox": None,
                }

            print(f"[VOTE]     '{title}' → FINAL: '{final_category}'")

            # ── YOLO: find all boxes ──────────────────────────────────────────
            clothing    = self.clothing_detector.detect(image, self._classify_crop)
            accessories = self.accessory_detector.detect(image, self._classify_crop)
            detections  = _nms(clothing + accessories, iou_threshold=0.3)

            # ── 3-tier box selection ──────────────────────────────────────────
            selected_box = None
            box_source   = None

            if detections:
                exact = [d for d in detections if d.get("search_label") == final_category]
                if exact:
                    selected_box = max(exact, key=lambda d: d["score"])
                    box_source   = selected_box["source"] + "_exact"
                    print(f"[CROP T1]  Exact '{final_category}' conf={selected_box['score']:.2f}")

                if selected_box is None:
                    accepted   = CATEGORY_ALIASES.get(final_category, [final_category])
                    alias_hits = [d for d in detections if d.get("search_label") in accepted]
                    if alias_hits:
                        selected_box = max(alias_hits, key=lambda d: d["score"])
                        box_source   = selected_box["source"] + "_alias"
                        print(f"[CROP T2]  Alias '{selected_box['search_label']}' "
                              f"for '{final_category}' conf={selected_box['score']:.2f}")

                if selected_box is None:
                    if final_category in _ACCESSORY_CATEGORIES:
                        same_family = [d for d in detections
                                       if d["score"] >= 0.50
                                       and d.get("search_label") in _ACCESSORY_CATEGORIES]
                    else:
                        same_family = [d for d in detections
                                       if d["score"] >= 0.50
                                       and d.get("search_label") not in _ACCESSORY_CATEGORIES]
                    if same_family:
                        selected_box = max(same_family, key=lambda d: d["score"])
                        box_source   = selected_box["source"] + "_best_available"
                        print(f"[CROP T3]  Best available '{selected_box['search_label']}' "
                              f"for '{final_category}' conf={selected_box['score']:.2f}")

            if selected_box is None:
                if final_category in _ACCESSORY_CATEGORIES:
                    # Accessory product photos (single shoe/bag/hat on white bg) have no
                    # garment context for YOLO to anchor on — detection reliably fails.
                    # The whole image IS the product, so embed it directly.
                    selected_box = {"bbox": [0, 0, W, H], "label": final_category, "source": "full_image_fallback"}
                    box_source   = "full_image_fallback"
                    print(f"[CROP T4]  No YOLO box for accessory '{final_category}' '{title}' — full image fallback")
                else:
                    print(f"[CROP]     No usable box for '{title}' → skip")
                    return {
                        "skipped": True, "skip_reason": "no_box_found",
                        "all_detections": detections,
                        "selected_bbox":  None,
                    }

            # ── Crop & embed ──────────────────────────────────────────────────
            bx1, by1, bx2, by2 = selected_box["bbox"]
            bx1 = max(0, int(bx1)); by1 = max(0, int(by1))
            bx2 = min(W, int(bx2)); by2 = min(H, int(by2))
            crop = image.crop((bx1, by1, bx2, by2))

            if max(crop.size) > 512:
                crop = crop.copy()
                crop.thumbnail((512, 512))

            vector_normal, _, _ = self._clip_embed(crop, final_category)

            # Hybrid vector for accessories: mix visual + CLIP text(title).
            # CLIP text space cleanly separates heel vs sneaker; visual space does not.
            # Only applied when a title is available; falls back silently to visual-only.
            if final_category in _ACCESSORY_CATEGORIES and title and title.strip():
                try:
                    text_vec      = torch.tensor(self.encode_text(title.strip()), dtype=torch.float32)
                    vis_vec       = torch.tensor(vector_normal, dtype=torch.float32)
                    mixed         = (1.0 - _HYBRID_ALPHA) * vis_vec + _HYBRID_ALPHA * text_vec
                    mixed         = mixed / mixed.norm(p=2)
                    vector_normal = mixed.tolist()
                    print(f"[HYBRID-IDX] '{title[:50]}' ({final_category}): visual+text mixed (α={_HYBRID_ALPHA})")
                except Exception as _he:
                    print(f"[HYBRID-IDX] fallback to visual-only: {_he}")

            print(f"[INDEX]    '{title}' done in {time.time()-t0:.2f}s  "
                  f"cat={final_category}  box={box_source}")

            # Compute shoe_style when the product is a shoe — used to enable
            # sub-collection filtering (sneaker/boot/heel/sandal) at search time.
            computed_shoe_style = None
            if final_category == "shoes" and selected_box is not None:
                # Always derive shoe_style from the product title — YOLO prompts are
                # detection anchors ("sneaker trainer running shoe"), not style classifiers,
                # so using the box label gives wrong styles for heels/boots/sandals.
                computed_shoe_style = shoe_style_from_label(title) if title else "other"
                print(f"[SHOE]     shoe_style='{computed_shoe_style}'  (from title: '{title}')")

            return {
                "skipped":         False,
                "skip_reason":     None,
                "vector_normal":   vector_normal,
                "category":        final_category,
                "box_source":      box_source,
                "shoe_style":      computed_shoe_style,
                "selected_bbox":   [bx1, by1, bx2, by2],
                "all_detections":  [
                    {
                        "bbox":         d["bbox"],
                        "search_label": d.get("search_label", ""),
                        "score":        round(d["score"], 3),
                    }
                    for d in detections
                ],
            }

        except Exception as e:
            print(f"index_product() error for '{title}': {e}")
            return {
                "skipped": True, "skip_reason": str(e),
                "all_detections": [], "selected_bbox": None,
            }

    # =========================================================================
    # PRIVATE: _vote()
    # =========================================================================
    def _vote(self, title_cat, title_conf, clip_cat, clip_conf, title_method=""):

        if title_method == "whitelist" and title_cat and title_cat != "not_fashion":
            print(f"[VOTE] Whitelist hit '{title_cat}' — final, CLIP irrelevant")
            return title_cat

        if title_cat == "not_fashion":
            print(f"[VOTE] Title says not_fashion → skip")
            return "not_fashion"

        if clip_cat == "not_fashion" and (title_cat is None or title_cat == "not_fashion"):
            print(f"[VOTE] CLIP says not_fashion, no title override → skip")
            return "not_fashion"

        t = title_cat if (title_cat and title_cat != "not_fashion") else None
        c = clip_cat  if (clip_cat  and clip_cat  != "not_fashion") else None

        if t and c and t == c:
            print(f"[VOTE] Title + CLIP agree on '{t}'")
            return t

        if t and title_conf >= 0.70 and c is None:
            print(f"[VOTE] Title ST confident ({title_conf:.2f}), CLIP absent → '{t}'")
            return t

        if c and clip_conf >= 0.90 and t is None:
            print(f"[VOTE] CLIP confident ({clip_conf:.2f}), title absent → '{c}'")
            return c

        if t and title_conf >= 0.70 and c and clip_conf < 0.85:
            print(f"[VOTE] Title ST ({title_conf:.2f}) over uncertain CLIP ({clip_conf:.2f}) → '{t}'")
            return t

        print(f"[VOTE] No consensus — title='{title_cat}'({title_conf:.2f}) "
              f"clip='{clip_cat}'({clip_conf:.2f}) → skip")
        return None

    # =========================================================================
    # PRIVATE: _classify_title()
    #
    # Tie-break rule: when two categories score equally, the one whose token
    # appeared LAST in the title wins. Brand names ("Top Ten", "Nike") and
    # generic descriptors ("Women", "Men", "Lifestyle") appear first;
    # the actual garment word ("Tight", "Jacket", "Dress") appears later.
    # Example: "Top Ten Stretchy Women Lifestyle Tight Black"
    #   → "top" (index 0) ties with "leggings" via "tight" (index 5)
    #   → last_seen picks leggings. Correct.
    # =========================================================================
    def _classify_title(self, title: str):
        if not title or len(title.strip()) < 2:
            return None, "empty", 0.0

        tokens = title.lower().replace("-", " ").replace("/", " ").split()

        not_fashion_hits = 0
        last_fashion_cat = None   # category of the last whitelist token seen
        last_fashion_pos = -1     # position of that token

        for idx, token in enumerate(tokens):
            cat = self.token_map.get(token)
            if cat is None:
                continue
            if cat == "not_fashion":
                not_fashion_hits += 1
            else:
                last_fashion_cat = cat
                last_fashion_pos = idx

        # Bigrams override single tokens at the same or earlier position
        # (position is offset past all single tokens so bigrams always rank last)
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        for i, bigram in enumerate(bigrams):
            cat = self.token_map.get(bigram)
            if cat is None:
                continue
            pos = len(tokens) + i
            if cat == "not_fashion":
                not_fashion_hits += 2
            elif pos > last_fashion_pos:
                last_fashion_cat = cat
                last_fashion_pos = pos

        if last_fashion_cat is not None:
            if not_fashion_hits > 0:
                return "not_fashion", "whitelist", 1.0
            return last_fashion_cat, "whitelist", 1.0

        if not_fashion_hits > 0:
            return "not_fashion", "whitelist", 1.0

        title_emb = self.st_model.encode(
            title, convert_to_tensor=True, normalize_embeddings=True
        )
        scores = util.cos_sim(title_emb, self._desc_embeddings)[0]

        not_fashion_idx   = self._all_categories.index("not_fashion")
        not_fashion_score = float(scores[not_fashion_idx])

        best_fashion_score = -1.0
        best_fashion_cat   = None
        for i, cat in enumerate(self._all_categories):
            if cat == "not_fashion":
                continue
            s = float(scores[i])
            if s > best_fashion_score:
                best_fashion_score = s
                best_fashion_cat   = cat

        if not_fashion_score >= 0.40 and not_fashion_score >= best_fashion_score:
            return "not_fashion", "sentence_transformer", not_fashion_score

        if best_fashion_score >= 0.30:
            return best_fashion_cat, "sentence_transformer", best_fashion_score

        return None, "none", 0.0

    # =========================================================================
    # PRIVATE: _clip_embed()
    # =========================================================================
    def _clip_embed(self, pil_image: "Image.Image", label_hint: str = "", query_mode: bool = False):
        clip_inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            vision_out     = self.clip_model.vision_model(**clip_inputs)
            image_features = self.clip_model.visual_projection(vision_out.pooler_output)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        vector = image_features[0].tolist()

        # query_mode=True uses prompts tuned for real-world photos ("a person wearing X").
        # query_mode=False (default) uses product-photo prompts for index-time classification.
        text_feats    = self.query_text_features if query_mode else self.text_features
        similarity    = (100.0 * image_features @ text_feats.T).softmax(dim=-1)
        scores_tensor = similarity[0]

        category_scores = {
            label: round(scores_tensor[i].item(), 4)
            for i, label in enumerate(self.clip_labels)
        }

        if label_hint and label_hint.strip() in self.clip_labels:
            # Hard override: caller explicitly confirmed a category (user has ground truth).
            category_tag = label_hint.strip()
        else:
            clip_conf  = scores_tensor.max().item()
            clip_label = self.clip_labels[scores_tensor.argmax().item()]
            if query_mode:
                # Always return a best guess — user can correct in ConfirmView.
                # No confidence floor: a low-confidence suggestion beats returning None.
                category_tag = clip_label
            else:
                category_tag = clip_label if clip_conf >= 0.45 else None

        return vector, category_tag, category_scores

    # =========================================================================
    # PRIVATE: _classify_crop()
    # =========================================================================
    def _classify_crop(self, pil_image):
        _, label, scores = self._clip_embed(pil_image)
        best_conf = max(scores.values()) if scores else 0.0
        return label or "", best_conf, scores

    # =========================================================================
    # PUBLIC: encode_text() — CLIP text embedding for /refine queries
    # =========================================================================
    def encode_text(self, text: str) -> list[float]:
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            out      = self.clip_model.text_model(**inputs)
            features = self.clip_model.text_projection(out.pooler_output)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features[0].tolist()

    # =========================================================================
    # PUBLIC: classify_text() — debug endpoint
    # =========================================================================
    def classify_text(self, title: str):
        category, method, confidence = self._classify_title(title)
        return category, confidence