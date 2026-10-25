# =============================================================================
# detector_accessories.py
# Model: YOLO-World (open-vocabulary, text-prompted)
#
# Replaces YOLOS-Fashionpedia. YOLO-World is prompted with rich text
# descriptions per category so it can detect shoes, bags, and hats even
# when they are small in the frame or visually complex.
#
# Covers:
#   Clothing DeepFashion2 misses: sweater, cardigan, coat, jumpsuit
#   Accessories:                  shoes, bag, hat
#
# Architecture:
#   - One YOLO-World model (yolov8s-worldv2.pt, ~100MB)
#   - Classes are set once at startup with rich text prompts
#   - Each detected class maps to a canonical label via PROMPT_TO_CANONICAL
#   - MIN_CONFIDENCE intentionally lower than DeepFashion2 — YOLO-World
#     zero-shot scores are inherently lower than trained-label scores
# =============================================================================

import torch
from ultralytics import YOLOWorld
from ultralytics.nn.tasks import WorldModel
from PIL import Image

torch.serialization.add_safe_globals([WorldModel])

MIN_CONFIDENCE = 0.25   # YOLO-World zero-shot scores are lower than supervised; 0.25 filters ghost boxes while still catching small accessories
MIN_AREA       = 800    # px² — smaller than DF2 to catch small accessories

# Rich text prompts — order determines class_id in results.
# Keep each prompt specific enough to avoid false positives from clothing.
PROMPTS = [
    # ── clothing DeepFashion2 misses ──────────────────────────────────────
    "sweater",                                                  # 0  → sweater
    "cardigan",                                                 # 1  → sweater
    "hoodie sweatshirt",                                        # 2  → sweater
    "coat overcoat",                                            # 3  → jacket
    "jumpsuit romper playsuit",                                 # 4  → jumpsuit
    # ── dress/skirt fallback (DF2 misses complex silhouettes) ─────────────
    "maxi dress long dress floor length dress",                 # 5  → dress
    "midi dress knee length dress wrap dress shirt dress",      # 6  → dress
    "mini dress short dress bodycon dress",                     # 7  → dress
    "skirt midi skirt maxi skirt mini skirt",                   # 8  → skirt
    # ── shoes ─────────────────────────────────────────────────────────────
    "sneaker trainer running shoe",                             # 9  → shoes
    "boot ankle boot chelsea boot combat boot",                 # 10 → shoes
    "high heel pump stiletto",                                  # 11 → shoes
    "sandal strappy sandal flat sandal",                        # 12 → shoes
    "loafer flat shoe ballet flat mule espadrille",             # 13 → shoes
    "platform shoe wedge clog slip-on",                         # 14 → shoes
    # ── bag ───────────────────────────────────────────────────────────────
    "handbag shoulder bag tote bag",                            # 15 → bag
    "crossbody bag satchel clutch purse",                       # 16 → bag
    "backpack rucksack",                                        # 17 → bag
    # ── hat ───────────────────────────────────────────────────────────────
    "baseball cap snapback trucker cap",                        # 18 → hat
    "straw hat sun hat wide brim hat",                          # 19 → hat
    "beanie bobble hat winter hat",                             # 20 → hat
    "fedora panama bucket hat beret",                           # 21 → hat
]

# Maps prompt index → canonical label
PROMPT_TO_CANONICAL = {
    0:  "sweater",   # sweater
    1:  "sweater",   # cardigan
    2:  "sweater",   # hoodie sweatshirt
    3:  "jacket",    # coat overcoat
    4:  "jumpsuit",  # jumpsuit romper
    5:  "dress",     # maxi dress long dress
    6:  "dress",     # midi dress wrap dress
    7:  "dress",     # mini dress bodycon
    8:  "skirt",     # skirt midi maxi mini
    9:  "shoes",     # sneaker
    10: "shoes",     # boot
    11: "shoes",     # heel pump
    12: "shoes",     # sandal
    13: "shoes",     # loafer flat
    14: "shoes",     # platform wedge
    15: "bag",       # handbag shoulder
    16: "bag",       # crossbody clutch
    17: "bag",       # backpack
    18: "hat",       # baseball cap
    19: "hat",       # straw hat
    20: "hat",       # beanie
    21: "hat",       # fedora bucket
}

# Max image size fed to YOLO-World — larger images hurt detection of small items
MAX_SIDE = 640


class AccessoryDetector:
    def __init__(self):
        print("=" * 50)
        print("Loading YOLO-World (open-vocabulary accessory detector)")
        print("Covers: sweater, coat, jumpsuit + shoes, bag, hat")
        print("=" * 50)
        self.model = YOLOWorld("yolov8s-worldv2.pt")
        self.model.set_classes(PROMPTS)
        print(f"  Classes set: {len(PROMPTS)} prompts")
        print("YOLO-World ready.")

    def detect(self, pil_image, classify_fn):
        """
        Runs YOLO-World on a PIL image.

        Resizes to MAX_SIDE before inference (improves small-item detection),
        then scales bbox coords back to original image space.

        Args:
            pil_image:   PIL.Image — the full photo to scan
            classify_fn: callable — passed for interface compatibility,
                         used for CLIP relabel in detect_objects()

        Returns:
            list of dicts:
                bbox         [x1, y1, x2, y2] in original image pixels
                label        prompt string (shown in UI)
                search_label canonical label (used for Qdrant filter)
                score        YOLO confidence
                source       "yolo_world"
        """
        detections = []
        try:
            orig_w, orig_h = pil_image.size

            # Resize for inference if needed
            if max(orig_w, orig_h) > MAX_SIDE:
                scale = MAX_SIDE / max(orig_w, orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                infer_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            else:
                scale = 1.0
                infer_image = pil_image

            results = self.model(infer_image, conf=MIN_CONFIDENCE, verbose=False)[0]

            for box in results.boxes:
                class_id = int(box.cls[0])
                conf     = float(box.conf[0])
                canonical = PROMPT_TO_CANONICAL.get(class_id)
                if canonical is None:
                    continue

                # Scale coords back to original image space
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if scale != 1.0:
                    x1 /= scale; y1 /= scale
                    x2 /= scale; y2 /= scale

                x1 = max(0, int(x1)); y1 = max(0, int(y1))
                x2 = min(orig_w, int(x2)); y2 = min(orig_h, int(y2))

                if (x2 - x1) * (y2 - y1) < MIN_AREA:
                    continue

                prompt_label = PROMPTS[class_id]
                detections.append({
                    "bbox":         [x1, y1, x2, y2],
                    "label":        prompt_label,
                    "search_label": canonical,
                    "score":        round(conf, 3),
                    "source":       "yolo_world"
                })

            print(f"  YOLO-World: {len(detections)} items found")

        except Exception as e:
            print(f"  YOLO-World error: {e}")

        return detections
