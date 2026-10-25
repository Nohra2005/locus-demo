# =============================================================================
# detector_clothing.py
# Model 1: DeepFashion2 YOLOv8
# Detects: shirt, jacket, dress, skirt, pants, shorts
# Does NOT detect: shoes, bags, accessories, sweaters, coats — see detector_accessories.py
#
# CHANGES vs previous version:
#   - MIN_CONFIDENCE raised 0.30 → 0.50 to reduce over-detection
# =============================================================================

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from huggingface_hub import hf_hub_download
from PIL import Image

from clip_labels import YOLO_TO_CANONICAL

# PyTorch 2.6 changed torch.load default to weights_only=True, which blocks
# loading legacy ultralytics segmentation checkpoints. Allowlisting the class
# is the fix recommended by the PyTorch error message itself.
torch.serialization.add_safe_globals([SegmentationModel])

# Raw DeepFashion2 class labels — fixed by model weights, do not reorder
DEEPFASHION2_LABELS = {
    0:  "short sleeved shirt",
    1:  "long sleeved shirt",
    2:  "short sleeved outwear",
    3:  "long sleeved outwear",
    4:  "vest",
    5:  "sling",
    6:  "shorts",
    7:  "trousers",
    8:  "skirt",
    9:  "short sleeved dress",
    10: "long sleeved dress",
    11: "vest dress",
    12: "sling dress",
}

MIN_CONFIDENCE = 0.50   # raised from 0.30 — reduces duplicate/weak boxes
MIN_AREA       = 1500   # px²


class ClothingDetector:
    def __init__(self):
        print("=" * 50)
        print("Loading Model 1: DeepFashion2 (Clothing)")
        print("Covers: shirt, jacket, dress, skirt, pants, shorts")
        print("=" * 50)
        model_path = hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="deepfashion2_yolov8s-seg.pt"
        )
        self.model = YOLO(model_path)
        print("Model 1 ready.")

    def detect(self, pil_image, classify_fn):
        """
        Runs DeepFashion2 on a PIL image.

        Args:
            pil_image:   PIL.Image — the full photo to scan
            classify_fn: kept for interface compatibility, not called here.
                         YOLO already knows the category — mapped via YOLO_TO_CANONICAL.

        Returns:
            list of dicts:
                bbox         [x1, y1, x2, y2]
                label        raw DeepFashion2 label (shown in UI)
                search_label canonical label (used for Qdrant filter)
                score        YOLO bbox confidence
                source       "deepfashion2"
        """
        detections = []
        try:
            results = self.model(pil_image, conf=MIN_CONFIDENCE, verbose=False)[0]

            for box in results.boxes:
                class_id        = int(box.cls[0])
                conf            = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if (x2 - x1) * (y2 - y1) < MIN_AREA:
                    continue

                df2_label = DEEPFASHION2_LABELS.get(class_id, "clothing")
                canonical = YOLO_TO_CANONICAL.get(df2_label)

                if canonical is None:
                    print(f"  WARNING: no canonical mapping for '{df2_label}', skipping")
                    continue

                detections.append({
                    "bbox":         [x1, y1, x2, y2],
                    "label":        df2_label,
                    "search_label": canonical,
                    "score":        round(conf, 3),
                    "source":       "deepfashion2"
                })

            print(f"  DeepFashion2: {len(detections)} clothing items found")

        except Exception as e:
            print(f"  DeepFashion2 error: {e}")

        return detections