#!/usr/bin/env python3
"""
Run inference with a trained Detectron2 Mask‑R CNN model and export the
predictions to COCO‑style *test‑results.json*.

Main steps
----------
1.  Build a `DefaultPredictor` from the saved config / weights.
2.  Loop through every test `.tif` image, predict instances.
3.  Save visualisation overlays for quick sanity‑check.
4.  Convert masks → RLE, boxes → [x, y, w, h] and dump to JSON.
"""

# ‑‑‑ standard library
import json
import os
from typing import Dict, Tuple, List

# ‑‑‑ third‑party
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# ---------------------------------------------------------------------#
# ----------------------------  CONSTANTS  ----------------------------#
# ---------------------------------------------------------------------#
TEST_FOLDER = "dataset/test"
MODEL_DIR = "checkpoints/models/T3_R101_DC53x" 
OUTPUT_DIR = "results/models/T3_R101_DC53x"
MAPPING_JSON_PATH = "dataset/test_image_name_to_ids.json"

CONFIDENCE_THRESHOLD = 0.05
DEVICE = "cuda"                     # "cpu" for CPU inference
VIS_SCALE = 0.5                     # smaller → faster draw

# ---------------------------------------------------------------------#
# ---------------------------  UTILITIES  -----------------------------#
# ---------------------------------------------------------------------#
def build_predictor() -> DefaultPredictor:
    """Construct a Detectron2 `DefaultPredictor` from saved config/weights."""
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(MODEL_DIR, "config.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(MODEL_DIR, "model_best.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.DEVICE = DEVICE
    return DefaultPredictor(cfg)


def save_visualisation(
    img_bgr: np.ndarray, instances, save_path: str, scale: float = 0.5
) -> None:
    """Overlay predicted masks / boxes on the image and write to disk."""
    viz = Visualizer(img_bgr[:, :, ::-1], scale=scale)
    vis_output = viz.draw_instance_predictions(instances)
    cv2.imwrite(save_path, vis_output.get_image()[:, :, ::-1])


def load_test_mapping(
    mapping_json: str,
) -> Tuple[Dict[str, int], Dict[str, Tuple[int, int]]]:
    """Read filename → image_id and filename → (H, W) from provided JSON."""
    with open(mapping_json, "r") as handle:
        mapping_list = json.load(handle)

    name_to_id = {item["file_name"]: item["id"] for item in mapping_list}
    size_lookup = {
        item["file_name"]: (item["height"], item["width"]) for item in mapping_list
    }
    return name_to_id, size_lookup


def encode_binary_mask(mask: np.ndarray) -> Dict:
    """Convert a binary mask (H, W) → COCO‑compatible RLE (with utf‑8 counts)."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")  # bytes → str for JSON
    return rle


# ---------------------------------------------------------------------#
# -----------------------------  MAIN  --------------------------------#
# ---------------------------------------------------------------------#
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vis_dir = os.path.join(OUTPUT_DIR, "visualise")
    os.makedirs(vis_dir, exist_ok=True)

    predictor = build_predictor()
    name_to_id, size_lookup = load_test_mapping(MAPPING_JSON_PATH)

    # Collect all test images
    test_imgs: List[str] = sorted(
        f
        for f in os.listdir(TEST_FOLDER)
        if f.lower().endswith((".tif"))
    )

    coco_results = []

    for fname in tqdm(test_imgs, desc="Inference", leave=False, unit="image"):
        if fname not in name_to_id:  # skip stray files
            continue

        img_id = name_to_id[fname]
        height, width = size_lookup[fname]

        img_path = os.path.join(TEST_FOLDER, fname)
        img_bgr = cv2.imread(img_path)

        # Predict instances
        outputs = predictor(img_bgr)
        instances = outputs["instances"].to("cpu")

        # Save overlay for qualitative check
        vis_path = os.path.join(vis_dir, fname).replace(".tif", ".png")
        save_visualisation(img_bgr, instances, vis_path, scale=VIS_SCALE)

        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        # Iterate through all detected instances
        for box, mask, score, cls in zip(boxes, masks, scores, classes):
            # Convert box from [x1, y1, x2, y2] → [x, y, w, h]
            x1, y1, x2, y2 = box
            bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            rle = encode_binary_mask(mask)

            coco_results.append(
                {
                    "image_id": img_id,
                    "bbox": bbox_xywh,
                    "score": float(score),
                    "category_id": int(cls) + 1,  # 1‑based for COCO
                    "segmentation": {"size": [height, width], "counts": rle["counts"]},
                }
            )

    # Sort by image_id for reproducibility
    coco_results.sort(key=lambda d: d["image_id"])

    json_out = os.path.join(OUTPUT_DIR, "test-results.json")
    with open(json_out, "w") as handle:
        json.dump(coco_results, handle)
    print(f"[Inference] Predictions saved: {json_out}")


if __name__ == "__main__":
    main()
