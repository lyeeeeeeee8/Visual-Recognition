#!/usr/bin/env python3
"""
Convert raw `.tif` masks into a single COCO‑format annotation file.

Dataset directory structure (per training image)
------------------------------------------------
dataset_root/
└── train/
    ├── 0001/
    │   ├── image.tif
    │   ├── class1.tif
    │   ├── class2.tif
    │   ├── class3.tif
    │   └── class4.tif
    ├── 0002/
    │   └── ...
    └── ...

Each `class<i>.tif` encodes *multiple* instances of class‑i as
unique grayscale values (background = 0).
The script loops through every sub‑folder, extracts each instance,
encodes it as RLE and writes a single `train.json`.
"""

# ‑‑‑ standard library
import json
import os
from typing import Dict, List

# ‑‑‑ third‑party
import numpy as np
from PIL import Image
from skimage import io as skio
from pycocotools import mask as mask_utils

# ---------------------------------------------------------------------#
# ---------------------------  CONSTANTS  -----------------------------#
# ---------------------------------------------------------------------#
CATEGORIES: List[Dict] = [
    {"id": 1, "name": "class1"},
    {"id": 2, "name": "class2"},
    {"id": 3, "name": "class3"},
    {"id": 4, "name": "class4"},
]

# ---------------------------------------------------------------------#
# -----------------------------  MAIN  --------------------------------#
# ---------------------------------------------------------------------#
def create_coco_json(dataset_root: str, out_json_path: str) -> None:
    """Scan `dataset_root/train/*` and export a COCO annotation JSON."""
    train_dir = os.path.join(dataset_root, "train")

    images: List[Dict] = []
    annotations: List[Dict] = []

    ann_id = 1
    img_id = 1

    for sub in sorted(os.listdir(train_dir)):
        sub_path = os.path.join(train_dir, sub)
        if not os.path.isdir(sub_path):
            continue  # skip stray files

        # ---- read raw image to get width / height ----
        img_path = os.path.join(sub_path, "image.tif")
        img_pil = Image.open(img_path)
        width, height = img_pil.size

        images.append(
            {
                "id": img_id,
                "file_name": os.path.join(sub, "image.tif"),
                "width": width,
                "height": height,
            }
        )

        # ---- iterate over 4 class masks ----
        for cat in CATEGORIES:
            mask_path = os.path.join(sub_path, f"class{cat['id']}.tif")
            if not os.path.exists(mask_path):
                continue  # some images may lack certain classes

            mask = skio.imread(mask_path).astype(np.uint8)

            # each unique gray value (≠ 0) represents one instance
            for inst_val in np.unique(mask):
                if inst_val == 0:
                    continue  # skip background

                binary = (mask == inst_val).astype(np.uint8)

                # RLE encode (Fortran order required by pycocotools)
                rle = mask_utils.encode(np.asfortranarray(binary))
                rle["counts"] = rle["counts"].decode("ascii")

                area = float(mask_utils.area(rle))
                bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat["id"],
                        "segmentation": rle,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        img_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }

    with open(out_json_path, "w") as handle:
        json.dump(coco_dict, handle, indent=2)

    print(
        f"[annotation] COCO JSON saved: {out_json_path} | "
        f"{len(images)} images, {len(annotations)} instances"
    )


if __name__ == "__main__":
    DATASET_ROOT = "/home/hscc/EN/hw3_detectron/dataset"
    OUTPUT_JSON = os.path.join(DATASET_ROOT, "train.json")
    create_coco_json(DATASET_ROOT, OUTPUT_JSON)
