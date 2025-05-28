"""inference.py – Generate pred.npz for submission
==================================================
Usage:
    python inference.py \
        --ckpt checkpoints/best.pth \
        --test_dir dataset/test/degraded \
        --out pred.npz

• Supports full image or patch-based sliding window inference (--patch 256)
• Preprocess: normalize (0‑255 → −1..1) as in training
• Output: (3, H, W) uint8, keyed by original filename
"""

from __future__ import annotations
import argparse
import os
import math
import yaml
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F
from tqdm import tqdm

import zipfile

# --------------------------------------------------------
# Load PromptIR model
# --------------------------------------------------------
try:
    from model import PromptIR
except ImportError:
    raise ImportError("Cannot find model.py. Please ensure it is in the same directory or PYTHONPATH.")

# --------------------------------------------------------
# Preprocessing / Postprocessing helpers
# --------------------------------------------------------

def preprocess(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to normalized torch tensor with shape (1, 3, H, W)
    Pixel values are scaled from [0, 255] → [-1, 1]
    """
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
    x = x * 2.0 - 1.0
    return x.unsqueeze(0)  # Add batch dimension


def postprocess(t: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor back to uint8 numpy image
    Shape: (1, 3, H, W) → (3, H, W)
    Value range: [-1, 1] → [0, 255]
    """
    t = (t.clamp_(-1, 1) + 1.0) * 127.5
    t = t.squeeze(0).cpu().numpy().round().astype(np.uint8)
    return t

# --------------------------------------------------------
# Sliding-window inference (for large images)
# --------------------------------------------------------

def sliding_window(model, im: Image.Image, patch: int, overlap: int = 20) -> np.ndarray:
    """
    Perform inference by splitting large image into overlapping patches.
    Result is stitched back into full-size image.
    Returns: (3, H, W) uint8 numpy image
    """
    w, h = im.size
    stride = patch - overlap

    # Pad image to ensure full coverage
    pad_w = (math.ceil(w / stride) * stride + overlap) - w
    pad_h = (math.ceil(h / stride) * stride + overlap) - h
    im_pad = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
    im_pad.paste(im, (0, 0))

    out_canvas = np.zeros((3, h + pad_h, w + pad_w), dtype=np.float32)
    weight_canvas = np.zeros_like(out_canvas, dtype=np.float32)

    for y in range(0, h + pad_h - overlap, stride):
        for x in range(0, w + pad_w - overlap, stride):
            crop = im_pad.crop((x, y, x + patch, y + patch))
            with torch.no_grad(), torch.cuda.amp.autocast():
                inp = preprocess(crop).to(device)
                pred = model(inp)
            pred_np = pred.squeeze(0).cpu().numpy()
            out_canvas[:, y:y + patch, x:x + patch] += pred_np
            weight_canvas[:, y:y + patch, x:x + patch] += 1.0

    out_canvas /= np.maximum(weight_canvas, 1.0)
    out_canvas = (np.clip(out_canvas, -1, 1) + 1.0) * 127.5
    return out_canvas[:, :h, :w].round().astype(np.uint8)

# --------------------------------------------------------
# Main Inference Pipeline
# --------------------------------------------------------

def main(args):
    # 1. Load model and weights
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptIR(inp_channels=3, out_channels=3, decoder=True).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 2. Load test images
    test_path = Path(args.test_dir)
    img_files = sorted([
        f for f in test_path.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])
    images_dict: Dict[str, Any] = {}

    # 3. Run inference on each image
    for fp in tqdm(img_files, desc="Infer", leave=False):
        img = Image.open(fp).convert("RGB")
        if args.patch == 0:
            # Run full image inference
            with torch.no_grad(), torch.amp.autocast('cuda'):
                pred = model(preprocess(img).to(device))
            pred_np = postprocess(pred)
        else:
            # Use sliding window inference
            pred_np = sliding_window(model, img, patch=args.patch, overlap=args.overlap)
        images_dict[fp.name] = pred_np

    # 4. Save pred.npz to output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "pred.npz"
    np.savez_compressed(npz_path, **images_dict)
    print(f"Saved {len(images_dict)} images to {npz_path}")

    # --------------------------------------------------------
    # Save first 10 predictions as PNG for visualization
    # --------------------------------------------------------
    viz_dir = out_dir / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for i, (fname, pred_np) in enumerate(images_dict.items()):
        if i >= 10:
            break
        # Convert shape (3, H, W) → (H, W, 3) for PIL
        img = Image.fromarray(np.transpose(pred_np, (1, 2, 0)))
        out_name = Path(fname).stem + ".png"
        img.save(viz_dir / out_name)
    print(f"Saved visualization of first 10 images to {viz_dir}")

    # --------------------------------------------------------
    # Compress output folder into a ZIP file
    # --------------------------------------------------------
    zip_name = out_dir.name + ".zip"
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(npz_path, arcname="pred.npz")
    print(f"Compressed only pred.npz to: {zip_path}")

# --------------------------------------------------------
# Entry Point
# --------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PromptIR inference → pred.npz")
    p.add_argument("--ckpt", default="checkpoints/best.pth", help="Path to model checkpoint")
    p.add_argument("--test_dir", default="dataset/test/degraded", help="Directory of degraded test images")
    p.add_argument("--out", default="results/0519", help="Output directory")
    p.add_argument("--patch", type=int, default=0, help="Patch size (0 = full image)")
    p.add_argument("--overlap", type=int, default=20, help="Overlap size in pixels for sliding window")
    args = p.parse_args()

    main(args)
