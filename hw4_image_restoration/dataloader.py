import os
import glob
from typing import Dict, Any

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


class PromptIRDataset(Dataset):
    """
    Dataset for loading paired (degraded, clean) images.

    - If split='train': returns {'degraded', 'clean', 'label'}
    - If split='test' : returns {'degraded', 'fname'}
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        patch_size: int = 256,
    ) -> None:
        super().__init__()
        assert split in ("train", "test")
        self.split = split
        self.root_dir = root_dir
        self.patch_size = patch_size

        if split == "train":
            # Load all degraded image paths for training
            self.deg_paths = sorted(
                glob.glob(os.path.join(root_dir, "train/degraded", "*.png"))
            )
            self.clean_root = os.path.join(root_dir, "train/clean")

            # Joint augmentation for degraded and clean images
            self.transform = A.Compose(
                [
                    # A.RandomCrop(patch_size, patch_size),  # optional

                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    # A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),  # optional

                    A.Normalize(mean=(0.5,) * 3, std=(0.5,) * 3),
                    ToTensorV2(),
                ],
                additional_targets={"clean": "image"},  # Ensure same transform is applied to 'clean'
            )
        else:  # test split
            # Load all degraded test images
            self.deg_paths = sorted(
                glob.glob(os.path.join(root_dir, "test/degraded", "*.png"))
            )
            self.transform = A.Compose(
                [
                    A.Normalize(mean=(0.5,) * 3, std=(0.5,) * 3),
                    ToTensorV2(),
                ]
            )

    # --------------------------------------------------------
    # Private Utilities
    # --------------------------------------------------------

    def _paired_clean_path(self, deg_path: str) -> str:
        """
        Derive clean image path from degraded filename.

        Examples:
            rain-123.png  -> rain_clean-123.png
            snow-045.png  -> snow_clean-045.png
        """
        fname = os.path.basename(deg_path)
        if fname.startswith("rain"):
            clean_name = fname.replace("rain", "rain_clean")
        else:
            clean_name = fname.replace("snow", "snow_clean")
        return os.path.join(self.clean_root, clean_name)

    # --------------------------------------------------------
    # torch Dataset Interface
    # --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.deg_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        deg_path = self.deg_paths[idx]
        deg_img = np.array(Image.open(deg_path).convert("RGB"))

        if self.split == "train":
            clean_path = self._paired_clean_path(deg_path)
            clean_img = np.array(Image.open(clean_path).convert("RGB"))

            # Apply the same augmentation to both degraded and clean images
            aug = self.transform(image=deg_img, clean=clean_img)
            deg_img_t = aug["image"]        # Tensor shape: [3, H, W]
            clean_img_t = aug["clean"]      # Tensor shape: [3, H, W]
            label = 0 if "rain" in deg_path else 1

            return {
                "degraded": deg_img_t,
                "clean": clean_img_t,
                "label": torch.tensor(label, dtype=torch.long),
            }

        else:  # test mode
            aug = self.transform(image=deg_img)
            deg_img_t = aug["image"]

            return {
                "degraded": deg_img_t,
                "fname": os.path.basename(deg_path)
            }

# --------------------------------------------------------
# Sanity check: confirm dataloader and tensor shapes
# --------------------------------------------------------

def run_sanity_check():
    """
    Load a random batch and confirm:
    1. Shapes of degraded / clean images are [B, 3, H, W]
    2. Labels are correct (0=rain, 1=snow)
    3. Data can be moved to GPU without issue
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds = PromptIRDataset(root_dir="dataset", split="train", patch_size=256)
    dl = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    batch = next(iter(dl))
    degraded = batch["degraded"].to(device)
    clean = batch["clean"].to(device)
    label = batch["label"].to(device)

    print(f"degraded: {degraded.shape} {degraded.dtype}")
    print(f"clean   : {clean.shape}")
    print(f"label   : {label}")

    # Value check: should roughly be in range (-1, 1) after normalization
    print(
        f"degraded pixel range ~ [{degraded.min():.2f}, {degraded.max():.2f}]"
    )

    # Estimate memory usage for the current batch
    print(
        f"Batch memory footprint = {degraded.nelement() * degraded.element_size() / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    run_sanity_check()
