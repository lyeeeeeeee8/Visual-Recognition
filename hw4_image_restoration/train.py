"""train2.py – PromptIR (official) with warm-up & EMA
====================================================
Enhancements
------------
1. 5-epoch linear warm-up followed by cosine LR decay.
2. EMA (Exponential Moving Average) starts updating after epoch 10.
3. Maintains early stopping, curve saving, TensorBoard logging.
4. Fixes CUDA device to GPU 1.
5. PEP8-compliant formatting.
"""
from __future__ import annotations

import argparse
import json
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from dataloader import PromptIRDataset
from model import PromptIR
from torchmetrics.image import StructuralSimilarityIndexMeasure

# --------------------------------------------------------
# Basic setup: device and SSIM metric initialization
# --------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SSIM_METRIC = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
print("Current CUDA device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# --------------------------------------------------------
# Utilities
# --------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def psnr(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Calculate Peak Signal-to-Noise Ratio between prediction and target."""
    mse = F.mse_loss(pred, target, reduction="mean")
    return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))


class EarlyStopper:
    """Helper class to stop training early if metric doesn't improve."""

    def __init__(self, patience: int = 20, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best = -math.inf
        self.counter = 0

    def step(self, metric: float) -> bool:
        """Check if training should stop based on current metric."""
        if metric > self.best + self.delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# --------------------------------------------------------
# Builders for dataloaders, model, optimizer, scheduler
# --------------------------------------------------------

def build_dataloaders(cfg: Dict[str, Any], val_ratio: float) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders."""
    ds_full = PromptIRDataset("dataset", split="train", patch_size=cfg["patch_size"])
    val_sz = int(len(ds_full) * val_ratio)
    train_sz = len(ds_full) - val_sz
    train_ds, val_ds = torch.utils.data.random_split(
        ds_full,
        [train_sz, val_sz],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    dl_kw = {"num_workers": cfg["num_workers"], "pin_memory": True}
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **dl_kw)
    return train_dl, val_dl


def build_model() -> nn.Module:
    """Initialize the PromptIR model and move to device."""
    model = PromptIR(inp_channels=3, out_channels=3, decoder=True)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {params_m:.2f} M")
    return model.to(DEVICE)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    """Create AdamW optimizer with config parameters."""
    opt_cfg = cfg["optim"]
    return torch.optim.AdamW(
        model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"]
    )


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    """Create learning rate scheduler with warm-up and cosine decay."""
    warm_up = 5
    total = cfg["max_epoch"]

    def lr_lambda(epoch: int):
        if epoch < warm_up:
            return float(epoch + 1) / warm_up
        progress = (epoch - warm_up) / max(1, total - warm_up)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

# --------------------------------------------------------
# Loss function combining L1 loss and SSIM loss
# --------------------------------------------------------

def loss_fn(out: Tensor, gt: Tensor, l1_w: float, ssim_w: float):
    """Calculate combined L1 and SSIM loss."""
    l1 = F.l1_loss(out, gt)
    ssim_val = SSIM_METRIC(out, gt)
    ssim_loss = 1 - ssim_val
    return l1_w * l1 + ssim_w * ssim_loss, l1.item()

# --------------------------------------------------------
# Plotting and logging utilities
# --------------------------------------------------------

def save_curves(curves: Dict[str, List[float]], save_dir: Path) -> None:
    """Plot and save training/validation loss and PSNR curves."""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(curves["train_loss"], label="train_loss", color="tab:blue")
    ax1.plot(curves["val_loss"], label="val_loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.grid()
    ax1.legend()

    ax2.plot(curves["val_psnr"], label="val_psnr", color="tab:green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR")
    ax2.set_title("Val PSNR")
    ax2.grid()
    ax2.legend()

    fig.tight_layout()
    plt.savefig(save_dir / "curves.png")
    plt.close(fig)

    with open(save_dir / "curves.json", "w", encoding="utf-8") as fp:
        json.dump(curves, fp)


def append_log(text: str, log_file: Path) -> None:
    """Append a line of text to the log file."""
    with open(log_file, "a", encoding="utf-8") as fp:
        fp.write(text + "\n")

# --------------------------------------------------------
# Training and validation loops
# --------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dl: DataLoader,
    optimizer,
    scaler: GradScaler,
    cfg: Dict[str, Any],
    epoch: int,
    writer: SummaryWriter,
) -> float:
    """Run one epoch of training."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(dl, leave=False, desc=f"Train {epoch:03d}")
    for step, batch in enumerate(pbar):
        inp = batch["degraded"].to(DEVICE)
        gt = batch["clean"].to(DEVICE)
        with autocast():
            out = model(inp)
            loss, _ = loss_fn(out, gt, cfg["loss"]["l1"], cfg["loss"]["ssim"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        global_step = epoch * len(dl) + step
        writer.add_scalar("train/loss", loss.item(), global_step)

    return running_loss / len(dl)


@torch.no_grad()
def validate(
    model: nn.Module,
    dl: DataLoader,
    cfg: Dict[str, Any],
    epoch: int,
    writer: SummaryWriter,
) -> Tuple[float, float]:
    """Run validation and return average loss and PSNR."""
    model.eval()
    tot_loss = 0.0
    tot_psnr = 0.0
    for batch in dl:
        inp = batch["degraded"].to(DEVICE)
        gt = batch["clean"].to(DEVICE)
        out = model(inp)
        loss, _ = loss_fn(out, gt, cfg["loss"]["l1"], cfg["loss"]["ssim"])
        tot_loss += loss.item()
        tot_psnr += psnr(out, gt).item()
    avg_loss = tot_loss / len(dl)
    avg_psnr = tot_psnr / len(dl)
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/psnr", avg_psnr, epoch)
    return avg_loss, avg_psnr

# --------------------------------------------------------
# Main entry point
# --------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base2.yaml")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if args.epochs is not None:
        cfg["max_epoch"] = args.epochs
    cfg.setdefault("seed", 42)

    seed_everything(cfg["seed"])

    save_dir = Path(cfg.get("save_dir", "runs/train2"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "log.txt"

    # Build components
    train_dl, val_dl = build_dataloaders(cfg, args.val_ratio)
    model = build_model()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=DEVICE.type == "cuda")
    writer = SummaryWriter(log_dir=save_dir / "tb")

    # EMA setup: starts updating after epoch 10
    ema_model = AveragedModel(model)
    use_ema_after = 10

    curves = {"train_loss": [], "val_loss": [], "val_psnr": []}
    best_psnr = -math.inf
    stopper = EarlyStopper(patience=cfg.get("early_stop", 20))

    try:
        for epoch in range(cfg["max_epoch"]):
            t0 = time.time()

            train_loss = train_one_epoch(
                model, train_dl, optimizer, scaler, cfg, epoch, writer
            )

            # Update LR scheduler after epoch
            scheduler.step()

            # Validation using EMA or raw model
            if epoch >= use_ema_after:
                ema_model.update_parameters(model)
                val_loss, val_psnr = validate(
                    ema_model.module, val_dl, cfg, epoch, writer
                )
            else:
                val_loss, val_psnr = validate(model, val_dl, cfg, epoch, writer)

            curves["train_loss"].append(train_loss)
            curves["val_loss"].append(val_loss)
            curves["val_psnr"].append(val_psnr)

            msg = (
                f"Epoch {epoch:03d}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_psnr={val_psnr:.2f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"time={time.time() - t0:.1f}s"
            )
            print(msg)
            append_log(msg, log_file)

            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), save_dir / "best.pth")

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save(model.state_dict(), save_dir / f"epoch_{epoch:03d}.pth")

            # Early stopping check
            if stopper.step(val_psnr):
                print("Early stopping triggered.")
                break

            # Save last checkpoint every epoch
            torch.save(model.state_dict(), save_dir / "last.pth")

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        save_curves(curves, save_dir)
        writer.close()
        print(f"Finished. Best PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    main()
