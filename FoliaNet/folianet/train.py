"""
Train the FoliaNet image classifier on an ImageFolder-style dataset.

    python -m folianet.train --config configs/default.yaml

Saves a checkpoint containing the model weights AND the class list + metadata, so
inference stays consistent without re-specifying classes.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from folianet.config import load_config
from folianet.data.dataset import CropDiseaseDataset, build_transforms
from folianet.models.fusion_model import FusionModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_full = CropDiseaseDataset(
        cfg.data.root, cfg.data.crops,
        transform=build_transforms(cfg.data.img_size, train=True))
    val_full = CropDiseaseDataset(
        cfg.data.root, cfg.data.crops,
        transform=build_transforms(cfg.data.img_size, train=False))

    n_total = len(train_full)
    n_val = max(1, int(n_total * cfg.data.val_split))
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_set = torch.utils.data.Subset(train_full, train_idx)
    val_set = torch.utils.data.Subset(val_full, val_idx)
    n_train = len(train_set)

    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_set, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=(device == "cuda"))

    classes = train_full.classes
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train: {n_train}  Val: {n_val}")

    model = FusionModel(
        num_classes=len(classes),
        tabular_dim=cfg.model.tabular_dim,
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {va_acc:.3f} "
              f"| train loss {tr_loss:.3f} | val loss {va_loss:.3f}")
        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "backbone": cfg.model.backbone,
                "tabular_dim": cfg.model.tabular_dim,
                "img_size": cfg.data.img_size,
                "val_acc": va_acc,
            }, ckpt_dir / "folianet_best.pt")
            print(f"  saved checkpoint (val acc {va_acc:.3f})")

    print(f"Done. Best val acc: {best_acc:.3f}")


if __name__ == "__main__":
    main()
