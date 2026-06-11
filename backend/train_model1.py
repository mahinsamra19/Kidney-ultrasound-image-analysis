"""
Model 1 Training Script — Self-supervised pretraining + fine-tuning
=======================================================================
Strategy:
  1. SimCLR self-supervised pretraining on unlabeled ultrasound images
  2. Linear probing / fine-tuning on any available labeled subset
  3. Saves checkpoint compatible with inference.py

Usage:
  # Pretrain (unlabeled data):
  python train_model1.py --phase pretrain --data_dir /path/to/unlabeled_images

  # Fine-tune (with labels):
  python train_model1.py --phase finetune --data_dir /path/to/labeled --ckpt weights/model1_pretrained.pth

  # Evaluate:
  python train_model1.py --phase eval --data_dir /path/to/test --ckpt weights/model1_resnet50.pth
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASSES = ["normal", "stone", "hydronephrosis", "cyst", "tumor"]
IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Augmentations ─────────────────────────────────────────────────────────────

# For SimCLR — strong augmentations to create two views of each image
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# For supervised fine-tuning
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Datasets ──────────────────────────────────────────────────────────────────

class UnlabeledUltrasoundDataset(Dataset):
    """Dataset for self-supervised pretraining — returns two augmented views."""

    def __init__(self, root: str, transform=None):
        self.root      = Path(root)
        self.transform = transform or simclr_transform
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        self.files = [p for p in self.root.rglob("*") if p.suffix.lower() in exts]
        logger.info(f"UnlabeledDataset: {len(self.files)} images from {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), self.transform(img)   # two random views


class LabeledUltrasoundDataset(Dataset):
    """
    Labeled dataset — expects directory layout:
      root/
        normal/      *.png
        stone/       *.png
        hydronephrosis/*.png
        cyst/        *.png
        tumor/       *.png
    """

    def __init__(self, root: str, transform=None, classes=CLASSES):
        self.root      = Path(root)
        self.transform = transform
        self.classes   = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []
        for cls in classes:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for f in cls_dir.iterdir():
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((f, self.class_to_idx[cls]))
        logger.info(f"LabeledDataset: {len(self.samples)} samples, classes={classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── SimCLR ────────────────────────────────────────────────────────────────────

class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class SimCLRModel(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, proj_dim: int = 128):
        super().__init__()
        self.backbone   = backbone
        self.projector  = SimCLRProjectionHead(feat_dim, proj_dim)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent contrastive loss (SimCLR)."""
    N  = z1.size(0)
    z  = torch.cat([z1, z2], dim=0)    # [2N, D]
    sim = torch.mm(z, z.T) / temperature
    sim.fill_diagonal_(float("-inf"))   # mask self-similarity

    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, labels)


def pretrain_simclr(data_dir: str, epochs: int = 100, batch_size: int = 64, lr: float = 3e-4):
    dataset    = UnlabeledUltrasoundDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    backbone = models.resnet50(weights="IMAGENET1K_V1")
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    model     = SimCLRModel(backbone, feat_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (x1, x2) in dataloader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss  = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        scheduler.step()
        logger.info(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            Path("weights").mkdir(exist_ok=True)
            torch.save({"epoch": epoch, "model_state_dict": model.backbone.state_dict(), "loss": best_loss},
                       "weights/model1_pretrained.pth")
            logger.info(f"  Saved best checkpoint (loss={best_loss:.4f})")

    return model.backbone


def finetune(
    data_dir:    str,
    ckpt_path:   str  = None,
    epochs:      int  = 50,
    batch_size:  int  = 32,
    lr:          float = 1e-4,
    num_classes: int  = 5,
):
    dataset = LabeledUltrasoundDataset(data_dir, transform=train_transform)
    n_val   = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    val_ds.dataset    = LabeledUltrasoundDataset(data_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # Build model
    model = models.resnet50(weights=None)
    feat_dim = model.fc.in_features
    model.fc = nn.Identity()

    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        logger.info(f"Loaded pretrained weights from {ckpt_path}")

    # Add classification head
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))
    model    = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds   = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total * 100
        scheduler.step()
        logger.info(f"Epoch {epoch}/{epochs} | TrainLoss: {train_loss/len(train_loader):.4f} | ValAcc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path("weights").mkdir(exist_ok=True)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc},
                       "weights/model1_resnet50.pth")
            logger.info(f"  Saved best model (val_acc={val_acc:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",       choices=["pretrain", "finetune", "eval"], default="pretrain")
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--ckpt",        default=None)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--num_classes", type=int,   default=5)
    args = parser.parse_args()

    if args.phase == "pretrain":
        pretrain_simclr(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.phase == "finetune":
        finetune(args.data_dir, ckpt_path=args.ckpt, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, num_classes=args.num_classes)
    else:
        logger.info("Eval phase — run your evaluation loop here")
