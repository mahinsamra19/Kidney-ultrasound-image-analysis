"""
Model 2 Training Script — Supervised Binary Classifier (stone vs normal)
=========================================================================
Backbone: EfficientNet-B3 with class-weighted loss to handle imbalance.

Usage:
  python train_model2.py --data_dir /path/to/labeled_data
                         --epochs 60
                         --batch_size 32
                         --lr 3e-4

Directory layout expected:
  data_dir/
    normal/  *.png
    stone/   *.png

The script:
  - Applies strong medical-image-specific augmentations
  - Uses class-weighted CrossEntropyLoss for imbalanced datasets
  - Evaluates with AUC, sensitivity, specificity (clinically meaningful metrics)
  - Saves best checkpoint by AUC
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASSES = ["normal", "stone"]
IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Augmentation pipeline ─────────────────────────────────────────────────────
# Medical imaging augmentations: avoid color distortions that change clinical meaning
# but simulate realistic scan variability (probe angle, depth, gain)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 48, IMG_SIZE + 48)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    # Simulate gain/brightness variation (common in ultrasound)
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),   # simulate scan artifacts
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class BinaryKidneyDataset(Dataset):
    def __init__(self, root: str, transform=None, classes=CLASSES):
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []
        for cls in classes:
            d = Path(root) / cls
            if not d.exists():
                logger.warning(f"Class directory not found: {d}")
                continue
            for f in d.iterdir():
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((f, self.class_to_idx[cls]))
        logger.info(f"Dataset: {len(self.samples)} samples from {root}")
        self._log_class_counts()

    def _log_class_counts(self):
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        for idx, cls in enumerate(CLASSES):
            logger.info(f"  {cls}: {counts.get(idx, 0)} samples")

    def get_class_weights(self) -> torch.Tensor:
        """Returns per-sample weights for WeightedRandomSampler (handles imbalance)."""
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        total  = len(self.samples)
        weights = [total / counts[label] for _, label in self.samples]
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = "IMAGENET1K_V1" if pretrained else None
    model   = models.efficientnet_b3(weights=weights)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(in_feat, 256),
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def evaluate(model, loader, device) -> dict:
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    acc = (all_preds == all_labels).mean() * 100

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, len(all_labels)))
    sensitivity = tp / (tp + fn + 1e-8) * 100
    specificity = tn / (tn + fp + 1e-8) * 100

    return {
        "auc":         auc,
        "accuracy":    acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def train(
    data_dir:   str,
    epochs:     int   = 60,
    batch_size: int   = 32,
    lr:         float = 3e-4,
    save_dir:   str   = "weights",
):
    Path(save_dir).mkdir(exist_ok=True)

    full_dataset = BinaryKidneyDataset(data_dir, transform=train_transform)
    n_val   = max(1, int(0.15 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    val_ds_eval = BinaryKidneyDataset(data_dir, transform=val_transform)
    val_indices = val_ds.indices
    val_ds_eval.samples = [val_ds_eval.samples[i] for i in val_indices]

    # Weighted sampler for class imbalance on training set
    train_weights = full_dataset.get_class_weights()[train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds_eval, batch_size=batch_size, shuffle=False, num_workers=4)

    model     = build_model(num_classes=2, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 5,
        steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.1,
    )

    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        metrics = evaluate(model, val_loader, DEVICE)
        logger.info(
            f"Epoch {epoch:03d}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | "
            f"AUC: {metrics['auc']:.4f} | Acc: {metrics['accuracy']:.1f}% | "
            f"Sens: {metrics['sensitivity']:.1f}% | Spec: {metrics['specificity']:.1f}%"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            ckpt = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "metrics":          metrics,
                "classes":          CLASSES,
            }
            torch.save(ckpt, f"{save_dir}/model2_efficientnet.pth")
            logger.info(f"  Saved best model — AUC: {best_auc:.4f}")

    logger.info(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    logger.info(f"Model saved to {save_dir}/model2_efficientnet.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--save_dir",   default="weights")
    args = parser.parse_args()

    train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, save_dir=args.save_dir)
