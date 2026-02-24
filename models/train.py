"""
models/train.py — Training Script for DMS CNN + LSTM Models
Fine-tunes the MobileNetV3 CNN on a drowsiness dataset, then trains
the GRU temporal model on extracted feature sequences.

Dataset structure expected at config.TRAIN_DATA_DIR:
  data/
    alert/      ← face crop images (jpg/png)
    drowsy/     ← face crop images
    yawning/    ← face crop images

Run:
  python -m models.train --mode cnn
  python -m models.train --mode lstm
  python -m models.train --mode all
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models.cnn_classifier import DrowsinessClassifier
from models.lstm_model import TemporalDrowsinessModel

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

CLASS_MAP = {"alert": 0, "drowsy": 1, "yawning": 2}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DrowsinessDataset(Dataset):
    """
    Loads face crop images from a folder-per-class directory structure.
    Applies ImageNet normalization + augmentation during training.
    """

    def __init__(self, data_dir: str, augment: bool = True):
        self.samples: list[tuple[Path, int]] = []
        for class_name, label in CLASS_MAP.items():
            class_dir = Path(data_dir) / class_name
            if not class_dir.is_dir():
                print(f"[Train] WARNING: Missing class folder: {class_dir}")
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(
                f"[Train] No images found in {data_dir}. "
                "Create subfolders: alert/, drowsy/, yawning/"
            )

        print(f"[Train] Loaded {len(self.samples)} images from {data_dir}")

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ──────────────────────────────────────────────────────────────────────────────
# CNN Training
# ──────────────────────────────────────────────────────────────────────────────

def train_cnn(device: str = "cpu"):
    """Fine-tune the MobileNetV3 classifier on the drowsiness dataset."""
    print("\n─── CNN Training ─────────────────────────────────────")
    dataset = DrowsinessDataset(config.TRAIN_DATA_DIR, augment=True)

    val_size = int(len(dataset) * config.TRAIN_VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    # Use non-augmented transform for validation
    val_ds.dataset = DrowsinessDataset(config.TRAIN_DATA_DIR, augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.TRAIN_BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = DrowsinessClassifier(pretrained=True).to(device)

    # Optionally unfreeze only the head for the first N epochs (transfer learning)
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.TRAIN_LR,
        weight_decay=config.TRAIN_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAIN_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_count = 0

    for epoch in range(1, config.TRAIN_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        # Unfreeze backbone after epoch 5
        if epoch == 5:
            print("[Train] Unfreezing backbone for full fine-tuning.")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(),
                                    lr=config.TRAIN_LR * 0.1,
                                    weight_decay=config.TRAIN_WEIGHT_DECAY)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        train_acc = correct / total
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / val_total
        print(
            f"Epoch {epoch:3d}/{config.TRAIN_EPOCHS} | "
            f"Train Loss: {train_loss/total:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.3f}"
        )

        # ── Early stopping + checkpointing ─────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            model.save(config.CNN_MODEL_PATH)
            print(f"  ✓ New best val acc: {val_acc:.3f} — checkpoint saved.")
        else:
            patience_count += 1
            if patience_count >= config.TRAIN_EARLY_STOPPING_PATIENCE:
                print(f"[Train] Early stopping at epoch {epoch}.")
                break

    print(f"\nCNN training complete. Best val accuracy: {best_val_acc:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# LSTM Training (simplified — uses precomputed feature sequences)
# ──────────────────────────────────────────────────────────────────────────────

def train_lstm(device: str = "cpu"):
    """
    Train the GRU temporal model on pre-extracted feature sequences.
    
    Expects numpy files at config.TRAIN_DATA_DIR:
      sequences_X.npy  — shape (N, T, feature_dim)
      sequences_y.npy  — shape (N,) binary labels {0=alert, 1=drowsy}
    
    Generate these with: python utils/extract_sequences.py
    """
    print("\n─── LSTM Training ────────────────────────────────────")
    x_path = os.path.join(config.TRAIN_DATA_DIR, "sequences_X.npy")
    y_path = os.path.join(config.TRAIN_DATA_DIR, "sequences_y.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(
            "[Train] LSTM data not found. Please generate sequences first:\n"
            "  python utils/extract_sequences.py\n"
            "  (Requires trained CNN weights.)"
        )
        return

    X = torch.from_numpy(np.load(x_path)).float()
    y = torch.from_numpy(np.load(y_path)).float().unsqueeze(1)
    print(f"[Train] Loaded sequences: X={X.shape}, y={y.shape}")

    N = len(X)
    val_n = int(N * config.TRAIN_VAL_SPLIT)
    train_n = N - val_n
    indices = torch.randperm(N)
    train_idx, val_idx = indices[:train_n], indices[train_n:]

    train_loader = DataLoader(
        list(zip(X[train_idx], y[train_idx])),
        batch_size=config.TRAIN_BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        list(zip(X[val_idx], y[val_idx])),
        batch_size=config.TRAIN_BATCH_SIZE,
    )

    model = TemporalDrowsinessModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN_LR,
                             weight_decay=config.TRAIN_WEIGHT_DECAY)
    criterion = nn.BCELoss()
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, config.TRAIN_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for seq, label in train_loader:
            seq, label = seq.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(seq)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, label in val_loader:
                seq, label = seq.to(device), label.to(device)
                val_loss += criterion(model(seq), label).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            model.save(config.LSTM_MODEL_PATH)
            print(f"  ✓ Best val loss: {val_loss:.4f}")
        else:
            patience_count += 1
            if patience_count >= config.TRAIN_EARLY_STOPPING_PATIENCE:
                print("[Train] Early stopping.")
                break

    print(f"\nLSTM training complete. Best val loss: {best_val_loss:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMS Model Training")
    parser.add_argument("--mode", choices=["cnn", "lstm", "all"], default="cnn",
                        help="Which model to train.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device: cpu | cuda | mps")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        device = "cpu"

    print(f"[Train] Device: {device}")

    if args.mode in ("cnn", "all"):
        train_cnn(device)
    if args.mode in ("lstm", "all"):
        train_lstm(device)
