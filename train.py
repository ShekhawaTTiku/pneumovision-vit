"""
train.py - Full training script for the PyTorch ViT.

Usage:
    .\venv311\Scripts\python train.py

Mirrors the original notebook training configuration:
  - AdamW (lr=0.001, weight_decay=1e-4)
  - CrossEntropyLoss on 2-class logits
  - 30 epochs, batch_size=16
  - Best checkpoint saved on val F1 (same as original ModelCheckpoint)
  - Early stopping patience=20 on val accuracy (same as original)
  - Metrics: loss, accuracy, F1, precision, recall (train + val per epoch)
"""

import os
import csv
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for headless/Windows)
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from vit_model import VisionTransformer, count_parameters
from dataset_utils import get_dataloaders, dataset_summary


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  train", leave=False, ncols=80):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    avg_loss = running_loss / n
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Evaluation (val or test)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  eval ", leave=False, ncols=80):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    avg_loss = running_loss / n
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Metrics (using sklearn for reliability - same stats as original notebook)
# ---------------------------------------------------------------------------
def compute_metrics(labels, preds):
    """Returns dict with accuracy, f1, precision, recall."""
    labels = np.array(labels)
    preds  = np.array(preds)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="binary", pos_label=1, zero_division=0),
        "precision": precision_score(labels, preds, average="binary", pos_label=1, zero_division=0),
        "recall":    recall_score(labels, preds, average="binary", pos_label=1, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_history(history: list[dict], save_path: str):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History - ViT Chest X-Ray", fontsize=14)

    metrics = [
        ("loss",      "Loss"),
        ("accuracy",  "Accuracy"),
        ("f1",        "F1 Score"),
        ("precision", "Precision"),
    ]
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.plot(epochs, [r[f"train_{key}"] for r in history], label="Train")
        ax.plot(epochs, [r[f"val_{key}"]   for r in history], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved training curves -> {save_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main(sanity_check: bool = False):
    set_seed(config.SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    print("\nLoading data...")
    dataset_summary()
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"Classes: {class_names}")

    if sanity_check:
        # Quick sanity: 3 mini-epochs over 5 batches each
        print("\n=== SANITY CHECK MODE (5 batches x 3 epochs) ===")
        from itertools import islice
        mini_train = list(islice(train_loader, 5))
        mini_val   = list(islice(val_loader,   3))

        class MiniLoader:
            def __init__(self, data): self.data = data; self.dataset = data
            def __iter__(self): return iter(self.data)
            def __len__(self): return len(self.data)

        train_loader = MiniLoader(mini_train)
        val_loader   = MiniLoader(mini_val)
        epochs = 3
    else:
        epochs = config.EPOCHS

    # Model
    print("\nInstantiating ViT...")
    model = VisionTransformer().to(device)
    print(f"  Trainable parameters: {count_parameters(model):,}")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    # Dirs
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR,    exist_ok=True)

    history           = []
    best_val_f1       = -1.0
    early_stop_counter = 0
    best_val_acc      = -1.0
    history_csv_path  = os.path.join(config.RESULTS_DIR, "training_history.csv")
    curves_path       = os.path.join(config.RESULTS_DIR, "training_curves.png")

    print(f"\nStarting training for up to {epochs} epoch(s)...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_m   = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        row = {
            "epoch":          epoch,
            "train_loss":     round(train_loss, 5),
            "train_accuracy": round(train_m["accuracy"], 5),
            "train_f1":       round(train_m["f1"], 5),
            "train_precision":round(train_m["precision"], 5),
            "train_recall":   round(train_m["recall"], 5),
            "val_loss":       round(val_loss, 5),
            "val_accuracy":   round(val_m["accuracy"], 5),
            "val_f1":         round(val_m["f1"], 5),
            "val_precision":  round(val_m["precision"], 5),
            "val_recall":     round(val_m["recall"], 5),
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{epochs} [{elapsed:.0f}s]  "
            f"train_loss={train_loss:.4f}  train_acc={train_m['accuracy']:.4f}  train_f1={train_m['f1']:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_m['accuracy']:.4f}  val_f1={val_m['f1']:.4f}"
        )

        # Save best checkpoint (monitor val F1 - same as original ModelCheckpoint)
        if not sanity_check and val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":     best_val_f1,
                "val_acc":    val_m["accuracy"],
                "class_names": class_names,
                "config": {
                    "IMAGE_DIM":    config.IMAGE_DIM,
                    "PATCH_SIZE":   config.PATCH_SIZE,
                    "NUM_PATCHES":  config.NUM_PATCHES,
                    "PROJ_DIM":     config.PROJ_DIM,
                    "NUM_HEADS":    config.NUM_HEADS,
                    "TRANS_LAYERS": config.TRANS_LAYERS,
                    "NUM_CLASSES":  config.NUM_CLASSES,
                },
            }, config.BEST_CHECKPOINT)
            print(f"  *** Saved best checkpoint (val_f1={best_val_f1:.4f}) -> {config.BEST_CHECKPOINT}")

        # Early stopping (monitor val accuracy - same as original EarlyStopping)
        if val_m["accuracy"] > best_val_acc:
            best_val_acc      = val_m["accuracy"]
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if not sanity_check and early_stop_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no val_acc improvement for {config.EARLY_STOP_PATIENCE} epochs).")
                break

    # Save CSV
    if history:
        fieldnames = list(history[0].keys())
        with open(history_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        print(f"\nTraining history saved -> {history_csv_path}")

    # Save plot
    if len(history) > 1 and not sanity_check:
        plot_history(history, curves_path)

    if sanity_check:
        print("\n=== SANITY CHECK COMPLETE ===")
        losses = [r["train_loss"] for r in history]
        print(f"  Epoch losses: {losses}")
        if losses[-1] < losses[0]:
            print("  Loss decreased - training pipeline OK.")
        else:
            print("  WARNING: Loss did not decrease. Check data loading and model init.")
    else:
        print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
        print(f"Best checkpoint: {config.BEST_CHECKPOINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="Run short sanity check only")
    args = parser.parse_args()
    main(sanity_check=args.sanity)
