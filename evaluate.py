"""
evaluate.py - Evaluate the best saved checkpoint on the held-out test set.

Usage:
    .\venv311\Scripts\python evaluate.py

Loads checkpoints/best_vit.pt and evaluates on chest_xray/test/ only.
Prints a full sklearn classification report.
The test set is NEVER used during training or model selection.
"""

import os
import json

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from tqdm import tqdm

import config
from vit_model import VisionTransformer
from dataset_utils import get_dataloaders


@torch.no_grad()
def evaluate_on_test(checkpoint_path: str = config.BEST_CHECKPOINT):
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please run train.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model from checkpoint config
    saved_cfg = checkpoint.get("config", {})
    model = VisionTransformer(
        image_dim    = saved_cfg.get("IMAGE_DIM",    config.IMAGE_DIM),
        patch_size   = saved_cfg.get("PATCH_SIZE",   config.PATCH_SIZE),
        num_patches  = saved_cfg.get("NUM_PATCHES",  config.NUM_PATCHES),
        proj_dim     = saved_cfg.get("PROJ_DIM",     config.PROJ_DIM),
        num_heads    = saved_cfg.get("NUM_HEADS",     config.NUM_HEADS),
        trans_layers = saved_cfg.get("TRANS_LAYERS", config.TRANS_LAYERS),
        num_classes  = saved_cfg.get("NUM_CLASSES",  config.NUM_CLASSES),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = checkpoint.get("class_names", config.CLASSES)
    saved_epoch = checkpoint.get("epoch", "?")
    saved_val_f1= checkpoint.get("val_f1", "?")
    print(f"Checkpoint from epoch {saved_epoch}, val_f1={saved_val_f1}")
    print(f"Classes: {class_names}")

    # Get test loader
    _, _, test_loader, _ = get_dataloaders()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    print("\nRunning test set evaluation...")
    for images, labels in tqdm(test_loader, ncols=80):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1])   # P(PNEUMONIA)

    n = len(test_loader.dataset)
    test_loss = running_loss / n

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)
    prec = precision_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Accuracy:       {acc:.4f}  ({acc*100:.2f}%)")
    print(f"F1 (PNEUMONIA): {f1:.4f}")
    print(f"Precision:      {prec:.4f}")
    print(f"Recall:         {rec:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"               {class_names[0]:>12}  {class_names[1]:>12}")
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>12}  {row[0]:>12}  {row[1]:>12}")

    # Save results
    results = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": saved_epoch,
        "checkpoint_val_f1": str(saved_val_f1),
        "test_loss": round(test_loss, 5),
        "test_accuracy": round(acc, 5),
        "test_f1": round(f1, 5),
        "test_precision": round(prec, 5),
        "test_recall": round(rec, 5),
        "num_test_samples": n,
        "class_names": class_names,
    }
    results_path = os.path.join(config.RESULTS_DIR, "test_results.json")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(results_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nResults saved -> {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_on_test()
