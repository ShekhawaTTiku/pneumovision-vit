"""
dataset_utils.py - DataLoader and transform helpers.

Validation strategy (mirrors original notebook):
  - Uses 85% of train/ for training and 15% for validation.
  - The disk val/ folder (16 images) is NOT used - too small to be meaningful.
  - The test/ folder is held out and NEVER touched during training.
"""

import os
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config


def get_transforms(split: str) -> transforms.Compose:
    """
    Returns transforms for the given split.

    - train: resize, optional augmentation, ToTensor, Normalize
    - val/test: resize, ToTensor, Normalize (no augmentation)

    Normalization applied consistently to all splits.
    Original notebook lacked consistent rescaling on the training generator
    (likely a bug). We fix that here.
    """
    resize = transforms.Resize((config.IMAGE_DIM, config.IMAGE_DIM))
    to_tensor = transforms.ToTensor()                     # -> [0, 1]
    normalize = transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)

    if split == "train" and config.USE_DATA_AUG:
        aug = [
            transforms.RandomRotation(degrees=config.DA_ROTATION),
            transforms.RandomAffine(
                degrees=0,
                translate=(config.DA_TRANSLATE, config.DA_TRANSLATE),
                shear=config.DA_SHEAR * 45,
                scale=(1 - config.DA_ZOOM, 1 + config.DA_ZOOM),
            ),
        ]
        if config.DA_HFLIP:
            aug.append(transforms.RandomHorizontalFlip())
        if config.DA_VFLIP:
            aug.append(transforms.RandomVerticalFlip())
        return transforms.Compose([resize] + aug + [to_tensor, normalize])

    return transforms.Compose([resize, to_tensor, normalize])


def _make_split_indices(dataset_size: int, val_fraction: float, seed: int):
    """Reproducible 85/15 index split."""
    indices = list(range(dataset_size))
    random.seed(seed)
    random.shuffle(indices)
    split_point = int(dataset_size * (1 - val_fraction))
    return indices[:split_point], indices[split_point:]


def get_dataloaders():
    """
    Returns (train_loader, val_loader, test_loader, class_names).

    class_names: list of str in label-index order, e.g. ["NORMAL", "PNEUMONIA"].
    The folder alphabetical order determines label indices
    (torchvision ImageFolder convention):
      NORMAL    -> 0
      PNEUMONIA -> 1
    """
    # Full training dataset (no augmentation yet - applied via Subset below)
    full_train_dataset = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=get_transforms("val"),   # neutral transform first for splitting
    )
    class_names = full_train_dataset.classes  # ["NORMAL", "PNEUMONIA"] alphabetically

    # Build train/val index split reproducibly
    train_idx, val_idx = _make_split_indices(
        len(full_train_dataset), config.VAL_SPLIT, config.SEED
    )

    # Re-build datasets with the correct transforms for each subset
    train_dataset_aug = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=get_transforms("train"),
    )
    val_dataset = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=get_transforms("val"),
    )
    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR,
        transform=get_transforms("test"),
    )

    train_subset = Subset(train_dataset_aug, train_idx)
    val_subset   = Subset(val_dataset,       val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # 0 = safe on Windows (avoids multiprocessing issues)
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


def dataset_summary():
    """Print dataset statistics for verification."""
    import math
    full = datasets.ImageFolder(root=config.TRAIN_DIR)
    train_idx, val_idx = _make_split_indices(len(full), config.VAL_SPLIT, config.SEED)

    test_ds = datasets.ImageFolder(root=config.TEST_DIR)

    print("=" * 55)
    print("DATASET SUMMARY")
    print("=" * 55)
    print(f"Classes (alphabetical): {full.classes}")
    print(f"Class -> label index:   {full.class_to_idx}")
    print()
    print(f"Total in train/:        {len(full):>6}")
    print(f"  -> training split:    {len(train_idx):>6}  (~{(1-config.VAL_SPLIT)*100:.0f}%)")
    print(f"  -> validation split:  {len(val_idx):>6}  (~{config.VAL_SPLIT*100:.0f}%)")
    print()

    # Per-class breakdown in each split
    all_labels = [label for _, label in full.samples]
    tr_labels  = [all_labels[i] for i in train_idx]
    va_labels  = [all_labels[i] for i in val_idx]
    for idx, cls in enumerate(full.classes):
        tr_cnt = tr_labels.count(idx)
        va_cnt = va_labels.count(idx)
        print(f"  {cls:<12} train={tr_cnt}  val={va_cnt}")

    print()
    print(f"Test set (test/):       {len(test_ds):>6}")
    te_labels = [label for _, label in test_ds.samples]
    for idx, cls in enumerate(test_ds.classes):
        print(f"  {cls:<12} test={te_labels.count(idx)}")

    print()
    train_batches = math.ceil(len(train_idx) / config.BATCH_SIZE)
    val_batches   = math.ceil(len(val_idx)   / config.BATCH_SIZE)
    print(f"Train batches/epoch:    {train_batches}")
    print(f"Val batches/epoch:      {val_batches}")
    print()
    print("NOTE: disk val/ folder (16 images) is IGNORED - too small.")
    print("=" * 55)


if __name__ == "__main__":
    dataset_summary()
