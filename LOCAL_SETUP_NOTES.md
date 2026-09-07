# Local Setup Notes
## Vision Transformer for X-Ray Pneumonia Detection — PyTorch Adaptation

---

## Python Environment

| Item | Value |
|---|---|
| Python version | 3.11.9 (64-bit) — installed alongside existing 3.14 |
| Virtual environment | `venv311/` inside the project directory |
| Activation (Windows) | `.\venv311\Scripts\Activate.ps1` |
| PyTorch version | 2.6.0+cu124 |
| CUDA driver | 13.4 UMD (driver 616.64) |
| GPU | NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM) |

> Note: Python 3.14 was the system default but PyTorch has no stable wheel for it yet.
> Python 3.11 was installed and isolated in venv311/ so the system default is unchanged.

---

## Installed Dependencies

```
torch==2.6.0+cu124        (GPU-enabled)
torchvision==0.21.0+cu124
tqdm
numpy
pillow
scikit-learn
matplotlib
pandas
```

---

## Dataset Structure (Unchanged)

```
chest_xray/                   <- relative to project root (..)
├── train/
│   ├── NORMAL/      1,341 images
│   └── PNEUMONIA/   3,875 images
├── val/
│   ├── NORMAL/          8 images   <- NOT USED (too small)
│   └── PNEUMONIA/       8 images   <- NOT USED
└── test/
    ├── NORMAL/        234 images   <- HELD OUT (evaluation only)
    └── PNEUMONIA/     390 images   <- HELD OUT
```

No files were moved, renamed, or deleted.

---

## Validation Strategy

**Same as original notebook**: 15% internal split from `train/` used as validation.

- Training split: 85% of `train/` (~4,434 images)
- Validation split: 15% of `train/` (~782 images)
- The disk `val/` folder (16 images) is **ignored** — the Kaggle default val folder
  is too small to provide statistically meaningful validation metrics.
- The `test/` folder is **held out completely** until `evaluate.py` is run manually.
  It is never seen during training or model selection.

Split is performed using `random_split` with `seed=123` (identical to original).

---

## Preprocessing

| Step | Value |
|---|---|
| Resize | 288 × 288 (same as original) |
| ToTensor | Divides by 255, outputs [0, 1] float32 |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Data augmentation | OFF by default (same as original notebook default) |
| Color mode | RGB (same as original) |

> **Bug fix**: The original notebook's `train_generator` used `ImageDataGenerator(validation_split=0.15)`
> without `rescale=1./255`. Only `val_datagen` had it. This inconsistency (likely unintentional)
> meant training images were NOT rescaled to [0,1] while validation images were.
> In this PyTorch port, `transforms.ToTensor()` consistently maps all images to [0,1] across
> ALL splits, followed by ImageNet normalization.

---

## TF/Keras -> PyTorch Architecture Changes

| TF/Keras Component | PyTorch Equivalent | Notes |
|---|---|---|
| `tensorflow_addons.AdamW` | `torch.optim.AdamW` | Native PyTorch, no addon needed |
| `layers.MultiHeadAttention(num_heads=4, key_dim=16)` | `nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)` | head_dim=4 (proj_dim/num_heads) |
| `layers.Embedding(256, 16)` | `nn.Embedding(256, 16)` | Identical learnable positional embeddings |
| `layers.Dense` | `nn.Linear` | Direct equivalent |
| `layers.LayerNormalization(epsilon=1e-6)` | `nn.LayerNorm(eps=1e-6)` | Direct equivalent |
| `layers.Dropout` | `nn.Dropout` | Direct equivalent |
| `tf.nn.gelu` | `nn.GELU()` | Direct equivalent |
| `tf.image.extract_patches` | `tensor.unfold()` | Equivalent non-overlapping patch extraction |
| `ImageDataGenerator` | `torchvision.datasets.ImageFolder` + `DataLoader` | More explicit, no magic |
| `BinaryCrossentropy(from_logits=True)` | `nn.CrossEntropyLoss()` | Mathematically equivalent for 2-class logits |
| `model.compile / model.fit` | Manual training loop | Standard PyTorch |
| Keras custom F1/precision/recall | `sklearn.metrics` | Computed on epoch-end accumulated predictions |
| `ModelCheckpoint` | `torch.save` on best val F1 | |
| `EarlyStopping(patience=20)` | Manual counter on val accuracy | Same logic |

---

## Model Architecture

```
VisionTransformer
  Image:   (B, 3, 288, 288)
  Patches: (B, 256, 972)       <- 18x18 patch = 18*18*3=972 dims, 256 patches
  Encoder: (B, 256, 16)        <- Dense proj + position embedding
  x4 TransformerBlock:
    LayerNorm(eps=1e-6)
    MultiHeadAttention(heads=4, embed_dim=16, dropout=0.1)
    Residual Add
    LayerNorm(eps=1e-6)
    MLP: Linear(16->32)->GELU->Dropout(0.1)->Linear(32->16)->GELU->Dropout(0.1)
    Residual Add
  LayerNorm
  Flatten: (B, 4096)           <- 256*16
  Dropout(0.5)
  MLP Head: Linear(4096->1024)->GELU->Dropout(0.5)->Linear(1024->512)->GELU->Dropout(0.5)
  Classifier: Linear(512->2)
  Output: (B, 2) raw logits
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Loss | CrossEntropyLoss (on 2-class logits) |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Max epochs | 30 |
| Early stopping | patience=20 (monitors val accuracy) |
| Checkpoint metric | val F1 (PNEUMONIA class, binary) |
| CUDA | Auto-detected |

---

## Project Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters and paths |
| `dataset_utils.py` | DataLoaders and transforms |
| `vit_model.py` | PyTorch ViT architecture |
| `train.py` | Training script |
| `evaluate.py` | Test-set evaluation |
| `predict.py` | Standalone inference CLI |
| `checkpoints/best_vit.pt` | Best saved checkpoint (created by train.py) |
| `results/training_history.csv` | Per-epoch metrics |
| `results/training_curves.png` | Loss/accuracy/F1 plots |
| `results/test_results.json` | Final test set metrics |

---

## Checkpoint Location

```
checkpoints/best_vit.pt
```

Contains: `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_f1`, `val_acc`, `class_names`, `config`.

---

## How to Run

### Activate environment
```powershell
.\venv311\Scripts\Activate.ps1
```

### Sanity check (fast - 3 epochs x 5 batches)
```powershell
python train.py --sanity
```

### Full training
```powershell
python train.py
```

### Evaluate on test set
```powershell
python evaluate.py
```

### Single image inference
```powershell
python predict.py --image ..\chest_xray\test\NORMAL\IM-0001-0001.jpeg
python predict.py --image ..\chest_xray\test\PNEUMONIA\person1_bacteria_1.jpeg
```

---

## Actual Training Results

> To be filled after training completes.
> Results will be added from results/training_history.csv and results/test_results.json.

---

## Known Limitations / Issues

1. **Projection dimension=16 is very small** — the original ViT paper used 768 for ViT-Base.
   This matches the original notebook exactly but limits model capacity.
2. **Class imbalance** (1,341 NORMAL vs 3,875 PNEUMONIA) is not explicitly compensated;
   the original notebook also did not use class weights. Binary F1 on PNEUMONIA class is
   the primary metric, which is robust to this imbalance.
3. **num_workers=0** on Windows to avoid multiprocessing pickling issues with DataLoader.
   This may make data loading slower but is safe and correct.
4. **4GB VRAM** — batch_size=16 at 288x288 RGB is borderline. If OOM occurs,
   reduce BATCH_SIZE to 8 in config.py (optionally add gradient accumulation).
