"""
config.py - Central hyperparameter configuration.

Mirrors the original TF/Keras params dict from the notebook exactly.
All other modules import constants from here so changes propagate everywhere.
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative - works from any working directory inside the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "chest_xray")
TRAIN_DIR      = os.path.join(DATASET_ROOT, "train")
VAL_DIR        = os.path.join(DATASET_ROOT, "val")   
TEST_DIR       = os.path.join(DATASET_ROOT, "test")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")

BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_vit.pt")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
CLASSES      = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES  = 2
SEED         = 123
VAL_SPLIT    = 0.15    # 15% of train/ used as validation (same as original notebook)

# ---------------------------------------------------------------------------
# Image & Patch parameters  (identical to original notebook)
# ---------------------------------------------------------------------------
IMAGE_DIM    = 288                            # square input: 288x288
PATCH_SIZE   = 18                             # 18x18 pixel patches
NUM_PATCHES  = (IMAGE_DIM // PATCH_SIZE) ** 2 # = 256

# ---------------------------------------------------------------------------
# ViT architecture  (identical to original notebook)
# ---------------------------------------------------------------------------
PROJ_DIM       = 16                            # projection / embedding dimension
NUM_HEADS      = 4                             # multi-head attention heads
TRANS_LAYERS   = 4                             # number of transformer encoder blocks
TRANS_UNITS    = [PROJ_DIM * 2, PROJ_DIM]     # MLP units inside each transformer block: [32, 16]
MLP_HEAD_UNITS = [1024, 512]                   # classification head MLP units
DROPOUT_T      = 0.1                           # dropout in transformer blocks
DROPOUT_H      = 0.5                           # dropout before classification head

# ---------------------------------------------------------------------------
# Training  (identical to original notebook)
# ---------------------------------------------------------------------------
BATCH_SIZE   = 16
EPOCHS       = 30
LR           = 1e-3           # 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 20      # patience in epochs (monitors val accuracy, same as original)

# ---------------------------------------------------------------------------
# Data augmentation (optional - off by default, same as original)
# ---------------------------------------------------------------------------
USE_DATA_AUG   = False
DA_ROTATION    = 20
DA_TRANSLATE   = 0.1          # width/height shift
DA_SHEAR       = 0.05
DA_ZOOM        = 0.05
DA_HFLIP       = False
DA_VFLIP       = False

# ---------------------------------------------------------------------------
# ImageNet normalization stats (standard for RGB models)
# NOTE: Original notebook did NOT rescale training images consistently (likely
# a bug - the training ImageDataGenerator lacked rescale=1./255).
# We apply proper normalization: ToTensor (-> [0,1]) + Normalize.
# ---------------------------------------------------------------------------
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
