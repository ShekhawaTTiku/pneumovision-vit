"""
predict.py - Standalone inference script.

Usage:
    ./venv311/Scripts/python predict.py --image path/to/xray.jpg
    ./venv311/Scripts/python predict.py --image path/to/xray.jpg --checkpoint checkpoints/best_vit.pt

Output:
    Predicted class : NORMAL  (or PNEUMONIA)
    Confidence      : 0.9423
    Checkpoint used : checkpoints/best_vit.pt

The preprocessing pipeline is IDENTICAL to training (same resize, normalize).
"""
import argparse
import os

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import config
from vit_model import VisionTransformer


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint. Reconstructs architecture from saved config."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to produce a checkpoint."
        )

    checkpoint  = torch.load(checkpoint_path, map_location=device)
    saved_cfg   = checkpoint.get("config", {})
    class_names = checkpoint.get("class_names", config.CLASSES)

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

    return model, class_names, checkpoint.get("epoch", "?"), checkpoint.get("val_f1", "?")


def get_inference_transform():
    """
    Inference transform - MUST match the val/test transform used during training.
    Any mismatch here would silently degrade accuracy.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_DIM, config.IMAGE_DIM)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


@torch.no_grad()
def predict(image_path: str, checkpoint_path: str = config.BEST_CHECKPOINT):
    """
    Run inference on a single image.

    Args:
        image_path:      Path to the input X-ray image (JPEG or PNG).
        checkpoint_path: Path to the saved .pt checkpoint.

    Returns:
        dict with keys: predicted_class, confidence, probabilities, checkpoint_used
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, epoch, val_f1 = load_model(checkpoint_path, device)
    transform = get_inference_transform()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)   # (1, 3, 288, 288)

    logits = model(tensor)                              # (1, 2)
    probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx   = int(probs.argmax())
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "predicted_class": pred_class,
        "confidence":      round(confidence, 4),
        "probabilities":   {cls: round(float(p), 4) for cls, p in zip(class_names, probs)},
        "checkpoint_used": checkpoint_path,
        "checkpoint_epoch": epoch,
        "checkpoint_val_f1": str(val_f1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict NORMAL or PNEUMONIA from a chest X-ray image."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input X-ray image (JPEG, PNG, etc.)"
    )
    parser.add_argument(
        "--checkpoint", default=config.BEST_CHECKPOINT,
        help=f"Path to checkpoint .pt file (default: {config.BEST_CHECKPOINT})"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        return

    result = predict(args.image, args.checkpoint)

    print()
    print(f"  Image           : {args.image}")
    print(f"  Predicted class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']:.4f}  ({result['confidence']*100:.1f}%)")
    print(f"  Probabilities   : {result['probabilities']}")
    print(f"  Checkpoint used : {result['checkpoint_used']}")
    print(f"  Checkpoint epoch: {result['checkpoint_epoch']}")
    print(f"  Checkpoint val_F1: {result['checkpoint_val_f1']}")
    print()


if __name__ == "__main__":
    main()
