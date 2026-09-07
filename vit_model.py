"""
vit_model.py - Vision Transformer (ViT) in pure PyTorch.

Architecture faithfully mirrors the original TF/Keras implementation from:
  vision_transformer_xray_pneumonia_detection.ipynb

Key design decisions:
  - Patches extracted via torch.Tensor.unfold (equivalent to tf.image.extract_patches)
  - Positional embeddings: nn.Embedding(num_patches, proj_dim) - learnable, same as Keras
  - Transformer blocks: LayerNorm -> MultiHeadAttention -> residual, then
                         LayerNorm -> MLP(GELU) -> residual  (Pre-LN style)
  - Classification head: Flatten -> Dropout -> MLP -> Linear(num_classes)
  - Output: raw logits (no softmax) - loss function handles that

Differences from original (unavoidable):
  - PyTorch MultiheadAttention uses batch_first=True convention
  - No explicit key_dim parameter; head_dim = proj_dim // num_heads = 4
    (original had key_dim=proj_dim=16 per head, which is unusual;
     here we follow standard ViT convention: proj_dim is split across heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# Helper: MLP block (used inside transformer blocks and classification head)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """
    Two-layer MLP with GELU activation and dropout after each layer.
    Mirrors the `mlp()` function in the original notebook.
    """
    def __init__(self, in_features: int, hidden_units: list[int], dropout_rate: float):
        super().__init__()
        layers = []
        prev = in_features
        for units in hidden_units:
            layers.append(nn.Linear(prev, units))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev = units
        self.net = nn.Sequential(*layers)
        self.out_features = prev

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Patch extractor (mirrors the Patches layer in the notebook)
# ---------------------------------------------------------------------------
class Patches(nn.Module):
    """
    Splits an image into non-overlapping square patches.

    Input:  (B, C, H, W)
    Output: (B, num_patches, patch_dim)  where patch_dim = patch_size^2 * C

    Equivalent to tf.image.extract_patches with VALID padding.
    """
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, (
            f"Image size ({H},{W}) must be divisible by patch_size ({p})"
        )
        # unfold H dimension then W dimension
        x = x.unfold(2, p, p).unfold(3, p, p)
        # x shape: (B, C, H//p, W//p, p, p)
        x = x.permute(0, 2, 3, 4, 5, 1).contiguous()
        # x shape: (B, H//p, W//p, p, p, C)
        B, nh, nw, ph, pw, C = x.shape
        x = x.view(B, nh * nw, ph * pw * C)
        # x shape: (B, num_patches, patch_dim)
        return x


# ---------------------------------------------------------------------------
# Patch encoder (mirrors the PatchEncoder layer in the notebook)
# ---------------------------------------------------------------------------
class PatchEncoder(nn.Module):
    """
    Linearly projects patches to proj_dim and adds learnable positional embeddings.

    Mirrors the PatchEncoder class in the original notebook exactly:
      self.projection = Dense(units=projection_dim)
      self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)
    """
    def __init__(self, num_patches: int, proj_dim: int, patch_dim: int):
        super().__init__()
        self.projection       = nn.Linear(patch_dim, proj_dim)
        self.position_embedding = nn.Embedding(num_patches, proj_dim)
        self.num_patches      = num_patches

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: (B, num_patches, patch_dim)
        projected = self.projection(patches)                           # (B, num_patches, proj_dim)
        positions = torch.arange(self.num_patches, device=patches.device)
        pos_emb   = self.position_embedding(positions)                 # (num_patches, proj_dim)
        return projected + pos_emb                                     # broadcast over B


# ---------------------------------------------------------------------------
# Single Transformer encoder block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer encoder block.

    Structure:
      x -> LN -> MultiHeadAttn -> + x  (residual)
        -> LN -> MLP            -> + x  (residual)

    Mirrors the transformer block loop in create_vit_classifier().
    """
    def __init__(self, proj_dim: int, num_heads: int, mlp_units: list[int], dropout: float):
        super().__init__()
        self.norm1   = nn.LayerNorm(proj_dim, eps=1e-6)
        self.attn    = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,          # (B, seq, dim) convention
        )
        self.norm2   = nn.LayerNorm(proj_dim, eps=1e-6)
        self.mlp     = MLP(proj_dim, mlp_units, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-block
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1)
        x = x + attn_out                  # residual

        # MLP sub-block
        x2 = self.norm2(x)
        x = x + self.mlp(x2)             # residual

        return x


# ---------------------------------------------------------------------------
# Full ViT classifier
# ---------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    """
    Vision Transformer for binary chest X-ray classification.

    Closely mirrors create_vit_classifier() from the original notebook.

    Input:  (B, 3, IMAGE_DIM, IMAGE_DIM)
    Output: (B, num_classes)  - raw logits, no softmax
    """
    def __init__(
        self,
        image_dim:     int = config.IMAGE_DIM,
        patch_size:    int = config.PATCH_SIZE,
        num_patches:   int = config.NUM_PATCHES,
        proj_dim:      int = config.PROJ_DIM,
        num_heads:     int = config.NUM_HEADS,
        trans_layers:  int = config.TRANS_LAYERS,
        trans_units:   list = config.TRANS_UNITS,
        mlp_head_units:list = config.MLP_HEAD_UNITS,
        dropout_t:     float = config.DROPOUT_T,
        dropout_h:     float = config.DROPOUT_H,
        num_classes:   int = config.NUM_CLASSES,
        in_channels:   int = 3,
    ):
        super().__init__()

        patch_dim = patch_size * patch_size * in_channels  # 18*18*3 = 972

        self.patches        = Patches(patch_size)
        self.patch_encoder  = PatchEncoder(num_patches, proj_dim, patch_dim)

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(proj_dim, num_heads, trans_units, dropout_t)
            for _ in range(trans_layers)
        ])

        self.norm       = nn.LayerNorm(proj_dim, eps=1e-6)
        self.flatten    = nn.Flatten()
        self.dropout_h  = nn.Dropout(dropout_h)

        # Classification MLP head
        # Input to head: num_patches * proj_dim = 256 * 16 = 4096
        head_in = num_patches * proj_dim
        self.mlp_head = MLP(head_in, mlp_head_units, dropout_h)

        # Final linear projection to class logits
        self.classifier = nn.Linear(mlp_head_units[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        patches  = self.patches(x)                    # (B, 256, 972)
        encoded  = self.patch_encoder(patches)         # (B, 256, 16)
        encoded  = self.transformer_blocks(encoded)    # (B, 256, 16)
        rep      = self.norm(encoded)                  # (B, 256, 16)
        rep      = self.flatten(rep)                   # (B, 4096)
        rep      = self.dropout_h(rep)                 # (B, 4096)
        features = self.mlp_head(rep)                  # (B, 512)
        logits   = self.classifier(features)           # (B, 2)
        return logits


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = VisionTransformer()
    print(model)
    total = count_parameters(model)
    print(f"\nTotal trainable parameters: {total:,}")

    # Forward pass check
    B = 2
    dummy = torch.zeros(B, 3, config.IMAGE_DIM, config.IMAGE_DIM)
    out   = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}  (expected: ({B}, {config.NUM_CLASSES}))")
    assert out.shape == (B, config.NUM_CLASSES), "Output shape mismatch!"
    print("Forward pass OK.")
