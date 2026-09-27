"""B1: shared Scale-MAE ViT-Large, temporal concatenation, FPN, and task heads."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import VisionTransformer


def _sincos_1d(embed_dim: int, positions: Tensor) -> Tensor:
    if embed_dim % 2:
        raise ValueError("1D sin/cos embedding dimension must be even")
    omega = torch.arange(embed_dim // 2, device=positions.device, dtype=positions.dtype)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    angles = positions.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


def gsd_positional_embedding(embed_dim: int, grid_size: int, gsd_m: Tensor) -> Tensor:
    """Build Scale-MAE's GSD-aware sin/cos positions for each image in a batch."""
    if embed_dim % 4:
        raise ValueError("2D sin/cos embedding dimension must be divisible by four")
    if gsd_m.ndim != 1 or torch.any(gsd_m <= 0):
        raise ValueError("gsd_m must be a positive [batch] tensor")
    axis = torch.arange(grid_size, device=gsd_m.device, dtype=gsd_m.dtype)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
    x = grid_x.reshape(1, -1) * gsd_m.reshape(-1, 1)
    y = grid_y.reshape(1, -1) * gsd_m.reshape(-1, 1)
    x_embed = torch.stack([_sincos_1d(embed_dim // 2, row) for row in x])
    y_embed = torch.stack([_sincos_1d(embed_dim // 2, row) for row in y])
    spatial = torch.cat((x_embed, y_embed), dim=-1)
    cls = spatial.new_zeros((spatial.shape[0], 1, embed_dim))
    return torch.cat((cls, spatial), dim=1)


class ScaleMAEViTLarge(nn.Module):
    """ViT-Large encoder with the official Scale-MAE GSD positional encoding."""

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        feature_blocks: tuple[int, ...] = (5, 11, 17, 23),
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.encoder = VisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=0,
            global_pool="",
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.feature_blocks = tuple(feature_blocks)
        if not self.feature_blocks or max(self.feature_blocks) >= len(self.encoder.blocks):
            raise ValueError("feature_blocks must index existing transformer blocks")
        self.gradient_checkpointing = gradient_checkpointing
        # The checkpoint's fixed-resolution positional tensor is intentionally bypassed.
        if self.encoder.pos_embed is not None:
            self.encoder.pos_embed.requires_grad_(False)

    def load_pretrained(self, checkpoint_path: Path) -> dict[str, int]:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint_data
        if isinstance(state, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise ValueError("Scale-MAE checkpoint does not contain a state dictionary")

        target = self.encoder.state_dict()
        compatible: dict[str, Tensor] = {}
        for key, value in state.items():
            clean_key = str(key)
            for prefix in ("module.", "model.", "encoder."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            # Scale-specific positional embeddings are regenerated from each image's GSD.
            if clean_key == "pos_embed" or clean_key not in target:
                continue
            if target[clean_key].shape == value.shape:
                compatible[clean_key] = value

        incompatible = self.encoder.load_state_dict(compatible, strict=False)
        required_prefixes = ("patch_embed.proj.", "blocks.0.", "blocks.23.", "norm.")
        absent = [
            prefix for prefix in required_prefixes
            if not any(key.startswith(prefix) for key in compatible)
        ]
        if absent:
            raise ValueError(
                "checkpoint does not match a Scale-MAE ViT-Large encoder; "
                f"missing required parameter groups: {absent}"
            )
        return {
            "loaded": len(compatible),
            "missing": len(incompatible.missing_keys),
            "unexpected": len(incompatible.unexpected_keys),
        }

    def forward(self, image: Tensor, gsd_m: Tensor) -> list[Tensor]:
        encoder = self.encoder
        tokens = encoder.patch_embed(image)
        batch, count, _ = tokens.shape
        grid = int(count**0.5)
        if grid * grid != count:
            raise ValueError(f"expected square ViT token grid, got {count} tokens")
        position = gsd_positional_embedding(1024, grid, gsd_m).to(dtype=tokens.dtype)
        if encoder.cls_token is not None:
            cls = encoder.cls_token.expand(batch, -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
        tokens = encoder.pos_drop(tokens + position)
        if hasattr(encoder, "norm_pre"):
            tokens = encoder.norm_pre(tokens)

        selected: list[Tensor] = []
        for index, block in enumerate(encoder.blocks):
            if self.gradient_checkpointing and self.training and tokens.requires_grad:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if index in self.feature_blocks:
                feature = encoder.norm(tokens[:, 1:])
                feature = feature.transpose(1, 2).reshape(batch, 1024, grid, grid)
                selected.append(feature)
        return selected


class TemporalFPN(nn.Module):
    """Project four temporal concatenations and merge a stride-16 to stride-128 pyramid."""

    def __init__(self, input_channels: int = 2048, channels: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(input_channels, channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, channels),
                nn.GELU(),
            )
            for _ in range(4)
        )
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, channels),
                nn.GELU(),
            )
            for _ in range(4)
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        if len(features) != 4:
            raise ValueError(f"FPN requires four transformer features, got {len(features)}")
        base_h, base_w = features[0].shape[-2:]
        sizes = [(base_h // (2**i), base_w // (2**i)) for i in range(4)]
        lateral_features: list[Tensor] = []
        for index, (feature, size) in enumerate(zip(features, sizes)):
            if feature.shape[-2:] != size:
                feature = nn.functional.adaptive_avg_pool2d(feature, size)
            lateral_features.append(self.lateral[index](feature))
        pyramid: list[Tensor | None] = [None] * 4
        for index in range(3, -1, -1):
            merged = lateral_features[index]
            if index < 3:
                assert pyramid[index + 1] is not None
                merged = merged + nn.functional.interpolate(
                    pyramid[index + 1], size=sizes[index], mode="nearest"
                )
            pyramid[index] = self.smooth[index](merged)
        return [value for value in pyramid if value is not None]


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        channels = (128, 64, 32, 32)
        blocks: list[nn.Module] = []
        current = in_channels
        for output in channels:
            blocks.extend(
                [
                    nn.Conv2d(current, output, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(8, output),
                    nn.GELU(),
                ]
            )
            current = output
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Conv2d(current, out_channels, kernel_size=1)

    def forward(self, feature: Tensor, output_size: tuple[int, int]) -> Tensor:
        x = feature
        for layer in self.blocks:
            if isinstance(layer, nn.Conv2d):
                x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = layer(x)
        x = self.output(x)
        return nn.functional.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class ScaleMAESiameseFPN(nn.Module):
    """B1 baseline: shared pre/post encoder, simple concatenation, and dual masks."""

    def __init__(
        self,
        image_size: int = 512,
        feature_blocks: tuple[int, ...] = (5, 11, 17, 23),
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = ScaleMAEViTLarge(
            image_size=image_size,
            feature_blocks=feature_blocks,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.fpn = TemporalFPN(input_channels=2048, channels=256)
        self.localization_head = SegmentationHead(256, 1)
        self.damage_head = SegmentationHead(256, 5)

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not frozen)

    def unfreeze_last_blocks(self, count: int = 6) -> None:
        if count < 0 or count > len(self.backbone.encoder.blocks):
            raise ValueError("count must be between zero and the ViT depth")
        self.freeze_backbone(True)
        blocks = self.backbone.encoder.blocks[-count:] if count else []
        for block in blocks:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.backbone.encoder.norm.parameters():
            parameter.requires_grad_(True)

    def forward(
        self,
        pre_image: Tensor,
        post_image: Tensor,
        pre_gsd_m: Tensor,
        post_gsd_m: Tensor,
    ) -> dict[str, Tensor]:
        pre_features = self.backbone(pre_image, pre_gsd_m)
        post_features = self.backbone(post_image, post_gsd_m)
        fused = [torch.cat((pre, post), dim=1) for pre, post in zip(pre_features, post_features)]
        pyramid = self.fpn(fused)
        output_size = tuple(post_image.shape[-2:])
        return {
            "localization": self.localization_head(pyramid[0], output_size),
            "damage": self.damage_head(pyramid[0], output_size),
        }
