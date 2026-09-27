"""Loss functions for the dual-head building assessment model."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def localization_loss(logits: Tensor, target: Tensor, dice_smooth: float = 1.0) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - ((2.0 * intersection + dice_smooth) / (denominator + dice_smooth)).mean()
    return 0.5 * bce + 0.5 * dice


def damage_loss(
    logits: Tensor,
    target: Tensor,
    class_weights: Tensor | None = None,
    ignore_index: int = 255,
) -> Tensor:
    weights = class_weights.to(device=logits.device, dtype=logits.dtype) if class_weights is not None else None
    return F.cross_entropy(logits, target.long(), weight=weights, ignore_index=ignore_index)


def multitask_loss(
    predictions: dict[str, Tensor],
    localization_target: Tensor,
    damage_target: Tensor,
    class_weights: Tensor | None = None,
    task_weights: tuple[float, float] = (0.5, 0.5),
) -> tuple[Tensor, dict[str, Tensor]]:
    loc = localization_loss(predictions["localization"], localization_target)
    damage = damage_loss(predictions["damage"], damage_target, class_weights)
    total = task_weights[0] * loc + task_weights[1] * damage
    return total, {"localization": loc.detach(), "damage": damage.detach()}

