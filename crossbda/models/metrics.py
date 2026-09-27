"""Streaming confusion-matrix metrics for localization and damage maps."""

from __future__ import annotations

import torch
from torch import Tensor


class SegmentationConfusion:
    def __init__(self, damage_classes: int = 5, ignore_index: int = 255) -> None:
        self.damage_classes = damage_classes
        self.ignore_index = ignore_index
        self.loc = torch.zeros((2, 2), dtype=torch.int64)
        self.damage = torch.zeros((damage_classes, damage_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(
        self,
        loc_logits: Tensor,
        loc_target: Tensor,
        damage_logits: Tensor,
        damage_target: Tensor,
    ) -> None:
        loc_pred = (loc_logits.sigmoid() >= 0.5).long().reshape(-1).cpu()
        loc_true = loc_target.long().reshape(-1).cpu()
        valid_loc = (loc_true == 0) | (loc_true == 1)
        loc_bins = 2 * loc_true[valid_loc] + loc_pred[valid_loc]
        self.loc += torch.bincount(loc_bins, minlength=4).reshape(2, 2)

        pred = damage_logits.argmax(dim=1).reshape(-1).long().cpu()
        target = damage_target.reshape(-1).long().cpu()
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.damage_classes)
        bins = self.damage_classes * target[valid] + pred[valid]
        self.damage += torch.bincount(
            bins, minlength=self.damage_classes * self.damage_classes
        ).reshape(self.damage_classes, self.damage_classes)

    @staticmethod
    def _f1(confusion: Tensor, class_id: int) -> float:
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum()) - tp
        fn = int(confusion[class_id, :].sum()) - tp
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator else 0.0

    def compute(self) -> dict[str, object]:
        loc_f1 = self._f1(self.loc, 1)
        damage_f1_by_class = {str(k): self._f1(self.damage, k) for k in range(1, self.damage_classes)}
        class_f1 = list(damage_f1_by_class.values())
        damage_f1_macro = sum(class_f1) / max(1, len(class_f1))
        epsilon = 1e-8
        damage_f1_harmonic = (
            len(class_f1) / sum(1.0 / max(value, epsilon) for value in class_f1)
            if class_f1
            else 0.0
        )
        return {
            "localization_f1": loc_f1,
            "damage_f1_harmonic": damage_f1_harmonic,
            "damage_f1_macro": damage_f1_macro,
            "damage_f1_by_class": damage_f1_by_class,
            "xview2_score": 0.3 * loc_f1 + 0.7 * damage_f1_harmonic,
            "localization_iou": self._iou(self.loc, 1),
            "localization_confusion": self.loc.tolist(),
            "damage_confusion": self.damage.tolist(),
        }

    @staticmethod
    def _iou(confusion: Tensor, class_id: int) -> float:
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum()) - tp
        fn = int(confusion[class_id, :].sum()) - tp
        denominator = tp + fp + fn
        return tp / denominator if denominator else 0.0
