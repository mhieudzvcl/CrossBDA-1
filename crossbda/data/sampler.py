"""Event-balanced weighted sampling with additional weight for rare damage presence."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Sampler


class EventClassBalancedSampler(Sampler[int]):
    """Balance events, then add a smaller sampling mass for chips with rare damage."""

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        root: str | Path = ".",
        rare_classes: tuple[int, ...] = (2, 3, 4),
        rare_mass: float = 0.5,
        seed: int = 42,
    ) -> None:
        if not rows:
            raise ValueError("sampler requires at least one row")
        if not 0.0 <= rare_mass <= 1.0:
            raise ValueError("rare_mass must be in [0, 1]")
        self.num_samples = len(rows)
        self.seed = seed
        self.epoch = 0
        root = Path(root)
        event_counts: Counter[str] = Counter(row["canonical_name"] for row in rows)
        class_counts: Counter[int] = Counter()
        classes_per_row: list[set[int]] = []
        for row in rows:
            path = Path(row["post_damage_mask"])
            if not path.is_absolute():
                path = root / path
            with Image.open(path) as mask:
                histogram = mask.histogram()
            sample_classes = {value for value in rare_classes if histogram[value] > 0}
            classes_per_row.append(sample_classes)
            class_counts.update(sample_classes)

        event_total = len(event_counts)
        rare_weight_total = sum(
            1.0 / class_counts[class_id]
            for sample_classes in classes_per_row
            for class_id in sample_classes
        )
        weights: list[float] = []
        for row, sample_classes in zip(rows, classes_per_row):
            event_weight = 1.0 / event_counts[row["canonical_name"]]
            rare_weight = 0.0
            if sample_classes and rare_weight_total:
                rare_weight = event_total * sum(
                    1.0 / class_counts[class_id] for class_id in sample_classes
                ) / rare_weight_total
            weights.append((1.0 - rare_mass) * event_weight + rare_mass * rare_weight)
        self.weights = torch.tensor(weights, dtype=torch.double)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=generator
        )
        return iter(indices.tolist())
