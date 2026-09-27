"""Paired 512px chip dataset for localization and damage segmentation."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(image.height, image.width, 3).permute(2, 0, 1).float()
    if tensor.shape[0] != 3:
        raise ValueError(f"expected RGB image at {path}")
    return (tensor / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def read_mask(path: Path) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        return torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(image.height, image.width).long()


def synchronized_geometry(items: list[Tensor]) -> list[Tensor]:
    if random.random() < 0.5:
        items = [torch.flip(item, dims=(-1,)) for item in items]
    if random.random() < 0.5:
        items = [torch.flip(item, dims=(-2,)) for item in items]
    quarter_turns = random.randrange(4)
    if quarter_turns:
        items = [torch.rot90(item, quarter_turns, dims=(-2, -1)) for item in items]
    return items


class PairedChipDataset(Dataset[dict[str, Any]]):
    """Read train/validation/test rows from the processed chip manifest."""

    def __init__(
        self,
        manifest: str | Path = "data/processed/512/training_manifest.csv",
        split: str = "train",
        augment: bool = False,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split {split!r}")
        self.root = Path.cwd()
        self.rows = [row for row in read_manifest(Path(manifest)) if row["split"] == split]
        self.augment = augment and split == "train"
        if not self.rows:
            raise ValueError(f"no chip rows found for split {split!r}")

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        pre_image = read_rgb(self._resolve(row["pre_image"]))
        post_image = read_rgb(self._resolve(row["post_image"]))
        post_loc = read_mask(self._resolve(row["post_localization_mask"]))
        post_damage = read_mask(self._resolve(row["post_damage_mask"]))
        sizes = {tuple(value.shape[-2:]) for value in (pre_image, post_image, post_loc, post_damage)}
        if len(sizes) != 1:
            raise ValueError(f"paired input/target sizes disagree for {row['chip_id']}: {sizes}")
        if torch.any((post_loc != 0) & (post_loc != 1)):
            raise ValueError(f"localization mask must contain only 0/1: {row['chip_id']}")
        if torch.any(~((post_damage <= 4) | (post_damage == 255))):
            raise ValueError(f"damage mask contains an unsupported label: {row['chip_id']}")

        if self.augment:
            pre_image, post_image, post_loc, post_damage = synchronized_geometry(
                [pre_image, post_image, post_loc, post_damage]
            )
        return {
            "pre_image": pre_image,
            "post_image": post_image,
            "pre_gsd_m": torch.tensor(float(row["pre_gsd_m"]), dtype=torch.float32),
            "post_gsd_m": torch.tensor(float(row["post_gsd_m"]), dtype=torch.float32),
            "localization_target": post_loc.float().unsqueeze(0),
            "damage_target": post_damage,
            "canonical_name": row["canonical_name"],
            "event_id": row["event_id"],
            "chip_id": row["chip_id"],
        }
