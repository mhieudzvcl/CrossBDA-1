"""Evaluate a locked B1 checkpoint on validation or the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossbda.data.paired_chips import PairedChipDataset
from crossbda.models.b1_scalemae_fpn import ScaleMAESiameseFPN
from crossbda.models.losses import multitask_loss
from crossbda.models.metrics import SegmentationConfusion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config or args.checkpoint.parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScaleMAESiameseFPN(
        image_size=int(config["data"]["image_size"]),
        feature_blocks=tuple(config["model"]["feature_blocks"]),
        gradient_checkpointing=False,
    )
    model.backbone.load_pretrained(Path(config["model"]["pretrained_checkpoint"]))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()

    dataset = PairedChipDataset(config["data"]["manifest"], split=args.split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["training"]["num_workers"]) > 0,
    )
    weights_data = json.loads(Path(config["data"]["class_weights"]).read_text(encoding="utf-8"))
    class_weights = torch.tensor(
        [weights_data["class_weights"][str(i)] for i in range(5)], dtype=torch.float32, device=device
    )
    task_weights = tuple(float(value) for value in config["loss"]["task_weights"])
    amp_enabled = device.type == "cuda" and config["training"]["amp"] == "fp16"
    confusion = SegmentationConfusion()
    loss_sum = 0.0
    count = 0
    with torch.inference_mode():
        for raw in loader:
            batch: dict[str, Tensor] = {
                key: value.to(device, non_blocking=True)
                for key, value in raw.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(
                    batch["pre_image"], batch["post_image"],
                    batch["pre_gsd_m"], batch["post_gsd_m"],
                )
                loss, _ = multitask_loss(
                    predictions,
                    batch["localization_target"],
                    batch["damage_target"],
                    class_weights=class_weights,
                    task_weights=task_weights,
                )
            confusion.update(
                predictions["localization"], batch["localization_target"],
                predictions["damage"], batch["damage_target"],
            )
            loss_sum += float(loss)
            count += 1
    result: dict[str, Any] = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "chips": len(dataset),
        "mean_loss_per_batch": loss_sum / max(1, count),
        **confusion.compute(),
    }
    output = args.output or args.checkpoint.parent / f"metrics_{args.split}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
