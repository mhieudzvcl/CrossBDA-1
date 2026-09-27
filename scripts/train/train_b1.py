"""Train or preflight the B1 Scale-MAE Siamese concatenation/FPN baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossbda.data.paired_chips import PairedChipDataset
from crossbda.data.sampler import EventClassBalancedSampler
from crossbda.models.b1_scalemae_fpn import ScaleMAESiameseFPN
from crossbda.models.losses import multitask_loss
from crossbda.models.metrics import SegmentationConfusion


TENSOR_KEYS = {
    "pre_image", "post_image", "pre_gsd_m", "post_gsd_m",
    "localization_target", "damage_target",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if key in TENSOR_KEYS}


def forward_loss(
    model: ScaleMAESiameseFPN,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    class_weights: torch.Tensor,
    task_weights: tuple[float, float],
    amp_enabled: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        predictions = model(
            batch["pre_image"], batch["post_image"], batch["pre_gsd_m"], batch["post_gsd_m"]
        )
        loss, parts = multitask_loss(
            predictions,
            batch["localization_target"],
            batch["damage_target"],
            class_weights=class_weights,
            task_weights=task_weights,
        )
    return predictions, loss, parts


def train_epoch(
    model: ScaleMAESiameseFPN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    class_weights: torch.Tensor,
    task_weights: tuple[float, float],
    accumulation: int,
    amp_enabled: bool,
    max_steps: int = 0,
    epoch: int = 0,
) -> dict[str, Any]:
    model.train()
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    total_steps = min(len(loader), max_steps) if max_steps else len(loader)
    optimizer.zero_grad(set_to_none=True)
    metrics = SegmentationConfusion()
    loss_sum = 0.0
    loc_sum = 0.0
    damage_sum = 0.0
    started = time.perf_counter()

    for step, raw_batch in enumerate(loader):
        if step >= total_steps:
            break
        batch = move_batch(raw_batch, device)
        predictions, loss, parts = forward_loss(
            model, batch, device, class_weights, task_weights, amp_enabled
        )
        scaler.scale(loss / accumulation).backward()
        should_step = (step + 1) % accumulation == 0 or step + 1 == total_steps
        if should_step:
            scaler.unscale_(optimizer)
            clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        metrics.update(
            predictions["localization"], batch["localization_target"],
            predictions["damage"], batch["damage_target"],
        )
        loss_sum += float(loss.detach())
        loc_sum += float(parts["localization"])
        damage_sum += float(parts["damage"])

    steps = max(1, total_steps)
    elapsed = time.perf_counter() - started
    return {
        "loss": loss_sum / steps,
        "localization_loss": loc_sum / steps,
        "damage_loss": damage_sum / steps,
        "steps": total_steps,
        "images_per_second": total_steps * loader.batch_size / max(elapsed, 1e-9),
        "metrics": metrics.compute(),
    }


@torch.inference_mode()
def evaluate(
    model: ScaleMAESiameseFPN,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    task_weights: tuple[float, float],
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    metrics = SegmentationConfusion()
    loss_sum = 0.0
    count = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        predictions, loss, _ = forward_loss(
            model, batch, device, class_weights, task_weights, amp_enabled
        )
        metrics.update(
            predictions["localization"], batch["localization_target"],
            predictions["damage"], batch["damage_target"],
        )
        loss_sum += float(loss)
        count += 1
    return {"loss": loss_sum / max(1, count), **metrics.compute()}


def make_loader(
    dataset: PairedChipDataset,
    batch_size: int,
    num_workers: int,
    sampler: EventClassBalancedSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and dataset.augment,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def save_checkpoint(
    path: Path,
    model: ScaleMAESiameseFPN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    score: float,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "validation_score": score,
    }, temporary)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model/b1_scalemae_fpn.yaml"))
    parser.add_argument("--preflight", action="store_true", help="load weights and check data/model setup; never trains")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    checkpoint = Path(config["model"]["pretrained_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Scale-MAE checkpoint not found: {checkpoint}")
    train_set = PairedChipDataset(config["data"]["manifest"], split="train", augment=True)
    validation_set = PairedChipDataset(config["data"]["manifest"], split="validation")
    model = ScaleMAESiameseFPN(
        image_size=int(config["data"]["image_size"]),
        feature_blocks=tuple(config["model"]["feature_blocks"]),
        gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
    )
    load_summary = model.backbone.load_pretrained(checkpoint)
    model.to(device)

    if args.preflight:
        sample = train_set[0]
        if tuple(sample["pre_image"].shape) != (3, 512, 512):
            raise ValueError(f"unexpected pre-image shape: {tuple(sample['pre_image'].shape)}")
        if tuple(sample["damage_target"].shape) != (512, 512):
            raise ValueError(f"unexpected damage mask shape: {tuple(sample['damage_target'].shape)}")
        model.eval()
        amp_enabled = device.type == "cuda" and config["training"]["amp"] == "fp16"
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            predictions = model(
                sample["pre_image"].unsqueeze(0).to(device),
                sample["post_image"].unsqueeze(0).to(device),
                sample["pre_gsd_m"].reshape(1).to(device),
                sample["post_gsd_m"].reshape(1).to(device),
            )
        expected_size = (1, 1, 512, 512)
        if tuple(predictions["localization"].shape) != expected_size:
            raise ValueError(f"unexpected localization output: {tuple(predictions['localization'].shape)}")
        if tuple(predictions["damage"].shape) != (1, 5, 512, 512):
            raise ValueError(f"unexpected damage output: {tuple(predictions['damage'].shape)}")
        if not all(torch.isfinite(value).all().item() for value in predictions.values()):
            raise ValueError("model preflight produced non-finite output")
        print(json.dumps({
            "status": "preflight_passed_no_training_performed",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "pretrained_parameter_groups": load_summary,
            "train_chips": len(train_set),
            "validation_chips": len(validation_set),
            "sample_pre_gsd_m": sample["pre_gsd_m"].item(),
            "sample_post_gsd_m": sample["post_gsd_m"].item(),
            "output_shapes": {key: list(value.shape) for key, value in predictions.items()},
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        }, indent=2))
        return 0

    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")
    manifest_path = Path(config["data"]["manifest"])
    (output_dir / "manifest.sha256").write_text(sha256_file(manifest_path) + "\n", encoding="utf-8")
    (output_dir / "git_commit.txt").write_text(git_revision() + "\n", encoding="utf-8")
    (output_dir / "seed.txt").write_text(f"{seed}\n", encoding="utf-8")
    (output_dir / "pretrained_checkpoint.sha256").write_text(
        sha256_file(checkpoint) + "\n", encoding="utf-8"
    )
    weights_data = json.loads(Path(config["data"]["class_weights"]).read_text(encoding="utf-8"))
    class_weights = torch.tensor(
        [weights_data["class_weights"][str(class_id)] for class_id in range(5)],
        dtype=torch.float32,
        device=device,
    )
    tasks = tuple(float(value) for value in config["loss"]["task_weights"])
    train_sampler = EventClassBalancedSampler(
        train_set.rows,
        root=train_set.root,
        rare_classes=tuple(config["training"]["sampler_rare_classes"]),
        rare_mass=float(config["training"]["sampler_rare_mass"]),
        seed=seed,
    )
    train_loader = make_loader(
        train_set,
        int(config["training"]["batch_size"]),
        int(config["training"]["num_workers"]),
        sampler=train_sampler,
    )
    validation_loader = make_loader(
        validation_set,
        int(config["training"]["batch_size"]),
        int(config["training"]["num_workers"]),
    )
    amp_enabled = device.type == "cuda" and config["training"]["amp"] == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    metrics_path = output_dir / "metrics.jsonl"
    best_score = float("-inf")
    global_epoch = 0

    sanity_count = min(
        int(config["training"]["stages"]["sanity_overfit_samples"]), len(train_set)
    )
    sanity_steps = int(config["training"]["stages"]["sanity_overfit_steps"])
    sanity_subset = Subset(train_set, list(range(sanity_count)))
    sanity_sampler = WeightedRandomSampler(
        torch.ones(sanity_count, dtype=torch.double),
        num_samples=sanity_steps,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    sanity_loader = DataLoader(
        sanity_subset,
        batch_size=1,
        sampler=sanity_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.freeze_backbone(True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["head_learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    sanity_metrics = train_epoch(
        model, sanity_loader, optimizer, scaler, device, class_weights, tasks,
        int(config["training"]["gradient_accumulation_steps"]), amp_enabled,
        max_steps=sanity_steps, epoch=0,
    )
    sanity_record = {"stage": "sanity_overfit", "train": sanity_metrics}
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sanity_record) + "\n")
    save_checkpoint(output_dir / "sanity.ckpt", model, optimizer, 0, float("nan"))

    def record(stage: str, train_metrics: dict[str, Any], val_metrics: dict[str, Any]) -> None:
        nonlocal best_score, global_epoch
        global_epoch += 1
        row = {
            "epoch": global_epoch,
            "stage": stage,
            "train": train_metrics,
            "validation": val_metrics,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        score = float(val_metrics["xview2_score"])
        save_checkpoint(output_dir / "last.ckpt", model, optimizer, global_epoch, score)
        if score > best_score:
            best_score = score
            save_checkpoint(output_dir / "best.ckpt", model, optimizer, global_epoch, score)
        print(json.dumps(row), flush=True)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["head_learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    frozen_epochs = int(config["training"]["stages"]["frozen_epochs"])
    frozen_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, frozen_epochs))
    for frozen_epoch in range(frozen_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, class_weights, tasks,
            int(config["training"]["gradient_accumulation_steps"]), amp_enabled,
            epoch=frozen_epoch + 1,
        )
        val_metrics = evaluate(model, validation_loader, device, class_weights, tasks, amp_enabled)
        record("frozen_backbone", train_metrics, val_metrics)
        frozen_schedule.step()

    model.unfreeze_last_blocks(int(config["training"]["stages"]["partial_unfreeze_last_blocks"]))
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad],
             "lr": float(config["training"]["backbone_learning_rate"])},
            {"params": list(model.fpn.parameters()) + list(model.localization_head.parameters()) + list(model.damage_head.parameters()),
             "lr": float(config["training"]["head_learning_rate"])},
        ],
        weight_decay=float(config["training"]["weight_decay"]),
    )
    partial_epochs = int(config["training"]["stages"]["partial_unfreeze_epochs"])
    partial_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, partial_epochs))
    for partial_epoch in range(partial_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, class_weights, tasks,
            int(config["training"]["gradient_accumulation_steps"]), amp_enabled,
            epoch=frozen_epochs + partial_epoch + 1,
        )
        val_metrics = evaluate(model, validation_loader, device, class_weights, tasks, amp_enabled)
        record("partial_unfreeze", train_metrics, val_metrics)
        partial_schedule.step()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
