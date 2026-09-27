"""Compute weighted-CE class weights from training-split post-damage pixels only."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("data/processed/512/preparation_report.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/512/damage_class_weights.json")
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    counts: Counter[int] = Counter()
    for key, count in report["mask_pixel_counts_by_source_task_class"].items():
        parts = key.split(":")
        if len(parts) != 5:
            raise ValueError(f"unexpected mask count key: {key}")
        _, _, split, task, class_value = parts
        if split == "train" and task == "post_damage":
            value = int(class_value)
            if value in range(5):
                counts[value] += int(count)
    if set(counts) != set(range(5)):
        raise ValueError(f"training post-damage masks do not contain all five classes: {dict(counts)}")

    total = sum(counts.values())
    raw = {class_id: math.sqrt(total / counts[class_id]) for class_id in range(5)}
    mean = sum(raw.values()) / len(raw)
    weights = {str(class_id): raw[class_id] / mean for class_id in range(5)}
    result = {
        "source_report": str(args.report),
        "split_used": "train",
        "target_mask": "post_damage",
        "method": "inverse_sqrt_frequency_normalized_to_mean_1",
        "pixel_counts": {str(class_id): counts[class_id] for class_id in range(5)},
        "class_weights": weights,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(f"Wrote training-only damage class weights: {args.out}")
    print("class_weights", weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
