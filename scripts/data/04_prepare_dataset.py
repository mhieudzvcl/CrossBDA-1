"""Rasterize, crop, and normalize CrossBDA parent samples into 512px chips."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

CHIP_SIZE = 512
DAMAGE = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}
COORDINATE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def polygon_points(wkt: str) -> list[tuple[float, float]]:
    # xBD footprint annotations use a single exterior POLYGON ring.
    start = wkt.find("((")
    if start < 0:
        return []
    end = wkt.find("))", start + 2)
    if end < 0:
        return []
    ring = wkt[start + 2 : end]
    return [(float(x), float(y)) for x, y in COORDINATE.findall(ring)]


def rasterize_xbd(annotation_path: Path, size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    loc = Image.new("L", size, 0)
    damage = Image.new("L", size, 0)
    loc_draw, damage_draw = ImageDraw.Draw(loc), ImageDraw.Draw(damage)
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    features: list[dict[str, Any]] = document.get("features", {}).get("xy", [])
    for feature in features:
        properties = feature.get("properties", {})
        if properties.get("feature_type") != "building":
            continue
        points = polygon_points(feature.get("wkt", ""))
        if len(points) < 3:
            continue
        loc_draw.polygon(points, fill=1)
        subtype = str(properties.get("subtype", "")).lower()
        damage_draw.polygon(points, fill=DAMAGE.get(subtype, 255))
    return loc, damage


def normalize_ebd_mask(path: Path, size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    with Image.open(path) as opened:
        if opened.size != size:
            raise ValueError(f"mask/image size mismatch: {path} {opened.size} != {size}")
        if opened.mode == "L":
            damage = opened.copy()
        elif opened.mode == "P":
            damage = Image.frombytes("L", opened.size, opened.tobytes())
        else:
            raise ValueError(f"unsupported categorical mask mode {opened.mode}: {path}")
    values = set(damage.getdata())
    invalid = values - {0, 1, 2, 3, 4, 255}
    if invalid:
        raise ValueError(f"unexpected EBD mask class values {sorted(invalid)[:20]}: {path}")
    loc = damage.point(lambda value: 1 if value not in (0, 255) else 0)
    return loc, damage


def load_pair(row: dict[str, str]) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    with Image.open(row["pre_path"]) as image:
        pre = image.convert("RGB")
    with Image.open(row["post_path"]) as image:
        post = image.convert("RGB")
    if pre.size != post.size:
        raise ValueError(f"pre/post size mismatch for {row['sample_id']}: {pre.size} != {post.size}")
    if row["source"] == "xbd":
        pre_loc, pre_damage = rasterize_xbd(Path(row["pre_annotation_path"]), pre.size)
        post_loc, post_damage = rasterize_xbd(Path(row["post_annotation_path"]), post.size)
    else:
        pre_loc, pre_damage = normalize_ebd_mask(Path(row["pre_annotation_path"]), pre.size)
        post_loc, post_damage = normalize_ebd_mask(Path(row["post_annotation_path"]), post.size)
    return pre, post, pre_loc, post_loc, pre_damage, post_damage


def load_pair_with_retry(
    row: dict[str, str], attempts: int = 3
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    for attempt in range(attempts):
        try:
            return load_pair(row)
        except OSError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError("unreachable retry state")


def save_chip(
    image: Image.Image, path: Path, source_path: str, box: tuple[int, int, int, int]
) -> None:
    crop = image.crop(box)
    suffix = Path(source_path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        suffix = ".png"
    path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = (
        {"quality": 95, "subsampling": 0}
        if suffix in {".jpg", ".jpeg"}
        else {"compress_level": 1} if suffix == ".png" else {}
    )
    crop.save(path, **options)


def chip_boxes(size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    width, height = size
    if width < CHIP_SIZE or height < CHIP_SIZE:
        raise ValueError(f"source image smaller than {CHIP_SIZE}px ({size})")
    if size == (CHIP_SIZE, CHIP_SIZE):
        return [(0, 0, CHIP_SIZE, CHIP_SIZE)]
    if size == (2 * CHIP_SIZE, 2 * CHIP_SIZE):
        return [
            (0, 0, CHIP_SIZE, CHIP_SIZE),
            (CHIP_SIZE, 0, 2 * CHIP_SIZE, CHIP_SIZE),
            (0, CHIP_SIZE, CHIP_SIZE, 2 * CHIP_SIZE),
            (CHIP_SIZE, CHIP_SIZE, 2 * CHIP_SIZE, 2 * CHIP_SIZE),
        ]
    raise ValueError(f"unsupported source dimensions {width}x{height}")


def chip_record(
    row: dict[str, str], output_root: Path, quadrant: int, box: tuple[int, int, int, int]
) -> dict[str, Any]:
    chip_id = f"{row['sample_id']}_q{quadrant}"
    event_dir = Path(row["source"]) / row["source_subset"] / row["event_id"]
    image_dir = output_root / event_dir / "images"
    mask_dir = output_root / event_dir / "masks"
    pre_suffix = Path(row["pre_path"]).suffix.lower()
    post_suffix = Path(row["post_path"]).suffix.lower()
    if pre_suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        pre_suffix = ".png"
    if post_suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        post_suffix = ".png"
    return {
        "chip_id": chip_id,
        "parent_id": row["parent_id"],
        "sample_id": row["sample_id"],
        "source": row["source"],
        "source_subset": row["source_subset"],
        "event_id": row["event_id"],
        "quadrant": quadrant,
        "x": box[0],
        "y": box[1],
        "width": CHIP_SIZE,
        "height": CHIP_SIZE,
        "allowed_train": row["allowed_train"],
        "external_target": row["external_target"],
        "split": row.get("split", ""),
        "pre_image": str(image_dir / f"{chip_id}_pre{pre_suffix}"),
        "post_image": str(image_dir / f"{chip_id}_post{post_suffix}"),
        "pre_localization_mask": str(mask_dir / f"{chip_id}_pre_localization.png"),
        "post_localization_mask": str(mask_dir / f"{chip_id}_post_localization.png"),
        "pre_damage_mask": str(mask_dir / f"{chip_id}_pre_damage.png"),
        "post_damage_mask": str(mask_dir / f"{chip_id}_post_damage.png"),
    }


def add_mask_counts(
    counts: Counter[str], row: dict[str, str], name: str, histogram: list[int]
) -> None:
    prefix = f"{row['source']}:{row['event_id']}:{row.get('split', '')}:{name}:"
    for value in (0, 1, 2, 3, 4, 255):
        count = histogram[value]
        if count:
            counts[prefix + str(value)] += count


def resume_parent(
    row: dict[str, str], output_root: Path, boxes: list[tuple[int, int, int, int]]
) -> tuple[list[dict[str, Any]], Counter[str]] | None:
    chips: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    mask_fields = (
        ("pre_localization", "pre_localization_mask"),
        ("post_localization", "post_localization_mask"),
        ("pre_damage", "pre_damage_mask"),
        ("post_damage", "post_damage_mask"),
    )
    for quadrant, box in enumerate(boxes):
        chip = chip_record(row, output_root, quadrant, box)
        expected = [chip["pre_image"], chip["post_image"]] + [chip[field] for _, field in mask_fields]
        if any(not Path(path).is_file() or Path(path).stat().st_size == 0 for path in expected):
            return None
        for path in expected[:2]:
            with Image.open(path) as image:
                image.verify()
        for name, field in mask_fields:
            with Image.open(chip[field]) as mask:
                if mask.size != (CHIP_SIZE, CHIP_SIZE):
                    return None
                add_mask_counts(class_counts, row, name, mask.histogram())
        chips.append(chip)
    return chips, class_counts


def prepare_parent(
    row: dict[str, str], output_root: Path
) -> tuple[list[dict[str, Any]], Counter[str], str | None, bool]:
    if row["qc_status"] == "missing_annotation":
        return [], Counter(), f"{row['sample_id']}: missing annotation path", False
    try:
        expected_size = (int(row["width"]), int(row["height"]))
        boxes = chip_boxes(expected_size)
        cached = resume_parent(row, output_root, boxes)
        if cached is not None:
            chips, counts = cached
            return chips, counts, None, True

        pre, post, pre_loc, post_loc, pre_damage, post_damage = load_pair_with_retry(row)
        if pre.size != expected_size or post.size != expected_size:
            raise ValueError(f"source image dimensions disagree with manifest for {row['sample_id']}")

        chips: list[dict[str, Any]] = []
        class_counts: Counter[str] = Counter()
        for quadrant, box in enumerate(boxes):
            chip = chip_record(row, output_root, quadrant, box)
            save_chip(pre, Path(chip["pre_image"]).with_suffix(""), row["pre_path"], box)
            save_chip(post, Path(chip["post_image"]).with_suffix(""), row["post_path"], box)
            for name, mask in (
                ("pre_localization", pre_loc), ("post_localization", post_loc),
                ("pre_damage", pre_damage), ("post_damage", post_damage),
            ):
                path = Path(chip[f"{name}_mask"])
                path.parent.mkdir(parents=True, exist_ok=True)
                cropped = mask.crop(box)
                cropped.save(path, compress_level=1)
                add_mask_counts(class_counts, row, name, cropped.histogram())
            chips.append(chip)
        return chips, class_counts, None, False
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [], Counter(), f"{row['sample_id']}: {error}", False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/splits_v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/512"))
    parser.add_argument("--limit", type=int, default=0, help="process the first N rows (0 means all)")
    parser.add_argument("--source", choices=("xbd", "ebd"), help="process only one dataset source")
    parser.add_argument("--include-excluded", action="store_true", help="also preprocess events excluded from training")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    if args.source:
        rows = [row for row in rows if row["source"] == args.source]
    if args.limit:
        rows = rows[: args.limit]
    skipped_excluded = sum(row["allowed_train"] == "0" for row in rows) if not args.include_excluded else 0
    active_rows = [row for row in rows if args.include_excluded or row["allowed_train"] != "0"]
    chips: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    errors: list[str] = []
    resumed_parents = 0
    batch_size = max(64, max(1, args.workers) * 8)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for start in range(0, len(active_rows), batch_size):
            batch = active_rows[start : start + batch_size]
            futures = [pool.submit(prepare_parent, row, args.out) for row in batch]
            for future in as_completed(futures):
                try:
                    parent_chips, counts, error, resumed = future.result()
                    chips.extend(parent_chips)
                    class_counts.update(counts)
                    resumed_parents += int(resumed)
                    if error:
                        errors.append(error)
                except Exception as error:
                    errors.append(f"unexpected worker failure: {error}")
            print(
                f"Prepared {min(start + len(batch), len(active_rows))}/{len(active_rows)} eligible parents "
                f"({len(chips)} chips; reused {resumed_parents} parents), errors={len(errors)}",
                flush=True,
            )
    chips.sort(key=lambda row: (row["sample_id"], row["quadrant"]))
    errors.sort()

    args.out.mkdir(parents=True, exist_ok=True)
    chip_fields = list(chips[0]) if chips else []
    chips_path = args.out / "chips.csv"
    temporary_chips = chips_path.with_suffix(".csv.tmp")
    with temporary_chips.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=chip_fields)
        writer.writeheader()
        writer.writerows(chips)
    temporary_chips.replace(chips_path)
    report = {
        "input_manifest": str(args.manifest),
        "parents_requested": len(rows),
        "excluded_parents_skipped": skipped_excluded,
        "parents_reused_from_existing_output": resumed_parents,
        "chips_written": len(chips),
        "mask_pixel_counts_by_source_task_class": dict(sorted(class_counts.items())),
        "errors": errors,
    }
    report_path = args.out / "preparation_report.json"
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)
    print(f"Wrote {len(chips)} chips; parents with errors={len(errors)}; output={args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
