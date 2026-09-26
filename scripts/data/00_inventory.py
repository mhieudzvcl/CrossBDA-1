"""Inventory source dataset files without modifying them."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PRE_MARKERS = ("_pre_disaster", "_pre-disaster", "_pre", "-pre")
POST_MARKERS = ("_post_disaster", "_post-disaster", "_post", "-post")
AUXILIARY_DIRS = {
    "label", "labels", "annotation", "annotations", "mask", "masks",
    "target", "targets", "ground_truth", "ground-truth",
}


def _pair_key(path: Path, root: Path, marker: str) -> str:
    relative = path.relative_to(root).with_suffix("").as_posix()
    lowered = relative.lower()
    position = lowered.rfind(marker)
    if position >= 0:
        relative = relative[:position] + relative[position + len(marker) :]
    return re.sub(r"[_\-.]+", "_", relative).strip("_").lower()


def _source_event(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) >= 3 and relative.parts[0].lower() not in {"images", "image", "imagery"}:
        return relative.parts[0]
    return path.stem.split("_")[0]


def _verify_raster(path: Path) -> tuple[str, str | None]:
    try:
        with Image.open(path) as image:
            size = f"{image.width}x{image.height}"
            image.load()
        return size, None
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return "", f"{path.as_posix()}: {error}"


def inspect_root(
    name: str, root: Path, verify_images: bool = True, workers: int = 12
) -> dict[str, Any]:
    if not root.exists():
        return {"source": name, "root": str(root), "error": "directory does not exist"}
    if not root.is_dir():
        return {"source": name, "root": str(root), "error": "path is not a directory"}

    dimensions: Counter[str] = Counter()
    corrupt: list[str] = []
    pre: dict[str, str] = {}
    post: dict[str, str] = {}
    event_pair_counts: Counter[str] = Counter()
    image_count = 0
    auxiliary_raster_count = 0
    raster_count = 0
    image_bytes = 0
    label_count = 0
    label_examples: list[str] = []
    scan_errors: list[str] = []
    raster_paths: list[Path] = []

    def on_walk_error(error: OSError) -> None:
        scan_errors.append(str(error))

    for directory, _, filenames in os.walk(root, onerror=on_walk_error):
        for filename in filenames:
            path = Path(directory) / filename
            path_lower = path.as_posix().lower()
            if any(part in path_lower for part in ("label", "annotation", "mask")):
                label_count += 1
                if len(label_examples) < 20:
                    label_examples.append(path.as_posix())
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            raster_count += 1
            raster_paths.append(path)
            relative_parts = path.relative_to(root).parts[:-1]
            is_auxiliary = any(part.lower() in AUXILIARY_DIRS for part in relative_parts)
            if is_auxiliary:
                auxiliary_raster_count += 1
            else:
                image_count += 1
            try:
                file_size = path.stat().st_size
                if not is_auxiliary:
                    image_bytes += file_size
            except OSError as error:
                scan_errors.append(f"{path.as_posix()}: {error}")
            if raster_count % 1000 == 0:
                print(f"{name}: indexed {raster_count} raster files...", flush=True)

            if not is_auxiliary:
                pre_marker = next((m for m in PRE_MARKERS if m in path_lower), None)
                post_marker = next((m for m in POST_MARKERS if m in path_lower), None)
                if pre_marker:
                    pre[_pair_key(path, root, pre_marker)] = path.as_posix()
                    event_pair_counts[_source_event(path, root)] += 1
                elif post_marker:
                    post[_pair_key(path, root, post_marker)] = path.as_posix()

    if verify_images:
        chunk_size = max(128, workers * 8)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for start in range(0, len(raster_paths), chunk_size):
                chunk = raster_paths[start : start + chunk_size]
                futures = [pool.submit(_verify_raster, path) for path in chunk]
                for future in as_completed(futures):
                    size, error = future.result()
                    if error:
                        corrupt.append(error)
                    else:
                        dimensions[size] += 1
                completed = min(start + len(chunk), len(raster_paths))
                print(f"{name}: content-checked {completed}/{len(raster_paths)} rasters...", flush=True)

    missing_post = sorted(set(pre) - set(post))
    missing_pre = sorted(set(post) - set(pre))

    return {
        "source": name,
        "root": str(root),
        "raster_file_count": raster_count,
        "image_file_count": image_count,
        "auxiliary_raster_count": auxiliary_raster_count,
        "image_file_bytes": image_bytes,
        "pre_image_count": len(pre),
        "post_image_count": len(post),
        "label_candidate_count": label_count,
        "paired_pre_post_count": len(set(pre) & set(post)),
        "event_pair_counts": dict(sorted(event_pair_counts.items())),
        "missing_post_pair_ids": missing_post,
        "missing_pre_pair_ids": missing_pre,
        "image_content_checked": verify_images,
        "corrupt_images": corrupt if verify_images else None,
        "scan_errors": scan_errors,
        "dimensions": dict(sorted(dimensions.items())),
        "label_candidate_examples": label_examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbd-train", type=Path)
    parser.add_argument("--xbd-tier3", type=Path)
    parser.add_argument("--ebd", type=Path)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="count filenames and pair markers without opening image contents",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    roots = [(name, path) for name, path in (
        ("xbd_train", args.xbd_train),
        ("xbd_tier3", args.xbd_tier3),
        ("ebd", args.ebd),
    ) if path is not None]
    if not roots:
        parser.error("provide at least one source root")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            inspect_root(name, path, verify_images=not args.metadata_only, workers=args.workers)
            for name, path in roots
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(f"Wrote inventory: {args.out}")

    errors = [entry for entry in report["sources"] if "error" in entry]
    for entry in report["sources"]:
        if "error" in entry:
            print(f"ERROR {entry['source']}: {entry['error']}", file=sys.stderr)
        else:
            unpaired = len(entry["missing_post_pair_ids"]) + len(entry["missing_pre_pair_ids"])
            print(
                f"{entry['source']}: images={entry['image_file_count']}, "
                f"auxiliary_rasters={entry['auxiliary_raster_count']}, "
                f"pre/post pairs={entry['paired_pre_post_count']}, "
                f"content_checked={entry['image_content_checked']}, "
                f"corrupt={len(entry['corrupt_images'] or [])}, "
                f"unpaired={unpaired}"
            )
            if unpaired:
                errors.append(entry)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
