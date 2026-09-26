"""Build a parent-level manifest from xBD and EBD filenames, without opening imagery."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PRE_MARKERS = ("_pre_disaster", "_pre-disaster", "_pre", "-pre")
POST_MARKERS = ("_post_disaster", "_post-disaster", "_post", "-post")
IMAGE_DIRS = {"image", "images", "imagery"}

FIELDS = [
    "sample_id", "parent_id", "source", "source_subset", "source_pair_id",
    "event_id", "canonical_name", "hazard", "country", "pre_path", "post_path",
    "pre_annotation_path", "post_annotation_path", "width", "height",
    "allowed_train", "external_target", "qc_status", "split",
]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def read_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["event_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate event_id values in registry: {path}")
    return by_id


def event_token(path: Path, image_root: Path, source: str) -> str:
    if source == "ebd":
        return image_root.parent.name
    return path.stem.split("_")[0]


def annotation_path(path: Path, image_root: Path, source: str) -> Path:
    annotation_dir = "masks" if source == "ebd" else "labels"
    annotation_root = image_root.parent / annotation_dir
    return annotation_root / path.name if source == "ebd" else annotation_root / path.with_suffix(".json").name


def collect_source(
    source: str,
    subset: str,
    root: Path,
    registry: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not root.is_dir():
        return [], [f"{source}/{subset}: source directory not found: {root}"]

    if source == "ebd":
        image_roots = [p for p in root.glob("*/images") if p.is_dir()]
    else:
        candidate = root / "images"
        image_roots = [candidate] if candidate.is_dir() else []
    if not image_roots:
        return [], [f"{source}/{subset}: no images directory found under {root}"]

    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for image_root in image_roots:
        files = [
            p for p in image_root.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        print(f"{source}/{subset}: indexing {image_root.parent.name} ({len(files)} raster files)", flush=True)
        annotation_dir = image_root.parent / ("masks" if source == "ebd" else "labels")
        if annotation_dir.is_dir():
            annotation_names = {p.name.lower() for p in annotation_dir.iterdir() if p.is_file()}
        else:
            annotation_names = set()
        pairs: dict[str, dict[str, Path]] = defaultdict(dict)
        for path in files:
            stem_lower = path.stem.lower()
            marker = next((m for m in PRE_MARKERS if stem_lower.endswith(m)), None)
            side = "pre"
            if marker is None:
                marker = next((m for m in POST_MARKERS if stem_lower.endswith(m)), None)
                side = "post"
            if marker is None:
                continue
            key = path.stem[: -len(marker)].lower()
            if side in pairs[key]:
                errors.append(f"duplicate {side} image for pair {key}: {path}")
            pairs[key][side] = path

        for pair_key, sides in pairs.items():
            if set(sides) != {"pre", "post"}:
                errors.append(f"unpaired image {source}/{subset}: {pair_key}")
                continue

            pre_path, post_path = sides["pre"], sides["post"]
            raw_event = event_token(pre_path, image_root, source)
            event_id = f"{source}_{slug(raw_event)}"
            event = registry.get(event_id)
            if event is None:
                errors.append(f"event is not registered: {event_id} (source folder/name: {raw_event})")
                continue

            pre_annotation = annotation_path(pre_path, image_root, source)
            post_annotation = annotation_path(post_path, image_root, source)
            pre_annotation_found = pre_annotation.name.lower() in annotation_names
            post_annotation_found = post_annotation.name.lower() in annotation_names
            missing_annotation = not pre_annotation_found or not post_annotation_found
            relative_pair = pre_path.stem
            for marker in PRE_MARKERS:
                if relative_pair.lower().endswith(marker):
                    relative_pair = relative_pair[: -len(marker)]
                    break
            source_pair_id = relative_pair
            sample_id = f"{source}_{subset}_{slug(source_pair_id)}"
            rows.append({
                "sample_id": sample_id,
                "parent_id": sample_id,
                "source": source,
                "source_subset": subset,
                "source_pair_id": source_pair_id,
                "event_id": event_id,
                "canonical_name": event["canonical_name"],
                "hazard": event["hazard"],
                "country": event["country"],
                "pre_path": str(pre_path),
                "post_path": str(post_path),
                "pre_annotation_path": str(pre_annotation) if pre_annotation_found else "",
                "post_annotation_path": str(post_annotation) if post_annotation_found else "",
                "width": "",
                "height": "",
                "allowed_train": event["allowed_train"],
                "external_target": event["external_target"],
                "qc_status": "missing_annotation" if missing_annotation else "pending_content_qc",
                "split": "",
            })
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbd-train", type=Path, required=True)
    parser.add_argument("--xbd-tier3", type=Path, required=True)
    parser.add_argument("--ebd", type=Path, required=True)
    parser.add_argument("--event-registry", type=Path, default=Path("data/manifests/event_registry.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/manifests/parent_samples.csv"))
    args = parser.parse_args()

    registry = read_registry(args.event_registry)
    all_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for source, subset, root in (
        ("xbd", "train", args.xbd_train),
        ("xbd", "tier3", args.xbd_tier3),
        ("ebd", "events", args.ebd),
    ):
        rows, source_errors = collect_source(source, subset, root, registry)
        all_rows.extend(rows)
        errors.extend(source_errors)
        counts = Counter(row["event_id"] for row in rows)
        print(f"{source}/{subset}: {len(rows)} parent pairs across {len(counts)} events")

    ids = [row["sample_id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate sample_id values found")

    if errors:
        print(f"Manifest not written; found {len(errors)} issue(s):", file=sys.stderr)
        for error in errors[:100]:
            print(f"- {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"- ... {len(errors) - 100} additional issue(s)", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} parent pairs: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
