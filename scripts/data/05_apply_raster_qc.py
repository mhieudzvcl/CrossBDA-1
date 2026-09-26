"""Apply a successful complete raster QC report to the parent manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SOURCE_KEY = {"xbd_train": ("xbd", "train"), "xbd_tier3": ("xbd", "tier3"), "ebd": ("ebd", "events")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("data/reports/inventory_content_qc.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/parent_samples.csv"))
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    by_source = {item["source"]: item for item in report["sources"]}
    if set(by_source) != set(SOURCE_KEY):
        raise ValueError(f"Expected QC sources {sorted(SOURCE_KEY)}, got {sorted(by_source)}")
    for name, item in by_source.items():
        if not item.get("image_content_checked"):
            raise ValueError(f"{name}: full image-content check is not marked complete")
        if item.get("corrupt_images") or item.get("scan_errors"):
            raise ValueError(
                f"{name}: QC found {len(item.get('corrupt_images') or [])} corrupt rasters and "
                f"{len(item.get('scan_errors') or [])} scan errors; manifest not updated"
            )
        if item.get("missing_pre_pair_ids") or item.get("missing_post_pair_ids"):
            raise ValueError(f"{name}: pre/post imagery is unpaired; manifest not updated")
        if item.get("raster_file_count") != sum(item.get("dimensions", {}).values()):
            raise ValueError(f"{name}: dimension histogram does not account for every raster")

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0].keys()) if rows else []
    manifest_counts = {
        key: sum((row["source"], row["source_subset"]) == key for row in rows)
        for key in SOURCE_KEY.values()
    }
    for report_name, source_key in SOURCE_KEY.items():
        reported_pairs = by_source[report_name]["paired_pre_post_count"]
        if manifest_counts[source_key] != reported_pairs:
            raise ValueError(
                f"{report_name}: QC found {reported_pairs} pairs but manifest has "
                f"{manifest_counts[source_key]} rows; manifest not updated"
            )
    for row in rows:
        source_key = (row["source"], row["source_subset"])
        report_name = next((name for name, key in SOURCE_KEY.items() if key == source_key), None)
        if report_name is None:
            raise ValueError(f"No raster QC scope for manifest row {row['sample_id']}: {source_key}")
        dimensions = by_source[report_name]["dimensions"]
        row["qc_status"] = "raster_qc_pass"
        if len(dimensions) == 1:
            dimension = next(iter(dimensions))
            width, height = dimension.split("x", 1)
            row["width"], row["height"] = width, height
        else:
            row["width"], row["height"] = "", ""

    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.manifest)
    print(f"Applied full raster QC status to {len(rows)} parent rows in {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
