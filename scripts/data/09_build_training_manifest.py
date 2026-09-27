"""Join processed chips with per-image or documented nominal GSD metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def xbd_gsd(annotation_path: str, sample_id: str, side: str) -> float:
    last_error: OSError | None = None
    for attempt in range(3):
        try:
            document = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
            break
        except OSError as error:
            last_error = error
            if attempt == 2:
                raise
            time.sleep(0.5 * (2**attempt))
    else:
        raise last_error or OSError(f"unable to read {annotation_path}")
    metadata: dict[str, Any] = document.get("metadata", {})
    value = metadata.get("gsd")
    if value is None:
        raise ValueError(f"missing {side} GSD in xBD annotation metadata: {sample_id}")
    gsd = float(value)
    if not math.isfinite(gsd) or gsd <= 0:
        raise ValueError(f"invalid {side} GSD {value!r}: {sample_id}")
    return gsd


def parent_gsd(row: dict[str, str], ebd_gsd: float) -> tuple[float, float, str]:
    sample_id = row["sample_id"]
    if row["source"] == "xbd":
        pre_gsd = xbd_gsd(row["pre_annotation_path"], sample_id, "pre")
        post_gsd = xbd_gsd(row["post_annotation_path"], sample_id, "post")
        return pre_gsd, post_gsd, "xbd_annotation_metadata_per_image"
    if row["source"] == "ebd":
        return ebd_gsd, ebd_gsd, "ebd_published_range_midpoint_approximation"
    raise ValueError(f"unsupported source {row['source']!r}: {sample_id}")


def write_checkpoint(path: Path, values: dict[str, tuple[float, float, str]], errors: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {key: list(value) for key, value in values.items()}
    temporary.write_text(
        json.dumps({"gsd_by_sample_id": payload, "errors": errors}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chips", type=Path, default=Path("data/processed/512/chips.csv"))
    parser.add_argument("--parents", type=Path, default=Path("data/manifests/splits_v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/512/training_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/processed/512/training_gsd_checkpoint.json"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--ebd-nominal-gsd-m",
        type=float,
        default=0.4,
        help="documented midpoint of EBD's published 0.3-0.5 m GSD range; not per-image metadata",
    )
    args = parser.parse_args()
    if not math.isfinite(args.ebd_nominal_gsd_m) or args.ebd_nominal_gsd_m <= 0:
        parser.error("--ebd-nominal-gsd-m must be a positive finite number")

    parents = {row["sample_id"]: row for row in read_rows(args.parents)}
    chips = read_rows(args.chips)
    cache: dict[str, tuple[float, float, str]] = {}
    errors: list[str] = []
    if args.checkpoint.is_file():
        try:
            payload = json.loads(args.checkpoint.read_text(encoding="utf-8"))
            cache = {
                sample_id: (float(values[0]), float(values[1]), str(values[2]))
                for sample_id, values in payload.get("gsd_by_sample_id", {}).items()
            }
            errors = [str(error) for error in payload.get("errors", [])]
        except (OSError, ValueError, KeyError, TypeError):
            cache, errors = {}, []
    used_ids = {chip["sample_id"] for chip in chips}
    if any(sample_id not in parents for sample_id in used_ids):
        missing = sorted(used_ids - parents.keys())
        raise ValueError(f"chip parents absent from split manifest: {missing[:10]}")
    pending = [parents[sample_id] for sample_id in used_ids if sample_id not in cache]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(parent_gsd, row, args.ebd_nominal_gsd_m): row for row in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            sample_id = row["sample_id"]
            try:
                cache[sample_id] = future.result()
                error_prefix = f"{sample_id}:"
                errors = [error for error in errors if not error.startswith(error_prefix)]
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{sample_id}: {error}")
            if index % 500 == 0:
                write_checkpoint(args.checkpoint, cache, errors)
                print(f"Read GSD metadata for {len(cache)}/{len(used_ids)} parents; errors={len(errors)}", flush=True)
        if pending:
            write_checkpoint(args.checkpoint, cache, errors)
    if errors or len(cache) != len(used_ids):
        print(f"GSD metadata incomplete: {len(cache)}/{len(used_ids)} parents; errors={len(errors)}")
        return 1

    enriched: list[dict[str, str]] = []
    gsd_values: Counter[str] = Counter()
    for chip in chips:
        sample_id = chip["sample_id"]
        parent = parents.get(sample_id)
        if parent is None:
            raise ValueError(f"chip parent absent from split manifest: {sample_id}")
        pre_gsd, post_gsd, provenance = cache[sample_id]
        enriched.append({
            **chip,
            "canonical_name": parent["canonical_name"],
            "hazard": parent["hazard"],
            "country": parent["country"],
            "pre_gsd_m": f"{pre_gsd:.8g}",
            "post_gsd_m": f"{post_gsd:.8g}",
            "gsd_provenance": provenance,
        })
        gsd_values[f"{parent['source']}:{chip['split']}:pre:{pre_gsd:.6g}"] += 1
        gsd_values[f"{parent['source']}:{chip['split']}:post:{post_gsd:.6g}"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(enriched[0]) if enriched else [])
        writer.writeheader()
        writer.writerows(enriched)
    temporary.replace(args.out)

    report = {
        "chips": len(enriched),
        "parents_with_gsd": len(cache),
        "ebd_nominal_gsd_m": args.ebd_nominal_gsd_m,
        "ebd_nominal_gsd_note": "Approximation: midpoint of published 0.3-0.5 m range; source files have no per-image GSD metadata.",
        "gsd_chip_counts_by_source_split_side_value": dict(sorted(gsd_values.items())),
        "output": str(args.out),
    }
    report_path = args.out.with_name("training_manifest_report.json")
    temp_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temp_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp_report.replace(report_path)
    args.checkpoint.unlink(missing_ok=True)
    print(f"Wrote {len(enriched)} training-manifest rows for {len(cache)} parents: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
