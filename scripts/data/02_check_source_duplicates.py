"""Find exact duplicate xBD image files across train and Tier3.

Files with different byte sizes cannot be exact duplicates, so only paths in a
byte-size group shared by both source subsets are read and SHA-256 hashed.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import itertools
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/parent_samples.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/reports/xbd_exact_duplicates.json"))
    parser.add_argument(
        "--links-out",
        type=Path,
        default=Path("data/manifests/cross_event_duplicate_links.csv"),
    )
    parser.add_argument("--plan-only", action="store_true", help="count same-size candidates without opening image data")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    records_by_directory: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["source"] != "xbd" or row["source_subset"] not in {"train", "tier3"}:
                continue
            for side in ("pre", "post"):
                value = row[f"{side}_path"]
                path = Path(value)
                records_by_directory[path.parent].append({
                    "path": str(path),
                    "filename": path.name.lower(),
                    "source_subset": row["source_subset"],
                    "event_id": row["event_id"],
                    "sample_id": row["sample_id"],
                    "side": side,
                })

    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for directory, records in records_by_directory.items():
        try:
            with os.scandir(directory) as entries:
                sizes = {entry.name.lower(): entry.stat(follow_symlinks=False).st_size
                         for entry in entries if entry.is_file(follow_symlinks=False)}
        except OSError as error:
            missing.append(f"{directory}: {error}")
            continue
        for entry in records:
            size = sizes.get(entry["filename"])
            if size is None:
                missing.append(entry["path"])
                continue
            entry["size"] = size
            by_size[size].append(entry)
        print(f"Indexed {len(records)} manifest image paths under {directory}", flush=True)

    candidate_groups = [
        files for files in by_size.values()
        if {entry["source_subset"] for entry in files} == {"train", "tier3"}
    ]
    candidates = [entry for group in candidate_groups for entry in group]
    print(
        f"Hashing {len(candidates)} same-size candidates "
        f"({sum(entry['size'] for entry in candidates):,} bytes)",
        flush=True,
    )
    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hash_errors: list[str] = []
    total_bytes_hashed = 0

    if not args.plan_only:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            future_entries = {pool.submit(sha256_file, Path(entry["path"])): entry for entry in candidates}
            for index, future in enumerate(as_completed(future_entries), start=1):
                entry = future_entries[future]
                path = Path(entry["path"])
                try:
                    entry["sha256"] = future.result()
                    total_bytes_hashed += entry["size"]
                    hashes[entry["sha256"]].append(entry)
                except OSError as error:
                    hash_errors.append(f"{path}: {error}")
                if index % 10 == 0:
                    print(f"Hashed {index}/{len(candidates)} same-size candidates", flush=True)

    duplicate_groups = [
        entries for entries in hashes.values()
        if len(entries) > 1 and len({entry["source_subset"] for entry in entries}) > 1
    ]
    links: list[dict[str, str]] = []
    for entries in duplicate_groups:
        train_entries = [entry for entry in entries if entry["source_subset"] == "train"]
        tier3_entries = [entry for entry in entries if entry["source_subset"] == "tier3"]
        for train_entry, tier3_entry in itertools.product(train_entries, tier3_entries):
            links.append({
                "link_group_id": f"sha256_{train_entry['sha256'][:16]}",
                "sha256": train_entry["sha256"],
                "duplicate_side": train_entry["side"],
                "train_sample_id": train_entry["sample_id"],
                "tier3_sample_id": tier3_entry["sample_id"],
                "train_event_id": train_entry["event_id"],
                "tier3_event_id": tier3_entry["event_id"],
                "train_path": train_entry["path"],
                "tier3_path": tier3_entry["path"],
            })
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "scope": "xBD train vs Tier3 source image files; pre and post imagery",
        "unique_file_sizes": len(by_size),
        "cross_subset_same_size_groups": len(candidate_groups),
        "candidate_files": len(candidates),
        "candidate_bytes": sum(entry["size"] for entry in candidates),
        "candidate_examples": [
            {key: entry[key] for key in ("path", "source_subset", "size")}
            for entry in candidates[:10]
        ],
        "candidate_files_hashed": len(candidates) - len(hash_errors) if not args.plan_only else 0,
        "bytes_hashed": total_bytes_hashed,
        "exact_cross_subset_duplicate_groups": duplicate_groups if not args.plan_only else None,
        "missing_or_unreadable_files": missing,
        "hash_errors": hash_errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.links_out.parent.mkdir(parents=True, exist_ok=True)
    link_fields = [
        "link_group_id", "sha256", "duplicate_side", "train_sample_id", "tier3_sample_id",
        "train_event_id", "tier3_event_id", "train_path", "tier3_path",
    ]
    with args.links_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=link_fields)
        writer.writeheader()
        writer.writerows(links)
    print(
        f"Hashed {report['candidate_files_hashed']} candidates; "
        f"exact cross-subset duplicate groups={len(duplicate_groups)}; "
        f"links={len(links)}; report={args.out}"
    )
    return 1 if missing or hash_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
