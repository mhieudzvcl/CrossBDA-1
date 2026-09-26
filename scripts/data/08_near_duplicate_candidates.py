"""Find cross-subset near-duplicate xBD images for manual review using dHash."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        values = list(small.getdata())
    result = 0
    for y in range(8):
        offset = y * 9
        for x in range(8):
            result = (result << 1) | int(values[offset + x] > values[offset + x + 1])
    return result


def distance(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


@dataclass
class BKNode:
    value: int
    records: list[dict[str, Any]] = field(default_factory=list)
    children: dict[int, "BKNode"] = field(default_factory=dict)


class BKTree:
    def __init__(self) -> None:
        self.root: BKNode | None = None

    def add(self, value: int, record: dict[str, Any]) -> None:
        if self.root is None:
            self.root = BKNode(value, [record])
            return
        node = self.root
        while True:
            gap = distance(value, node.value)
            if gap == 0:
                node.records.append(record)
                return
            child = node.children.get(gap)
            if child is None:
                node.children[gap] = BKNode(value, [record])
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, dict[str, Any]]]:
        if self.root is None:
            return []
        found: list[tuple[int, dict[str, Any]]] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            gap = distance(value, node.value)
            if gap <= radius:
                found.extend((gap, row) for row in node.records)
            low, high = gap - radius, gap + radius
            pending.extend(child for edge, child in node.children.items() if low <= edge <= high)
        return found


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/splits_v1.csv"))
    parser.add_argument("--exact-links", type=Path, default=Path("data/manifests/cross_event_duplicate_links.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/reports/xbd_near_duplicates_candidates.json"))
    parser.add_argument("--radius", type=int, default=4, help="maximum Hamming distance across 64 dHash bits")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    records: list[dict[str, Any]] = []
    for row in rows:
        if row["source"] != "xbd" or row["source_subset"] not in {"train", "tier3"}:
            continue
        for side in ("pre", "post"):
            records.append({
                "sample_id": row["sample_id"],
                "event_id": row["event_id"],
                "canonical_name": row["canonical_name"],
                "source_subset": row["source_subset"],
                "side": side,
                "split": row["split"],
                "path": row[f"{side}_path"],
            })

    errors: list[str] = []
    fingerprints: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(dhash, Path(record["path"])): record for record in records}
        for index, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            try:
                value = future.result()
                record["dhash"] = value
                fingerprints.setdefault(value, []).append(record)
            except (OSError, ValueError) as error:
                errors.append(f"{record['path']}: {error}")
            if index % 1000 == 0:
                print(f"Fingerprinted {index}/{len(records)} xBD images", flush=True)

    tier3_tree = BKTree()
    for value, grouped in fingerprints.items():
        for record in grouped:
            if record["source_subset"] == "tier3":
                tier3_tree.add(value, record)

    exact_keys: set[tuple[str, str, str]] = set()
    for link in read_rows(args.exact_links):
        exact_keys.add((link["train_sample_id"], link["duplicate_side"], link["tier3_sample_id"]))

    candidates: list[dict[str, Any]] = []
    for value, grouped in fingerprints.items():
        for train_record in grouped:
            if train_record["source_subset"] != "train":
                continue
            for gap, tier3_record in tier3_tree.query(value, args.radius):
                key = (train_record["sample_id"], train_record["side"], tier3_record["sample_id"])
                if key in exact_keys:
                    continue
                candidates.append({
                    "hamming_distance": gap,
                    "train_sample_id": train_record["sample_id"],
                    "train_event_id": train_record["event_id"],
                    "train_split": train_record["split"],
                    "train_side": train_record["side"],
                    "train_path": train_record["path"],
                    "tier3_sample_id": tier3_record["sample_id"],
                    "tier3_event_id": tier3_record["event_id"],
                    "tier3_split": tier3_record["split"],
                    "tier3_side": tier3_record["side"],
                    "tier3_path": tier3_record["path"],
                })
    candidates.sort(key=lambda item: (item["hamming_distance"], item["train_sample_id"], item["tier3_sample_id"]))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "method": "64-bit difference hash; candidate generation only, manual review required",
        "maximum_hamming_distance": args.radius,
        "images_fingerprinted": sum(len(group) for group in fingerprints.values()),
        "unique_hashes": len(fingerprints),
        "exact_matches_excluded": len(exact_keys),
        "near_duplicate_candidate_pairs": len(candidates),
        "candidates_crossing_split": sum(item["train_split"] != item["tier3_split"] for item in candidates),
        "read_errors": errors,
        "candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(f"Wrote {len(candidates)} near-duplicate candidates; read errors={len(errors)}; report={args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
