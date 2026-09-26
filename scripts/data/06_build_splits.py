"""Build and validate an event-level split manifest with duplicate links enforced."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/parent_samples.csv"))
    parser.add_argument("--links", type=Path, default=Path("data/manifests/cross_event_duplicate_links.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/data/splits_v1.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/manifests/splits_v1.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/splits_v1.json"))
    args = parser.parse_args()

    rows, fields = read_csv(args.manifest)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assignment_lists = config["assignments"]
    event_split: dict[str, str] = {}
    for split, names in assignment_lists.items():
        for canonical_name in names:
            if canonical_name in event_split:
                raise ValueError(f"Event {canonical_name} assigned more than once")
            event_split[canonical_name] = split

    observed_allowed = {row["canonical_name"] for row in rows if row["allowed_train"] == "1"}
    observed_excluded = {row["canonical_name"] for row in rows if row["allowed_train"] != "1"}
    assigned_allowed = {name for name, split in event_split.items() if split != "excluded"}
    assigned_excluded = {name for name, split in event_split.items() if split == "excluded"}
    if observed_allowed != assigned_allowed:
        raise ValueError(
            f"Eligible-event assignment mismatch; missing={sorted(observed_allowed - assigned_allowed)}, "
            f"unknown={sorted(assigned_allowed - observed_allowed)}"
        )
    if observed_excluded != assigned_excluded:
        raise ValueError(
            f"Excluded-event assignment mismatch; missing={sorted(observed_excluded - assigned_excluded)}, "
            f"unknown={sorted(assigned_excluded - observed_excluded)}"
        )

    seen_ids: set[str] = set()
    for row in rows:
        if row["sample_id"] in seen_ids:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        seen_ids.add(row["sample_id"])
        row["split"] = event_split[row["canonical_name"]]
        if row["allowed_train"] == "0" and row["split"] != "excluded":
            raise ValueError(f"Disallowed sample assigned to an active split: {row['sample_id']}")

    by_id = {row["sample_id"]: row for row in rows}
    links, _ = read_csv(args.links)
    checked_links = 0
    for link in links:
        train_row = by_id.get(link["train_sample_id"])
        tier3_row = by_id.get(link["tier3_sample_id"])
        if train_row is None or tier3_row is None:
            raise ValueError(f"Duplicate link refers to a missing parent: {link}")
        if train_row["split"] != tier3_row["split"]:
            raise ValueError(
                f"Exact duplicate leakage across split boundary: {link['train_sample_id']} "
                f"({train_row['split']}) vs {link['tier3_sample_id']} ({tier3_row['split']})"
            )
        checked_links += 1

    event_counts: Counter[tuple[str, str]] = Counter()
    source_counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        event_counts[(row["split"], row["canonical_name"])] += 1
        source_counts[(row["split"], row["source"], row["source_subset"])] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary_out = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary_out.replace(args.out)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "unit": config["unit"],
        "rows": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "source_counts": [
            {"split": split, "source": source, "source_subset": subset, "parents": count}
            for (split, source, subset), count in sorted(source_counts.items())
        ],
        "event_counts": [
            {"split": split, "canonical_name": event, "parents": count}
            for (split, event), count in sorted(event_counts.items())
        ],
        "exact_duplicate_links_checked": checked_links,
        "near_duplicate_check": "pending",
        "external_target_policy": "ida-BD is outside the parent training manifest; EBD Ida is excluded",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_report.replace(args.report)
    print(f"Wrote {len(rows)} rows; split counts={dict(Counter(row['split'] for row in rows))}; exact links={checked_links}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
