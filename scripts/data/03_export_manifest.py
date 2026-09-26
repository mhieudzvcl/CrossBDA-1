"""Export the canonical CSV manifest to a typed, versioned Parquet file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


STRING_FIELDS = [
    "sample_id", "parent_id", "source", "source_subset", "source_pair_id",
    "event_id", "canonical_name", "hazard", "country", "pre_path", "post_path",
    "pre_annotation_path", "post_annotation_path", "qc_status", "split",
]

SCHEMA = pa.schema([
    *[pa.field(name, pa.string(), nullable=False) for name in STRING_FIELDS],
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("allowed_train", pa.bool_(), nullable=False),
    pa.field("external_target", pa.bool_(), nullable=False),
])


def optional_int(value: str) -> int | None:
    return int(value) if value else None


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/manifests/splits_v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/manifests/all_samples.parquet"))
    parser.add_argument("--require-content-qc", action="store_true")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Input manifest is empty: {args.input}")
    if args.require_content_qc:
        pending = sum(row["qc_status"] == "pending_content_qc" for row in rows)
        if pending:
            raise ValueError(f"Refusing Parquet export: {pending} rows still need content QC")
    records = []
    for row in rows:
        record = {name: row[name] for name in STRING_FIELDS}
        record.update({
            "width": optional_int(row["width"]),
            "height": optional_int(row["height"]),
            "allowed_train": as_bool(row["allowed_train"]),
            "external_target": as_bool(row["external_target"]),
        })
        records.append(record)

    metadata = {
        b"crossbda.manifest_version": b"1",
        b"crossbda.source_csv": str(args.input).encode("utf-8"),
    }
    schema = SCHEMA.with_metadata(metadata)
    table = pa.Table.from_pylist(records, schema=schema)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd", version="2.6")
    temporary.replace(args.out)
    print(f"Wrote {table.num_rows} rows to {args.out} ({args.out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
