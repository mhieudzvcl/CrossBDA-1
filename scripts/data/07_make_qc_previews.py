"""Create deterministic, event-stratified 512px visual QA previews."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {
    1: (255, 255, 255),
    2: (255, 230, 0),
    3: (255, 128, 0),
    4: (255, 0, 0),
    255: (160, 0, 200),
}


def overlay(image: Image.Image, mask: Image.Image, localization: bool = False) -> Image.Image:
    base = image.convert("RGBA")
    rgba = Image.new("RGBA", image.size, (0, 0, 0, 0))
    source = mask.convert("L")
    pixels = []
    for value in source.getdata():
        if localization:
            pixels.append((255, 0, 0, 105) if value else (0, 0, 0, 0))
        else:
            color = COLORS.get(value)
            pixels.append((*color, 130) if color else (0, 0, 0, 0))
    rgba.putdata(pixels)
    return Image.alpha_composite(base, rgba).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chips", type=Path, default=Path("data/processed/512/chips.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qc-previews"))
    parser.add_argument("--per-event", type=int, default=3)
    parser.add_argument("--seed", type=str, default="2026")
    args = parser.parse_args()

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    with args.chips.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] in {"train", "validation", "test"}:
                groups[(row["split"], row["source"], row["event_id"])].append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, str]] = []
    for (split, source, event_id), rows in sorted(groups.items()):
        rows.sort(key=lambda row: hashlib.sha256(f"{args.seed}:{row['chip_id']}".encode()).digest())
        selected.extend(rows[: args.per_event])

    for row in selected:
        with Image.open(row["pre_image"]) as image:
            pre = image.convert("RGB")
        with Image.open(row["post_image"]) as image:
            post = image.convert("RGB")
        with Image.open(row["post_localization_mask"]) as image:
            loc = image.copy()
        with Image.open(row["post_damage_mask"]) as image:
            damage = image.copy()
        panels = [
            ("Pre-disaster", pre),
            ("Post-disaster", post),
            ("Post building mask overlay", overlay(post, loc, localization=True)),
            ("Post damage classes", overlay(post, damage)),
        ]
        canvas = Image.new("RGB", (1024, 1024), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (title, panel) in enumerate(panels):
            x = (index % 2) * 512
            y = (index // 2) * 512
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + 240, y + 20), fill=(0, 0, 0))
            draw.text((x + 5, y + 3), title, fill=(255, 255, 255))
        output = args.out / f"{row['split']}_{row['source']}_{row['event_id']}_{row['chip_id']}.jpg"
        canvas.save(output, quality=92)

    summary = {
        "input_chips": str(args.chips),
        "seed": args.seed,
        "per_event": args.per_event,
        "event_groups": len(groups),
        "previews_written": len(selected),
        "groups": [
            {"split": split, "source": source, "event_id": event_id, "chips": len(rows)}
            for (split, source, event_id), rows in sorted(groups.items())
        ],
    }
    (args.out / "preview_index.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} previews across {len(groups)} split/source/event groups to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
