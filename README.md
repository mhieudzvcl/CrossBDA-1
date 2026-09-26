# CrossBDA

Research code for cross-disaster building damage assessment from paired very-high-resolution optical imagery.

## Scope

- Input: 512 x 512 RGB pre-disaster and post-disaster image pairs.
- Targets: building localization (`0/1`) and damage (`0` background, `1` no damage, `2` minor, `3` major, `4` destroyed).
- Primary work: harmonize xBD/Tier3/EBD into a reproducible event-aware benchmark, establish leak-free splits, then compare a Scale-MAE/FPN baseline with temporal fusion variants.
- External evaluation: ida-BD is eligible only when all Hurricane Ida source data, including EBD Ida, is excluded from training and model selection.
- Myanmar 2025, Sentinel data, and other extensions remain out of scope until the core benchmark and baseline are stable.

Scale-MAE, FPN, and the source datasets are prior work. CrossBDA's research contribution must be supported by the benchmark protocol, model ablations, and cross-event evidence; do not describe source imagery as newly collected.

## Getting started

The project environment lives at `.envs/crossbda` (Python 3.9). Activate it in PowerShell:

```powershell
.\.envs\crossbda\Scripts\Activate.ps1
python -m pip install -e .
```

Place source datasets outside version control, then run inventory with the actual dataset roots:

```powershell
python scripts/data/00_inventory.py `
  --xbd-train "G:\My Drive\KhoaLuan_Data\data\xBD\train" `
  --xbd-tier3 "G:\My Drive\KhoaLuan_Data\data\xBD\Tier3\tier3\tier3" `
  --ebd "G:\My Drive\KhoaLuan_Data\data\EBD" `
  --metadata-only `
  --out data/reports/inventory_metadata.json
```

The metadata-only inventory avoids opening source imagery from the Google Drive mount. Full raster verification writes a separate content-QC report; `--workers` controls concurrent image reads (the local run uses 64). It does not alter source data.

After full raster QA reports zero corrupt files and scan errors, update the parent manifest and export its typed Parquet version:

```powershell
python scripts/data/05_apply_raster_qc.py
python scripts/data/06_build_splits.py
python scripts/data/03_export_manifest.py --require-content-qc
```

The 512px preprocessing script is `scripts/data/04_prepare_dataset.py`. It rasterizes xBD building polygons and crops paired images/masks together; EBD categorical masks are preserved and checked against classes `0..4` (with `255` reserved for ignored pixels). After processing the full manifest, `scripts/data/07_make_qc_previews.py` writes deterministic event-stratified image/mask previews under ignored `outputs/qc-previews/`.

After the content-QC run, `scripts/data/08_near_duplicate_candidates.py` can scan xBD train/Tier3 images for dHash matches. Those are review candidates, not automatic exclusions; do not change split membership until the images are inspected.

Once QC and the split manifest are finalized, run full preparation and generate previews:

```powershell
python scripts/data/08_near_duplicate_candidates.py
# Review data/reports/xbd_near_duplicates_candidates.json before changing splits.
python scripts/data/04_prepare_dataset.py
python scripts/data/07_make_qc_previews.py
```

## Current milestone

M1 is complete: 103,163 source rasters were fully decoded with zero corrupt files or scan errors; dimensions were applied to the parent manifest, the event-aware split manifest was rebuilt, and the typed Parquet export was written. Full 512px preprocessing and the xBD train/Tier3 near-duplicate scan are currently paused; status and partial outputs are recorded in [docs/ROADMAP.md](docs/ROADMAP.md). No final chip manifest or near-duplicate candidate report exists yet. Full preview/class-statistics review and model training remain incomplete. The source audit found cross-event duplicate pre-disaster images, so linked samples must remain together in future splits. See [docs/DATASET.md](docs/DATASET.md), [docs/ROADMAP.md](docs/ROADMAP.md), and [docs/DECISIONS.md](docs/DECISIONS.md).

## Data handling

Raw imagery, processed imagery, masks, reports generated from local data, and model weights are excluded from Git. Commit source code, configuration, documented manifests, and reports only when redistribution and privacy/licensing terms allow.
