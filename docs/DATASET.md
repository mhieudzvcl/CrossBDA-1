# Dataset inventory status

Inventory was collected from `G:/My Drive/KhoaLuan_Data/data` on 2026-09-26. All 103,163 raster files were decoded with Pillow; no corrupt rasters or scan errors were found. The report is `data/reports/inventory_content_qc.json`.

| Source | Parent pre/post pairs | Events | Raster files (images + auxiliary masks/targets) |
| --- | ---: | ---: | ---: |
| xBD train | 2,799 | 10 | 11,196 |
| xBD Tier3 | 6,369 | 9 | 19,107 |
| EBD | 18,215 | 12 | 72,860 |
| **Total** | **27,383** | **31** | **103,163** |

The inventory matched pre/post filenames for all 27,383 pairs and found the expected annotation filename for each pre/post image. Full content QC found no unpaired images, and confirmed dimensions of 1024x1024 for all xBD train/Tier3 rasters and 512x512 for all EBD rasters. `parent_samples.csv` now records these dimensions and `qc_status=raster_qc_pass`; `all_samples.parquet` contains the QC'd, split-assigned manifest.

## Tier3 count reconciliation

The supplied `xBD/Tier3/tier3/tier3/images` folder contains 6,369 pairs. This is consistent with published xBD counts: the RescueNet split table reports 6,369 Tier3 pairs, and an ICCV 2023 supplement reports 9,168 train-plus-Tier3 pairs, matching the local 2,799 + 6,369 total. The 5,600 Tier3 figure in the blueprint does not match these references and should be corrected before reuse ([RescueNet paper](https://openreview.net/pdf/bc30d84b497e256f77612ced40bb8d3389d363d1.pdf), [ICCV 2023 supplement](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zheng_Scalable_Multi-Temporal_Remote_ICCV_2023_supplemental.pdf)).

## Current files

- `data/reports/inventory_metadata.json`: per-source counts, event counts, and filesystem paths.
- `data/manifests/event_registry.csv`: xBD and EBD event metadata; EBD Ida has `allowed_train=0` and shares `canonical_name=hurricane-ida` with the ida-BD target entry.
- `data/manifests/parent_samples.csv`: parent-level image/annotation paths, dimensions, and raster QC state; Ida rows are retained for traceability but marked `allowed_train=0`.
- `data/manifests/all_samples.parquet`: typed versioned manifest with split assignments.

xBD-S12 and ida-BD have not been included in the training manifest. xBD-S12 remains deferred by the scope decision; ida-BD is reserved for external evaluation.

## Exact duplicate image links

The Tier1/train and Tier3 file names have no overlapping `source_pair_id` values. A SHA-256 audit of all cross-subset same-size candidates found **426 exact duplicate groups**. Every match is a pre-disaster image linking one `socal-fire` train pair to one `woolsey-fire` Tier3 pair; no post-disaster image matched. This links 426 parent pairs on each side.

Keep each linked pair on the same side of any train/validation/test boundary. For an event-held-out protocol, `socal-fire` and `woolsey-fire` must be assigned together or the linked examples must be excluded from one side. The pair-level links are recorded in `data/manifests/cross_event_duplicate_links.csv`; hash details are in `data/reports/xbd_exact_duplicates.json`.

## Event-held-out split v1

`data/manifests/splits_v1.csv` is generated from `configs/data/splits_v1.yaml`. It assigns Hurricane Ian to validation; Pakistan flooding and both linked California fire events to test; the remaining eligible events to train; and EBD Hurricane Ida to `excluded`. Counts are 13,863 train, 5,641 validation, 5,241 test, and 2,638 excluded parents. The builder checked all 426 exact-duplicate links and found no split crossing. Near-duplicate checking remains open.

The preprocessing pilot wrote eight xBD chips from two parents and two EBD chips from two parents with no errors. Pilot outputs are local artifacts under ignored `data/processed/`; they do not represent the full processed benchmark. The full preprocessing was paused by the user at the 1,792-parent logged checkpoint (24,745 eligible parents total), with no errors reported so far; partial chip files exist, but there is no final `chips.csv` or preparation report. The near-duplicate dHash scan was paused after fingerprinting 12,000 of 18,336 images, before it wrote a candidate report. Both full runs need to be restarted after work resumes; preprocessing currently reruns from the start and does not resume from a checkpoint.
