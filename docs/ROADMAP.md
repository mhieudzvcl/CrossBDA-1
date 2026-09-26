# Implementation roadmap

## M0 - Repository foundation

- Establish package layout, configuration loading, and data-safe Git ignores.
- Add research scope and decision log.
- Do not add raw data or weights to Git.

## M1 - Inventory and canonical manifest

- [x] Inventory xBD train, xBD Tier3, and EBD source folders by filenames and filesystem metadata.
- [x] Reconcile observed source event names to a canonical event registry.
- [x] Build a parent-pair CSV manifest with explicit image and annotation paths.
- [x] Confirm no unpaired pre/post image names and report counts by event.
- [x] Reconcile the local 6,369 Tier3 count with published xBD counts; correct the blueprint's stale 5,600 figure.
- [x] Verify full raster readability and record dimensions (103,163 rasters fully decoded; zero corrupt files or scan errors).
- [x] Implement the typed, versioned Parquet exporter.
- [x] Apply the successful content-QC report to the parent manifest and export Parquet.
- Do not train before M1 is complete.

## M2 - Processed 512 dataset

- [x] Implement xBD polygon rasterization, paired 512 crops, and EBD mask normalization.
- [x] Run small xBD and EBD preprocessing pilots.
- [x] Implement deterministic event-stratified preview generation and per-class pixel counting.
- [ ] Process the full eligible manifest and inspect the generated previews/class statistics (paused by user; last logged checkpoint 1,792/24,745 parents, zero errors. Partial files are present under `data/processed/512`; no final `chips.csv` or report was written.)

## M3 - Leak-free splits

- [x] Build a versioned split by canonical disaster event and parent pair.
- [x] Validate event assignments, unique parent IDs, and all 426 exact-duplicate links.
- [x] Keep linked `socal-fire`/`woolsey-fire` samples together in test.
- [x] Keep EBD Hurricane Ida out of active splits for ida-BD evaluation.
- [x] Implement a cross-subset dHash near-duplicate candidate scanner.
- [ ] Run and manually review near-duplicate candidates; check exact source-file reuse beyond the known links (paused by user after 12,000/18,336 fingerprints; interrupted scan did not write a candidate report; manual review pending.)

## M4-M6 - Baseline, proposed fusion, and external evaluation

- Train and lock the Scale-MAE/FPN concatenation baseline before BG-MTDF ablations.
- Evaluate on a held-out event, then run fixed zero-shot ida-BD evaluation if source exclusion is clean.
- Consider additional modalities or Myanmar only after these milestones.
