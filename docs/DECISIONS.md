# Decision log

Initial decisions transcribed from the project blueprint. Record changes here with a reason instead of silently changing experiment assumptions.

| ID | Decision |
| --- | --- |
| D01 | Task: paired VHR optical building localization and four-level damage classification. |
| D02 | Working input size: 512 x 512 RGB. |
| D03 | Crop xBD 1024 x 1024 samples into deterministic 512 x 512 quadrants; do not resize for the main pipeline. |
| D04 | Scale-MAE ViT-Large pretrained checkpoint is a backbone initialization, not a contribution. |
| D05 | Main data contribution is harmonization, metadata, quality control, and a reproducible event-level protocol over public sources. |
| D06 | Exclude EBD Hurricane Ida whenever ida-BD is the external target. |
| D07 | Baseline fusion: concatenate pre/post features, followed by FPN and localization/damage heads. |
| D08 | Proposed direction: explicit temporal change features with learnable/building-guided gating; novelty claims await literature review. |
| D09 | Report xView2-style damage score, per-class F1, localization metrics, and the in-domain to unseen-event gap. |
| D10 | Myanmar curation is optional and deferred until the core milestones are complete. |
| D11 | Split protocol v1 is event-held-out: Hurricane Ian is validation; Pakistan flooding and the linked Socal/Woolsey fires are test; EBD Hurricane Ida is excluded. The assignment is explicit in `configs/data/splits_v1.yaml` and must be versioned if changed. |
| D12 | Scale-MAE receives per-image pre/post GSD from xBD annotation metadata. EBD files have no per-image GSD field, so B1 uses a clearly flagged 0.4 m nominal value (midpoint of the published 0.3-0.5 m range); keep provenance in the training manifest and revisit if finer metadata is obtained. |

The event registry was reconciled against the local inventories. Split v1 is an initial reproducible protocol choice for this rebuild, not a result claimed by the blueprint.
