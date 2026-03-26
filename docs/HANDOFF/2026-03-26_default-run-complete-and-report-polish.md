# Handoff: Default Run Complete and Report Polish

- Author: Codex
- Date: 2026-03-26
- Branch: current workspace
- Scope: completed real-data default run plus final manuscript-report styling/consort polish

## Current State

- The main default-config real-data path under `/media/volume/ncs_inspire_data/ncs_aki/artifacts/default` has completed end to end through manuscript reporting.
- The current manuscript-facing report layer now includes:
  - Graphviz-rendered consort output with direct orthogonal final split arrows
  - legacy-style fold/run performance tables with fixed model ordering
  - `ASA Rule` restricted to the preop section
  - subtle monochrome HTML performance-cell shading with bold best-in-column values
  - full `html` / `md` / `csv` table outputs and `png` / `svg` figure outputs

## Default Run Evidence

- [`train_tabular.json`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/default/manifests/train_tabular.json) timestamp: `2026-03-25 20:11:28 UTC`
- [`train_sequence.json`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/default/manifests/train_sequence.json) timestamp: `2026-03-25 22:46:56 UTC`
- [`evaluate_calibration.json`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/default/manifests/evaluate_calibration.json) timestamp: `2026-03-26 00:39:58 UTC`
- [`evaluate_metrics.json`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/default/manifests/evaluate_metrics.json) timestamp: `2026-03-26 00:58:21 UTC`
- [`evaluate_delong.json`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/default/manifests/evaluate_delong.json) timestamp: `2026-03-26 00:58:37 UTC`
- [`evaluate_dca.json`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/default/manifests/evaluate_dca.json) timestamp: `2026-03-26 01:00:20 UTC`
- [`report_manuscript.json`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/default/manifests/report_manuscript.json) timestamp: `2026-03-26 02:40:51 UTC`
- [`report_manuscript.json`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/default/manifests/report_manuscript.json) currently records `141` emitted outputs.

## Latest Report Polish

- [`src/inspire_aki/reporting/consort.py`](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/reporting/consort.py)
  - DOT is now the canonical consort layout source.
  - `consort.png` / `consort.svg` render directly from Graphviz.
  - The final split now uses direct orthogonal arrows from the final labeled cohort to `No postoperative AKI` and `Postoperative AKI`.
- [`src/inspire_aki/reporting/rendering.py`](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/reporting/rendering.py)
  - performance-table HTML cells now use gentle monochrome vertical washes, with column-wise intensity and bold best values.

## Fast Iteration Commands

Use these from the repo root:

```bash
source .venv/bin/activate
inspire-aki report consort --config configs/aki/default.yaml
inspire-aki report tables --config configs/aki/default.yaml
inspire-aki report manuscript --config configs/aki/default.yaml
```

Notes:

- `report consort` is the tightest loop for consort-layout work.
- `report tables` is the tightest loop for performance-table styling, but it still recomputes fold/run manuscript summaries from the saved prediction artifacts.
- Report stages overwrite the canonical files under `reports/` in place; deleting old files first is not required.

## Verification

- Full suite previously passed in this workspace: `./.venv/bin/pytest -q`
  - result: `95 passed`
- Latest focused regression check after the final consort/table styling changes:
  - `./.venv/bin/pytest tests/test_preprocess_and_reports.py -q`
  - result: `5 passed`
