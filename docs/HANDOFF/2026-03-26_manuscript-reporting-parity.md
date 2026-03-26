# Handoff: Manuscript Reporting Parity

- Author: Codex
- Date: 2026-03-26
- Branch: current workspace
- Scope: legacy-style manuscript reporting rebuilt on top of corrected refactor artifacts

## What Changed

- Added a dedicated manuscript rendering layer in [src/inspire_aki/reporting/rendering.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/reporting/rendering.py).
- Extended evaluation with:
  - [src/inspire_aki/evaluation/reclassification.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/evaluation/reclassification.py)
  - FDR-corrected DeLong outputs in [src/inspire_aki/evaluation/delong.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/evaluation/delong.py)
  - DCA bootstrap CI outputs in [src/inspire_aki/evaluation/dca.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/evaluation/dca.py)
- Added CLI and orchestration support for `inspire-aki evaluate reclassification` in [src/inspire_aki/cli.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/cli.py).
- Rebuilt manuscript reporting:
  - performance tables now come from fold/run aggregation, not pooled `metrics_summary.csv`
  - performance tables now keep a fixed manuscript model order, suppress `ASA Rule` outside the preop section, and add subtle monochrome column-wise gradients in HTML
  - report tables emit `html`, `md`, and `csv`
  - report figures emit `png` and `svg`
  - ROC / PR curves now use foldwise aggregation with uncertainty bands
  - DCA now emits per-model figures plus cross-dataset comparison figures
  - consort now emits table formats plus a top-down branched Graphviz `consort.dot`, and renders `consort.png` / `consort.svg` from that DOT source with explicit exclusion summaries and final AKI / non-AKI terminal nodes
- Extended explicit legacy export in [src/inspire_aki/io/compat.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/io/compat.py).
- Updated shipped config defaults and the canonical docs for the new report contract.
- Raised the shipped default runtime stage caps in [configs/aki/default.yaml](/home/exouser/Inspire-AKI-prediction/configs/aki/default.yaml) so heavy CPU-bound preprocessing, evaluation, report, and SHAP stages use more of the 32-core host while keeping the existing runtime reserve and `nested_blas_threads: 1`.
- Tightened [src/inspire_aki/runtime.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/runtime.py) so explicit worker/thread caps are clipped to each host's safe usable CPU budget instead of being treated as unsafe absolutes on smaller machines.

## Canonical Report Surface

Tables under `reports/tables/` now include:

- `performance_table.*`
- `performance_table_calibrated.*`
- `cohort_characteristics.*`
- `fill_rate_table.*`
- `consort_audit.*`
- `metrics_ci.*`
- `delong_raw.*`
- `delong_fdr_corrected.*`
- `reclassification_report.*`

Figures under `reports/figures/` now include:

- `consort.{png,svg}`
- `roc_curves_<dataset>.{png,svg}`
- `pr_curves_<dataset>.{png,svg}`
- `calibration_curves_<dataset>.{png,svg}`
- `dca_curve_<dataset>_<model>*.{png,svg}`
- `dca_datasource_comparison_<model>.{png,svg}`
- SHAP beeswarm figures in both formats

## Intentional Differences Preserved

- Grouped isotonic calibration on `op_id` is still in place.
- Manuscript tables now label `Balanced Accuracy` correctly rather than reusing the legacy `Accuracy` heading for that metric.
- Exact historical values can still drift because the reporting layer now sits on corrected refactor predictions rather than the legacy leakage-prone notebook path.

## Verification

- Full suite passed: `./.venv/bin/pytest -q`
- Result: `95 passed`

## Next Useful Real-Data Continuation

If real-data artifacts are already present and training/evaluation need rerun:

```bash
source .venv/bin/activate
inspire-aki evaluate calibrate --config configs/aki/default.yaml
inspire-aki evaluate metrics --config configs/aki/default.yaml
inspire-aki evaluate delong --config configs/aki/default.yaml
inspire-aki evaluate dca --config configs/aki/default.yaml
inspire-aki evaluate reclassification --config configs/aki/default.yaml
inspire-aki report manuscript --config configs/aki/default.yaml
inspire-aki compat export-legacy --config configs/aki/default.yaml
```
