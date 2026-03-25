# Handoff: Low-CPU Runtime and Calibration Fixes

- Author: Codex
- Date: 2026-03-25
- Branch: `justin`
- Current `HEAD`: `2d3cc8e` `justin signoff 3/24/26 PM improved multithreading performance, fixed calibration data leakage`

## What Changed Today

- Landed the low-CPU tabular runtime patch and the calibration leakage fix in `HEAD` `2d3cc8e`.
- Updated agent-facing status in [AGENTS.md](/home/exouser/Inspire-AKI-prediction/AGENTS.md) so the current continuation point and test status are no longer stale.

The main code changes already captured in `HEAD` are:

- grouped calibration CV on `op_id` in [src/inspire_aki/evaluation/calibration.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/evaluation/calibration.py), which fixes repeated-row leakage during isotonic calibration
- durable per-study tabular HPO outputs and resume under [src/inspire_aki/pipelines/tune.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/pipelines/tune.py) and `artifacts/tuning/tabular_studies/`
- narrow low-CPU tabular concurrency:
  - `svm` HPO fans out across `preop` / `intraop` / `combined`
  - `svm` training fans out across repeat tasks
  - `log_reg` remains serial with a moderate BLAS cap
- SVM training convergence aligned to legacy `tol=0.01` in [src/inspire_aki/models/tabular.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/models/tabular.py)
- targeted benchmark filters in [src/inspire_aki/benchmarking.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/benchmarking.py) and the CLI

## Documentation Status

The canonical docs are already updated for the important behavior changes:

- [README.md](/home/exouser/Inspire-AKI-prediction/README.md)
- [docs/current/pipeline.md](/home/exouser/Inspire-AKI-prediction/docs/current/pipeline.md)
- [docs/refactor/behavior_drift.md](/home/exouser/Inspire-AKI-prediction/docs/refactor/behavior_drift.md)
- [docs/legacy/07_manuscript_alignment.md](/home/exouser/Inspire-AKI-prediction/docs/legacy/07_manuscript_alignment.md)
- [docs/legacy/08_reproducibility_and_known_gaps.md](/home/exouser/Inspire-AKI-prediction/docs/legacy/08_reproducibility_and_known_gaps.md)

Those docs now explicitly cover:

- the grouped-calibration leakage fix
- the durable `tabular_studies` HPO resume behavior
- the intentionally narrow `svm`-only low-CPU concurrency policy
- the fact that evaluation is still non-nested

## Verification

- Full test suite passed: `.venv/bin/pytest -q`
- Result: `82 passed`

Residual warnings during tests:

- `joblib` / `fork()` deprecation warnings in multithreaded test paths
- existing `torch.load(..., weights_only=False)` future warning in the sequence bundle test

## Run Status / Next Step

- The prior real-data run should be treated as stale because tuning/training policy changed during today’s work.
- Resume from `tune`, not `train`.

Recommended command chain:

```bash
source .venv/bin/activate
ts=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p logs
set -euo pipefail

inspire-aki tune tabular --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_tune_tabular.log
inspire-aki tune sequence --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_tune_sequence.log
inspire-aki train tabular --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_train_tabular.log
inspire-aki train sequence --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_train_sequence.log
inspire-aki evaluate calibrate --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_evaluate_calibrate.log
inspire-aki evaluate metrics --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_evaluate_metrics.log
inspire-aki evaluate delong --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_evaluate_delong.log
inspire-aki evaluate dca --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_evaluate_dca.log
inspire-aki report manuscript --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_report_manuscript.log
```

Useful live monitors:

- `tail -f artifacts/logs/tune_tabular_progress.jsonl`
- `tail -f artifacts/logs/tune_sequence_progress.jsonl`
- `tail -f artifacts/logs/train_tabular_progress.jsonl`
- `tail -f artifacts/logs/train_sequence_progress.jsonl`
- `tail -f artifacts/logs/run_all_events.jsonl`

## Risks / Notes

- The calibration leakage fix is intentional drift from the legacy notebook behavior and should be treated as a correctness fix, not a bug.
- The current repeated-CV evaluation is still optimistic relative to nested CV; that limitation is documented but intentionally unchanged in this patch.
